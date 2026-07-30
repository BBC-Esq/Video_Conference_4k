import numpy as np
import threading
import queue
import time
from typing import TypeVar, Optional, Callable, Union
from numpy.typing import NDArray

from ..utils.common import (
    get_logger,
    import_dependency_safe,
    log_version,
)
from .jitter import JitterBuffer
from .aec import EchoCanceller

sd = import_dependency_safe("sounddevice", error="silent")

logger = get_logger("AudioCapture")

T = TypeVar("T", bound="AudioCapture")

ROOM_REVERB_ALLOWANCE_MS = 80.0


class AudioCapture:

    def __init__(
        self,
        input_device: Optional[Union[int, str]] = None,
        output_device: Optional[Union[int, str]] = None,
        sample_rate: int = 48000,
        channels: int = 1,
        chunk_size: int = 960,
        dtype: str = "int16",
        enable_input: bool = True,
        enable_output: bool = True,
        output_jitter_ms: float = 0.0,
        echo_cancellation: bool = True,
        echo_tail_ms: Optional[float] = None,
        logging: bool = False,
        **options: dict
    ):
        self.__logging = logging if isinstance(logging, bool) else False

        log_version(logging=self.__logging)

        import_dependency_safe("sounddevice" if sd is None else "")

        self.__sample_rate = sample_rate
        self.__channels = channels
        self.__chunk_size = chunk_size
        self.__dtype = dtype
        self.__enable_input = enable_input
        self.__enable_output = enable_output

        self.__input_device = input_device
        self.__output_device = output_device

        self.__input_queue = queue.Queue(maxsize=100)
        self.__output_queue = queue.Queue(maxsize=100)
        self.__output_residual = None
        self.__subscribers = []
        self.__subscribers_lock = threading.Lock()

        self.__jitter = None
        self.__jitter_lock = threading.Lock()
        if enable_output and output_jitter_ms and output_jitter_ms > 0:
            self.__jitter = JitterBuffer(
                sample_rate=sample_rate,
                channels=channels,
                target_ms=output_jitter_ms,
                max_ms=output_jitter_ms + 120.0,
                dtype=dtype,
            )

        self.__input_stream = None
        self.__output_stream = None
        self.__duplex_stream = None

        self.__aec = None
        self.__aec_failures = 0
        self.__echo_cancellation = bool(echo_cancellation) and enable_input and enable_output
        self.__echo_tail_ms = echo_tail_ms

        self.__terminate = threading.Event()
        self.__lifecycle_lock = threading.RLock()
        self.__is_running = False

        self.__on_audio_callback = None
        self.__callback_queue = queue.Queue(maxsize=16)
        self.__callback_thread = None
        self.__callback_drops = 0

        options = {str(k).strip(): v for k, v in options.items()}

        if "latency" in options:
            self.__latency = options["latency"]
        else:
            self.__latency = "low"

        if "blocksize" in options:
            self.__chunk_size = options["blocksize"]

        if self.__echo_cancellation and not self.__chunk_size:
            logger.warning(
                "Echo cancellation needs a fixed block size and this stream has "
                "none; it will be disabled."
            )
            self.__echo_cancellation = False

        self.__logging and logger.debug(
            "AudioCapture initialized with sample_rate={}, channels={}, chunk_size={}, dtype={}".format(
                self.__sample_rate, self.__channels, self.__chunk_size, self.__dtype
            )
        )

    @property
    def sample_rate(self) -> int:
        return self.__sample_rate

    @property
    def channels(self) -> int:
        return self.__channels

    @property
    def chunk_size(self) -> int:
        return self.__chunk_size

    @property
    def dtype(self) -> str:
        return self.__dtype

    @property
    def is_running(self) -> bool:
        return self.__is_running

    def playout_pts_ns(self):
        if self.__jitter is None:
            return None
        with self.__jitter_lock:
            return self.__jitter.playout_pts_ns

    def jitter_depth_ms(self) -> Optional[float]:
        if self.__jitter is None:
            return None
        with self.__jitter_lock:
            return round(self.__jitter.depth_ms, 1)

    @property
    def echo_cancellation(self) -> bool:
        return self.__aec is not None

    @property
    def echo_reduction_db(self) -> Optional[float]:
        """How much echo is currently being removed. None when not cancelling."""
        return self.__aec.erle_db if self.__aec is not None else None

    @property
    def duplex(self) -> bool:
        """Whether capture and playback share one device clock."""
        return self.__duplex_stream is not None

    def jitter_underruns(self) -> int:
        """Playback callbacks that were served silence because audio had not arrived."""
        if self.__jitter is None:
            return 0
        with self.__jitter_lock:
            return self.__jitter.underruns

    @staticmethod
    def get_devices() -> dict:
        """Enumerate audio devices, naming the host API each one belongs to.

        The same headset appears once per host API with different capabilities,
        so the name alone is not enough to pick the right one.
        """
        import_dependency_safe("sounddevice" if sd is None else "")
        try:
            devices = sd.query_devices()
        except Exception as e:
            logger.error("Could not enumerate audio devices: {}".format(e))
            return {"input": [], "output": []}

        try:
            hostapis = sd.query_hostapis()
        except Exception:
            hostapis = []

        try:
            default_input, default_output = sd.default.device
        except Exception:
            default_input, default_output = None, None

        def describe(index, dev, channels_key):
            api_index = dev.get("hostapi")
            api = ""
            if isinstance(api_index, int) and 0 <= api_index < len(hostapis):
                api = hostapis[api_index].get("name", "")
            return {
                "index": index,
                "name": dev.get("name", ""),
                "channels": dev[channels_key],
                "hostapi": api,
                "default_samplerate": dev.get("default_samplerate"),
            }

        input_devices = []
        output_devices = []
        for i, dev in enumerate(devices):
            try:
                if dev["max_input_channels"] > 0:
                    entry = describe(i, dev, "max_input_channels")
                    entry["is_default"] = (i == default_input)
                    input_devices.append(entry)
                if dev["max_output_channels"] > 0:
                    entry = describe(i, dev, "max_output_channels")
                    entry["is_default"] = (i == default_output)
                    output_devices.append(entry)
            except Exception:
                continue
        return {"input": input_devices, "output": output_devices}

    def set_audio_callback(self, callback: Callable[[NDArray], None]) -> None:
        if callable(callback):
            self.__on_audio_callback = callback
            self.__logging and logger.debug("Audio callback registered.")
        else:
            logger.warning("Invalid callback provided. Must be callable.")

    def __input_callback(self, indata, frames, time_info, status):
        if status:
            self.__logging and logger.warning("Input status: {}".format(status))
        self.__handle_input(indata, frames)

    def __duplex_callback(self, indata, outdata, frames, time_info, status):
        """Capture and playback in one callback, sharing one device clock.

        Two separate streams each run on their own clock and drift apart, which
        leaves no fixed relationship between what was played and what the
        microphone heard a moment later. Echo cancellation needs exactly that
        relationship, so the two directions are handled together here: outdata
        is filled first and then handed to the canceller as the reference for
        what it is about to hear come back.
        """
        if status:
            self.__logging and logger.warning("Duplex status: {}".format(status))

        self.__fill_output(outdata, frames)

        mic = indata
        if self.__aec is not None:
            try:
                mic = self.__aec.process(indata, outdata)
            except Exception as e:
                self.__aec_failures += 1
                if self.__aec_failures == 1:
                    logger.error("Echo cancellation failed, passing the microphone "
                                 "through untouched: {}".format(e))
                mic = indata

        self.__handle_input(mic, frames)

    def __handle_input(self, indata, frames):
        if not self.__terminate.is_set():
            audio_data = np.asarray(indata).copy()
            pts_ns = time.perf_counter_ns() - int(frames / self.__sample_rate * 1e9)
            try:
                self.__input_queue.put_nowait(audio_data)
            except queue.Full:
                pass
            payload = (audio_data, pts_ns)
            with self.__subscribers_lock:
                subscribers = list(self.__subscribers)
            for subscriber in subscribers:
                try:
                    subscriber.put_nowait(payload)
                except queue.Full:
                    try:
                        subscriber.get_nowait()
                        subscriber.put_nowait(payload)
                    except (queue.Empty, queue.Full):
                        pass
            if self.__on_audio_callback is not None:
                try:
                    self.__callback_queue.put_nowait(audio_data)
                except queue.Full:
                    self.__callback_drops += 1

    def __output_callback(self, outdata, frames, time_info, status):
        if status:
            self.__logging and logger.warning("Output status: {}".format(status))
        self.__fill_output(outdata, frames)

    def __fill_output(self, outdata, frames):
        out_channels = outdata.shape[1] if outdata.ndim > 1 else 1
        needed = outdata.shape[0]

        if self.__jitter is not None:
            with self.__jitter_lock:
                samples = self.__jitter.pop(needed)
            outdata[:] = self.__adapt_channels(samples, out_channels)[:needed]
            if np.issubdtype(outdata.dtype, np.floating):
                np.nan_to_num(outdata, copy=False, nan=0.0, posinf=1.0, neginf=-1.0)
                np.clip(outdata, -1.0, 1.0, out=outdata)
            return

        filled = 0

        while filled < needed:
            source = self.__output_residual
            if source is None or source.shape[0] == 0:
                try:
                    source = self.__adapt_channels(self.__output_queue.get_nowait(), out_channels)
                except queue.Empty:
                    break
                self.__output_residual = None

            take = min(needed - filled, source.shape[0])
            outdata[filled:filled + take] = source[:take]
            filled += take
            self.__output_residual = source[take:] if take < source.shape[0] else None

        if filled < needed:
            outdata[filled:] = 0

        if np.issubdtype(outdata.dtype, np.floating):
            np.nan_to_num(outdata, copy=False, nan=0.0, posinf=1.0, neginf=-1.0)
            np.clip(outdata, -1.0, 1.0, out=outdata)

    def __adapt_channels(self, data: NDArray, out_channels: int) -> NDArray:
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        in_channels = data.shape[1]
        if in_channels == out_channels:
            return data

        if in_channels == 1 and out_channels > 1:
            return np.tile(data, (1, out_channels))
        if in_channels > 1 and out_channels == 1:
            return data.mean(axis=1, keepdims=True).astype(data.dtype)
        if in_channels > out_channels:
            return data[:, :out_channels]
        return np.pad(data, ((0, 0), (0, out_channels - in_channels)), mode="constant")

    def __callback_worker(self) -> None:
        """Run the user's audio callback away from the PortAudio thread.

        Anything slow or blocking in that callback would otherwise stall the
        device thread and glitch capture for every subscriber.
        """
        while not self.__terminate.is_set():
            try:
                audio_data = self.__callback_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            callback = self.__on_audio_callback
            if callback is None:
                continue
            try:
                callback(audio_data)
            except Exception as e:
                logger.error("Error in audio callback: {}".format(e))

    @property
    def callback_drops(self) -> int:
        return self.__callback_drops

    def start(self) -> T:
        with self.__lifecycle_lock:
            return self.__start_locked()

    def __build_canceller(self, stream) -> None:
        """Size the echo filter from the delay this device actually reports.

        The filter can only remove echo that arrives within its reach, and how
        far behind the echo is depends entirely on the hardware: a few
        milliseconds on a good interface, a hundred on ordinary Windows
        playback, several hundred over Bluetooth. A fixed guess is wrong on most
        machines, so the device is asked, and only the room reverberation is
        left as an assumption. Reach is not free either, since a longer filter
        spreads its learning thinner, which is why this is measured rather than
        simply made generous.
        """
        if not self.__echo_cancellation:
            return

        tail_ms = self.__echo_tail_ms
        if tail_ms is None:
            try:
                latency_in, latency_out = stream.latency
                device_ms = (float(latency_in) + float(latency_out)) * 1000.0
            except Exception:
                device_ms = 120.0
            tail_ms = min(400.0, max(120.0, device_ms + ROOM_REVERB_ALLOWANCE_MS))

        self.__aec = EchoCanceller(
            block_size=self.__chunk_size,
            sample_rate=self.__sample_rate,
            tail_ms=tail_ms,
            logging=self.__logging,
        )
        self.__logging and logger.debug(
            "Echo canceller sized to {:.0f} ms for this device.".format(tail_ms)
        )

    def __start_callback_worker(self) -> None:
        self.__callback_thread = threading.Thread(
            target=self.__callback_worker, daemon=True, name="AudioUserCallback"
        )
        self.__callback_thread.start()

    def __start_locked(self) -> T:
        if self.__is_running:
            self.__logging and logger.warning("AudioCapture is already running.")
            return self

        self.__terminate.clear()

        if self.__enable_input and self.__enable_output:
            try:
                self.__duplex_stream = sd.Stream(
                    device=(self.__input_device, self.__output_device),
                    samplerate=self.__sample_rate,
                    channels=self.__channels,
                    dtype=self.__dtype,
                    blocksize=self.__chunk_size,
                    latency=self.__latency,
                    callback=self.__duplex_callback,
                )
                self.__duplex_stream.start()
                self.__build_canceller(self.__duplex_stream)
                self.__logging and logger.debug(
                    "Duplex stream started; echo cancellation {}.".format(
                        "on" if self.__aec is not None else "off"
                    )
                )
                self.__start_callback_worker()
                self.__is_running = True
                return self
            except Exception as e:
                # A single handle cannot always span two different devices, and
                # exclusive-mode drivers may refuse one outright. Two streams
                # still carry a call; they only cost the shared clock that echo
                # cancellation depends on, so it is switched off rather than
                # left running against a reference it cannot trust.
                logger.warning(
                    "Could not open one duplex stream ({}); falling back to separate "
                    "capture and playback streams. Echo cancellation is unavailable "
                    "in that mode.".format(e)
                )
                self.__duplex_stream = None
                self.__aec = None

        if self.__enable_input:
            try:
                self.__input_stream = sd.InputStream(
                    device=self.__input_device,
                    samplerate=self.__sample_rate,
                    channels=self.__channels,
                    dtype=self.__dtype,
                    blocksize=self.__chunk_size,
                    latency=self.__latency,
                    callback=self.__input_callback,
                )
                self.__input_stream.start()
                self.__logging and logger.debug("Input stream started.")
            except Exception as e:
                logger.error("Failed to start input stream: {}".format(e))
                raise

        if self.__enable_output:
            try:
                self.__output_stream = sd.OutputStream(
                    device=self.__output_device,
                    samplerate=self.__sample_rate,
                    channels=self.__channels,
                    dtype=self.__dtype,
                    blocksize=self.__chunk_size,
                    latency=self.__latency,
                    callback=self.__output_callback,
                )
                self.__output_stream.start()
                self.__logging and logger.debug("Output stream started.")
            except Exception as e:
                logger.error("Failed to start output stream: {}".format(e))
                if self.__input_stream is not None:
                    try:
                        self.__input_stream.stop()
                        self.__input_stream.close()
                    except Exception:
                        pass
                    self.__input_stream = None
                self.__output_stream = None
                raise

        self.__start_callback_worker()

        self.__is_running = True
        self.__logging and logger.debug("AudioCapture started successfully.")
        return self

    def subscribe(self, maxsize: int = 100) -> "queue.Queue":
        subscriber = queue.Queue(maxsize=maxsize)
        with self.__subscribers_lock:
            self.__subscribers.append(subscriber)
        return subscriber

    def unsubscribe(self, subscriber: "queue.Queue") -> None:
        with self.__subscribers_lock:
            if subscriber in self.__subscribers:
                self.__subscribers.remove(subscriber)

    def read(self, timeout: Optional[float] = None) -> Optional[NDArray]:
        if not self.__enable_input:
            logger.warning("Input is not enabled.")
            return None
        try:
            return self.__input_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def write_timed(self, audio_data: NDArray, pts_ns: int) -> bool:
        if not self.__enable_output:
            logger.warning("Output is not enabled.")
            return False
        if not isinstance(audio_data, np.ndarray):
            logger.warning("Invalid audio data type. Expected numpy array.")
            return False
        if self.__jitter is not None:
            with self.__jitter_lock:
                self.__jitter.insert(audio_data, pts_ns)
            return True
        return self.write(audio_data)

    def write(self, audio_data: NDArray) -> bool:
        if not self.__enable_output:
            logger.warning("Output is not enabled.")
            return False
        if not isinstance(audio_data, np.ndarray):
            logger.warning("Invalid audio data type. Expected numpy array.")
            return False
        if self.__jitter is not None:
            with self.__jitter_lock:
                self.__jitter.insert(audio_data, None)
            return True
        try:
            self.__output_queue.put_nowait(audio_data)
            return True
        except queue.Full:
            pass

        try:
            self.__output_queue.get_nowait()
        except queue.Empty:
            pass
        try:
            self.__output_queue.put_nowait(audio_data)
            self.__logging and logger.debug(
                "Playback buffer full; discarded the oldest chunk to keep latency bounded."
            )
            return True
        except queue.Full:
            return False

    def clear_output_queue(self) -> None:
        if self.__jitter is not None:
            with self.__jitter_lock:
                self.__jitter.reset()
        while not self.__output_queue.empty():
            try:
                self.__output_queue.get_nowait()
            except queue.Empty:
                break

    def clear_input_queue(self) -> None:
        while not self.__input_queue.empty():
            try:
                self.__input_queue.get_nowait()
            except queue.Empty:
                break

    def stop(self) -> None:
        with self.__lifecycle_lock:
            self.__stop_locked()

    def __stop_locked(self) -> None:
        self.__logging and logger.debug("Stopping AudioCapture.")
        self.__terminate.set()
        self.__is_running = False

        if self.__duplex_stream is not None:
            try:
                self.__duplex_stream.stop()
                self.__duplex_stream.close()
                self.__logging and logger.debug("Duplex stream stopped.")
            except Exception as e:
                logger.error("Error stopping duplex stream: {}".format(e))
            self.__duplex_stream = None

        if self.__input_stream is not None:
            try:
                self.__input_stream.stop()
                self.__input_stream.close()
                self.__logging and logger.debug("Input stream stopped.")
            except Exception as e:
                logger.error("Error stopping input stream: {}".format(e))
            self.__input_stream = None

        if self.__output_stream is not None:
            try:
                self.__output_stream.stop()
                self.__output_stream.close()
                self.__logging and logger.debug("Output stream stopped.")
            except Exception as e:
                logger.error("Error stopping output stream: {}".format(e))
            self.__output_stream = None

        if self.__callback_thread is not None:
            self.__callback_thread.join(timeout=1.0)
            self.__callback_thread = None

        self.clear_input_queue()
        self.clear_output_queue()

        self.__logging and logger.debug("AudioCapture stopped successfully.")