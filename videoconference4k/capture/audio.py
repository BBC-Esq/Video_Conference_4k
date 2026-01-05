import numpy as np
import logging as log
import threading
import queue
from typing import TypeVar, Optional, Callable, Union
from numpy.typing import NDArray

from ..utils.common import (
    logger_handler,
    import_dependency_safe,
    log_version,
)

sd = import_dependency_safe("sounddevice", error="silent")

logger = log.getLogger("AudioCapture")
logger.propagate = False
logger.addHandler(logger_handler())
logger.setLevel(log.DEBUG)

T = TypeVar("T", bound="AudioCapture")


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

        self.__input_stream = None
        self.__output_stream = None

        self.__terminate = threading.Event()
        self.__is_running = False

        self.__on_audio_callback = None

        options = {str(k).strip(): v for k, v in options.items()}

        if "latency" in options:
            self.__latency = options["latency"]
        else:
            self.__latency = "low"

        if "blocksize" in options:
            self.__chunk_size = options["blocksize"]

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

    @staticmethod
    def get_devices() -> dict:
        import_dependency_safe("sounddevice" if sd is None else "")
        devices = sd.query_devices()
        input_devices = []
        output_devices = []
        for i, dev in enumerate(devices):
            if dev["max_input_channels"] > 0:
                input_devices.append({"index": i, "name": dev["name"], "channels": dev["max_input_channels"]})
            if dev["max_output_channels"] > 0:
                output_devices.append({"index": i, "name": dev["name"], "channels": dev["max_output_channels"]})
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
        if not self.__terminate.is_set():
            audio_data = indata.copy()
            try:
                self.__input_queue.put_nowait(audio_data)
            except queue.Full:
                pass
            if self.__on_audio_callback is not None:
                try:
                    self.__on_audio_callback(audio_data)
                except Exception as e:
                    logger.error("Error in audio callback: {}".format(e))

    def __output_callback(self, outdata, frames, time_info, status):
        if status:
            self.__logging and logger.warning("Output status: {}".format(status))
        try:
            data = self.__output_queue.get_nowait()
            if data.ndim == 1:
                data = data.reshape(-1, 1)
            samples_to_copy = min(data.shape[0], outdata.shape[0])
            outdata[:samples_to_copy] = data[:samples_to_copy]
            if samples_to_copy < outdata.shape[0]:
                outdata[samples_to_copy:] = 0
        except queue.Empty:
            outdata.fill(0)

    def start(self) -> T:
        if self.__is_running:
            self.__logging and logger.warning("AudioCapture is already running.")
            return self

        self.__terminate.clear()

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
                raise

        self.__is_running = True
        self.__logging and logger.debug("AudioCapture started successfully.")
        return self

    def read(self, timeout: Optional[float] = None) -> Optional[NDArray]:
        if not self.__enable_input:
            logger.warning("Input is not enabled.")
            return None
        try:
            return self.__input_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def write(self, audio_data: NDArray) -> bool:
        if not self.__enable_output:
            logger.warning("Output is not enabled.")
            return False
        if not isinstance(audio_data, np.ndarray):
            logger.warning("Invalid audio data type. Expected numpy array.")
            return False
        try:
            self.__output_queue.put_nowait(audio_data)
            return True
        except queue.Full:
            return False

    def clear_output_queue(self) -> None:
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
        self.__logging and logger.debug("Stopping AudioCapture.")
        self.__terminate.set()
        self.__is_running = False

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

        self.clear_input_queue()
        self.clear_output_queue()

        self.__logging and logger.debug("AudioCapture stopped successfully.")