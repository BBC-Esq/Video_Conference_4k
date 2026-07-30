"""Chooses which echo canceller to use, best available first.

Three tiers, all presenting the same `process(mic, reference)` call:

  localvqe  a small neural canceller run through its own compiled library.
            The strongest of the three by a wide margin, and the only one that
            removes more echo without taking more of the local voice with it.
            Needs a library built for this platform, so it is not always here.
  webrtc    the canceller used by every browser, reached through the livekit
            package. Installs from a wheel on every platform, no build step.
  numpy     the one in this repository. Weakest at removing echo but the only
            one that leaves the near speaker completely untouched, and the only
            one that needs nothing installed at all.

Measured on identical signals, echo arriving 165 ms late, with a near-end
talker overlapping (echo removed / near voice kept, both dB):

  localvqe v1.2   +57.9 / -1.5      1.98 ms per 20 ms block
  webrtc AEC3     +27.3 / -3.8      0.12 ms
  numpy           +13.4 / +0.0      0.12 ms

The neural models run at 16 kHz, so audio is rate-converted around them and
nothing above 8 kHz is cancelled; for speech that is not a meaningful loss.
"""
import os
from typing import Optional

import numpy as np

from .aec import EchoCanceller
from ..utils.common import get_logger

logger = get_logger("EchoBackend")

NEURAL_RATE = 16000

LOCALVQE_DIR_ENV = "VIDEOCONFERENCE4K_LOCALVQE"
LOCALVQE_MODEL_ENV = "VIDEOCONFERENCE4K_LOCALVQE_MODEL"
DEFAULT_LOCALVQE_MODEL = "localvqe-v1.2-1.3M-f32.gguf"


def _lowpass(cutoff_ratio: float, taps: int = 97) -> np.ndarray:
    """Windowed-sinc low pass, so rate conversion needs nothing installed.

    Rate converting by dropping or repeating samples folds high frequencies
    back into the audible band, which a canceller then tries to model and
    cannot. The filter is what makes the conversion honest.
    """
    n = np.arange(taps) - (taps - 1) / 2.0
    h = 2.0 * cutoff_ratio * np.sinc(2.0 * cutoff_ratio * n)
    h *= np.hamming(taps)
    return h / h.sum()


class _Resampler:
    """Integer-ratio conversion between the caller's rate and the model's."""

    def __init__(self, ratio: int):
        self.ratio = int(ratio)
        if self.ratio > 1:
            # Cut just under the model's Nyquist before decimating.
            self._down_h = _lowpass(0.5 / self.ratio * 0.9)
            self._up_h = _lowpass(0.5 / self.ratio * 0.9) * self.ratio
            self._down_tail = np.zeros(len(self._down_h) - 1)
            self._up_tail = np.zeros(len(self._up_h) - 1)

    def down(self, x: np.ndarray) -> np.ndarray:
        if self.ratio == 1:
            return x
        padded = np.concatenate((self._down_tail, x))
        self._down_tail = padded[-(len(self._down_h) - 1):]
        y = np.convolve(padded, self._down_h, mode="valid")
        return y[::self.ratio]

    def up(self, x: np.ndarray) -> np.ndarray:
        if self.ratio == 1:
            return x
        spread = np.zeros(len(x) * self.ratio)
        spread[::self.ratio] = x
        padded = np.concatenate((self._up_tail, spread))
        self._up_tail = padded[-(len(self._up_h) - 1):]
        return np.convolve(padded, self._up_h, mode="valid")


class _NeuralBase:
    """Shared plumbing for the 16 kHz backends: rates, dtypes, hop alignment."""

    name = "neural"

    def __init__(self, block_size: int, sample_rate: int):
        self._n = int(block_size)
        self._sr = int(sample_rate)
        self._ratio = self._sr // NEURAL_RATE if self._sr >= NEURAL_RATE else 1
        if self._ratio * NEURAL_RATE != self._sr:
            raise ValueError(
                "{} needs a sample rate that is a whole multiple of {}".format(
                    self.name, NEURAL_RATE))
        self._rs_mic = _Resampler(self._ratio)
        self._rs_ref = _Resampler(self._ratio)
        self._rs_out = _Resampler(self._ratio)
        self._pend_mic = np.zeros(0)
        self._pend_ref = np.zeros(0)
        self._spill = np.zeros(0)
        self.erle_db = 0.0

    def _hop(self) -> int:
        raise NotImplementedError

    def _run_hop(self, mic_hop: np.ndarray, ref_hop: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def process(self, mic, reference):
        arr = np.asarray(mic)
        shape, dtype = arr.shape, arr.dtype
        want = arr.reshape(-1).shape[0]
        m = arr.reshape(-1).astype(np.float64)
        r = np.asarray(reference).reshape(-1).astype(np.float64)
        integral = np.issubdtype(dtype, np.integer)
        if integral:
            m = m / 32767.0
            r = r / 32767.0
        if r.shape[0] != m.shape[0]:
            r = np.resize(r, m.shape[0])

        self._pend_mic = np.concatenate((self._pend_mic, self._rs_mic.down(m)))
        self._pend_ref = np.concatenate((self._pend_ref, self._rs_ref.down(r)))

        hop = self._hop()
        produced = []
        while len(self._pend_mic) >= hop and len(self._pend_ref) >= hop:
            produced.append(self._run_hop(self._pend_mic[:hop], self._pend_ref[:hop]))
            self._pend_mic = self._pend_mic[hop:]
            self._pend_ref = self._pend_ref[hop:]

        if produced:
            self._spill = np.concatenate(
                (self._spill, self._rs_out.up(np.concatenate(produced))))

        out = self._spill[:want]
        self._spill = self._spill[want:]
        if len(out) < want:
            # Priming: the model has not emitted a full block yet.
            out = np.concatenate((out, np.zeros(want - len(out))))

        heard = float(np.dot(m, m))
        left = float(np.dot(out, out))
        if heard > 1e-9:
            sample = 10.0 * np.log10(heard / max(left, 1e-12))
            self.erle_db = round(sample if self.erle_db == 0.0
                                 else self.erle_db * 0.9 + sample * 0.1, 1)

        if integral:
            out = np.clip(out * 32767.0, -32768, 32767).astype(dtype)
        return out.reshape(shape).astype(dtype, copy=False)

    def reset(self):
        self._pend_mic = np.zeros(0)
        self._pend_ref = np.zeros(0)
        self._spill = np.zeros(0)


class LocalVQEBackend(_NeuralBase):
    """The neural canceller, through its compiled streaming library."""

    name = "localvqe"

    def __init__(self, block_size: int, sample_rate: int,
                 library_dir: Optional[str] = None, model: Optional[str] = None):
        super().__init__(block_size, sample_rate)
        import ctypes

        library_dir = library_dir or os.environ.get(LOCALVQE_DIR_ENV)
        if not library_dir:
            raise RuntimeError("{} is not set".format(LOCALVQE_DIR_ENV))
        dll = os.path.join(library_dir, "localvqe.dll")
        if not os.path.exists(dll):
            dll = os.path.join(library_dir, "liblocalvqe.so")
        if not os.path.exists(dll):
            raise FileNotFoundError("no localvqe library in {}".format(library_dir))

        model = model or os.environ.get(LOCALVQE_MODEL_ENV) or os.path.join(
            library_dir, DEFAULT_LOCALVQE_MODEL)
        if not os.path.exists(model):
            raise FileNotFoundError("no localvqe model at {}".format(model))

        if hasattr(os, "add_dll_directory"):
            self._dll_cookie = os.add_dll_directory(library_dir)
        self._lib = ctypes.CDLL(dll)
        c = ctypes.c_void_p
        self._lib.localvqe_new.restype = ctypes.c_size_t
        self._lib.localvqe_new.argtypes = [ctypes.c_char_p]
        self._lib.localvqe_free.argtypes = [ctypes.c_size_t]
        self._lib.localvqe_hop_length.restype = ctypes.c_int
        self._lib.localvqe_hop_length.argtypes = [ctypes.c_size_t]
        self._lib.localvqe_process_frame_f32.restype = ctypes.c_int
        self._lib.localvqe_process_frame_f32.argtypes = [
            ctypes.c_size_t, c, c, ctypes.c_int, c]
        self._lib.localvqe_reset.argtypes = [ctypes.c_size_t]

        self._ctx = self._lib.localvqe_new(model.encode("utf-8"))
        if not self._ctx:
            raise RuntimeError("localvqe refused the model {}".format(model))
        self._hop_n = self._lib.localvqe_hop_length(self._ctx)
        self._out = np.zeros(self._hop_n, dtype=np.float32)
        self.model_path = model

    def _hop(self) -> int:
        return self._hop_n

    def _run_hop(self, mic_hop, ref_hop):
        import ctypes
        m = np.ascontiguousarray(mic_hop, dtype=np.float32)
        r = np.ascontiguousarray(ref_hop, dtype=np.float32)
        rc = self._lib.localvqe_process_frame_f32(
            self._ctx,
            m.ctypes.data_as(ctypes.c_void_p),
            r.ctypes.data_as(ctypes.c_void_p),
            self._hop_n,
            self._out.ctypes.data_as(ctypes.c_void_p))
        if rc != 0:
            raise RuntimeError("localvqe frame failed ({})".format(rc))
        return self._out.astype(np.float64)

    def reset(self):
        super().reset()
        if getattr(self, "_ctx", 0):
            self._lib.localvqe_reset(self._ctx)

    def close(self):
        if getattr(self, "_ctx", 0):
            self._lib.localvqe_free(self._ctx)
            self._ctx = 0

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class WebRTCBackend(_NeuralBase):
    """The browser echo canceller, reached through the livekit package."""

    name = "webrtc"

    def __init__(self, block_size: int, sample_rate: int,
                 stream_delay_ms: int = 100, noise_suppression: bool = True):
        super().__init__(block_size, sample_rate)
        from livekit import rtc
        from livekit.rtc.apm import AudioProcessingModule

        self._rtc = rtc
        self._apm = AudioProcessingModule(
            echo_cancellation=True,
            noise_suppression=noise_suppression,
            high_pass_filter=True,
            auto_gain_control=False,
        )
        self._apm.set_stream_delay_ms(int(stream_delay_ms))
        # This module insists on exactly ten milliseconds per call.
        self._hop_n = NEURAL_RATE // 100

    def _hop(self) -> int:
        return self._hop_n

    def _run_hop(self, mic_hop, ref_hop):
        to16 = lambda a: np.clip(np.asarray(a) * 32767.0, -32768, 32767).astype(np.int16)
        self._apm.process_reverse_stream(
            self._rtc.AudioFrame(to16(ref_hop).tobytes(), NEURAL_RATE, 1, self._hop_n))
        cap = self._rtc.AudioFrame(to16(mic_hop).tobytes(), NEURAL_RATE, 1, self._hop_n)
        self._apm.process_stream(cap)
        return np.frombuffer(cap.data, dtype=np.int16).astype(np.float64) / 32767.0

    def set_stream_delay_ms(self, delay_ms: int) -> None:
        self._apm.set_stream_delay_ms(int(delay_ms))


def _numpy_backend(block_size, sample_rate, tail_ms, logging):
    ec = EchoCanceller(block_size=block_size, sample_rate=sample_rate,
                       tail_ms=tail_ms, logging=logging)
    ec.name = "numpy"
    return ec


ORDER = ("localvqe", "webrtc", "numpy")


def create_canceller(block_size: int, sample_rate: int, tail_ms: float,
                     backend: str = "auto", logging: bool = False,
                     stream_delay_ms: Optional[int] = None,
                     library_dir: Optional[str] = None,
                     model: Optional[str] = None):
    """Build the best canceller available, or the one asked for by name.

    `auto` walks the order above and takes the first that loads. Naming one
    explicitly raises if it cannot be had, so a deliberate choice never
    degrades quietly into something weaker.
    """
    wanted = (backend or "auto").strip().lower()
    if wanted not in ORDER and wanted != "auto":
        raise ValueError("unknown echo backend {!r}; expected one of {} or auto".format(
            backend, ", ".join(ORDER)))

    delay = int(stream_delay_ms if stream_delay_ms is not None else max(40.0, tail_ms * 0.6))
    candidates = ORDER if wanted == "auto" else (wanted,)

    for name in candidates:
        try:
            if name == "localvqe":
                ec = LocalVQEBackend(block_size, sample_rate,
                                     library_dir=library_dir, model=model)
            elif name == "webrtc":
                ec = WebRTCBackend(block_size, sample_rate, stream_delay_ms=delay)
            else:
                ec = _numpy_backend(block_size, sample_rate, tail_ms, logging)
            logger.info("Echo cancellation using the {} backend.".format(name))
            return ec
        except Exception as exc:
            if wanted != "auto":
                raise
            logging and logger.debug("{} backend unavailable: {}".format(name, exc))

    return _numpy_backend(block_size, sample_rate, tail_ms, logging)


def available_backends() -> dict:
    """Which backends this machine could actually use, for diagnosis."""
    found = {}
    for name in ORDER:
        try:
            ec = create_canceller(960, 48000, 200.0, backend=name)
            found[name] = True
            close = getattr(ec, "close", None)
            if callable(close):
                close()
        except Exception as exc:
            found[name] = str(exc)[:120]
    return found
