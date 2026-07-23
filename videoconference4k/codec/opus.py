import numpy as np
from typing import Optional
from numpy.typing import NDArray

from ..utils.common import get_logger, import_dependency_safe

av = import_dependency_safe("av", error="silent")

logger = get_logger("OpusCodec")


def has_opus_codec() -> bool:
    if av is None:
        return False
    try:
        av.codec.Codec("libopus", "w")
        av.codec.Codec("libopus", "r")
        return True
    except Exception:
        return False


class OpusEncoder:

    def __init__(
        self,
        sample_rate: int = 48000,
        channels: int = 1,
        bitrate: int = 32000,
        application: str = "voip",
        fec: bool = True,
        packet_loss: int = 10,
        logging: bool = False,
    ):
        import_dependency_safe("av" if av is None else "", error="raise")
        self._sample_rate = sample_rate
        self._channels = channels
        self._logging = logging
        self._pts = 0
        self._layout = "mono" if channels == 1 else "stereo"

        self._ctx = av.AudioCodecContext.create("libopus", "w")
        self._ctx.sample_rate = sample_rate
        self._ctx.format = "s16"
        self._ctx.layout = self._layout
        self._ctx.bit_rate = bitrate

        options = {"application": application, "vbr": "on"}
        if fec:
            options["fec"] = "1"
            options["packet_loss"] = str(packet_loss)
        self._ctx.options = options
        self._ctx.open()

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def channels(self) -> int:
        return self._channels

    def encode(self, pcm: NDArray) -> bytes:
        if self._ctx is None:
            return b""
        if pcm.dtype != np.int16:
            pcm = pcm.astype(np.int16)
        interleaved = np.ascontiguousarray(pcm.reshape(1, -1))

        frame = av.AudioFrame.from_ndarray(interleaved, format="s16", layout=self._layout)
        frame.sample_rate = self._sample_rate
        frame.pts = self._pts
        self._pts += interleaved.shape[1] // self._channels

        try:
            packets = self._ctx.encode(frame)
        except Exception as e:
            self._logging and logger.error("Opus encode error: {}".format(e))
            return b""
        return b"".join(bytes(p) for p in packets)

    def flush(self) -> bytes:
        if self._ctx is None:
            return b""
        try:
            packets = self._ctx.encode(None)
        except Exception:
            return b""
        return b"".join(bytes(p) for p in packets)

    def close(self) -> None:
        self._ctx = None


class OpusDecoder:

    def __init__(
        self,
        sample_rate: int = 48000,
        channels: int = 1,
        logging: bool = False,
    ):
        import_dependency_safe("av" if av is None else "", error="raise")
        self._sample_rate = sample_rate
        self._channels = channels
        self._logging = logging
        self._layout = "mono" if channels == 1 else "stereo"

        self._ctx = av.AudioCodecContext.create("libopus", "r")
        self._ctx.sample_rate = sample_rate
        self._ctx.format = "s16"
        self._ctx.layout = self._layout
        self._ctx.open()

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def channels(self) -> int:
        return self._channels

    def decode(self, data: bytes) -> Optional[NDArray]:
        if self._ctx is None or not data:
            return None
        try:
            packet = av.Packet(data)
            frames = self._ctx.decode(packet)
        except Exception as e:
            self._logging and logger.error("Opus decode error: {}".format(e))
            return None

        chunks = []
        for frame in frames:
            arr = frame.to_ndarray()
            chunks.append(arr.reshape(-1, self._channels))
        if not chunks:
            return None
        return np.concatenate(chunks, axis=0)

    def close(self) -> None:
        self._ctx = None
