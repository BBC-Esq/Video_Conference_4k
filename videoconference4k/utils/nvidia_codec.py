import numpy as np
import logging as log
from typing import Optional
from numpy.typing import NDArray

from .common import logger_handler, import_dependency_safe, set_cuda_paths

set_cuda_paths()

nvc = import_dependency_safe("PyNvVideoCodec", error="silent")

logger = log.getLogger("NvidiaCodec")
logger.propagate = False
logger.addHandler(logger_handler())
logger.setLevel(log.DEBUG)


def has_nvidia_codec() -> bool:
    if nvc is None:
        return False
    try:
        encoder = nvc.CreateEncoder(256, 256, "NV12", True, codec="h264")
        del encoder
        return True
    except Exception:
        return False


def get_nvidia_info() -> dict:
    info = {
        "available": False,
        "pynvvideocodec_installed": nvc is not None,
        "encoder_functional": False,
    }

    if nvc is None:
        return info

    info["available"] = True

    try:
        encoder = nvc.CreateEncoder(256, 256, "NV12", True, codec="h264")
        del encoder
        info["encoder_functional"] = True
    except Exception as e:
        info["encoder_error"] = str(e)

    return info


def bgr_to_nv12(bgr_frame: NDArray) -> NDArray:
    import cv2

    height, width = bgr_frame.shape[:2]

    yuv_i420 = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2YUV_I420)

    y = yuv_i420[:height, :]
    u = yuv_i420[height:height + height // 4].reshape(height // 2, width // 2)
    v = yuv_i420[height + height // 4:].reshape(height // 2, width // 2)

    uv = np.empty((height // 2, width), dtype=np.uint8)
    uv[:, 0::2] = u
    uv[:, 1::2] = v

    nv12 = np.vstack([y, uv])
    return nv12.flatten().astype(np.uint8)


def nv12_to_bgr(nv12_data: NDArray, width: int, height: int) -> NDArray:
    import cv2
    nv12_image = nv12_data.reshape((height * 3 // 2, width))
    bgr = cv2.cvtColor(nv12_image, cv2.COLOR_YUV2BGR_NV12)
    return bgr


class NvidiaEncoder:

    def __init__(
        self,
        width: int,
        height: int,
        framerate: int = 30,
        bitrate: int = 8000000,
        codec: str = "h264",
        preset: str = "P1",
        tuning: str = "ultra_low_latency",
        rate_control: str = "cbr",
        gpu_id: int = 0,
        logging: bool = False,
    ):
        self._logging = logging
        self._width = width
        self._height = height
        self._framerate = framerate
        self._codec = codec
        self._encoder = None

        import_dependency_safe("PyNvVideoCodec" if nvc is None else "")

        encoder_params = {
            "preset": preset,
            "tuning_info": tuning,
            "frame_rate": framerate,
            "gop_length": framerate,
        }

        if rate_control.lower() == "cbr":
            encoder_params["rate_control_mode"] = "cbr"
            encoder_params["average_bit_rate"] = bitrate
            encoder_params["max_bit_rate"] = bitrate
        elif rate_control.lower() == "vbr":
            encoder_params["rate_control_mode"] = "vbr"
            encoder_params["average_bit_rate"] = bitrate
            encoder_params["max_bit_rate"] = int(bitrate * 1.5)

        if gpu_id > 0:
            encoder_params["gpu_id"] = gpu_id

        try:
            self._encoder = nvc.CreateEncoder(
                width,
                height,
                "NV12",
                True,
                codec=codec,
                **encoder_params
            )
        except Exception as e:
            self._encoder = nvc.CreateEncoder(
                width,
                height,
                "NV12",
                True,
                codec=codec,
            )
            self._logging and logger.warning(
                "Created encoder with default parameters: {}".format(e)
            )

        self._logging and logger.debug(
            "NvidiaEncoder initialized: {}x{} @ {}fps, codec={}, rate_control={}".format(
                width, height, framerate, codec, rate_control
            )
        )

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def codec(self) -> str:
        return self._codec

    def encode(self, bgr_frame: NDArray) -> bytes:
        import cv2

        if bgr_frame.shape[1] != self._width or bgr_frame.shape[0] != self._height:
            bgr_frame = cv2.resize(bgr_frame, (self._width, self._height))

        nv12_data = bgr_to_nv12(bgr_frame)

        encoded_packets = self._encoder.Encode(nv12_data)

        if encoded_packets:
            if isinstance(encoded_packets, list):
                return b''.join(bytes(pkt) for pkt in encoded_packets)
            else:
                return bytes(encoded_packets)

        return b''

    def flush(self) -> bytes:
        if self._encoder is None:
            return b''
        flushed = self._encoder.Flush()
        if flushed:
            if isinstance(flushed, list):
                return b''.join(bytes(pkt) for pkt in flushed)
            else:
                return bytes(flushed)
        return b''

    def close(self) -> None:
        self._logging and logger.debug("Closing NvidiaEncoder")
        if self._encoder is not None:
            try:
                self.flush()
            except Exception:
                pass
            self._encoder = None


class NvidiaDecoder:

    def __init__(
        self,
        gpu_id: int = 0,
        codec: str = "h264",
        logging: bool = False,
    ):
        self._logging = logging
        self._gpu_id = gpu_id
        self._codec = codec
        self._decoder = None
        self._width = None
        self._height = None

        import_dependency_safe("PyNvVideoCodec" if nvc is None else "")

        codec_map = {
            "h264": nvc.cudaVideoCodec.H264,
            "hevc": nvc.cudaVideoCodec.HEVC,
            "h265": nvc.cudaVideoCodec.HEVC,
            "av1": nvc.cudaVideoCodec.AV1,
        }
        
        cuda_codec = codec_map.get(codec.lower(), nvc.cudaVideoCodec.H264)

        try:
            self._decoder = nvc.CreateDecoder(
                gpuid=gpu_id,
                codec=cuda_codec,
            )
            self._logging and logger.debug(
                "NvidiaDecoder initialized: gpu_id={}, codec={}".format(gpu_id, codec)
            )
        except Exception as e:
            logger.error("Failed to create decoder: {}".format(e))
            raise

    @property
    def width(self) -> Optional[int]:
        return self._width

    @property
    def height(self) -> Optional[int]:
        return self._height

    def decode(
        self,
        encoded_data: bytes,
        width: Optional[int] = None,
        height: Optional[int] = None
    ) -> Optional[NDArray]:
        if not encoded_data:
            return None

        if width is not None and height is not None:
            self._width = width
            self._height = height

        if self._decoder is None:
            return None

        try:
            decoded_frames = self._decoder.Decode(encoded_data)

            if not decoded_frames or len(decoded_frames) == 0:
                return None

            frame = decoded_frames[-1]

            if hasattr(frame, 'shape'):
                if len(frame.shape) == 1:
                    if self._width is None or self._height is None:
                        raise RuntimeError(
                            "Received 1D NV12 buffer but no dimensions available. "
                            "Ensure width and height are provided."
                        )
                    return nv12_to_bgr(np.array(frame), self._width, self._height)
                elif len(frame.shape) == 2:
                    h, w = frame.shape
                    actual_height = int(h * 2 / 3)
                    self._width = w
                    self._height = actual_height
                    return nv12_to_bgr(np.array(frame).flatten(), w, actual_height)
                elif len(frame.shape) == 3:
                    self._height, self._width = frame.shape[:2]
                    return np.array(frame)

            return np.array(frame)

        except Exception as e:
            self._logging and logger.error("Decode error: {}".format(e))
            return None

    def close(self) -> None:
        self._logging and logger.debug("Closing NvidiaDecoder")
        self._decoder = None