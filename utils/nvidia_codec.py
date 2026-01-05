import numpy as np
import logging as log
from typing import Optional, Tuple
from numpy.typing import NDArray

from .common import logger_handler, import_dependency_safe

nvc = import_dependency_safe("PyNvVideoCodec", error="silent")

logger = log.getLogger("NvidiaCodec")
logger.propagate = False
logger.addHandler(logger_handler())
logger.setLevel(log.DEBUG)


def has_nvidia_codec() -> bool:
    if nvc is None:
        return False
    try:
        encoder = nvc.CreateEncoder(64, 64, "NV12", True, codec="h264")
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
        encoder = nvc.CreateEncoder(64, 64, "NV12", True, codec="h264")
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
    
    y_size = width * height
    uv_size = width * height // 2
    
    y = nv12_data[:y_size].reshape(height, width)
    uv = nv12_data[y_size:y_size + uv_size].reshape(height // 2, width)
    
    u = uv[:, 0::2]
    v = uv[:, 1::2]
    
    u_full = np.repeat(np.repeat(u, 2, axis=0), 2, axis=1)
    v_full = np.repeat(np.repeat(v, 2, axis=0), 2, axis=1)
    
    yuv = np.dstack([y, u_full, v_full]).astype(np.uint8)
    bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    
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
        gpu_id: int = 0,
        logging: bool = False,
    ):
        self.__logging = logging
        self.__width = width
        self.__height = height
        self.__framerate = framerate
        self.__codec = codec
        
        import_dependency_safe("PyNvVideoCodec" if nvc is None else "")
        
        encoder_params = {
            "preset": preset,
            "tuning_info": tuning,
            "rate_control_mode": "cbr",
            "average_bit_rate": bitrate,
            "max_bit_rate": bitrate,
            "frame_rate": framerate,
            "gop_length": framerate,
        }
        
        if gpu_id > 0:
            encoder_params["gpu_id"] = gpu_id
        
        try:
            self.__encoder = nvc.CreateEncoder(
                width,
                height,
                "NV12",
                True,
                codec=codec,
                **encoder_params
            )
        except Exception as e:
            self.__encoder = nvc.CreateEncoder(
                width,
                height,
                "NV12",
                True,
                codec=codec,
            )
            self.__logging and logger.warning(
                "Created encoder with default parameters: {}".format(e)
            )
        
        self.__logging and logger.debug(
            "NvidiaEncoder initialized: {}x{} @ {}fps, codec={}".format(
                width, height, framerate, codec
            )
        )
    
    @property
    def width(self) -> int:
        return self.__width
    
    @property
    def height(self) -> int:
        return self.__height
    
    @property
    def codec(self) -> str:
        return self.__codec
    
    def encode(self, bgr_frame: NDArray) -> bytes:
        import cv2
        
        if bgr_frame.shape[1] != self.__width or bgr_frame.shape[0] != self.__height:
            bgr_frame = cv2.resize(bgr_frame, (self.__width, self.__height))
        
        nv12_data = bgr_to_nv12(bgr_frame)
        
        encoded_packets = self.__encoder.Encode(nv12_data)
        
        if encoded_packets:
            if isinstance(encoded_packets, list):
                return b''.join(bytes(pkt) for pkt in encoded_packets)
            else:
                return bytes(encoded_packets)
        
        return b''
    
    def flush(self) -> bytes:
        flushed = self.__encoder.Flush()
        if flushed:
            if isinstance(flushed, list):
                return b''.join(bytes(pkt) for pkt in flushed)
            else:
                return bytes(flushed)
        return b''
    
    def close(self) -> None:
        self.__logging and logger.debug("Closing NvidiaEncoder")
        try:
            self.flush()
        except Exception:
            pass
        self.__encoder = None


class NvidiaDecoder:
    
    def __init__(
        self,
        gpu_id: int = 0,
        logging: bool = False,
    ):
        self.__logging = logging
        self.__gpu_id = gpu_id
        self.__decoder = None
        self.__width = None
        self.__height = None
        
        import_dependency_safe("PyNvVideoCodec" if nvc is None else "")
        
        self.__logging and logger.debug("NvidiaDecoder initialized")
    
    @property
    def width(self) -> Optional[int]:
        return self.__width
    
    @property
    def height(self) -> Optional[int]:
        return self.__height
    
    def decode(self, encoded_data: bytes) -> Optional[NDArray]:
        if not encoded_data:
            return None
        
        try:
            if self.__decoder is None:
                decoder_params = {}
                if self.__gpu_id > 0:
                    decoder_params["gpu_id"] = self.__gpu_id
                
                self.__decoder = nvc.CreateDecoder(
                    encoded_stream=encoded_data,
                    **decoder_params
                )
                self.__logging and logger.debug("Decoder created from stream")
            
            decoded_frames = self.__decoder.Decode(encoded_data)
            
            if decoded_frames and len(decoded_frames) > 0:
                frame = decoded_frames[-1]
                
                if hasattr(frame, 'shape'):
                    if len(frame.shape) == 1:
                        if self.__width is None or self.__height is None:
                            return None
                        return nv12_to_bgr(frame, self.__width, self.__height)
                    elif len(frame.shape) == 2:
                        self.__height, self.__width = frame.shape
                        return nv12_to_bgr(frame.flatten(), self.__width, self.__height)
                    elif len(frame.shape) == 3:
                        self.__height, self.__width = frame.shape[:2]
                        return np.array(frame)
                
                return np.array(frame)
            
            return None
            
        except Exception as e:
            self.__logging and logger.error("Decode error: {}".format(e))
            return None
    
    def close(self) -> None:
        self.__logging and logger.debug("Closing NvidiaDecoder")
        self.__decoder = None