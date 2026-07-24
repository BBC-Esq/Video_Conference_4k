import numpy as np
from typing import Optional, Tuple, Dict, Any
from numpy.typing import NDArray

from ..utils.common import get_logger, import_dependency_safe
from ..codec import (
    has_nvidia_codec,
    has_x264,
    has_jpeg_codec,
    NvidiaEncoder,
    NvidiaDecoder,
    SoftwareEncoder,
    SoftwareDecoder,
    JpegEncoder,
    JpegDecoder,
)

simplejpeg = import_dependency_safe("simplejpeg", error="silent", min_version="1.6.1")

logger = get_logger("Compression")


class CompressionType:
    NVENC = "nvenc"
    SOFTWARE = "software"
    JPEG = "jpeg"
    NONE = "none"


class CompressionHandler:

    def __init__(
        self,
        gpu_accelerated: bool = False,
        gpu_id: int = 0,
        gpu_bitrate: int = 8000000,
        gpu_codec: str = "h264",
        jpeg_quality: int = 90,
        jpeg_colorspace: str = "BGR",
        jpeg_fastdct: bool = True,
        jpeg_fastupsample: bool = False,
        logging: bool = False,
    ):
        self._logging = logging
        self._gpu_id = gpu_id
        self._gpu_bitrate = gpu_bitrate
        self._gpu_codec = gpu_codec
        self._jpeg_quality = jpeg_quality
        self._jpeg_colorspace = jpeg_colorspace.upper()
        self._jpeg_fastdct = jpeg_fastdct
        self._jpeg_fastupsample = jpeg_fastupsample

        self._nvidia_encoder = None
        self._nvidia_decoder = None
        self._software_encoder = None
        self._software_decoder = None
        self._jpeg_encoder = None
        self._jpeg_decoder = None

        self._compression_type = CompressionType.NONE
        self._use_nvidia = False
        self._use_software = False
        self._use_jpeg = False

        if gpu_accelerated:
            if has_nvidia_codec():
                self._use_nvidia = True
                self._compression_type = CompressionType.NVENC
                self._logging and logger.info("GPU acceleration enabled with NVIDIA hardware encoding")
            elif has_x264():
                self._use_software = True
                self._compression_type = CompressionType.SOFTWARE
                logger.warning("GPU acceleration requested but NVIDIA codec not available. Using software fallback.")
                self._logging and logger.info("Software acceleration enabled with x264 encoding")
            elif has_jpeg_codec():
                self._use_jpeg = True
                self._compression_type = CompressionType.JPEG
                logger.warning("No hardware or software codec available. Falling back to JPEG compression.")
            else:
                logger.warning("No compression codec available.")
        elif has_jpeg_codec():
            self._use_jpeg = True
            self._compression_type = CompressionType.JPEG

    @property
    def compression_type(self) -> str:
        return self._compression_type

    @property
    def is_nvidia(self) -> bool:
        return self._use_nvidia

    @property
    def is_software(self) -> bool:
        return self._use_software

    @property
    def is_jpeg(self) -> bool:
        return self._use_jpeg

    @property
    def is_enabled(self) -> bool:
        return self._use_nvidia or self._use_software or self._use_jpeg

    @property
    def supports_dynamic_bitrate(self) -> bool:
        if self._use_nvidia and self._nvidia_encoder is not None:
            return self._nvidia_encoder.supports_dynamic_bitrate
        return False

    def reconfigure_bitrate(self, bitrate: int, maxbitrate: Optional[int] = None) -> bool:
        if self._use_nvidia and self._nvidia_encoder is not None:
            if self._nvidia_encoder.reconfigure_bitrate(bitrate, maxbitrate):
                self._gpu_bitrate = int(bitrate)
                return True
        return False

    def configure_jpeg(
        self,
        enabled: Optional[bool] = None,
        quality: Optional[int] = None,
        colorspace: Optional[str] = None,
        fastdct: Optional[bool] = None,
        fastupsample: Optional[bool] = None,
    ) -> None:
        """Apply JPEG settings parsed from transport options.

        Must be called before the first frame is encoded/decoded. No-op when
        a hardware/software video codec is active (JPEG is unused in those
        modes). Passing `enabled=False` disables JPEG entirely, causing
        frames to be transmitted raw (lossless).
        """
        if self._use_nvidia or self._use_software:
            return

        if quality is not None:
            self._jpeg_quality = int(quality)
        if colorspace is not None:
            self._jpeg_colorspace = colorspace.upper()
        if fastdct is not None:
            self._jpeg_fastdct = fastdct
        if fastupsample is not None:
            self._jpeg_fastupsample = fastupsample

        if enabled is not None:
            if enabled and has_jpeg_codec():
                self._use_jpeg = True
                self._compression_type = CompressionType.JPEG
            else:
                self._use_jpeg = False
                self._compression_type = CompressionType.NONE

        # Drop any lazily-created JPEG codecs so new settings take effect.
        if self._jpeg_encoder is not None:
            self._jpeg_encoder.close()
            self._jpeg_encoder = None
        if self._jpeg_decoder is not None:
            self._jpeg_decoder.close()
            self._jpeg_decoder = None

    def _get_nvidia_encoder(self, width: int, height: int) -> NvidiaEncoder:
        if self._nvidia_encoder is not None and (
            self._nvidia_encoder.width != width or self._nvidia_encoder.height != height
        ):
            self._nvidia_encoder.close()
            self._nvidia_encoder = None
        if self._nvidia_encoder is None:
            self._nvidia_encoder = NvidiaEncoder(
                width=width,
                height=height,
                bitrate=self._gpu_bitrate,
                codec=self._gpu_codec,
                gpu_id=self._gpu_id,
                logging=self._logging,
            )
        return self._nvidia_encoder

    def _get_nvidia_decoder(self) -> NvidiaDecoder:
        if self._nvidia_decoder is None:
            self._nvidia_decoder = NvidiaDecoder(
                gpu_id=self._gpu_id,
                codec=self._gpu_codec,
                logging=self._logging,
            )
        return self._nvidia_decoder

    def _get_software_encoder(self, width: int, height: int) -> SoftwareEncoder:
        if self._software_encoder is not None and (
            self._software_encoder.width != width or self._software_encoder.height != height
        ):
            self._software_encoder.close()
            self._software_encoder = None
        if self._software_encoder is None:
            codec = "x264" if self._gpu_codec in ["h264", "x264"] else "x265"
            self._software_encoder = SoftwareEncoder(
                width=width,
                height=height,
                bitrate=self._gpu_bitrate,
                codec=codec,
                logging=self._logging,
            )
        return self._software_encoder

    def _get_software_decoder(self) -> SoftwareDecoder:
        if self._software_decoder is None:
            codec = "x264" if self._gpu_codec in ["h264", "x264"] else "x265"
            self._software_decoder = SoftwareDecoder(
                codec=codec,
                logging=self._logging,
            )
        return self._software_decoder

    def _get_jpeg_encoder(self, width: int, height: int) -> JpegEncoder:
        if self._jpeg_encoder is not None and (
            self._jpeg_encoder.width != width or self._jpeg_encoder.height != height
        ):
            self._jpeg_encoder.close()
            self._jpeg_encoder = None
        if self._jpeg_encoder is None:
            self._jpeg_encoder = JpegEncoder(
                width=width,
                height=height,
                quality=self._jpeg_quality,
                colorspace=self._jpeg_colorspace,
                fastdct=self._jpeg_fastdct,
                logging=self._logging,
            )
        return self._jpeg_encoder

    def _get_jpeg_decoder(self) -> JpegDecoder:
        if self._jpeg_decoder is None:
            self._jpeg_decoder = JpegDecoder(
                colorspace=self._jpeg_colorspace,
                fastdct=self._jpeg_fastdct,
                fastupsample=self._jpeg_fastupsample,
                logging=self._logging,
            )
        return self._jpeg_decoder

    def encode_frame(self, frame: NDArray) -> Tuple[bytes, Dict[str, Any]]:
        if not frame.flags["C_CONTIGUOUS"]:
            frame = np.ascontiguousarray(frame, dtype=frame.dtype)

        height, width = frame.shape[:2]

        if self._use_nvidia:
            encoder = self._get_nvidia_encoder(width, height)
            encoded = encoder.encode(frame)
            metadata = {
                "type": CompressionType.NVENC,
                "codec": self._gpu_codec,
                "width": width,
                "height": height,
            }
            return encoded, metadata

        elif self._use_software:
            encoder = self._get_software_encoder(width, height)
            encoded = encoder.encode(frame)
            metadata = {
                "type": CompressionType.SOFTWARE,
                "codec": self._gpu_codec,
                "width": width,
                "height": height,
            }
            return encoded, metadata

        elif self._use_jpeg:
            encoder = self._get_jpeg_encoder(width, height)
            encoded = encoder.encode(frame)
            metadata = encoder.get_compression_metadata()
            return encoded, metadata

        else:
            return frame.tobytes(), {
                "type": CompressionType.NONE,
                "dtype": str(frame.dtype),
                "shape": frame.shape,
            }

    def decode_frame(self, data: bytes, metadata: Dict[str, Any]) -> Optional[NDArray]:
        compression_type = metadata.get("type", CompressionType.NONE)

        if compression_type == CompressionType.NVENC:
            decoder = self._get_nvidia_decoder()
            return decoder.decode(
                data,
                width=metadata.get("width"),
                height=metadata.get("height"),
            )

        elif compression_type == CompressionType.SOFTWARE:
            decoder = self._get_software_decoder()
            return decoder.decode(
                data,
                width=metadata.get("width"),
                height=metadata.get("height"),
            )

        elif compression_type == CompressionType.JPEG:
            decoder = self._get_jpeg_decoder()
            return decoder.decode(
                data,
                colorspace=metadata.get("colorspace", self._jpeg_colorspace),
                fastdct=metadata.get("dct", self._jpeg_fastdct),
                fastupsample=metadata.get("ups", self._jpeg_fastupsample),
            )

        else:
            dtype = metadata.get("dtype", "uint8")
            shape = metadata.get("shape")
            if shape:
                return np.frombuffer(data, dtype=dtype).reshape(shape)
            return None

    def create_sync_compression_metadata(self) -> Any:
        if self._use_nvidia:
            return {
                "type": CompressionType.NVENC,
                "codec": self._gpu_codec,
            }
        elif self._use_software:
            return {
                "type": CompressionType.SOFTWARE,
                "codec": self._gpu_codec,
            }
        elif self._use_jpeg:
            return {
                "dct": self._jpeg_fastdct,
                "ups": self._jpeg_fastupsample,
                "colorspace": self._jpeg_colorspace,
            }
        return False

    def close(self) -> None:
        if self._nvidia_encoder is not None:
            self._nvidia_encoder.close()
            self._nvidia_encoder = None

        if self._nvidia_decoder is not None:
            self._nvidia_decoder.close()
            self._nvidia_decoder = None

        if self._software_encoder is not None:
            self._software_encoder.close()
            self._software_encoder = None

        if self._software_decoder is not None:
            self._software_decoder.close()
            self._software_decoder = None

        if self._jpeg_encoder is not None:
            self._jpeg_encoder.close()
            self._jpeg_encoder = None

        if self._jpeg_decoder is not None:
            self._jpeg_decoder.close()
            self._jpeg_decoder = None

        self._logging and logger.debug("CompressionHandler closed")


def decode_sync_frame(
    data: bytes,
    compression_info: Any,
    compression_handler: CompressionHandler,
    jpeg_fastdct: bool = True,
    jpeg_fastupsample: bool = False,
) -> Optional[NDArray]:
    if not compression_info:
        return None

    if isinstance(compression_info, dict):
        comp_type = compression_info.get("type")

        if comp_type == CompressionType.NVENC:
            return compression_handler.decode_frame(data, compression_info)

        elif comp_type == CompressionType.SOFTWARE:
            return compression_handler.decode_frame(data, compression_info)

        else:
            # Sender's encode settings take precedence; the receiver's own
            # settings are only a fallback when the metadata omits a key.
            metadata = {
                "type": CompressionType.JPEG,
                "colorspace": compression_info.get("colorspace", "BGR"),
                "dct": compression_info.get("dct", jpeg_fastdct),
                "ups": compression_info.get("ups", jpeg_fastupsample),
            }
            frame = compression_handler.decode_frame(data, metadata)

            if frame is not None and compression_info.get("colorspace") == "GRAY" and frame.ndim == 3:
                frame = np.squeeze(frame, axis=2)

            return frame

    return None


def encode_return_frame(
    frame: NDArray,
    compression_handler: CompressionHandler,
) -> Tuple[bytes, Dict[str, Any]]:
    if not frame.flags["C_CONTIGUOUS"]:
        frame = np.ascontiguousarray(frame, dtype=frame.dtype)

    return compression_handler.encode_frame(frame)