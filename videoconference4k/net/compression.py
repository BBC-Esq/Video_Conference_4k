import numpy as np
from typing import Optional, Tuple, Dict, Any
from numpy.typing import NDArray

from ..utils.common import get_logger, import_dependency_safe
from ..codec import (
    has_nvidia_codec,
    has_intel_codec,
    has_software_codec,
    has_x264,
    has_jpeg_codec,
    NvidiaEncoder,
    NvidiaDecoder,
    IntelEncoder,
    SoftwareEncoder,
    SoftwareDecoder,
    JpegEncoder,
    JpegDecoder,
)
from ..codec.base import normalize_codec
from ..codec.capability import encoder_implementation

simplejpeg = import_dependency_safe("simplejpeg", error="silent", min_version="1.6.1")

logger = get_logger("Compression")


class CompressionType:
    NVENC = "nvenc"
    INTEL_QSV = "intel_qsv"
    SOFTWARE = "software"
    JPEG = "jpeg"
    NONE = "none"


VIDEO_CODEC_TYPES = frozenset((
    CompressionType.NVENC,
    CompressionType.INTEL_QSV,
    CompressionType.SOFTWARE,
))


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
        self._intel_encoder = None
        self._software_encoder = None
        self._jpeg_encoder = None
        self._jpeg_decoder = None
        self._video_decoders = {}
        self._last_decoded_codec = None

        self._compression_type = CompressionType.NONE
        self._use_nvidia = False
        self._use_intel = False
        self._use_software = False
        self._use_jpeg = False

        if gpu_accelerated:
            if not self._select_implementation(gpu_codec):
                if has_jpeg_codec():
                    self._use_jpeg = True
                    self._compression_type = CompressionType.JPEG
                    logger.warning(
                        "Nothing here can encode {}; falling back to JPEG.".format(gpu_codec))
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

    def _select_implementation(self, codec: str) -> bool:
        """Resolve how this machine will encode the codec. Layer two, decided locally.

        Delegates to the capability module so this ladder and the capability a
        peer is told about can never disagree; they were separate answers before
        and the software tier claimed codecs it had no encoder for.
        """
        codec = normalize_codec(codec)
        implementation = encoder_implementation(codec)

        self._use_nvidia = implementation == CompressionType.NVENC
        self._use_intel = implementation == CompressionType.INTEL_QSV
        self._use_software = implementation == CompressionType.SOFTWARE

        if implementation is None:
            return False

        self._gpu_codec = codec
        self._compression_type = implementation
        self._logging and logger.info("Encoding {} with {}".format(codec, implementation))
        return True

    def set_codec(self, codec: str) -> bool:
        """Switch the codec being sent, re-choosing the implementation for it.

        The best implementation is a property of the codec, not the machine: an
        integrated GPU may encode H.264 in hardware yet leave HEVC to the CPU,
        so the ladder is walked again rather than assumed to still hold.
        """
        codec = normalize_codec(codec)
        if codec == normalize_codec(self._gpu_codec):
            return False
        if not (self._use_nvidia or self._use_intel or self._use_software):
            return False
        if encoder_implementation(codec) is None:
            logger.warning("Asked to send {} but nothing here can encode it; staying on {}."
                           .format(codec, self._gpu_codec))
            return False

        for encoder_attr in ("_nvidia_encoder", "_intel_encoder", "_software_encoder"):
            encoder = getattr(self, encoder_attr)
            if encoder is not None:
                try:
                    encoder.close()
                except Exception:
                    pass
                setattr(self, encoder_attr, None)

        return self._select_implementation(codec)

    @property
    def encoder_kind(self) -> str:
        """Which implementation is producing frames here, for display and diagnosis."""
        return self._compression_type

    @property
    def decoder_codec(self) -> Optional[str]:
        """The codec last seen arriving, which is the peer's choice rather than ours."""
        return self._last_decoded_codec

    @property
    def decoder_kind(self) -> Optional[str]:
        """Which decoder this machine picked for what is arriving."""
        decoder = self._video_decoders.get(self._last_decoded_codec)
        if decoder is not None:
            return type(decoder).__name__
        if self._jpeg_decoder is not None:
            return type(self._jpeg_decoder).__name__
        return None

    @property
    def is_intel(self) -> bool:
        return self._use_intel

    @property
    def is_software(self) -> bool:
        return self._use_software

    @property
    def is_jpeg(self) -> bool:
        return self._use_jpeg

    @property
    def is_enabled(self) -> bool:
        return self._use_nvidia or self._use_intel or self._use_software or self._use_jpeg

    @property
    def active_encoder(self):
        """The encoder currently producing frames, or None before the first one."""
        if self._use_nvidia:
            return self._nvidia_encoder
        if self._use_intel:
            return self._intel_encoder
        if self._use_software:
            return self._software_encoder
        if self._use_jpeg:
            return self._jpeg_encoder
        return None

    @property
    def encoder_alive(self) -> bool:
        """Whether the active encoder is still able to produce frames.

        A pipe encoder whose subprocess has exited keeps accepting frames and
        returning nothing, which looks exactly like a black picture unless the
        condition is reported.
        """
        encoder = self.active_encoder
        if encoder is None:
            return True
        return getattr(encoder, "is_alive", True)

    @property
    def encoder_error(self) -> str:
        encoder = self.active_encoder
        return getattr(encoder, "last_error", "") if encoder is not None else ""

    @property
    def supports_force_idr(self) -> bool:
        """Whether a keyframe request reaching this machine can actually be honoured.

        Reported rather than assumed, so a peer asking for one that will never
        arrive is a visible condition instead of a silent wait.
        """
        encoder = self.active_encoder
        return bool(encoder is not None and encoder.supports_force_idr)

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

    def _get_video_decoder(self, codec: Optional[str]):
        """Choose a decoder for the codec on the wire, preferring hardware.

        An encoded stream is a standard, not an implementation. Whatever produced
        it - NVENC, Quick Sync or a CPU - any conformant decoder can read it, so
        the choice belongs to the receiver and depends only on the codec.
        """
        codec = normalize_codec(codec)
        if codec in self._video_decoders:
            return self._video_decoders[codec]

        decoder = None
        if has_nvidia_codec():
            try:
                decoder = NvidiaDecoder(gpu_id=self._gpu_id, codec=codec, logging=self._logging)
            except Exception as exc:
                logger.warning(
                    "NVDEC could not be opened for {} ({}); using the software decoder.".format(
                        codec, exc))
                decoder = None

        if decoder is None and has_software_codec():
            decoder = SoftwareDecoder(codec=codec, logging=self._logging)

        if decoder is None:
            logger.error("No decoder available for {} on this machine.".format(codec))

        self._video_decoders[codec] = decoder
        return decoder

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

    def _get_intel_encoder(self, width: int, height: int) -> IntelEncoder:
        if self._intel_encoder is not None and (
            self._intel_encoder.width != width or self._intel_encoder.height != height
        ):
            self._intel_encoder.close()
            self._intel_encoder = None
        if self._intel_encoder is None:
            self._intel_encoder = IntelEncoder(
                width=width,
                height=height,
                bitrate=self._gpu_bitrate,
                codec=normalize_codec(self._gpu_codec),
                logging=self._logging,
            )
        return self._intel_encoder

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

    def encode_frame(self, frame: NDArray, force_idr: bool = False) -> Tuple[bytes, Dict[str, Any]]:
        if not frame.flags["C_CONTIGUOUS"]:
            frame = np.ascontiguousarray(frame, dtype=frame.dtype)

        height, width = frame.shape[:2]

        if self._use_nvidia:
            encoder = self._get_nvidia_encoder(width, height)
            encoded = encoder.encode(frame, force_idr=force_idr)
            metadata = {
                "type": CompressionType.NVENC,
                "codec": self._gpu_codec,
                "width": width,
                "height": height,
            }
            return encoded, metadata

        elif self._use_intel:
            encoder = self._get_intel_encoder(width, height)
            encoded = encoder.encode(frame, force_idr=force_idr)
            metadata = {
                "type": CompressionType.INTEL_QSV,
                "codec": normalize_codec(self._gpu_codec),
                "width": width,
                "height": height,
            }
            return encoded, metadata

        elif self._use_software:
            encoder = self._get_software_encoder(width, height)
            encoded = encoder.encode(frame, force_idr=force_idr)
            metadata = {
                "type": CompressionType.SOFTWARE,
                "codec": normalize_codec(self._gpu_codec),
                "width": width,
                "height": height,
            }
            return encoded, metadata

        elif self._use_jpeg:
            encoder = self._get_jpeg_encoder(width, height)
            encoded = encoder.encode(frame, force_idr=force_idr)
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

        if compression_type in VIDEO_CODEC_TYPES:
            self._last_decoded_codec = normalize_codec(metadata.get("codec"))
            decoder = self._get_video_decoder(metadata.get("codec"))
            if decoder is None:
                return None
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
                "codec": normalize_codec(self._gpu_codec),
            }
        elif self._use_intel:
            return {
                "type": CompressionType.INTEL_QSV,
                "codec": normalize_codec(self._gpu_codec),
            }
        elif self._use_software:
            return {
                "type": CompressionType.SOFTWARE,
                "codec": normalize_codec(self._gpu_codec),
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

        for decoder in self._video_decoders.values():
            if decoder is not None:
                try:
                    decoder.close()
                except Exception:
                    pass
        self._video_decoders = {}

        if self._intel_encoder is not None:
            self._intel_encoder.close()
            self._intel_encoder = None

        if self._software_encoder is not None:
            self._software_encoder.close()
            self._software_encoder = None

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

        if comp_type in VIDEO_CODEC_TYPES:
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