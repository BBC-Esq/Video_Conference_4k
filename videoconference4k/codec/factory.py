from enum import Enum, auto
from typing import Optional, Dict, Any

from .base import BaseEncoder, BaseDecoder
from .nvidia import has_nvidia_codec, NvidiaEncoder, NvidiaDecoder
from .intel import has_intel_codec, IntelEncoder
from .software import has_software_codec, has_x264, has_x265, SoftwareEncoder, SoftwareDecoder
from .jpeg import has_jpeg_codec, JpegEncoder, JpegDecoder
from ..utils.common import get_logger

logger = get_logger("CodecFactory")


class CodecType(Enum):
    NVENC = auto()
    INTEL_QSV = auto()
    SOFTWARE = auto()
    JPEG = auto()
    NONE = auto()


def get_available_codecs() -> Dict[str, bool]:
    return {
        "nvidia": has_nvidia_codec(),
        "intel_qsv": has_intel_codec(),
        "software_x264": has_x264(),
        "software_x265": has_x265(),
        "jpeg": has_jpeg_codec(),
    }


def get_best_encoder(
    width: int,
    height: int,
    framerate: int = 30,
    bitrate: int = 8000000,
    codec: str = "h264",
    quality: int = 90,
    preferred_type: Optional[CodecType] = None,
    fallback_to_jpeg: bool = True,
    logging: bool = False,
    **kwargs: Any,
) -> Optional[BaseEncoder]:

    if preferred_type == CodecType.NVENC:
        if has_nvidia_codec():
            return NvidiaEncoder(
                width=width,
                height=height,
                framerate=framerate,
                bitrate=bitrate,
                codec=codec,
                logging=logging,
                **kwargs
            )
        else:
            logger.warning("NVIDIA codec requested but not available")

    elif preferred_type == CodecType.INTEL_QSV:
        if has_intel_codec():
            return IntelEncoder(
                width=width,
                height=height,
                framerate=framerate,
                bitrate=bitrate,
                codec=codec,
                logging=logging,
                **kwargs
            )
        else:
            logger.warning("Intel QSV codec requested but not available")

    elif preferred_type == CodecType.SOFTWARE:
        if has_software_codec():
            sw_codec = "x264" if codec.lower() in ["h264", "x264"] else "x265"
            return SoftwareEncoder(
                width=width,
                height=height,
                framerate=framerate,
                bitrate=bitrate,
                codec=sw_codec,
                logging=logging,
                **kwargs
            )
        else:
            logger.warning("Software codec requested but not available")

    elif preferred_type == CodecType.JPEG:
        if has_jpeg_codec():
            return JpegEncoder(
                width=width,
                height=height,
                quality=quality,
                logging=logging,
                **kwargs
            )
        else:
            logger.warning("JPEG codec requested but not available")

    if preferred_type is None or preferred_type != CodecType.NONE:
        sw_codec = "x264" if codec.lower() in ["h264", "x264"] else "x265"
        ladder = [
            ("NVIDIA hardware", has_nvidia_codec, lambda: NvidiaEncoder(
                width=width, height=height, framerate=framerate, bitrate=bitrate,
                codec=codec, logging=logging, **kwargs
            )),
            ("Intel QSV", has_intel_codec, lambda: IntelEncoder(
                width=width, height=height, framerate=framerate, bitrate=bitrate,
                codec=codec, logging=logging, **kwargs
            )),
            ("software", has_software_codec, lambda: SoftwareEncoder(
                width=width, height=height, framerate=framerate, bitrate=bitrate,
                codec=sw_codec, logging=logging, **kwargs
            )),
        ]
        if fallback_to_jpeg:
            ladder.append(("JPEG", has_jpeg_codec, lambda: JpegEncoder(
                width=width, height=height, quality=quality, logging=logging, **kwargs
            )))

        for name, available, build in ladder:
            if not available():
                continue
            try:
                encoder = build()
            except Exception as e:
                logger.error(
                    "{} encoder is present but failed to initialise at {}x{} ({}); "
                    "trying the next backend.".format(name, width, height, e)
                )
                continue
            logging and logger.info("Using {} encoder".format(name))
            return encoder

    logger.error("No suitable encoder available")
    return None


def get_decoder_for_type(
    codec_type: str,
    codec: str = "h264",
    colorspace: str = "BGR",
    logging: bool = False,
    **kwargs: Any,
) -> Optional[BaseDecoder]:

    codec_type_lower = codec_type.lower()

    if codec_type_lower == "nvenc":
        if has_nvidia_codec():
            return NvidiaDecoder(
                codec=codec,
                logging=logging,
                **kwargs
            )
        else:
            logger.warning("NVIDIA decoder requested but not available")
            return None

    elif codec_type_lower == "intel_qsv":
        logger.warning("Intel QSV decoder not implemented, falling back to software")
        if has_software_codec():
            return SoftwareDecoder(
                codec=codec,
                logging=logging,
            )
        return None

    elif codec_type_lower == "software":
        if has_software_codec():
            return SoftwareDecoder(
                codec=codec,
                logging=logging,
            )
        else:
            logger.warning("Software decoder requested but not available")
            return None

    elif codec_type_lower == "jpeg":
        if has_jpeg_codec():
            return JpegDecoder(
                colorspace=colorspace,
                logging=logging,
                **kwargs
            )
        else:
            logger.warning("JPEG decoder requested but not available")
            return None

    else:
        logger.error("Unknown codec type: {}".format(codec_type))
        return None


def get_best_decoder(
    logging: bool = False,
    **kwargs: Any,
) -> Optional[BaseDecoder]:

    if has_nvidia_codec():
        logging and logger.info("Using NVIDIA hardware decoder")
        return NvidiaDecoder(logging=logging, **kwargs)

    if has_software_codec():
        logging and logger.info("Using software decoder")
        return SoftwareDecoder(logging=logging, **kwargs)

    if has_jpeg_codec():
        logging and logger.info("Using JPEG decoder")
        return JpegDecoder(logging=logging, **kwargs)

    logger.error("No suitable decoder available")
    return None