import numpy as np
from typing import Optional
from numpy.typing import NDArray

from .base import BaseEncoder, BaseDecoder
from ..utils.common import get_logger, import_dependency_safe

simplejpeg = import_dependency_safe("simplejpeg", error="silent", min_version="1.6.1")

logger = get_logger("JpegCodec")


def has_jpeg_codec() -> bool:
    return simplejpeg is not None


class JpegEncoder(BaseEncoder):

    def __init__(
        self,
        width: int,
        height: int,
        quality: int = 90,
        colorspace: str = "BGR",
        fastdct: bool = True,
        logging: bool = False,
    ):
        self._logging = logging
        self._width = width
        self._height = height
        self._quality = quality
        self._colorspace = colorspace.upper()
        self._fastdct = fastdct

        if simplejpeg is None:
            raise ImportError("simplejpeg is required for JpegEncoder")

        valid_colorspaces = [
            "RGB", "BGR", "RGBX", "BGRX", "XBGR", "XRGB",
            "GRAY", "RGBA", "BGRA", "ABGR", "ARGB", "CMYK"
        ]

        if self._colorspace not in valid_colorspaces:
            self._colorspace = "BGR"
            self._logging and logger.warning(
                "Invalid colorspace. Defaulting to BGR."
            )

        self._logging and logger.debug(
            "JpegEncoder initialized: {}x{}, quality={}, colorspace={}".format(
                width, height, quality, self._colorspace
            )
        )

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def codec_type(self) -> str:
        return "jpeg"

    @property
    def quality(self) -> int:
        return self._quality

    @property
    def colorspace(self) -> str:
        return self._colorspace

    def encode(self, frame: NDArray) -> bytes:
        import cv2

        if frame.shape[1] != self._width or frame.shape[0] != self._height:
            frame = cv2.resize(frame, (self._width, self._height))

        try:
            if self._colorspace == "GRAY":
                if frame.ndim == 2:
                    frame = frame[:, :, np.newaxis]
                encoded = simplejpeg.encode_jpeg(
                    frame,
                    quality=self._quality,
                    colorspace=self._colorspace,
                    fastdct=self._fastdct,
                )
            else:
                encoded = simplejpeg.encode_jpeg(
                    frame,
                    quality=self._quality,
                    colorspace=self._colorspace,
                    colorsubsampling="422",
                    fastdct=self._fastdct,
                )
            return encoded
        except Exception as e:
            self._logging and logger.error("Encode error: {}".format(e))
            return b''

    def flush(self) -> bytes:
        return b''

    def close(self) -> None:
        self._logging and logger.debug("Closing JpegEncoder")

    def get_compression_metadata(self) -> dict:
        return {
            "type": self.codec_type,
            "colorspace": self._colorspace,
            "dct": self._fastdct,
            "ups": False,
            "width": self._width,
            "height": self._height,
        }


class JpegDecoder(BaseDecoder):

    def __init__(
        self,
        colorspace: str = "BGR",
        fastdct: bool = True,
        fastupsample: bool = False,
        logging: bool = False,
    ):
        self._logging = logging
        self._colorspace = colorspace.upper()
        self._fastdct = fastdct
        self._fastupsample = fastupsample

        if simplejpeg is None:
            raise ImportError("simplejpeg is required for JpegDecoder")

        self._logging and logger.debug(
            "JpegDecoder initialized: colorspace={}, fastdct={}, fastupsample={}".format(
                self._colorspace, fastdct, fastupsample
            )
        )

    @property
    def codec_type(self) -> str:
        return "jpeg"

    def decode(
        self,
        encoded_data: bytes,
        width: int = None,
        height: int = None,
        colorspace: str = None,
        fastdct: bool = None,
        fastupsample: bool = None,
    ) -> Optional[NDArray]:
        if not encoded_data:
            return None

        cs = colorspace if colorspace else self._colorspace
        dct = fastdct if fastdct is not None else self._fastdct
        ups = fastupsample if fastupsample is not None else self._fastupsample

        try:
            frame = simplejpeg.decode_jpeg(
                encoded_data,
                colorspace=cs,
                fastdct=dct,
                fastupsample=ups,
            )

            if frame is None:
                return None

            if cs == "GRAY" and frame.ndim == 3:
                frame = np.squeeze(frame, axis=2)

            return frame

        except Exception as e:
            self._logging and logger.error("Decode error: {}".format(e))
            return None

    def close(self) -> None:
        self._logging and logger.debug("Closing JpegDecoder")