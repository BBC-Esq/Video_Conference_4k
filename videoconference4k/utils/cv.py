import cv2
from typing import Optional
from packaging.version import parse

from .common import get_logger

logger = get_logger("CVUtils")


def check_CV_version() -> int:
    if parse(cv2.__version__) >= parse("4"):
        return 4
    else:
        return 3


def capPropId(property: str, logging: bool = True) -> Optional[int]:
    integer_value = 0
    try:
        integer_value = getattr(cv2, property)
    except Exception as e:
        logging and logger.exception(str(e))
        logger.critical("`{}` is not a valid OpenCV property!".format(property))
        return None
    return integer_value