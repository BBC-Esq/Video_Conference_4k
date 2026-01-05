import cv2
import logging as log
from typing import TypeVar, Tuple, Union, Any
from numpy.typing import NDArray
from ..utils.common import logger_handler, log_version
from ..capture.video import VideoCapture

logger = log.getLogger("VideoStream")
logger.propagate = False
logger.addHandler(logger_handler())
logger.setLevel(log.DEBUG)

T = TypeVar("T", bound="VideoStream")


class VideoStream:
    def __init__(
        self,
        resolution: Tuple[int, int] = (640, 480),
        framerate: Union[int, float] = 30,
        source: Any = 0,
        backend: int = 0,
        time_delay: int = 0,
        colorspace: str = None,
        logging: bool = False,
        **options: dict
    ):
        self.__logging = logging if isinstance(logging, bool) else False
        log_version(logging=self.__logging)
        options = {str(k).strip(): v for k, v in options.items()}
        if "CAP_PROP_FRAME_WIDTH" not in options:
            options["CAP_PROP_FRAME_WIDTH"] = resolution[0]
        if "CAP_PROP_FRAME_HEIGHT" not in options:
            options["CAP_PROP_FRAME_HEIGHT"] = resolution[1]
        if "CAP_PROP_FPS" not in options and framerate > 0:
            options["CAP_PROP_FPS"] = framerate
        self.stream = VideoCapture(
            source=source,
            backend=backend,
            colorspace=colorspace,
            logging=logging,
            time_delay=time_delay,
            **options
        )
        self.framerate = self.stream.framerate

    @property
    def is_running(self) -> bool:
        return self.stream.is_running

    def start(self) -> T:
        self.stream.start()
        return self

    def read(self) -> NDArray:
        return self.stream.read()

    def stop(self) -> None:
        self.stream.stop()
        self.__logging and logger.debug("Terminating VideoStream.")