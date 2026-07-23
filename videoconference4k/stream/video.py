import cv2
from typing import TypeVar, Tuple, Union, Any
from numpy.typing import NDArray
from ..utils.common import get_logger, log_version
from ..capture.video import VideoCapture

logger = get_logger("VideoStream")

T = TypeVar("T", bound="VideoStream")

HIGH_RES_PIXELS = 1920 * 1080


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

        defaults = {}
        if (
            isinstance(source, int)
            and resolution[0] * resolution[1] >= HIGH_RES_PIXELS
            and "CAP_PROP_FOURCC" not in options
        ):
            defaults["CAP_PROP_FOURCC"] = cv2.VideoWriter_fourcc(*"MJPG")
            self.__logging and logger.debug(
                "Defaulting to MJPG FOURCC for high-resolution camera source; "
                "pass CAP_PROP_FOURCC to override."
            )
        if "CAP_PROP_FRAME_WIDTH" not in options:
            defaults["CAP_PROP_FRAME_WIDTH"] = resolution[0]
        if "CAP_PROP_FRAME_HEIGHT" not in options:
            defaults["CAP_PROP_FRAME_HEIGHT"] = resolution[1]
        if "CAP_PROP_FPS" not in options and framerate > 0:
            defaults["CAP_PROP_FPS"] = framerate

        options = {**defaults, **options}
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