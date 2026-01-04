import cv2
import time
import queue
import logging as log
from threading import Thread, Event, Lock
from typing import TypeVar, Optional, Any
from numpy.typing import NDArray

from ..utils.common import (
    capPropId,
    logger_handler,
    check_CV_version,
    import_dependency_safe,
    log_version,
)

logger = log.getLogger("VideoCapture")
logger.propagate = False
logger.addHandler(logger_handler())
logger.setLevel(log.DEBUG)

T = TypeVar("T", bound="VideoCapture")


class VideoCapture:

    def __init__(
        self,
        source: Any = 0,
        backend: int = 0,
        colorspace: str = None,
        logging: bool = False,
        time_delay: int = 0,
        **options: dict
    ):
        self.__logging = logging if isinstance(logging, bool) else False

        log_version(logging=self.__logging)

        self.__threaded_queue_mode = options.pop("THREADED_QUEUE_MODE", True)
        if not isinstance(self.__threaded_queue_mode, bool):
            self.__threaded_queue_mode = True
        self.__thread_timeout = options.pop("THREAD_TIMEOUT", None)
        if self.__thread_timeout and isinstance(self.__thread_timeout, (int, float)):
            self.__thread_timeout = float(self.__thread_timeout)
        else:
            self.__thread_timeout = None

        self.__default_queue_timeout = 1.0

        self.__queue = None
        if self.__threaded_queue_mode and isinstance(source, str):
            self.__queue = queue.Queue(maxsize=96)
            self.__logging and logger.debug(
                "Enabling Threaded Queue Mode for the current video source!"
            )
        else:
            self.__threaded_queue_mode = False
            self.__logging and logger.warning(
                "Threaded Queue Mode is disabled for the current video source!"
            )

        self.__thread_timeout and logger.info(
            "Setting Video-Thread Timeout to {}s.".format(self.__thread_timeout)
        )

        self.stream = None

        if backend and isinstance(backend, int):
            if check_CV_version() == 3:
                self.stream = cv2.VideoCapture(source + backend)
            else:
                self.stream = cv2.VideoCapture(source, backend)
            logger.info("Setting backend `{}` for this source.".format(backend))
        else:
            self.stream = cv2.VideoCapture(source)

        self.color_space = None

        options = {str(k).strip(): v for k, v in options.items()}
        for key, value in options.items():
            property = capPropId(key)
            not (property is None) and self.stream.set(property, value)

        if not (colorspace is None):
            self.color_space = capPropId(colorspace.strip())
            self.__logging and not (self.color_space is None) and logger.debug(
                "Enabling `{}` colorspace for this video stream!".format(
                    colorspace.strip()
                )
            )

        self.framerate = 0.0
        _fps = self.stream.get(cv2.CAP_PROP_FPS)
        if _fps > 1.0:
            self.framerate = _fps

        time_delay and isinstance(time_delay, (int, float)) and time.sleep(time_delay)

        (grabbed, self.frame) = self.stream.read()

        if grabbed:
            if not (self.color_space is None):
                self.frame = cv2.cvtColor(self.frame, self.color_space)

            self.__threaded_queue_mode and self.__queue.put(self.frame)
        else:
            raise RuntimeError(
                "[VideoCapture:ERROR] :: Source is invalid, VideoCapture failed to initialize stream on this source!"
            )

        self.__thread = None

        self.__terminate = Event()

        self.__stream_read = Event()

        self.__frame_lock = Lock()

        self.__is_running = False

    @property
    def is_running(self) -> bool:
        return self.__is_running

    def start(self) -> T:
        if self.__is_running and self.__thread is not None and self.__thread.is_alive():
            self.__logging and logger.warning("VideoCapture is already running.")
            return self
        self.__terminate.clear()
        self.__is_running = True
        self.__thread = Thread(target=self.__update, name="VideoCapture", args=())
        self.__thread.daemon = True
        self.__thread.start()
        return self

    def __update(self):
        while not self.__terminate.is_set():
            self.__stream_read.clear()

            (grabbed, frame) = self.stream.read()

            if not grabbed:
                if self.__threaded_queue_mode:
                    if self.__queue.empty():
                        break
                    else:
                        continue
                else:
                    break

            if not (self.color_space is None):
                color_frame = None
                try:
                    color_frame = cv2.cvtColor(frame, self.color_space)
                except Exception as e:
                    color_frame = None
                    self.color_space = None
                    self.__logging and logger.exception(str(e))
                    logger.warning("Assigned colorspace value is invalid. Discarding!")
                frame = color_frame if not (color_frame is None) else frame

            if self.__threaded_queue_mode:
                self.__queue.put(frame)
            else:
                with self.__frame_lock:
                    self.frame = frame

            self.__stream_read.set()

        self.__threaded_queue_mode and self.__queue.put(None)
        self.__threaded_queue_mode = False

        self.__terminate.set()
        self.__stream_read.set()
        self.__is_running = False

        self.stream.release()

    def read(self) -> Optional[NDArray]:
        if not self.__is_running:
            self.__logging and logger.warning(
                "VideoCapture is not running. Returning initial frame. Call start() for continuous capture."
            )
            with self.__frame_lock:
                return self.frame

        while self.__threaded_queue_mode and not self.__terminate.is_set():
            try:
                timeout = self.__thread_timeout if self.__thread_timeout else self.__default_queue_timeout
                frame = self.__queue.get(timeout=timeout)
                return frame
            except queue.Empty:
                if self.__terminate.is_set():
                    return None
                continue
        if not self.__terminate.is_set() and self.__stream_read.wait(timeout=self.__thread_timeout):
            with self.__frame_lock:
                return self.frame
        return None

    def stop(self) -> None:
        self.__logging and logger.debug("Terminating processes.")
        self.__threaded_queue_mode = False
        self.__is_running = False

        self.__stream_read.set()
        self.__terminate.set()

        if self.__thread is not None:
            if not (self.__queue is None):
                while not self.__queue.empty():
                    try:
                        self.__queue.get_nowait()
                    except queue.Empty:
                        continue
                    self.__queue.task_done()
            self.__thread.join()
            self.__thread = None