import cv2
import time
import queue
from threading import Thread, Event, Lock
from typing import TypeVar, Optional, Any
from numpy.typing import NDArray

from ..utils.common import (
    get_logger,
    import_dependency_safe,
    log_version,
)
from ..utils.cv import (
    capPropId,
    check_CV_version,
)

logger = get_logger("VideoCapture")

T = TypeVar("T", bound="VideoCapture")

DEFAULT_CAMERA_PRESETS = [
    (3840, 2160, 60),
    (3840, 2160, 30),
    (2560, 1440, 60),
    (1920, 1080, 60),
    (1920, 1080, 30),
    (1280, 720, 60),
    (1280, 720, 30),
]


def probe_camera(
    source: int = 0,
    presets: Optional[list] = None,
    backend: int = 0,
    warmup: int = 5,
    sample: int = 30,
    logging: bool = False,
) -> list:
    if presets is None:
        presets = DEFAULT_CAMERA_PRESETS

    report = []
    for (req_w, req_h, req_fps) in presets:
        entry = {
            "requested": (req_w, req_h, req_fps),
            "opened": False,
            "delivered": None,
            "measured_fps": 0.0,
            "fourcc": "",
        }

        if backend and isinstance(backend, int):
            if check_CV_version() == 3:
                cap = cv2.VideoCapture(source + backend)
            else:
                cap = cv2.VideoCapture(source, backend)
        else:
            cap = cv2.VideoCapture(source)

        try:
            if not cap.isOpened():
                report.append(entry)
                continue
            entry["opened"] = True

            if req_w * req_h >= 1920 * 1080:
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, req_w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, req_h)
            cap.set(cv2.CAP_PROP_FPS, req_fps)

            for _ in range(warmup):
                cap.read()

            frames = 0
            start = time.perf_counter()
            while frames < sample:
                grabbed, _ = cap.read()
                if not grabbed:
                    break
                frames += 1
            elapsed = time.perf_counter() - start

            entry["delivered"] = (
                int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            )
            entry["measured_fps"] = (frames / elapsed) if elapsed > 0 else 0.0
            cc = int(cap.get(cv2.CAP_PROP_FOURCC))
            entry["fourcc"] = (
                "".join(chr((cc >> (8 * i)) & 0xFF) for i in range(4)).strip()
                if cc else ""
            )
            logging and logger.info(
                "Probe {}x{}@{} -> delivered {} @ {:.1f} fps (FOURCC {}).".format(
                    req_w, req_h, req_fps, entry["delivered"],
                    entry["measured_fps"], entry["fourcc"] or "N/A"
                )
            )
        finally:
            cap.release()

        report.append(entry)

    return report


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
            if self.__logging:
                neg_w = int(self.stream.get(cv2.CAP_PROP_FRAME_WIDTH))
                neg_h = int(self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
                neg_fps = self.stream.get(cv2.CAP_PROP_FPS)
                neg_cc = int(self.stream.get(cv2.CAP_PROP_FOURCC))
                neg_cc = (
                    "".join(chr((neg_cc >> (8 * i)) & 0xFF) for i in range(4)).strip()
                    if neg_cc
                    else "N/A"
                )
                logger.info(
                    "Negotiated capture format: {}x{} @ {:.0f} fps (FOURCC: {}).".format(
                        neg_w, neg_h, neg_fps, neg_cc
                    )
                )

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