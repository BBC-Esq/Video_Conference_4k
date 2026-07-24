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


def _percentile(sorted_vals: list, pct: float) -> float:
    if not sorted_vals:
        return 0.0
    k = (len(sorted_vals) - 1) * pct / 100.0
    lo = int(k)
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo)


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
            "requested_fps": req_fps,
            "opened": False,
            "delivered": None,
            "negotiated_fps": 0.0,
            "measured_fps": 0.0,
            "worst_interval_ms": 0.0,
            "p99_interval_ms": 0.0,
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

            intervals = []
            frames = 0
            last = time.perf_counter()
            while frames < sample:
                grabbed, _ = cap.read()
                if not grabbed:
                    break
                now = time.perf_counter()
                intervals.append(now - last)
                last = now
                frames += 1

            intervals.sort()
            median = _percentile(intervals, 50)
            entry["delivered"] = (
                int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            )
            entry["negotiated_fps"] = round(cap.get(cv2.CAP_PROP_FPS), 2)
            entry["measured_fps"] = round(1.0 / median, 2) if median > 0 else 0.0
            entry["worst_interval_ms"] = round(max(intervals) * 1000.0, 2) if intervals else 0.0
            entry["p99_interval_ms"] = round(_percentile(intervals, 99) * 1000.0, 2)
            cc = int(cap.get(cv2.CAP_PROP_FOURCC))
            entry["fourcc"] = (
                "".join(chr((cc >> (8 * i)) & 0xFF) for i in range(4)).strip()
                if cc else ""
            )
            logging and logger.info(
                "Probe {}x{}@{} -> delivered {} negotiated {} measured {} fps, "
                "worst {}ms p99 {}ms (FOURCC {}).".format(
                    req_w, req_h, req_fps, entry["delivered"], entry["negotiated_fps"],
                    entry["measured_fps"], entry["worst_interval_ms"], entry["p99_interval_ms"],
                    entry["fourcc"] or "N/A"
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
        self.__capture_failed = False

        self.__max_read_failures = options.pop("MAX_READ_FAILURES", 5)
        self.__max_reopen_attempts = options.pop("MAX_REOPEN_ATTEMPTS", 3)

        options = {str(k).strip(): v for k, v in options.items()}

        self.__source = source
        self.__backend = backend
        self.__cap_options = dict(options)

        if backend and isinstance(backend, int):
            logger.info("Setting backend `{}` for this source.".format(backend))

        self.stream = self.__open_stream()

        self.color_space = None

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
            try:
                self.stream.release()
            except Exception:
                pass
            self.stream = None
            raise RuntimeError(
                "[VideoCapture:ERROR] :: Source is invalid, VideoCapture failed to initialize stream on this source!"
            )

        self.__thread = None

        self.__terminate = Event()

        self.__stream_read = Event()

        self.__frame_lock = Lock()

        self.__frame_seq = 0
        self.__frame_pts_ns = time.perf_counter_ns()
        self.__synth_seq = 0

        self.__is_running = False

    @property
    def is_running(self) -> bool:
        return self.__is_running

    @property
    def capture_failed(self) -> bool:
        return self.__capture_failed

    def __open_stream(self):
        if self.__backend and isinstance(self.__backend, int):
            if check_CV_version() == 3:
                stream = cv2.VideoCapture(self.__source + self.__backend)
            else:
                stream = cv2.VideoCapture(self.__source, self.__backend)
        else:
            stream = cv2.VideoCapture(self.__source)

        for key, value in self.__cap_options.items():
            prop = capPropId(key, logging=False)
            prop is not None and stream.set(prop, value)
        return stream

    def __reopen_stream(self) -> bool:
        try:
            if self.stream is not None:
                self.stream.release()
        except Exception:
            pass
        self.stream = None
        try:
            stream = self.__open_stream()
            grabbed, frame = stream.read()
            if grabbed:
                self.stream = stream
                with self.__frame_lock:
                    self.frame = frame
                return True
            stream.release()
        except Exception as e:
            logger.error("Camera reopen failed: {}".format(e))
        return False

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
        consecutive_failures = 0
        reopen_attempts = 0
        frame_period = 1.0 / self.framerate if self.framerate > 1.0 else 1.0 / 30.0

        while not self.__terminate.is_set():
            self.__stream_read.clear()

            grabbed, frame = False, None
            if self.stream is not None:
                (grabbed, frame) = self.stream.read()

            if not grabbed:
                if self.__threaded_queue_mode:
                    if self.__queue.empty():
                        break
                    else:
                        continue

                consecutive_failures += 1
                if consecutive_failures <= self.__max_read_failures:
                    self.__terminate.wait(frame_period)
                    continue

                if reopen_attempts >= self.__max_reopen_attempts:
                    logger.error(
                        "Camera stopped delivering frames and could not be recovered "
                        "after {} reopen attempts.".format(reopen_attempts)
                    )
                    self.__capture_failed = True
                    break

                reopen_attempts += 1
                logger.warning(
                    "Camera stopped delivering frames; reopening (attempt {} of {}).".format(
                        reopen_attempts, self.__max_reopen_attempts
                    )
                )
                if self.__reopen_stream():
                    logger.info("Camera reopened successfully.")
                    consecutive_failures = 0
                else:
                    self.__terminate.wait(frame_period * 5)
                continue

            consecutive_failures = 0
            reopen_attempts = 0

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
                    self.__frame_seq += 1
                    self.__frame_pts_ns = time.perf_counter_ns()

            self.__stream_read.set()

        self.__threaded_queue_mode and self.__queue.put(None)
        self.__threaded_queue_mode = False

        self.__terminate.set()
        self.__stream_read.set()
        self.__is_running = False

        if self.stream is not None:
            try:
                self.stream.release()
            except Exception:
                pass
            self.stream = None

    def read(self) -> Optional[NDArray]:
        if self.__capture_failed:
            return None

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

    def read_timed(self):
        if self.__capture_failed:
            return None, 0, -1

        if self.__threaded_queue_mode:
            frame = self.read()
            if frame is None:
                return None, 0, -1
            self.__synth_seq += 1
            return frame, time.perf_counter_ns(), self.__synth_seq

        if not self.__is_running:
            with self.__frame_lock:
                return self.frame, self.__frame_pts_ns, self.__frame_seq

        if not self.__terminate.is_set() and self.__stream_read.wait(timeout=self.__thread_timeout):
            with self.__frame_lock:
                return self.frame, self.__frame_pts_ns, self.__frame_seq
        return None, 0, -1

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

        if self.stream is not None:
            try:
                self.stream.release()
            except Exception:
                pass
            self.stream = None