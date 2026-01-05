import json
import base64
import zlib
import threading
import time
import logging as log
from typing import Optional, Callable, Tuple
import numpy as np
from numpy.typing import NDArray

from ..stream.video import VideoStream
from ..capture.audio import AudioCapture
from ..rtc.connection import RTCConnection
from ..utils.common import logger_handler, log_version

logger = log.getLogger("PeerConference")
logger.propagate = False
logger.addHandler(logger_handler())
logger.setLevel(log.DEBUG)

STUN_ONLY_SERVERS = [
    {"urls": ["stun:stun.l.google.com:19302"]},
    {"urls": ["stun:stun1.l.google.com:19302"]},
    {"urls": ["stun:stun.stunprotocol.org:3478"]},
]


def _compress_sdp(sdp_dict: dict) -> str:
    json_bytes = json.dumps(sdp_dict, separators=(',', ':')).encode('utf-8')
    compressed = zlib.compress(json_bytes, level=9)
    return base64.urlsafe_b64encode(compressed).decode('ascii')


def _decompress_sdp(code: str) -> dict:
    try:
        compressed = base64.urlsafe_b64decode(code.encode('ascii'))
        json_bytes = zlib.decompress(compressed)
        return json.loads(json_bytes.decode('utf-8'))
    except Exception as e:
        raise ValueError(
            "Invalid code. Make sure you copied the entire code correctly."
        ) from e


class PeerConference:

    def __init__(
        self,
        resolution: Tuple[int, int] = (1920, 1080),
        framerate: int = 30,
        enable_audio: bool = True,
        camera_id: int = 0,
        microphone_id: Optional[int] = None,
        use_stun: bool = True,
        logging: bool = False,
        **options: dict
    ):
        self.__logging = logging if isinstance(logging, bool) else False

        log_version(logging=self.__logging)

        self.__resolution = resolution
        self.__framerate = framerate
        self.__enable_audio = enable_audio

        self.__ice_servers = STUN_ONLY_SERVERS if use_stun else []

        options = {str(k).strip(): v for k, v in options.items()}

        self.__video = VideoStream(
            source=camera_id,
            resolution=resolution,
            framerate=framerate,
            logging=logging,
            **options
        )

        self.__audio = None
        if enable_audio:
            self.__audio = AudioCapture(
                input_device=microphone_id,
                sample_rate=48000,
                channels=1,
                enable_output=True,
                logging=logging,
            )

        self.__rtc: Optional[RTCConnection] = None

        self.__is_initiator = False
        self.__is_connected = False
        self.__is_streaming = False
        self.__media_started = False

        self.__on_remote_video: Optional[Callable[[NDArray], None]] = None
        self.__on_remote_audio: Optional[Callable[[NDArray], None]] = None
        self.__on_connected: Optional[Callable[[], None]] = None
        self.__on_disconnected: Optional[Callable[[], None]] = None

        self.__last_remote_frame: Optional[NDArray] = None
        self.__last_remote_audio: Optional[NDArray] = None
        self.__frame_lock = threading.Lock()

        self.__logging and logger.debug(
            "PeerConference initialized: {}x{} @ {}fps, audio={}".format(
                resolution[0], resolution[1], framerate, enable_audio
            )
        )

    @property
    def is_connected(self) -> bool:
        return self.__is_connected

    @property
    def is_streaming(self) -> bool:
        return self.__is_streaming

    @property
    def resolution(self) -> Tuple[int, int]:
        return self.__resolution

    @property
    def framerate(self) -> int:
        return self.__framerate

    def on_remote_video(self, callback: Callable[[NDArray], None]):
        self.__on_remote_video = callback
        return callback

    def on_remote_audio(self, callback: Callable[[NDArray], None]):
        self.__on_remote_audio = callback
        return callback

    def on_connected(self, callback: Callable[[], None]):
        self.__on_connected = callback
        return callback

    def on_disconnected(self, callback: Callable[[], None]):
        self.__on_disconnected = callback
        return callback

    def __start_local_media(self):
        if self.__media_started:
            return

        self.__logging and logger.debug("Starting local media devices")
        self.__video.start()
        if self.__audio:
            self.__audio.start()
        self.__media_started = True

    def __setup_rtc(self):
        self.__rtc = RTCConnection(
            video_source=self.__video,
            audio_source=self.__audio,
            framerate=self.__framerate,
            sample_rate=48000,
            audio_channels=1,
            enable_video=True,
            enable_audio=self.__enable_audio,
            ice_servers=self.__ice_servers,
            logging=self.__logging,
        )

        def handle_video(frame: NDArray):
            with self.__frame_lock:
                self.__last_remote_frame = frame
            if self.__on_remote_video:
                try:
                    self.__on_remote_video(frame)
                except Exception as e:
                    logger.error("Error in video callback: {}".format(e))

        def handle_audio(audio: NDArray):
            with self.__frame_lock:
                self.__last_remote_audio = audio
            if self.__on_remote_audio:
                try:
                    self.__on_remote_audio(audio)
                except Exception as e:
                    logger.error("Error in audio callback: {}".format(e))

        def handle_state(state: str):
            self.__logging and logger.info("Connection state: {}".format(state))
            if state == "connected":
                self.__is_connected = True
                logger.critical("Peer-to-peer connection established!")
                if self.__on_connected:
                    try:
                        self.__on_connected()
                    except Exception as e:
                        logger.error("Error in connected callback: {}".format(e))
            elif state in ("failed", "disconnected", "closed"):
                was_connected = self.__is_connected
                self.__is_connected = False
                if was_connected:
                    logger.warning("Connection lost: {}".format(state))
                    if self.__on_disconnected:
                        try:
                            self.__on_disconnected()
                        except Exception as e:
                            logger.error("Error in disconnected callback: {}".format(e))

        self.__rtc.on_video_frame(handle_video)
        self.__rtc.on_audio_frame(handle_audio)
        self.__rtc.on_connection_state(handle_state)

    def create_invite(self) -> str:
        if self.__rtc is not None:
            raise RuntimeError(
                "Already in a session. Call stop() first to start a new one."
            )

        self.__is_initiator = True

        self.__start_local_media()

        self.__setup_rtc()

        self.__logging and logger.debug("Creating WebRTC offer...")

        offer = self.__rtc.create_offer()

        invite_code = _compress_sdp(offer)

        self.__logging and logger.info(
            "Created invite code ({} characters)".format(len(invite_code))
        )

        return invite_code

    def accept_invite(self, invite_code: str) -> str:
        if self.__rtc is not None:
            raise RuntimeError(
                "Already in a session. Call stop() first to start a new one."
            )

        self.__is_initiator = False

        self.__start_local_media()

        self.__setup_rtc()

        self.__logging and logger.debug("Decoding invite code...")
        offer = _decompress_sdp(invite_code)

        self.__logging and logger.debug("Creating WebRTC answer...")

        answer = self.__rtc.create_answer(offer)

        response_code = _compress_sdp(answer)

        self.__logging and logger.info(
            "Created response code ({} characters)".format(len(response_code))
        )

        self.__is_streaming = True

        return response_code

    def complete_connection(self, response_code: str):
        if not self.__is_initiator:
            raise RuntimeError(
                "Only the person who created the invite should call complete_connection(). "
                "If you're joining, your connection starts automatically after accept_invite()."
            )

        if self.__rtc is None:
            raise RuntimeError(
                "Must call create_invite() first before complete_connection()."
            )

        self.__logging and logger.debug("Decoding response code...")
        answer = _decompress_sdp(response_code)

        self.__logging and logger.debug("Setting remote answer...")
        self.__rtc.set_answer(answer)

        self.__is_streaming = True
        self.__logging and logger.info(
            "Connection process started. Waiting for peer-to-peer link..."
        )

    def get_local_frame(self) -> Optional[NDArray]:
        if not self.__media_started:
            return None
        return self.__video.read()

    def get_remote_frame(self) -> Optional[NDArray]:
        with self.__frame_lock:
            return self.__last_remote_frame

    def get_remote_audio(self) -> Optional[NDArray]:
        with self.__frame_lock:
            return self.__last_remote_audio

    def wait_for_connection(self, timeout: float = 30.0) -> bool:
        start = time.time()
        while not self.__is_connected and (time.time() - start) < timeout:
            time.sleep(0.1)
        return self.__is_connected

    def stop(self):
        self.__logging and logger.debug("Stopping PeerConference")

        self.__is_streaming = False
        self.__is_connected = False

        if self.__rtc:
            self.__rtc.stop()
            self.__rtc = None

        if self.__media_started:
            self.__video.stop()
            if self.__audio:
                self.__audio.stop()
            self.__media_started = False

        with self.__frame_lock:
            self.__last_remote_frame = None
            self.__last_remote_audio = None

        self.__logging and logger.debug("PeerConference stopped")