import threading
import time
from typing import Optional, Callable, Tuple
import numpy as np
from numpy.typing import NDArray

from .base import (
    BaseConference,
    CallbackRegistrar,
    compress_sdp,
    decompress_sdp,
    STUN_ONLY_SERVERS,
    logger as base_logger,
)
from ..rtc.connection import RTCConnection
from ..utils.common import get_logger

logger = get_logger("PeerConference")


class PeerConference(BaseConference, CallbackRegistrar):

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
        ice_servers = STUN_ONLY_SERVERS if use_stun else []

        BaseConference.__init__(
            self,
            resolution=resolution,
            framerate=framerate,
            enable_audio=enable_audio,
            camera_id=camera_id,
            microphone_id=microphone_id,
            ice_servers=ice_servers,
            logging=logging,
            **options
        )

        CallbackRegistrar.__init__(self)

        self.__rtc: Optional[RTCConnection] = None

        self.__is_initiator = False
        self.__is_connected = False
        self.__is_streaming = False

        self.__on_connected: Optional[Callable[[], None]] = None
        self.__on_disconnected: Optional[Callable[[], None]] = None

        self.__last_remote_frame: Optional[NDArray] = None
        self.__last_remote_audio: Optional[NDArray] = None
        self.__frame_lock = threading.Lock()

        self._logging and logger.debug(
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

    def on_connected(self, callback: Callable[[], None]):
        self.__on_connected = callback
        return callback

    def on_disconnected(self, callback: Callable[[], None]):
        self.__on_disconnected = callback
        return callback

    def __setup_rtc(self):
        self.__rtc = RTCConnection(
            video_source=self._video,
            audio_source=self._audio,
            framerate=self._framerate,
            sample_rate=48000,
            audio_channels=1,
            enable_video=True,
            enable_audio=self._enable_audio,
            ice_servers=self._ice_servers,
            logging=self._logging,
        )

        def handle_video(frame: NDArray):
            with self.__frame_lock:
                self.__last_remote_frame = frame
            self._invoke_video_callback(frame, logger)

        def handle_audio(audio: NDArray):
            with self.__frame_lock:
                self.__last_remote_audio = audio
            self._invoke_audio_callback(audio, logger)

        def handle_state(state: str):
            self._logging and logger.info("Connection state: {}".format(state))
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

        self._start_local_media()
        self.__setup_rtc()

        self._logging and logger.debug("Creating WebRTC offer...")

        offer = self.__rtc.create_offer()
        invite_code = compress_sdp(offer)

        self._logging and logger.info(
            "Created invite code ({} characters)".format(len(invite_code))
        )

        return invite_code

    def accept_invite(self, invite_code: str) -> str:
        if self.__rtc is not None:
            raise RuntimeError(
                "Already in a session. Call stop() first to start a new one."
            )

        self.__is_initiator = False

        self._start_local_media()
        self.__setup_rtc()

        self._logging and logger.debug("Decoding invite code...")
        offer = decompress_sdp(invite_code)

        self._logging and logger.debug("Creating WebRTC answer...")

        answer = self.__rtc.create_answer(offer)
        response_code = compress_sdp(answer)

        self._logging and logger.info(
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

        self._logging and logger.debug("Decoding response code...")
        answer = decompress_sdp(response_code)

        self._logging and logger.debug("Setting remote answer...")
        self.__rtc.set_answer(answer)

        self.__is_streaming = True
        self._logging and logger.info(
            "Connection process started. Waiting for peer-to-peer link..."
        )

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
        self._logging and logger.debug("Stopping PeerConference")

        self.__is_streaming = False
        self.__is_connected = False

        if self.__rtc:
            self.__rtc.stop()
            self.__rtc = None

        self._stop_local_media()

        with self.__frame_lock:
            self.__last_remote_frame = None
            self.__last_remote_audio = None

        self._logging and logger.debug("PeerConference stopped")