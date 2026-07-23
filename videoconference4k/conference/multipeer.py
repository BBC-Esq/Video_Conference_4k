import threading
import time
from typing import Optional, Callable, Tuple, Dict
import numpy as np
from numpy.typing import NDArray

from .base import (
    BaseConference,
    compress_sdp,
    decompress_sdp,
    STUN_ONLY_SERVERS,
)
from ..rtc.connection import RTCConnection
from ..utils.common import get_logger

logger = get_logger("MultiPeerConference")


class MultiPeerConference(BaseConference):

    def __init__(
        self,
        resolution: Tuple[int, int] = (1920, 1080),
        framerate: int = 30,
        enable_audio: bool = True,
        camera_id: int = 0,
        microphone_id: Optional[int] = None,
        max_peers: int = 3,
        use_stun: bool = True,
        turn_servers: Optional[list] = None,
        logging: bool = False,
        **options: dict
    ):
        ice_servers = list(STUN_ONLY_SERVERS) if use_stun else []
        if turn_servers:
            ice_servers = ice_servers + list(turn_servers)
            logging and logger.debug(
                "TURN relay configured with {} server(s).".format(len(turn_servers))
            )

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

        self.__max_peers = max_peers

        self.__peers: Dict[str, RTCConnection] = {}
        self.__peer_frames: Dict[str, NDArray] = {}
        self.__peer_audio: Dict[str, NDArray] = {}
        self.__frame_lock = threading.Lock()

        self.__on_remote_video: Optional[Callable[[str, NDArray], None]] = None
        self.__on_remote_audio: Optional[Callable[[str, NDArray], None]] = None
        self.__on_peer_connected: Optional[Callable[[str], None]] = None
        self.__on_peer_disconnected: Optional[Callable[[str], None]] = None

        self._logging and logger.debug(
            "MultiPeerConference initialized: max {} peers".format(max_peers)
        )

    @property
    def peer_count(self) -> int:
        return len(self.__peers)

    @property
    def peer_names(self) -> list:
        return list(self.__peers.keys())

    @property
    def max_peers(self) -> int:
        return self.__max_peers

    def on_remote_video(self, callback: Callable[[str, NDArray], None]):
        self.__on_remote_video = callback
        return callback

    def on_remote_audio(self, callback: Callable[[str, NDArray], None]):
        self.__on_remote_audio = callback
        return callback

    def on_peer_connected(self, callback: Callable[[str], None]):
        self.__on_peer_connected = callback
        return callback

    def on_peer_disconnected(self, callback: Callable[[str], None]):
        self.__on_peer_disconnected = callback
        return callback

    def __create_rtc_for_peer(self, peer_name: str) -> RTCConnection:
        rtc = RTCConnection(
            video_source=self._video,
            audio_source=self._audio,
            framerate=self._framerate,
            enable_video=True,
            enable_audio=self._enable_audio,
            ice_servers=self._ice_servers,
            logging=self._logging,
        )

        def make_video_handler(name):
            def handler(frame):
                with self.__frame_lock:
                    self.__peer_frames[name] = frame
                if self.__on_remote_video:
                    try:
                        self.__on_remote_video(name, frame)
                    except Exception as e:
                        logger.error("Error in video callback: {}".format(e))
            return handler

        def make_audio_handler(name):
            def handler(audio):
                with self.__frame_lock:
                    self.__peer_audio[name] = audio
                if self.__on_remote_audio:
                    try:
                        self.__on_remote_audio(name, audio)
                    except Exception as e:
                        logger.error("Error in audio callback: {}".format(e))
            return handler

        def make_state_handler(name):
            def handler(state):
                self._logging and logger.info(
                    "Peer '{}' connection state: {}".format(name, state)
                )
                if state == "connected":
                    logger.critical("Connected to peer: {}".format(name))
                    if self.__on_peer_connected:
                        try:
                            self.__on_peer_connected(name)
                        except Exception as e:
                            logger.error("Error in connected callback: {}".format(e))
                elif state in ("failed", "disconnected", "closed"):
                    logger.warning("Peer '{}' disconnected".format(name))
                    if self.__on_peer_disconnected:
                        try:
                            self.__on_peer_disconnected(name)
                        except Exception as e:
                            logger.error("Error in disconnected callback: {}".format(e))
            return handler

        rtc.on_video_frame(make_video_handler(peer_name))
        rtc.on_audio_frame(make_audio_handler(peer_name))
        rtc.on_connection_state(make_state_handler(peer_name))

        return rtc

    def __check_peer_limits(self, peer_name: str) -> None:
        if peer_name in self.__peers:
            raise RuntimeError(
                "Already have a connection for peer '{}'. "
                "Use a different name or remove the existing one.".format(peer_name)
            )

        if len(self.__peers) >= self.__max_peers:
            raise RuntimeError(
                "Maximum {} peers reached. Cannot add more.".format(self.__max_peers)
            )

    def create_invite_for_peer(self, peer_name: str) -> str:
        self.__check_peer_limits(peer_name)

        self._start_local_media()

        rtc = self.__create_rtc_for_peer(peer_name)
        self.__peers[peer_name] = rtc

        offer = rtc.create_offer()
        invite_code = compress_sdp(offer)

        self._logging and logger.info(
            "Created invite for '{}' ({} chars)".format(peer_name, len(invite_code))
        )

        return invite_code

    def accept_invite_from_peer(self, peer_name: str, invite_code: str) -> str:
        self.__check_peer_limits(peer_name)

        self._start_local_media()

        rtc = self.__create_rtc_for_peer(peer_name)
        self.__peers[peer_name] = rtc

        offer = decompress_sdp(invite_code)
        answer = rtc.create_answer(offer)
        response_code = compress_sdp(answer)

        self._logging and logger.info(
            "Created response for '{}' ({} chars)".format(peer_name, len(response_code))
        )

        return response_code

    def complete_connection_with_peer(self, peer_name: str, response_code: str):
        if peer_name not in self.__peers:
            raise RuntimeError(
                "No pending connection for peer '{}'. "
                "Call create_invite_for_peer() first.".format(peer_name)
            )

        answer = decompress_sdp(response_code)
        self.__peers[peer_name].set_answer(answer)

        self._logging and logger.info(
            "Completed connection handshake with '{}'".format(peer_name)
        )

    def remove_peer(self, peer_name: str):
        if peer_name not in self.__peers:
            return

        self._logging and logger.debug("Removing peer: {}".format(peer_name))

        rtc = self.__peers.pop(peer_name)
        rtc.stop()

        with self.__frame_lock:
            self.__peer_frames.pop(peer_name, None)
            self.__peer_audio.pop(peer_name, None)

    def get_peer_frame(self, peer_name: str) -> Optional[NDArray]:
        with self.__frame_lock:
            return self.__peer_frames.get(peer_name)

    def get_all_peer_frames(self) -> Dict[str, NDArray]:
        with self.__frame_lock:
            return dict(self.__peer_frames)

    def get_peer_audio(self, peer_name: str) -> Optional[NDArray]:
        with self.__frame_lock:
            return self.__peer_audio.get(peer_name)

    def wait_for_peer(self, peer_name: str, timeout: float = 30.0) -> bool:
        if peer_name not in self.__peers:
            return False

        rtc = self.__peers[peer_name]
        start = time.time()
        while (time.time() - start) < timeout:
            if rtc.connection_state == "connected":
                return True
            time.sleep(0.1)
        return False

    def stop(self):
        self._logging and logger.debug("Stopping MultiPeerConference")

        for peer_name in list(self.__peers.keys()):
            self.remove_peer(peer_name)

        self._stop_local_media()

        self._logging and logger.debug("MultiPeerConference stopped")