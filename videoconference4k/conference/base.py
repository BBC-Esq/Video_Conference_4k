import json
import base64
import zlib
from typing import Optional, Tuple, Callable
from numpy.typing import NDArray

from ..stream.video import VideoStream
from ..capture.audio import AudioCapture
from ..utils.common import get_logger, log_version

logger = get_logger("ConferenceBase")

STUN_ONLY_SERVERS = [
    {"urls": ["stun:stun.l.google.com:19302"]},
    {"urls": ["stun:stun1.l.google.com:19302"]},
    {"urls": ["stun:stun.stunprotocol.org:3478"]},
]


def compress_sdp(sdp_dict: dict) -> str:
    json_bytes = json.dumps(sdp_dict, separators=(',', ':')).encode('utf-8')
    compressed = zlib.compress(json_bytes, level=9)
    return base64.urlsafe_b64encode(compressed).decode('ascii')


def decompress_sdp(code: str) -> dict:
    try:
        compressed = base64.urlsafe_b64decode(code.encode('ascii'))
        json_bytes = zlib.decompress(compressed)
        return json.loads(json_bytes.decode('utf-8'))
    except Exception as e:
        raise ValueError(
            "Invalid code. Make sure you copied the entire code correctly."
        ) from e


def make_turn_server(
    host: str,
    username: str,
    credential: str,
    port: int = 3478,
    tls: bool = False,
    transport: Optional[str] = None,
) -> dict:
    scheme = "turns" if tls else "turn"
    url = "{}:{}:{}".format(scheme, host, port)
    if transport:
        url = "{}?transport={}".format(url, transport)
    return {
        "urls": [url],
        "username": username,
        "credential": credential,
    }


class BaseConference:

    def __init__(
        self,
        resolution: Tuple[int, int] = (1920, 1080),
        framerate: int = 30,
        enable_audio: bool = True,
        camera_id: int = 0,
        microphone_id: Optional[int] = None,
        ice_servers: Optional[list] = None,
        logging: bool = False,
        **options: dict
    ):
        self._logging = logging if isinstance(logging, bool) else False

        log_version(logging=self._logging)

        self._resolution = resolution
        self._framerate = framerate
        self._enable_audio = enable_audio

        self._ice_servers = ice_servers if ice_servers is not None else STUN_ONLY_SERVERS

        options = {str(k).strip(): v for k, v in options.items()}

        self._video = VideoStream(
            source=camera_id,
            resolution=resolution,
            framerate=framerate,
            logging=logging,
            **options
        )

        self._audio = None
        if enable_audio:
            self._audio = AudioCapture(
                input_device=microphone_id,
                sample_rate=48000,
                channels=1,
                enable_output=False,
                logging=logging,
            )

        self._media_started = False

    @property
    def resolution(self) -> Tuple[int, int]:
        return self._resolution

    @property
    def framerate(self) -> int:
        return self._framerate

    @property
    def enable_audio(self) -> bool:
        return self._enable_audio

    def _start_local_media(self) -> None:
        if self._media_started:
            return

        self._logging and logger.debug("Starting local media devices")
        self._video.start()
        if self._audio:
            self._audio.start()
        self._media_started = True

    def _stop_local_media(self) -> None:
        if not self._media_started:
            return

        self._logging and logger.debug("Stopping local media devices")
        self._video.stop()
        if self._audio:
            self._audio.stop()
        self._media_started = False

    def get_local_frame(self) -> Optional[NDArray]:
        if not self._media_started:
            return None
        return self._video.read()


class CallbackRegistrar:

    def __init__(self):
        self._on_remote_video = None
        self._on_remote_audio = None

    def on_remote_video(self, callback: Callable[[NDArray], None]):
        self._on_remote_video = callback
        return callback

    def on_remote_audio(self, callback: Callable[[NDArray], None]):
        self._on_remote_audio = callback
        return callback

    def _invoke_video_callback(self, frame: NDArray, logger_ref=None) -> None:
        if self._on_remote_video is not None:
            try:
                self._on_remote_video(frame)
            except Exception as e:
                if logger_ref:
                    logger_ref.error("Error in video callback: {}".format(e))

    def _invoke_audio_callback(self, audio: NDArray, logger_ref=None) -> None:
        if self._on_remote_audio is not None:
            try:
                self._on_remote_audio(audio)
            except Exception as e:
                if logger_ref:
                    logger_ref.error("Error in audio callback: {}".format(e))