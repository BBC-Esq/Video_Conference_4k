import queue
import threading
import time
from typing import Optional, Tuple, Any, Union
from numpy.typing import NDArray

from ..stream.video import VideoStream
from ..capture.audio import AudioCapture
from ..net.sync import SyncTransport
from ..net.audio import AudioTransport
from ..net.upnp import UPnPPortMapper
from ..utils.common import get_logger, log_version

logger = get_logger("DirectConference")


class DirectConference:

    def __init__(
        self,
        peer_address: str = "localhost",
        video_port: str = "5555",
        audio_port: str = "5556",
        peer_video_port: str = None,
        peer_audio_port: str = None,
        resolution: Tuple[int, int] = (1920, 1080),
        framerate: int = 30,
        camera_id: int = 0,
        microphone_id: Optional[int] = None,
        video_source: Any = None,
        gpu_accelerated: bool = True,
        gpu_codec: str = "h264",
        gpu_bitrate: int = 8000000,
        enable_audio: bool = True,
        audio_bitrate: int = 32000,
        enable_upnp: bool = False,
        logging: bool = False,
    ):
        self.__logging = logging if isinstance(logging, bool) else False
        log_version(logging=self.__logging)

        self.__peer_address = peer_address
        self.__video_port = str(video_port)
        self.__audio_port = str(audio_port)
        self.__peer_video_port = str(peer_video_port) if peer_video_port is not None else str(video_port)
        self.__peer_audio_port = str(peer_audio_port) if peer_audio_port is not None else str(audio_port)

        self.__framerate = framerate
        self.__gpu_accelerated = gpu_accelerated
        self.__gpu_codec = gpu_codec
        self.__gpu_bitrate = gpu_bitrate
        self.__enable_audio = enable_audio
        self.__audio_bitrate = audio_bitrate
        self.__enable_upnp = enable_upnp

        self.__owns_video_source = False
        if video_source is None:
            self.__video_source = VideoStream(
                source=camera_id, resolution=resolution, framerate=framerate, logging=logging
            )
            self.__owns_video_source = True
        elif isinstance(video_source, int):
            self.__video_source = VideoStream(
                source=video_source, resolution=resolution, framerate=framerate, logging=logging
            )
            self.__owns_video_source = True
        elif hasattr(video_source, "read"):
            self.__video_source = video_source
        else:
            raise ValueError("video_source must be int, None, or an object with a read() method.")

        self.__audio = None
        if enable_audio:
            self.__audio = AudioCapture(
                input_device=microphone_id,
                sample_rate=48000,
                channels=1,
                enable_input=True,
                enable_output=True,
                logging=logging,
            )

        self.__send_video = None
        self.__recv_video = None
        self.__send_audio = None
        self.__recv_audio = None
        self.__audio_sub = None
        self.__upnp = None

        self.__remote_frame: Optional[NDArray] = None
        self.__local_frame: Optional[NDArray] = None
        self.__frame_lock = threading.Lock()

        self.__terminate = threading.Event()
        self.__threads = []
        self.__is_running = False
        self.__join_timeout = 6.0

    @property
    def is_running(self) -> bool:
        return self.__is_running

    def start(self) -> "DirectConference":
        if self.__is_running:
            return self

        if self.__enable_upnp:
            self.__upnp = UPnPPortMapper(description="VideoConference4k", logging=self.__logging)
            if self.__upnp.discover():
                self.__upnp.map_port(int(self.__video_port), "TCP")
                if self.__enable_audio:
                    self.__upnp.map_port(int(self.__audio_port), "TCP")
            else:
                self.__logging and logger.debug("No UPnP gateway; relying on direct/STUN/TURN reachability.")

        if hasattr(self.__video_source, "start"):
            if not getattr(self.__video_source, "is_running", False):
                self.__video_source.start()

        self.__recv_video = SyncTransport(
            address="*", port=self.__video_port, receive_mode=True,
            gpu_accelerated=self.__gpu_accelerated, gpu_codec=self.__gpu_codec,
            gpu_bitrate=self.__gpu_bitrate, logging=self.__logging,
        )
        self.__send_video = SyncTransport(
            address=self.__peer_address, port=self.__peer_video_port,
            gpu_accelerated=self.__gpu_accelerated, gpu_codec=self.__gpu_codec,
            gpu_bitrate=self.__gpu_bitrate, logging=self.__logging,
        )

        if self.__enable_audio:
            self.__audio.start()
            self.__audio_sub = self.__audio.subscribe()
            self.__recv_audio = AudioTransport(
                address="*", port=self.__audio_port, receive_mode=True,
                sample_rate=48000, channels=1, logging=self.__logging,
            )
            self.__send_audio = AudioTransport(
                address=self.__peer_address, port=self.__peer_audio_port,
                sample_rate=48000, channels=1, bitrate=self.__audio_bitrate, logging=self.__logging,
            )

        self.__terminate.clear()
        self.__threads = [
            threading.Thread(target=self.__video_send_loop, name="DirectVideoSend", daemon=True),
            threading.Thread(target=self.__video_recv_loop, name="DirectVideoRecv", daemon=True),
        ]
        if self.__enable_audio:
            self.__threads.append(threading.Thread(target=self.__audio_send_loop, name="DirectAudioSend", daemon=True))
            self.__threads.append(threading.Thread(target=self.__audio_recv_loop, name="DirectAudioRecv", daemon=True))

        for t in self.__threads:
            t.start()
        self.__is_running = True
        self.__logging and logger.debug("DirectConference started with peer {}.".format(self.__peer_address))
        return self

    def __video_send_loop(self) -> None:
        interval = 1.0 / self.__framerate if self.__framerate > 0 else 0.0
        while not self.__terminate.is_set():
            start = time.perf_counter()
            frame = self.__video_source.read()
            if frame is not None:
                with self.__frame_lock:
                    self.__local_frame = frame
                try:
                    self.__send_video.send(frame)
                except Exception as e:
                    self.__logging and logger.debug("Video send error: {}".format(e))
            wait = interval - (time.perf_counter() - start)
            if wait > 0:
                self.__terminate.wait(wait)

    def __video_recv_loop(self) -> None:
        while not self.__terminate.is_set():
            try:
                frame = self.__recv_video.recv()
            except Exception:
                break
            if frame is None:
                break
            with self.__frame_lock:
                self.__remote_frame = frame

    def __audio_send_loop(self) -> None:
        while not self.__terminate.is_set():
            try:
                chunk = self.__audio_sub.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                self.__send_audio.send(chunk)
            except Exception as e:
                self.__logging and logger.debug("Audio send error: {}".format(e))

    def __audio_recv_loop(self) -> None:
        while not self.__terminate.is_set():
            chunk = self.__recv_audio.recv()
            if chunk is not None:
                self.__audio.write(chunk)
            else:
                self.__terminate.wait(0.005)

    def get_remote_frame(self) -> Optional[NDArray]:
        with self.__frame_lock:
            return self.__remote_frame

    def get_local_frame(self) -> Optional[NDArray]:
        with self.__frame_lock:
            return self.__local_frame

    def stop(self) -> None:
        self.__terminate.set()

        transports = (self.__recv_video, self.__send_video, self.__recv_audio, self.__send_audio)

        for transport in transports:
            if transport is not None:
                try:
                    transport.signal_stop()
                except Exception:
                    pass

        for t in self.__threads:
            t.join(timeout=self.__join_timeout)
        self.__threads = []

        for transport in transports:
            if transport is not None:
                try:
                    transport.close()
                except Exception:
                    pass

        if self.__audio is not None:
            if self.__audio_sub is not None:
                self.__audio.unsubscribe(self.__audio_sub)
                self.__audio_sub = None
            self.__audio.stop()

        if self.__owns_video_source and hasattr(self.__video_source, "stop"):
            self.__video_source.stop()

        if self.__upnp is not None:
            self.__upnp.close()
            self.__upnp = None

        self.__send_video = self.__recv_video = None
        self.__send_audio = self.__recv_audio = None
        self.__is_running = False
        with self.__frame_lock:
            self.__remote_frame = None
            self.__local_frame = None
        self.__logging and logger.debug("DirectConference stopped.")
