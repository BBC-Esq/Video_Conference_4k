import asyncio
import threading
import time
import json
import logging as log
from typing import TypeVar, Optional, Callable, Any, Union
import numpy as np
from numpy.typing import NDArray

from ..utils.common import (
    logger_handler,
    import_dependency_safe,
    log_version,
)
from ..stream.video import VideoStream
from ..capture.audio import AudioCapture
from .tracks import LocalVideoTrack, LocalAudioTrack

aiortc = import_dependency_safe("aiortc", error="silent")
if aiortc is not None:
    from aiortc import (
        RTCPeerConnection,
        RTCSessionDescription,
        RTCIceCandidate,
        RTCConfiguration,
        RTCIceServer,
    )

logger = log.getLogger("RTCConnection")
logger.propagate = False
logger.addHandler(logger_handler())
logger.setLevel(log.DEBUG)

T = TypeVar("T", bound="RTCConnection")

AUDIO_SPECIFIC_OPTIONS = {
    "input_device",
    "output_device",
    "sample_rate",
    "channels",
    "chunk_size",
    "dtype",
    "enable_input",
    "enable_output",
    "latency",
    "blocksize",
}


class RTCConnection:

    def __init__(
        self,
        video_source: Any = None,
        audio_source: Union[AudioCapture, int, None] = None,
        framerate: Union[int, float] = 30,
        sample_rate: int = 48000,
        audio_channels: int = 1,
        enable_video: bool = True,
        enable_audio: bool = True,
        ice_servers: Optional[list] = None,
        logging: bool = False,
        **options: dict
    ):
        self.__logging = logging if isinstance(logging, bool) else False

        log_version(logging=self.__logging)

        import_dependency_safe("aiortc" if aiortc is None else "")

        if aiortc is None:
            raise ImportError("aiortc is required for RTCConnection. Install it with `pip install aiortc`.")

        self.__enable_video = enable_video
        self.__enable_audio = enable_audio
        self.__framerate = framerate
        self.__sample_rate = sample_rate
        self.__audio_channels = audio_channels

        self.__video_source = None
        self.__audio_source = None
        self.__owns_video_source = False
        self.__owns_audio_source = False

        options = {str(k).strip(): v for k, v in options.items()}

        video_options = {k: v for k, v in options.items() if k not in AUDIO_SPECIFIC_OPTIONS}

        if self.__enable_video:
            if video_source is None:
                self.__video_source = VideoStream(source=0, logging=logging, **video_options)
                self.__owns_video_source = True
            elif isinstance(video_source, int):
                self.__video_source = VideoStream(source=video_source, logging=logging, **video_options)
                self.__owns_video_source = True
            elif hasattr(video_source, "read") and callable(video_source.read):
                self.__video_source = video_source
                self.__owns_video_source = False
            else:
                raise ValueError("Invalid video_source. Must be int, None, or object with read() method.")

        if self.__enable_audio:
            if audio_source is None:
                self.__audio_source = AudioCapture(
                    sample_rate=self.__sample_rate,
                    channels=self.__audio_channels,
                    enable_output=False,
                    logging=logging,
                )
                self.__owns_audio_source = True
            elif isinstance(audio_source, int):
                self.__audio_source = AudioCapture(
                    input_device=audio_source,
                    sample_rate=self.__sample_rate,
                    channels=self.__audio_channels,
                    enable_output=False,
                    logging=logging,
                )
                self.__owns_audio_source = True
            elif isinstance(audio_source, AudioCapture):
                self.__audio_source = audio_source
                self.__owns_audio_source = False
            else:
                raise ValueError("Invalid audio_source. Must be int, None, or AudioCapture instance.")

        if ice_servers is None:
            self.__ice_servers = [
                RTCIceServer(urls=["stun:stun.l.google.com:19302"]),
                RTCIceServer(urls=["stun:stun1.l.google.com:19302"]),
            ]
        else:
            self.__ice_servers = []
            for server in ice_servers:
                if isinstance(server, dict):
                    self.__ice_servers.append(RTCIceServer(**server))
                elif isinstance(server, RTCIceServer):
                    self.__ice_servers.append(server)

        self.__pc = None
        self.__local_video_track = None
        self.__local_audio_track = None

        self.__on_video_frame_callback = None
        self.__on_audio_frame_callback = None
        self.__on_connection_state_callback = None
        self.__on_ice_candidate_callback = None

        self.__loop = None
        self.__thread = None
        self.__is_running = False
        self.__terminate = threading.Event()

        self.__pending_ice_candidates = []

        self.__audio_playback = None
        if self.__enable_audio:
            self.__audio_playback = AudioCapture(
                sample_rate=self.__sample_rate,
                channels=self.__audio_channels,
                enable_input=False,
                enable_output=True,
                logging=logging,
            )

        self.__logging and logger.debug("RTCConnection initialized.")

    @property
    def is_running(self) -> bool:
        return self.__is_running

    @property
    def connection_state(self) -> Optional[str]:
        if self.__pc is not None:
            return self.__pc.connectionState
        return None

    @property
    def ice_connection_state(self) -> Optional[str]:
        if self.__pc is not None:
            return self.__pc.iceConnectionState
        return None

    @property
    def ice_gathering_state(self) -> Optional[str]:
        if self.__pc is not None:
            return self.__pc.iceGatheringState
        return None

    def on_video_frame(self, callback: Callable[[NDArray], None]) -> None:
        if callable(callback):
            self.__on_video_frame_callback = callback
            self.__logging and logger.debug("Video frame callback registered.")
        else:
            logger.warning("Invalid callback. Must be callable.")

    def on_audio_frame(self, callback: Callable[[NDArray], None]) -> None:
        if callable(callback):
            self.__on_audio_frame_callback = callback
            self.__logging and logger.debug("Audio frame callback registered.")
        else:
            logger.warning("Invalid callback. Must be callable.")

    def on_connection_state(self, callback: Callable[[str], None]) -> None:
        if callable(callback):
            self.__on_connection_state_callback = callback
            self.__logging and logger.debug("Connection state callback registered.")
        else:
            logger.warning("Invalid callback. Must be callable.")

    def on_ice_candidate(self, callback: Callable[[dict], None]) -> None:
        if callable(callback):
            self.__on_ice_candidate_callback = callback
            self.__logging and logger.debug("ICE candidate callback registered.")
        else:
            logger.warning("Invalid callback. Must be callable.")

    def __ensure_loop(self):
        if self.__loop is None or self.__loop.is_closed():
            self.__loop = asyncio.new_event_loop()
            self.__thread = threading.Thread(target=self.__run_loop, daemon=True)
            self.__thread.start()
            time.sleep(0.1)

    def __run_loop(self):
        asyncio.set_event_loop(self.__loop)
        self.__loop.run_forever()

    def __run_coroutine(self, coro):
        self.__ensure_loop()
        future = asyncio.run_coroutine_threadsafe(coro, self.__loop)
        return future.result(timeout=30)

    async def __create_peer_connection(self):
        config = RTCConfiguration(iceServers=self.__ice_servers)
        self.__pc = RTCPeerConnection(configuration=config)

        @self.__pc.on("connectionstatechange")
        async def on_connectionstatechange():
            self.__logging and logger.debug("Connection state: {}".format(self.__pc.connectionState))
            if self.__on_connection_state_callback is not None:
                try:
                    self.__on_connection_state_callback(self.__pc.connectionState)
                except Exception as e:
                    logger.error("Error in connection state callback: {}".format(e))
            if self.__pc.connectionState == "failed":
                await self.__pc.close()

        @self.__pc.on("iceconnectionstatechange")
        async def on_iceconnectionstatechange():
            self.__logging and logger.debug("ICE connection state: {}".format(self.__pc.iceConnectionState))

        @self.__pc.on("icegatheringstatechange")
        async def on_icegatheringstatechange():
            self.__logging and logger.debug("ICE gathering state: {}".format(self.__pc.iceGatheringState))

        @self.__pc.on("icecandidate")
        async def on_icecandidate(candidate):
            if candidate and self.__on_ice_candidate_callback is not None:
                candidate_dict = {
                    "candidate": candidate.candidate,
                    "sdpMid": candidate.sdpMid,
                    "sdpMLineIndex": candidate.sdpMLineIndex,
                }
                try:
                    self.__on_ice_candidate_callback(candidate_dict)
                except Exception as e:
                    logger.error("Error in ICE candidate callback: {}".format(e))

        @self.__pc.on("track")
        async def on_track(track):
            self.__logging and logger.debug("Received track: {}".format(track.kind))
            if track.kind == "video":
                asyncio.ensure_future(self.__handle_video_track(track))
            elif track.kind == "audio":
                asyncio.ensure_future(self.__handle_audio_track(track))

    async def __handle_video_track(self, track):
        self.__logging and logger.debug("Handling incoming video track.")
        while not self.__terminate.is_set():
            try:
                frame = await asyncio.wait_for(track.recv(), timeout=1.0)
                img = frame.to_ndarray(format="bgr24")
                if self.__on_video_frame_callback is not None:
                    try:
                        self.__on_video_frame_callback(img)
                    except Exception as e:
                        logger.error("Error in video frame callback: {}".format(e))
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                if "MediaStreamError" not in str(type(e)):
                    logger.error("Error receiving video frame: {}".format(e))
                break

    async def __handle_audio_track(self, track):
        self.__logging and logger.debug("Handling incoming audio track.")
        if self.__audio_playback is not None and not self.__audio_playback.is_running:
            self.__audio_playback.start()
        while not self.__terminate.is_set():
            try:
                frame = await asyncio.wait_for(track.recv(), timeout=1.0)
                audio_data = frame.to_ndarray()
                if audio_data.ndim > 1:
                    audio_data = audio_data.T
                if self.__audio_playback is not None:
                    self.__audio_playback.write(audio_data.astype(np.int16))
                if self.__on_audio_frame_callback is not None:
                    try:
                        self.__on_audio_frame_callback(audio_data)
                    except Exception as e:
                        logger.error("Error in audio frame callback: {}".format(e))
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                if "MediaStreamError" not in str(type(e)):
                    logger.error("Error receiving audio frame: {}".format(e))
                break

    async def __add_local_tracks(self):
        if self.__enable_video and self.__video_source is not None:
            if LocalVideoTrack is None:
                raise RuntimeError("aiortc is not properly initialized. LocalVideoTrack is unavailable.")
            self.__local_video_track = LocalVideoTrack(
                video_source=self.__video_source,
                framerate=self.__framerate,
                logging=self.__logging,
            )
            self.__pc.addTrack(self.__local_video_track)
            self.__logging and logger.debug("Added local video track.")

        if self.__enable_audio and self.__audio_source is not None:
            if LocalAudioTrack is None:
                raise RuntimeError("aiortc is not properly initialized. LocalAudioTrack is unavailable.")
            self.__local_audio_track = LocalAudioTrack(
                audio_source=self.__audio_source,
                sample_rate=self.__sample_rate,
                channels=self.__audio_channels,
                logging=self.__logging,
            )
            self.__pc.addTrack(self.__local_audio_track)
            self.__logging and logger.debug("Added local audio track.")

    async def __create_offer_async(self) -> dict:
        await self.__create_peer_connection()
        await self.__add_local_tracks()
        offer = await self.__pc.createOffer()
        await self.__pc.setLocalDescription(offer)

        while self.__pc.iceGatheringState != "complete":
            await asyncio.sleep(0.1)

        return {
            "type": self.__pc.localDescription.type,
            "sdp": self.__pc.localDescription.sdp,
        }

    async def __create_answer_async(self, offer: dict) -> dict:
        await self.__create_peer_connection()
        await self.__add_local_tracks()

        remote_description = RTCSessionDescription(sdp=offer["sdp"], type=offer["type"])
        await self.__pc.setRemoteDescription(remote_description)

        for candidate in self.__pending_ice_candidates:
            await self.__pc.addIceCandidate(candidate)
        self.__pending_ice_candidates.clear()

        answer = await self.__pc.createAnswer()
        await self.__pc.setLocalDescription(answer)

        while self.__pc.iceGatheringState != "complete":
            await asyncio.sleep(0.1)

        return {
            "type": self.__pc.localDescription.type,
            "sdp": self.__pc.localDescription.sdp,
        }

    async def __set_answer_async(self, answer: dict):
        remote_description = RTCSessionDescription(sdp=answer["sdp"], type=answer["type"])
        await self.__pc.setRemoteDescription(remote_description)

        for candidate in self.__pending_ice_candidates:
            await self.__pc.addIceCandidate(candidate)
        self.__pending_ice_candidates.clear()

    async def __add_ice_candidate_async(self, candidate: dict):
        ice_candidate = RTCIceCandidate(
            candidate=candidate.get("candidate"),
            sdpMid=candidate.get("sdpMid"),
            sdpMLineIndex=candidate.get("sdpMLineIndex"),
        )
        if self.__pc is not None and self.__pc.remoteDescription is not None:
            await self.__pc.addIceCandidate(ice_candidate)
        else:
            self.__pending_ice_candidates.append(ice_candidate)

    def create_offer(self) -> dict:
        self.__logging and logger.debug("Creating offer.")
        self.__is_running = True
        return self.__run_coroutine(self.__create_offer_async())

    def create_answer(self, offer: dict) -> dict:
        self.__logging and logger.debug("Creating answer.")
        self.__is_running = True
        return self.__run_coroutine(self.__create_answer_async(offer))

    def set_answer(self, answer: dict) -> None:
        self.__logging and logger.debug("Setting answer.")
        self.__run_coroutine(self.__set_answer_async(answer))

    def add_ice_candidate(self, candidate: dict) -> None:
        self.__logging and logger.debug("Adding ICE candidate.")
        self.__run_coroutine(self.__add_ice_candidate_async(candidate))

    def offer_to_json(self, offer: dict) -> str:
        return json.dumps(offer)

    def json_to_offer(self, json_str: str) -> dict:
        return json.loads(json_str)

    def answer_to_json(self, answer: dict) -> str:
        return json.dumps(answer)

    def json_to_answer(self, json_str: str) -> dict:
        return json.loads(json_str)

    async def __close_async(self):
        if self.__local_video_track is not None:
            self.__local_video_track.stop()
            self.__local_video_track = None

        if self.__local_audio_track is not None:
            self.__local_audio_track.stop()
            self.__local_audio_track = None

        if self.__pc is not None:
            await self.__pc.close()
            self.__pc = None

    def stop(self) -> None:
        self.__logging and logger.debug("Stopping RTCConnection.")
        self.__terminate.set()

        if self.__loop is not None and self.__loop.is_running():
            try:
                future = asyncio.run_coroutine_threadsafe(self.__close_async(), self.__loop)
                future.result(timeout=5)
            except Exception as e:
                logger.error("Error during close: {}".format(e))

        if self.__owns_video_source and self.__video_source is not None:
            if hasattr(self.__video_source, "stop"):
                self.__video_source.stop()
            self.__video_source = None

        if self.__owns_audio_source and self.__audio_source is not None:
            if hasattr(self.__audio_source, "stop"):
                self.__audio_source.stop()
            self.__audio_source = None

        if self.__audio_playback is not None:
            self.__audio_playback.stop()
            self.__audio_playback = None

        if self.__loop is not None and self.__loop.is_running():
            self.__loop.call_soon_threadsafe(self.__loop.stop)

        if self.__thread is not None:
            self.__thread.join(timeout=2)
            self.__thread = None

        self.__loop = None
        self.__is_running = False
        self.__logging and logger.debug("RTCConnection stopped.")