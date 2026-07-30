import queue
import threading
import time
from collections import deque
from typing import Optional, Tuple, Any, Union
from numpy.typing import NDArray

from ..stream.video import VideoStream
from ..capture.audio import AudioCapture
from ..net.sync import SyncTransport
from ..net.audio import AudioTransport
from ..net.upnp import UPnPPortMapper
from ..codec import (
    local_capabilities,
    choose_send_codec,
    normalize_priority,
    describe_priority,
)
from ..codec.base import normalize_codec
from ..utils.common import get_logger, log_version, raise_timer_resolution, restore_timer_resolution

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
        speaker_id: Optional[int] = None,
        video_source: Any = None,
        gpu_accelerated: bool = True,
        gpu_codec: str = "h264",
        codec_priority: Optional[Tuple[str, ...]] = None,
        prefer_hardware_codec: bool = False,
        gpu_bitrate: int = 8000000,
        adaptive_bitrate: bool = True,
        min_bitrate: int = 0,
        enable_audio: bool = True,
        audio_bitrate: int = 32000,
        audio_jitter_ms: float = 80.0,
        echo_cancellation: bool = True,
        echo_tail_ms: Optional[float] = None,
        peer_wait_s: float = 900.0,
        lipsync: bool = True,
        audio_sync_offset_ms: float = 0.0,
        lipsync_deadband_ms: float = 40.0,
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
        self.__audio_jitter_ms = audio_jitter_ms
        self.__enable_upnp = enable_upnp

        # How long to keep waiting for a peer that has not arrived. The default
        # transport gives up after three silent polls, about thirty-six seconds,
        # which is shorter than it takes to walk to the other machine and start
        # it; the side that was started first would then be permanently deaf
        # while still reporting itself healthy. A receiver loses nothing by
        # waiting, and the stats now say plainly when nothing is arriving.
        self.__peer_wait_s = max(36.0, float(peer_wait_s))
        self.__peer_wait_retries = max(3, int(self.__peer_wait_s / 12.0))

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
                output_device=speaker_id,
                sample_rate=48000,
                channels=1,
                enable_input=True,
                enable_output=True,
                output_jitter_ms=self.__audio_jitter_ms,
                echo_cancellation=echo_cancellation,
                echo_tail_ms=echo_tail_ms,
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

        self.__lipsync = lipsync
        self.__audio_sync_offset_ns = int(audio_sync_offset_ms * 1e6)
        self.__lipsync_deadband_ns = int(lipsync_deadband_ms * 1e6)
        self.__video_hold = deque()
        residency_s = (audio_jitter_ms + max(0.0, audio_sync_offset_ms)) / 1000.0
        self.__video_hold_horizon_ns = int((1.0 + residency_s) * 1e9)
        self.__video_hold_cap = max(4, int(framerate * (1.0 + residency_s))) if framerate > 0 else 60
        self.__playout_stall_s = 0.25
        self.__playout_last_value = None
        self.__playout_last_advance = 0.0

        self.__terminate = threading.Event()
        self.__threads = []
        self.__is_running = False
        self.__join_timeout = 6.0
        self.__frames_skipped = 0
        self.__frames_dropped = 0
        self.__frames_lagged = 0
        self.__frames_received = 0
        self.__last_recv_at = None
        self.__audio_chunks_received = 0
        self.__last_audio_recv_at = None
        self.__decode_gaps = 0
        self.__stats_prev_recv = 0
        self.__stats_prev_recv_time = None
        self.__recv_fps = 0.0
        self.__stats_prev_bytes = 0
        self.__stats_prev_time = None
        self.__stats_lock = threading.Lock()
        self.__timer_raised = False

        self.__adaptive_bitrate = adaptive_bitrate
        self.__abr_max = gpu_bitrate
        self.__abr_min = int(min_bitrate) if min_bitrate else max(300000, gpu_bitrate // 8)
        self.__abr_target = gpu_bitrate
        self.__abr_interval = 1.0
        self.__abr_increase_interval = 4.0
        self.__abr_drop_threshold = 0.05
        self.__abr_last_check = None
        self.__abr_last_increase = 0.0
        self.__abr_prev_sent = 0
        self.__abr_prev_dropped = 0
        self.__abr_prev_bytes = 0
        self.__abr_probe_backoff = 4.0
        self.__abr_probe_backoff_max = 60.0
        self.__abr_probe_ceiling = None
        self.__latency_floor = None
        self.__latency_slack_ms = 30.0

        self.__shed_level = 0
        self.__shed_max = 3
        self.__shed_threshold = 0.15
        self.__shed_counter = 0
        self.__frames_source_shed = 0
        self.__want_remote_keyframe = False

        self.__codec_priority = normalize_priority(codec_priority)
        self.__prefer_hardware_codec = bool(prefer_hardware_codec)
        self.__local_caps = local_capabilities()
        self.__negotiated_codec = normalize_codec(gpu_codec)

    @property
    def is_running(self) -> bool:
        return self.__is_running

    @property
    def frames_skipped(self) -> int:
        return self.__frames_skipped

    @property
    def codec_priority(self) -> tuple:
        """The order codecs are preferred in, most preferred first."""
        return self.__codec_priority

    @codec_priority.setter
    def codec_priority(self, order) -> None:
        """Replace the preference order, during a call if need be.

        Written as a setting rather than a constant because the order is a user's
        choice, not the program's: a settings screen assigns whatever ranking the
        user has arranged and the next negotiation simply follows it.
        """
        self.__codec_priority = normalize_priority(order)
        self.__logging and logger.debug(
            "Codec priority set to {}".format(describe_priority(self.__codec_priority)))

    @property
    def prefer_hardware_codec(self) -> bool:
        """Whether a hardware-capable lower choice may outrank a software-only higher one.

        Off by default, so the priority order is honoured exactly as written. Turn
        it on and a codec this machine can encode in hardware wins over one it
        would have to encode on the CPU, even if the CPU one ranks higher.
        """
        return self.__prefer_hardware_codec

    @prefer_hardware_codec.setter
    def prefer_hardware_codec(self, value: bool) -> None:
        self.__prefer_hardware_codec = bool(value)

    def start(self) -> "DirectConference":
        if self.__is_running:
            return self

        self.__abr_target = self.__abr_max
        self.__abr_last_check = None
        self.__abr_last_increase = 0.0
        self.__abr_prev_sent = 0
        self.__abr_prev_dropped = 0
        self.__abr_prev_bytes = 0
        self.__abr_probe_backoff = self.__abr_increase_interval
        self.__abr_probe_ceiling = None
        self.__latency_floor = None
        self.__shed_level = 0
        self.__shed_counter = 0
        self.__frames_skipped = 0
        self.__frames_dropped = 0
        self.__frames_lagged = 0
        self.__frames_received = 0
        self.__last_recv_at = None
        self.__audio_chunks_received = 0
        self.__last_audio_recv_at = None
        self.__decode_gaps = 0
        self.__stats_prev_recv = 0
        self.__stats_prev_recv_time = None
        self.__recv_fps = 0.0
        self.__frames_source_shed = 0
        with self.__stats_lock:
            self.__stats_prev_bytes = 0
            self.__stats_prev_time = None
        self.__playout_last_value = None
        self.__playout_last_advance = 0.0

        self.__timer_raised = raise_timer_resolution(1)

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
            max_retries=self.__peer_wait_retries,
        )
        self.__send_video = SyncTransport(
            address=self.__peer_address, port=self.__peer_video_port,
            gpu_accelerated=self.__gpu_accelerated, gpu_codec=self.__gpu_codec,
            gpu_bitrate=self.__gpu_bitrate, logging=self.__logging,
            max_retries=self.__peer_wait_retries,
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

        self.__send_video.announce_capabilities(self.__local_caps)

        self.__terminate.clear()
        self.__want_remote_keyframe = True
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
        read_timed = getattr(self.__video_source, "read_timed", None)
        last_seq = -1
        while not self.__terminate.is_set():
            start = time.perf_counter()
            if read_timed is not None:
                frame, pts_ns, seq = read_timed()
            else:
                frame, pts_ns, seq = self.__video_source.read(), time.perf_counter_ns(), None
            proc_start = time.perf_counter()
            if frame is not None:
                with self.__frame_lock:
                    self.__local_frame = frame
                if self.__want_remote_keyframe:
                    self.__send_video.request_keyframe()
                    self.__want_remote_keyframe = False
                shed = False
                if self.__shed_level > 0:
                    self.__shed_counter += 1
                    shed = (self.__shed_counter % (self.__shed_level + 1)) != 0
                if shed:
                    self.__frames_source_shed += 1
                elif seq is None or seq != last_seq:
                    last_seq = seq
                    try:
                        self.__send_video.send(frame, pts_ns=pts_ns)
                    except Exception as e:
                        self.__logging and logger.debug("Video send error: {}".format(e))
                else:
                    self.__frames_skipped += 1
                if interval > 0 and (time.perf_counter() - proc_start) > interval:
                    self.__frames_lagged += 1
            self.__maybe_negotiate_codec()
            self.__maybe_adapt_bitrate(time.perf_counter())
            wait = interval - (time.perf_counter() - start)
            if wait > 0:
                self.__terminate.wait(wait)

    def __maybe_negotiate_codec(self) -> None:
        """Settle this direction's codec once the peer has said what it can decode.

        Runs on the send thread so the encoder is never swapped underneath a
        frame, and only ever narrows to something the far end can actually read.
        """
        if self.__recv_video is None or self.__send_video is None:
            return
        remote = self.__recv_video.peer_capabilities
        if not remote:
            return

        wanted = choose_send_codec(
            self.__local_caps, remote, self.__codec_priority,
            prefer_hardware=self.__prefer_hardware_codec)
        if wanted == self.__negotiated_codec:
            return
        if self.__send_video.set_codec(wanted):
            self.__negotiated_codec = wanted
            self.__want_remote_keyframe = True
            logger.info("Negotiated {} for this direction with the peer.".format(wanted))

    def _latency_congested(self, latency_ms: Optional[float]) -> bool:
        """Whether the peer is falling behind, judged before any frame is lost.

        The lowest latency seen on this call stands in for an uncongested link,
        and a sustained rise above it means queueing somewhere in between.
        """
        if latency_ms is None:
            return False
        if self.__latency_floor is None or latency_ms < self.__latency_floor:
            self.__latency_floor = latency_ms
            return False
        self.__latency_floor += (latency_ms - self.__latency_floor) * 0.01
        return latency_ms > self.__latency_floor * 1.6 + self.__latency_slack_ms

    def _abr_decision(self, drop_frac: float, goodput_bps: float, now: float,
                      latency_ms: Optional[float] = None):
        congested = drop_frac > self.__abr_drop_threshold or self._latency_congested(latency_ms)

        if congested:
            self.__abr_probe_ceiling = self.__abr_target
            self.__abr_probe_backoff = min(
                self.__abr_probe_backoff_max,
                max(self.__abr_increase_interval, self.__abr_probe_backoff * 2.0),
            )
            self.__abr_last_increase = now
            reference = goodput_bps if drop_frac > self.__abr_drop_threshold else self.__abr_target
            target = max(self.__abr_min, min(self.__abr_target, int(reference * 0.9)))
            if target < int(self.__abr_target * 0.95):
                return target
            return None

        if drop_frac > 0.0:
            return None

        if (now - self.__abr_last_increase) < self.__abr_probe_backoff:
            return None

        if goodput_bps < self.__abr_target * 0.85:
            self.__abr_last_increase = now
            return None

        step = self.__abr_max // 10
        if self.__abr_probe_ceiling is not None:
            headroom = self.__abr_probe_ceiling - self.__abr_target
            if headroom <= 0:
                step = max(self.__abr_max // 40, 1)
            else:
                step = max(1, min(step, headroom // 2))

        target = min(self.__abr_max, self.__abr_target + step)
        self.__abr_last_increase = now
        if target > self.__abr_target:
            self.__abr_probe_backoff = max(
                self.__abr_increase_interval, self.__abr_probe_backoff * 0.5
            )
            return target
        return None

    def __maybe_adapt_bitrate(self, now: float) -> None:
        if not self.__adaptive_bitrate or self.__send_video is None:
            return
        if self.__abr_last_check is None:
            self.__abr_last_check = now
            self.__abr_last_increase = now
            self.__abr_prev_sent = self.__send_video.frames_sent
            self.__abr_prev_dropped = self.__send_video.frames_pipe_dropped
            self.__abr_prev_bytes = self.__send_video.bytes_sent
            return
        elapsed = now - self.__abr_last_check
        if elapsed < self.__abr_interval:
            return

        sent = self.__send_video.frames_sent
        dropped = self.__send_video.frames_pipe_dropped
        bytes_now = self.__send_video.bytes_sent
        sent_delta = sent - self.__abr_prev_sent
        attempts = sent_delta + (dropped - self.__abr_prev_dropped)
        drop_frac = (dropped - self.__abr_prev_dropped) / attempts if attempts else 0.0
        goodput_bps = (bytes_now - self.__abr_prev_bytes) * 8.0 / elapsed if elapsed > 0 else 0.0

        self.__abr_last_check = now
        self.__abr_prev_sent = sent
        self.__abr_prev_dropped = dropped
        self.__abr_prev_bytes = bytes_now

        if attempts and sent_delta == 0:
            return

        latency_ms = self.__send_video.peer_latency_ms

        can_reconfigure = self.__send_video.supports_dynamic_bitrate
        lowered = False
        if can_reconfigure:
            new_target = self._abr_decision(drop_frac, goodput_bps, now, latency_ms)
            if new_target is not None and new_target != self.__abr_target:
                if self.__send_video.reconfigure_bitrate(new_target):
                    lowered = new_target < self.__abr_target
                    self.__abr_target = new_target
                    self.__logging and logger.debug(
                        "Adaptive bitrate -> {} kbps (drop {:.0%}, goodput {:.0f} kbps).".format(
                            new_target // 1000, drop_frac, goodput_bps / 1000
                        )
                    )

        if not can_reconfigure:
            congested = self._latency_congested(latency_ms)
        else:
            congested = False

        at_floor = not can_reconfigure or self.__abr_target <= int(self.__abr_min * 1.05)
        overloaded = drop_frac > self.__shed_threshold or congested
        if overloaded and at_floor and not lowered:
            self.__shed_level = min(self.__shed_max, self.__shed_level + 1)
        elif drop_frac <= 0.0 and not congested and self.__shed_level > 0:
            self.__shed_level = max(0, self.__shed_level - 1)

    def __video_recv_loop(self) -> None:
        while not self.__terminate.is_set():
            try:
                frame = self.__recv_video.recv()
            except Exception:
                break
            if frame is None:
                break
            self.__frames_received += 1
            self.__last_recv_at = time.perf_counter()
            if self.__recv_video.consume_keyframe_request() and self.__send_video is not None:
                self.__send_video.force_next_keyframe()
            pts_ns = self.__recv_video.last_video_pts
            with self.__frame_lock:
                self.__video_hold.append((pts_ns, frame))
                while len(self.__video_hold) > self.__video_hold_cap or (
                    pts_ns - self.__video_hold[0][0] > self.__video_hold_horizon_ns
                ):
                    self.__video_hold.popleft()
                    self.__frames_dropped += 1

    def __audio_send_loop(self) -> None:
        while not self.__terminate.is_set():
            try:
                chunk, pts_ns = self.__audio_sub.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                self.__send_audio.send(chunk, pts_ns)
            except Exception as e:
                self.__logging and logger.debug("Audio send error: {}".format(e))

    def __audio_recv_loop(self) -> None:
        while not self.__terminate.is_set():
            received = self.__recv_audio.recv()
            if received is not None:
                chunk, pts_ns = received
                self.__audio_chunks_received += 1
                self.__last_audio_recv_at = time.perf_counter()
                self.__audio.write_timed(chunk, pts_ns)
            else:
                self.__terminate.wait(0.005)

    def get_remote_frame(self) -> Optional[NDArray]:
        target = None
        if self.__lipsync and self.__audio is not None:
            playout = self.__audio.playout_pts_ns()
            if playout is not None:
                now = time.perf_counter()
                if playout != self.__playout_last_value:
                    self.__playout_last_value = playout
                    self.__playout_last_advance = now
                elif now - self.__playout_last_advance > self.__playout_stall_s:
                    playout = None
            if playout is not None:
                target = playout - self.__audio_sync_offset_ns

        with self.__frame_lock:
            if target is None:
                if self.__video_hold:
                    self.__remote_frame = self.__video_hold[-1][1]
                    self.__video_hold.clear()
                return self.__remote_frame

            deadline = target + self.__lipsync_deadband_ns
            released = 0
            while self.__video_hold and self.__video_hold[0][0] <= deadline:
                _, frame = self.__video_hold.popleft()
                self.__remote_frame = frame
                released += 1
            if released > 1:
                self.__frames_dropped += released - 1
            return self.__remote_frame

    def stats(self) -> dict:
        with self.__frame_lock:
            hold_depth = len(self.__video_hold)

        bytes_sent = self.__send_video.bytes_sent if self.__send_video is not None else 0
        frames_sent = self.__send_video.frames_sent if self.__send_video is not None else 0
        pipe_dropped = self.__send_video.frames_pipe_dropped if self.__send_video is not None else 0
        reconnects = 0
        for transport in (self.__send_video, self.__recv_video):
            if transport is not None:
                reconnects += transport.reconnects

        with self.__stats_lock:
            now = time.perf_counter()
            send_kbps = 0.0
            if self.__stats_prev_time is not None:
                elapsed = now - self.__stats_prev_time
                if elapsed > 0:
                    send_kbps = (bytes_sent - self.__stats_prev_bytes) * 8.0 / elapsed / 1000.0
            self.__stats_prev_time = now
            self.__stats_prev_bytes = bytes_sent

            frames_received = self.__frames_received
            if self.__stats_prev_recv_time is not None:
                elapsed = now - self.__stats_prev_recv_time
                if elapsed > 0:
                    self.__recv_fps = (frames_received - self.__stats_prev_recv) / elapsed
            self.__stats_prev_recv_time = now
            self.__stats_prev_recv = frames_received

        last_video = self.__last_recv_at
        last_audio = self.__last_audio_recv_at
        video_silent_s = (now - last_video) if last_video is not None else None
        audio_silent_s = (now - last_audio) if last_audio is not None else None

        return {
            "audio_playout_pts_ns": self.__audio.playout_pts_ns() if self.__audio is not None else None,
            "video_hold_depth": hold_depth,
            "frames_sent": frames_sent,
            "frames_skipped": self.__frames_skipped,
            "frames_dropped": self.__frames_dropped,
            "frames_lagged": self.__frames_lagged,
            "pipe_dropped": pipe_dropped,
            "reconnects": reconnects,
            "bytes_sent": bytes_sent,
            "send_kbps": round(send_kbps, 1),
            "target_bitrate": self.__abr_target,
            "adaptive_bitrate": self.__adaptive_bitrate,
            "shed_level": self.__shed_level,
            "frames_source_shed": self.__frames_source_shed,
            "capture_failed": bool(getattr(self.__video_source, "capture_failed", False)),
            "send_codec": self.__negotiated_codec,
            "send_impl": (self.__send_video.encoder_kind
                          if self.__send_video is not None else None),
            "peer_latency_ms": (round(self.__send_video.peer_latency_ms, 1)
                                if self.__send_video is not None
                                and self.__send_video.peer_latency_ms is not None else None),
            "acks_lost": (self.__send_video.acks_lost
                          if self.__send_video is not None else 0),
            "recv_transport_alive": (not self.__recv_video.abandoned
                                     if self.__recv_video is not None else False),
            "send_transport_alive": (not self.__send_video.abandoned
                                     if self.__send_video is not None else False),
            "encoder_alive": (self.__send_video.encoder_alive
                              if self.__send_video is not None else True),
            "encoder_error": (self.__send_video.encoder_error
                              if self.__send_video is not None else ""),
            "can_force_keyframe": (self.__send_video.supports_force_idr
                                   if self.__send_video is not None else None),
            "keyframe_requests_unmet": (self.__send_video.keyframe_requests_unmet
                                        if self.__send_video is not None else 0),
            "recv_codec": (self.__recv_video.decoder_codec
                           if self.__recv_video is not None else None),
            "recv_impl": (self.__recv_video.decoder_kind
                          if self.__recv_video is not None else None),
            "peer_capabilities": (self.__recv_video.peer_capabilities
                                  if self.__recv_video is not None else None),
            "frames_received": self.__frames_received,
            "recv_fps": round(self.__recv_fps, 1),
            "video_silent_s": (round(video_silent_s, 2)
                               if video_silent_s is not None else None),
            "receiving_video": bool(video_silent_s is not None and video_silent_s < 2.0),
            "audio_chunks_received": self.__audio_chunks_received,
            "audio_silent_s": (round(audio_silent_s, 2)
                               if audio_silent_s is not None else None),
            "receiving_audio": bool(audio_silent_s is not None and audio_silent_s < 2.0),
            "audio_jitter_depth_ms": (self.__audio.jitter_depth_ms()
                                      if self.__audio is not None else None),
            "echo_cancellation": (self.__audio.echo_cancellation
                                  if self.__audio is not None else False),
            "audio_duplex": (self.__audio.duplex
                             if self.__audio is not None else False),
            "echo_reduction_db": (self.__audio.echo_reduction_db
                                  if self.__audio is not None else None),
            "audio_underruns": (self.__audio.jitter_underruns()
                                if self.__audio is not None else 0),
            "audio_callback_drops": (self.__audio.callback_drops
                                     if self.__audio is not None else 0),
            "lipsync": self.__lipsync and self.__audio is not None,
        }

    def get_local_frame(self) -> Optional[NDArray]:
        with self.__frame_lock:
            return self.__local_frame

    def request_keyframe(self) -> None:
        self.__want_remote_keyframe = True

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
        if self.__timer_raised:
            restore_timer_resolution(1)
            self.__timer_raised = False
        with self.__frame_lock:
            self.__remote_frame = None
            self.__local_frame = None
            self.__video_hold.clear()
        self.__logging and logger.debug("DirectConference stopped.")
