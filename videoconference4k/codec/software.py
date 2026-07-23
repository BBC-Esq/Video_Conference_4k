import subprocess
import threading
from typing import Optional
import numpy as np
from numpy.typing import NDArray

from .base import (
    BaseDecoder,
    FFmpegPipeEncoder,
    FFmpegSyncEncoder,
    get_ffmpeg_path,
    get_ffmpeg_encoders,
)
from ..utils.common import get_logger

logger = get_logger("SoftwareCodec")


def has_x264() -> bool:
    return "libx264" in get_ffmpeg_encoders()


def has_x265() -> bool:
    return "libx265" in get_ffmpeg_encoders()


def has_software_codec() -> bool:
    return has_x264() or has_x265()


def get_software_info() -> dict:
    info = {
        "available": False,
        "ffmpeg_found": False,
        "encoders": [],
    }

    ffmpeg_path = get_ffmpeg_path()
    if not ffmpeg_path:
        return info

    info["ffmpeg_found"] = True
    info["ffmpeg_path"] = ffmpeg_path

    encoders_output = get_ffmpeg_encoders()
    encoders = []
    if "libx264" in encoders_output:
        encoders.append("x264")
    if "libx265" in encoders_output:
        encoders.append("x265")

    info["encoders"] = encoders
    info["available"] = len(encoders) > 0

    return info


class SoftwareEncoder(FFmpegPipeEncoder):

    def __init__(
        self,
        width: int,
        height: int,
        framerate: int = 30,
        bitrate: int = 8000000,
        codec: str = "x264",
        preset: str = "ultrafast",
        tune: str = "zerolatency",
        logging: bool = False,
    ):
        super().__init__(
            width=width,
            height=height,
            framerate=framerate,
            codec_type="software",
            logger=logger,
            logging=logging,
        )

        self._codec = codec.lower()
        self._bitrate = bitrate
        self._preset = preset
        self._tune = tune

        codec_map = {
            "x264": "libx264",
            "h264": "libx264",
            "x265": "libx265",
            "hevc": "libx265",
            "h265": "libx265",
        }

        self._ffmpeg_codec = codec_map.get(self._codec, "libx264")
        self._is_hevc = "265" in self._codec or "hevc" in self._codec

        cmd = self._build_ffmpeg_cmd()
        self._start_process(cmd)

        self._logging and logger.debug(
            "SoftwareEncoder initialized: {}x{} @ {}fps, codec={}, preset={}".format(
                width, height, framerate, self._ffmpeg_codec, preset
            )
        )

    def _build_ffmpeg_cmd(self) -> list:
        bitrate_k = self._bitrate // 1000
        framerate = self._framerate if self._framerate > 0 else 30
        bufsize_k = max(1, bitrate_k // framerate)

        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel", "error",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", "{}x{}".format(self._width, self._height),
            "-r", str(self._framerate),
            "-i", "pipe:0",
            "-c:v", self._ffmpeg_codec,
            "-preset", self._preset,
            "-tune", self._tune,
            "-b:v", "{}k".format(bitrate_k),
            "-maxrate", "{}k".format(bitrate_k),
            "-bufsize", "{}k".format(bufsize_k),
            "-g", str(self._framerate),
            "-f", "hevc" if self._is_hevc else "h264",
            "pipe:1"
        ]

        return cmd


class SoftwareEncoderSync(FFmpegSyncEncoder):

    def __init__(
        self,
        width: int,
        height: int,
        framerate: int = 30,
        bitrate: int = 8000000,
        codec: str = "x264",
        preset: str = "ultrafast",
        tune: str = "zerolatency",
        logging: bool = False,
    ):
        super().__init__(
            width=width,
            height=height,
            framerate=framerate,
            codec_type="software",
            logger=logger,
            logging=logging,
        )

        self._codec = codec.lower()
        self._bitrate = bitrate
        self._preset = preset
        self._tune = tune

        codec_map = {
            "x264": "libx264",
            "h264": "libx264",
            "x265": "libx265",
            "hevc": "libx265",
            "h265": "libx265",
        }

        self._ffmpeg_codec = codec_map.get(self._codec, "libx264")
        self._is_hevc = "265" in self._codec or "hevc" in self._codec

        self._logging and logger.debug(
            "SoftwareEncoderSync initialized: {}x{} @ {}fps, codec={}".format(
                width, height, framerate, self._ffmpeg_codec
            )
        )

    def _build_ffmpeg_cmd(self) -> list:
        bitrate_k = self._bitrate // 1000
        framerate = self._framerate if self._framerate > 0 else 30
        bufsize_k = max(1, bitrate_k // framerate)

        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel", "error",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", "{}x{}".format(self._width, self._height),
            "-r", str(self._framerate),
            "-i", "pipe:0",
            "-c:v", self._ffmpeg_codec,
            "-preset", self._preset,
            "-tune", self._tune,
            "-b:v", "{}k".format(bitrate_k),
            "-maxrate", "{}k".format(bitrate_k),
            "-bufsize", "{}k".format(bufsize_k),
            "-g", str(self._framerate),
            "-f", "hevc" if self._is_hevc else "h264",
            "pipe:1"
        ]

        return cmd


class SoftwareDecoder(BaseDecoder):

    def __init__(
        self,
        codec: str = "x264",
        logging: bool = False,
    ):
        self._logging = logging
        self._codec = codec.lower()
        self._is_hevc = "265" in self._codec or "hevc" in self._codec
        self._process = None
        self._width = None
        self._height = None
        self._frame_size = None
        self._out_buffer = bytearray()
        self._out_lock = threading.Lock()
        self._stdout_thread = None
        self._stderr_thread = None

        self._logging and logger.debug(
            "SoftwareDecoder initialized: codec={}".format(codec)
        )

    @property
    def codec_type(self) -> str:
        return "software"

    def _start_process(self, width: int, height: int) -> None:
        self._width = width
        self._height = height
        self._frame_size = width * height * 3

        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel", "error",
            "-probesize", "32",
            "-analyzeduration", "0",
            "-thread_type", "slice",
            "-f", "hevc" if self._is_hevc else "h264",
            "-i", "pipe:0",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", "{}x{}".format(width, height),
            "pipe:1"
        ]

        self._process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=self._frame_size * 2,
        )

        self._stdout_thread = threading.Thread(
            target=self._drain_stdout, args=(self._process.stdout,), daemon=True
        )
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr, args=(self._process.stderr,), daemon=True
        )
        self._stdout_thread.start()
        self._stderr_thread.start()

    def _drain_stdout(self, stream) -> None:
        try:
            while True:
                chunk = stream.read1(1048576)
                if not chunk:
                    break
                with self._out_lock:
                    self._out_buffer.extend(chunk)
        except Exception:
            pass

    def _drain_stderr(self, stream) -> None:
        try:
            while stream.read1(65536):
                pass
        except Exception:
            pass

    def decode(
        self,
        encoded_data: bytes,
        width: int,
        height: int
    ) -> Optional[NDArray]:
        if not width or not height:
            return None

        if self._process is not None and (width != self._width or height != self._height):
            self.close()

        if self._process is None or self._process.poll() is not None:
            if self._process is not None:
                self.close()
            try:
                self._start_process(width, height)
            except Exception as e:
                self._logging and logger.error("Failed to start decoder: {}".format(e))
                self._process = None
                return None

        if encoded_data:
            try:
                self._process.stdin.write(encoded_data)
                self._process.stdin.flush()
            except Exception as e:
                self._logging and logger.error("Decode error: {}".format(e))
                self.close()
                return None

        frame_bytes = None
        with self._out_lock:
            if len(self._out_buffer) >= self._frame_size:
                frame_bytes = bytes(self._out_buffer[:self._frame_size])
                del self._out_buffer[:self._frame_size]

        if frame_bytes is None:
            return None

        return np.frombuffer(frame_bytes, dtype=np.uint8).reshape((height, width, 3))

    def close(self) -> None:
        self._logging and logger.debug("Closing SoftwareDecoder")

        if self._process is not None:
            try:
                self._process.stdin.close()
            except Exception:
                pass
            try:
                self._process.terminate()
                self._process.wait(timeout=2)
            except Exception:
                try:
                    self._process.kill()
                except Exception:
                    pass
            self._process = None

        if self._stdout_thread is not None:
            self._stdout_thread.join(timeout=1)
            self._stdout_thread = None

        if self._stderr_thread is not None:
            self._stderr_thread.join(timeout=1)
            self._stderr_thread = None

        with self._out_lock:
            self._out_buffer.clear()
        self._width = None
        self._height = None