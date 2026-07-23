import subprocess
import shutil
import tempfile
import os
import time
from typing import Optional
import numpy as np
from numpy.typing import NDArray

from .base import BaseDecoder, FFmpegPipeEncoder, FFmpegSyncEncoder
from ..utils.common import get_logger

logger = get_logger("SoftwareCodec")


def has_x264() -> bool:
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        return False

    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            timeout=5
        )
        return "libx264" in result.stdout
    except Exception:
        return False


def has_x265() -> bool:
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        return False

    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            timeout=5
        )
        return "libx265" in result.stdout
    except Exception:
        return False


def has_software_codec() -> bool:
    return has_x264() or has_x265()


def get_software_info() -> dict:
    info = {
        "available": False,
        "ffmpeg_found": False,
        "encoders": [],
    }

    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        return info

    info["ffmpeg_found"] = True
    info["ffmpeg_path"] = ffmpeg_path

    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            timeout=5
        )

        encoders = []
        if "libx264" in result.stdout:
            encoders.append("x264")
        if "libx265" in result.stdout:
            encoders.append("x265")

        info["encoders"] = encoders
        info["available"] = len(encoders) > 0

    except Exception as e:
        info["error"] = str(e)

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
            "-g", str(self._framerate),
            "-f", "hevc" if self._is_hevc else "h264",
            "pipe:1"
        ]

        return cmd


def _safe_unlink(filepath: str, max_retries: int = 3, retry_delay: float = 0.1) -> None:
    for attempt in range(max_retries):
        try:
            if os.path.exists(filepath):
                os.unlink(filepath)
            return
        except PermissionError:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                pass
        except Exception:
            return


class SoftwareDecoder(BaseDecoder):

    def __init__(
        self,
        codec: str = "x264",
        logging: bool = False,
    ):
        self._logging = logging
        self._codec = codec.lower()
        self._is_hevc = "265" in self._codec or "hevc" in self._codec

        self._logging and logger.debug(
            "SoftwareDecoder initialized: codec={}".format(codec)
        )

    @property
    def codec_type(self) -> str:
        return "software"

    def decode(
        self,
        encoded_data: bytes,
        width: int,
        height: int
    ) -> Optional[NDArray]:
        if not encoded_data:
            return None

        suffix = ".hevc" if self._is_hevc else ".h264"
        tmp_input = None

        try:
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_in:
                tmp_in.write(encoded_data)
                tmp_input = tmp_in.name

            cmd = [
                "ffmpeg",
                "-hide_banner",
                "-loglevel", "error",
                "-i", tmp_input,
                "-f", "rawvideo",
                "-pix_fmt", "bgr24",
                "-frames:v", "1",
                "pipe:1"
            ]

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            output, stderr = process.communicate(timeout=10)

            if output and len(output) == width * height * 3:
                frame = np.frombuffer(output, dtype=np.uint8).reshape((height, width, 3))
                return frame

            return None

        except Exception as e:
            self._logging and logger.error("Decode error: {}".format(e))
            return None

        finally:
            if tmp_input is not None:
                _safe_unlink(tmp_input)

    def close(self) -> None:
        self._logging and logger.debug("Closing SoftwareDecoder")