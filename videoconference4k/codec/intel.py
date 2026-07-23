import subprocess
import shutil
from typing import Optional
from numpy.typing import NDArray

from .base import BaseDecoder, FFmpegPipeEncoder, FFmpegSyncEncoder
from ..utils.common import get_logger

logger = get_logger("IntelCodec")


def has_intel_codec() -> bool:
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
        return "h264_qsv" in result.stdout
    except Exception:
        return False


def get_intel_info() -> dict:
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

        qsv_encoders = []
        for line in result.stdout.split('\n'):
            if '_qsv' in line.lower():
                if 'h264_qsv' in line:
                    qsv_encoders.append("h264")
                elif 'hevc_qsv' in line:
                    qsv_encoders.append("hevc")
                elif 'av1_qsv' in line:
                    qsv_encoders.append("av1")

        info["encoders"] = qsv_encoders
        info["available"] = len(qsv_encoders) > 0

    except Exception as e:
        info["error"] = str(e)

    return info


class IntelEncoder(FFmpegPipeEncoder):

    def __init__(
        self,
        width: int,
        height: int,
        framerate: int = 30,
        bitrate: int = 8000000,
        codec: str = "h264",
        rate_control: str = "cbr",
        logging: bool = False,
    ):
        super().__init__(
            width=width,
            height=height,
            framerate=framerate,
            codec_type="intel_qsv",
            logger=logger,
            logging=logging,
        )

        self._codec = codec
        self._bitrate = bitrate
        self._rate_control = rate_control

        cmd = self._build_ffmpeg_cmd()
        self._start_process(cmd)

        self._logging and logger.debug(
            "IntelEncoder initialized: {}x{} @ {}fps, codec={}, rate_control={}".format(
                width, height, framerate, codec, rate_control
            )
        )

    def _build_ffmpeg_cmd(self) -> list:
        codec_map = {
            "h264": "h264_qsv",
            "hevc": "hevc_qsv",
            "h265": "hevc_qsv",
            "av1": "av1_qsv",
        }

        ffmpeg_codec = codec_map.get(self._codec.lower(), "h264_qsv")
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
            "-c:v", ffmpeg_codec,
        ]

        if self._rate_control.lower() == "cbr":
            cmd.extend([
                "-b:v", "{}k".format(bitrate_k),
                "-maxrate", "{}k".format(bitrate_k),
                "-minrate", "{}k".format(bitrate_k),
                "-bufsize", "{}k".format(bitrate_k * 2),
            ])
        else:
            cmd.extend([
                "-b:v", "{}k".format(bitrate_k),
                "-maxrate", "{}k".format(int(bitrate_k * 1.5)),
                "-bufsize", "{}k".format(bitrate_k * 2),
            ])

        cmd.extend([
            "-g", str(self._framerate),
            "-f", "h264" if self._codec.lower() == "h264" else "hevc",
            "pipe:1"
        ])

        return cmd


class IntelEncoderSync(FFmpegSyncEncoder):

    def __init__(
        self,
        width: int,
        height: int,
        framerate: int = 30,
        bitrate: int = 8000000,
        codec: str = "h264",
        rate_control: str = "cbr",
        logging: bool = False,
    ):
        super().__init__(
            width=width,
            height=height,
            framerate=framerate,
            codec_type="intel_qsv",
            logger=logger,
            logging=logging,
        )

        self._codec = codec
        self._bitrate = bitrate
        self._rate_control = rate_control

        codec_map = {
            "h264": "h264_qsv",
            "hevc": "hevc_qsv",
            "h265": "hevc_qsv",
            "av1": "av1_qsv",
        }

        self._ffmpeg_codec = codec_map.get(codec.lower(), "h264_qsv")

        self._logging and logger.debug(
            "IntelEncoderSync initialized: {}x{} @ {}fps, codec={}".format(
                width, height, framerate, codec
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
        ]

        if self._rate_control.lower() == "cbr":
            cmd.extend([
                "-b:v", "{}k".format(bitrate_k),
                "-maxrate", "{}k".format(bitrate_k),
                "-minrate", "{}k".format(bitrate_k),
            ])
        else:
            cmd.extend([
                "-b:v", "{}k".format(bitrate_k),
                "-maxrate", "{}k".format(int(bitrate_k * 1.5)),
            ])

        cmd.extend([
            "-g", str(self._framerate),
            "-f", "h264" if self._codec.lower() == "h264" else "hevc",
            "pipe:1"
        ])

        return cmd