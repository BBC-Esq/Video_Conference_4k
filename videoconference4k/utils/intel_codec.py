import subprocess
import shutil
import logging as log
from typing import Optional
from numpy.typing import NDArray

from .common import logger_handler

logger = log.getLogger("IntelCodec")
logger.propagate = False
logger.addHandler(logger_handler())
logger.setLevel(log.DEBUG)


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


class IntelEncoder:

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
        self._logging = logging
        self._width = width
        self._height = height
        self._framerate = framerate
        self._codec = codec
        self._bitrate = bitrate
        self._rate_control = rate_control
        self._process = None
        self._frame_size = width * height * 3
        
        codec_map = {
            "h264": "h264_qsv",
            "hevc": "hevc_qsv",
            "h265": "hevc_qsv",
            "av1": "av1_qsv",
        }
        
        ffmpeg_codec = codec_map.get(codec.lower(), "h264_qsv")
        bitrate_k = bitrate // 1000
        
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel", "error",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{width}x{height}",
            "-r", str(framerate),
            "-i", "pipe:0",
            "-c:v", ffmpeg_codec,
        ]
        
        if rate_control.lower() == "cbr":
            cmd.extend([
                "-b:v", f"{bitrate_k}k",
                "-maxrate", f"{bitrate_k}k",
                "-minrate", f"{bitrate_k}k",
                "-bufsize", f"{bitrate_k * 2}k",
            ])
        else:
            cmd.extend([
                "-b:v", f"{bitrate_k}k",
                "-maxrate", f"{int(bitrate_k * 1.5)}k",
                "-bufsize", f"{bitrate_k * 2}k",
            ])
        
        cmd.extend([
            "-g", str(framerate),
            "-f", "h264" if codec.lower() == "h264" else "hevc",
            "pipe:1"
        ])
        
        try:
            self._process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=self._frame_size * 2
            )
            
            self._logging and logger.debug(
                "IntelEncoder initialized: {}x{} @ {}fps, codec={}, rate_control={}".format(
                    width, height, framerate, codec, rate_control
                )
            )
        except Exception as e:
            logger.error("Failed to create Intel encoder: {}".format(e))
            raise

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def codec(self) -> str:
        return self._codec

    def encode(self, bgr_frame: NDArray) -> bytes:
        import cv2
        import select
        import sys
        
        if self._process is None or self._process.poll() is not None:
            return b''
        
        if bgr_frame.shape[1] != self._width or bgr_frame.shape[0] != self._height:
            bgr_frame = cv2.resize(bgr_frame, (self._width, self._height))
        
        try:
            self._process.stdin.write(bgr_frame.tobytes())
            self._process.stdin.flush()
            
            encoded_data = b''
            
            if sys.platform == 'win32':
                import msvcrt
                import os
                
                while True:
                    try:
                        if msvcrt.get_osfhandle(self._process.stdout.fileno()):
                            chunk = self._process.stdout.read(4096)
                            if chunk:
                                encoded_data += chunk
                            else:
                                break
                    except:
                        break
                    
                    if len(encoded_data) > 0:
                        try:
                            more = self._process.stdout.read(4096)
                            if more:
                                encoded_data += more
                            else:
                                break
                        except:
                            break
                    else:
                        break
            else:
                while True:
                    ready, _, _ = select.select([self._process.stdout], [], [], 0.001)
                    if ready:
                        chunk = self._process.stdout.read(4096)
                        if chunk:
                            encoded_data += chunk
                        else:
                            break
                    else:
                        break
            
            return encoded_data
            
        except Exception as e:
            self._logging and logger.error("Encode error: {}".format(e))
            return b''

    def flush(self) -> bytes:
        if self._process is None:
            return b''
        
        try:
            self._process.stdin.close()
            remaining, _ = self._process.communicate(timeout=5)
            return remaining if remaining else b''
        except Exception:
            return b''

    def close(self) -> None:
        self._logging and logger.debug("Closing IntelEncoder")
        
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


class IntelEncoderSync:

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
        self._logging = logging
        self._width = width
        self._height = height
        self._framerate = framerate
        self._codec = codec
        self._bitrate = bitrate
        self._rate_control = rate_control
        self._frames = []
        
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

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def codec(self) -> str:
        return self._codec

    def encode(self, bgr_frame: NDArray) -> bytes:
        import cv2
        
        if bgr_frame.shape[1] != self._width or bgr_frame.shape[0] != self._height:
            bgr_frame = cv2.resize(bgr_frame, (self._width, self._height))
        
        self._frames.append(bgr_frame.copy())
        return b''

    def encode_all(self) -> list:
        if not self._frames:
            return []
        
        bitrate_k = self._bitrate // 1000
        
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel", "error",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{self._width}x{self._height}",
            "-r", str(self._framerate),
            "-i", "pipe:0",
            "-c:v", self._ffmpeg_codec,
        ]
        
        if self._rate_control.lower() == "cbr":
            cmd.extend([
                "-b:v", f"{bitrate_k}k",
                "-maxrate", f"{bitrate_k}k",
                "-minrate", f"{bitrate_k}k",
            ])
        else:
            cmd.extend([
                "-b:v", f"{bitrate_k}k",
                "-maxrate", f"{int(bitrate_k * 1.5)}k",
            ])
        
        cmd.extend([
            "-g", str(self._framerate),
            "-f", "h264" if self._codec.lower() == "h264" else "hevc",
            "pipe:1"
        ])
        
        try:
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            
            input_data = b''.join(frame.tobytes() for frame in self._frames)
            output, stderr = process.communicate(input=input_data, timeout=60)
            
            return output
            
        except Exception as e:
            self._logging and logger.error("Encode error: {}".format(e))
            return b''

    def flush(self) -> bytes:
        return b''

    def close(self) -> None:
        self._logging and logger.debug("Closing IntelEncoderSync")
        self._frames = []