import subprocess
import shutil
import logging as log
from typing import Optional, Tuple
from numpy.typing import NDArray

from .common import logger_handler

logger = log.getLogger("SoftwareCodec")
logger.propagate = False
logger.addHandler(logger_handler())
logger.setLevel(log.DEBUG)


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


class SoftwareEncoder:

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
        self._logging = logging
        self._width = width
        self._height = height
        self._framerate = framerate
        self._codec = codec.lower()
        self._bitrate = bitrate
        self._preset = preset
        self._tune = tune
        self._process = None
        self._frame_size = width * height * 3
        
        codec_map = {
            "x264": "libx264",
            "h264": "libx264",
            "x265": "libx265",
            "hevc": "libx265",
            "h265": "libx265",
        }
        
        self._ffmpeg_codec = codec_map.get(self._codec, "libx264")
        self._is_hevc = "265" in self._codec or "hevc" in self._codec
        
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
            "-c:v", self._ffmpeg_codec,
            "-preset", preset,
            "-tune", tune,
            "-b:v", f"{bitrate_k}k",
            "-g", str(framerate),
            "-f", "hevc" if self._is_hevc else "h264",
            "pipe:1"
        ]
        
        try:
            self._process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=self._frame_size * 2
            )
            
            self._logging and logger.debug(
                "SoftwareEncoder initialized: {}x{} @ {}fps, codec={}, preset={}".format(
                    width, height, framerate, self._ffmpeg_codec, preset
                )
            )
        except Exception as e:
            logger.error("Failed to create software encoder: {}".format(e))
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
                
                while True:
                    try:
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
        self._logging and logger.debug("Closing SoftwareEncoder")
        
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


class SoftwareEncoderSync:

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
        self._logging = logging
        self._width = width
        self._height = height
        self._framerate = framerate
        self._codec = codec.lower()
        self._bitrate = bitrate
        self._preset = preset
        self._tune = tune
        self._frames = []
        
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

    def encode_all(self) -> bytes:
        if not self._frames:
            return b''
        
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
            "-preset", self._preset,
            "-tune", self._tune,
            "-b:v", f"{bitrate_k}k",
            "-g", str(self._framerate),
            "-f", "hevc" if self._is_hevc else "h264",
            "pipe:1"
        ]
        
        try:
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            
            input_data = b''.join(frame.tobytes() for frame in self._frames)
            output, stderr = process.communicate(input=input_data, timeout=120)
            
            return output
            
        except Exception as e:
            self._logging and logger.error("Encode error: {}".format(e))
            return b''

    def flush(self) -> bytes:
        return b''

    def close(self) -> None:
        self._logging and logger.debug("Closing SoftwareEncoderSync")
        self._frames = []


class SoftwareDecoder:

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

    def decode(
        self,
        encoded_data: bytes,
        width: int,
        height: int
    ) -> Optional[NDArray]:
        import numpy as np
        import tempfile
        import os
        
        if not encoded_data:
            return None
        
        suffix = ".hevc" if self._is_hevc else ".h264"
        
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
            
            os.unlink(tmp_input)
            
            if output and len(output) == width * height * 3:
                frame = np.frombuffer(output, dtype=np.uint8).reshape((height, width, 3))
                return frame
            
            return None
            
        except Exception as e:
            self._logging and logger.error("Decode error: {}".format(e))
            return None

    def close(self) -> None:
        self._logging and logger.debug("Closing SoftwareDecoder")