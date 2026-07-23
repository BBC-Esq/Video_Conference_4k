import subprocess
import sys
from abc import ABC, abstractmethod
from typing import Optional
from numpy.typing import NDArray


class BaseEncoder(ABC):

    @property
    @abstractmethod
    def width(self) -> int:
        pass

    @property
    @abstractmethod
    def height(self) -> int:
        pass

    @property
    @abstractmethod
    def codec_type(self) -> str:
        pass

    @abstractmethod
    def encode(self, frame: NDArray) -> bytes:
        pass

    @abstractmethod
    def flush(self) -> bytes:
        pass

    @abstractmethod
    def close(self) -> None:
        pass

    def get_compression_metadata(self) -> dict:
        return {
            "type": self.codec_type,
            "width": self.width,
            "height": self.height,
        }


class BaseDecoder(ABC):

    @property
    @abstractmethod
    def codec_type(self) -> str:
        pass

    @abstractmethod
    def decode(self, encoded_data: bytes, width: int, height: int) -> Optional[NDArray]:
        pass

    @abstractmethod
    def close(self) -> None:
        pass


class FFmpegPipeEncoder(BaseEncoder):

    def __init__(
        self,
        width: int,
        height: int,
        framerate: int,
        codec_type: str,
        logger,
        logging: bool = False,
    ):
        self._logging = logging
        self._width = width
        self._height = height
        self._framerate = framerate
        self._codec_type_str = codec_type
        self._logger = logger
        self._process = None
        self._frame_size = width * height * 3
        self._codec = None

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def codec_type(self) -> str:
        return self._codec_type_str

    @property
    def codec(self) -> str:
        return self._codec

    def _build_ffmpeg_cmd(self) -> list:
        raise NotImplementedError("Subclasses must implement _build_ffmpeg_cmd")

    def _start_process(self, cmd: list) -> None:
        try:
            self._process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=self._frame_size * 2
            )
        except Exception as e:
            self._logger.error("Failed to create encoder: {}".format(e))
            raise

    def encode(self, bgr_frame: NDArray) -> bytes:
        import cv2
        import select

        if self._process is None or self._process.poll() is not None:
            return b''

        if bgr_frame.shape[1] != self._width or bgr_frame.shape[0] != self._height:
            bgr_frame = cv2.resize(bgr_frame, (self._width, self._height))

        try:
            self._process.stdin.write(bgr_frame.tobytes())
            self._process.stdin.flush()

            encoded_data = b''

            if sys.platform == 'win32':
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
            self._logging and self._logger.error("Encode error: {}".format(e))
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
        self._logging and self._logger.debug("Closing {}".format(self.__class__.__name__))

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

    def get_compression_metadata(self) -> dict:
        return {
            "type": self.codec_type,
            "codec": self._codec,
            "width": self._width,
            "height": self._height,
        }


class FFmpegSyncEncoder(BaseEncoder):

    def __init__(
        self,
        width: int,
        height: int,
        framerate: int,
        codec_type: str,
        logger,
        logging: bool = False,
    ):
        self._logging = logging
        self._width = width
        self._height = height
        self._framerate = framerate
        self._codec_type_str = codec_type
        self._logger = logger
        self._frames = []
        self._codec = None

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def codec_type(self) -> str:
        return self._codec_type_str

    @property
    def codec(self) -> str:
        return self._codec

    def _build_ffmpeg_cmd(self) -> list:
        raise NotImplementedError("Subclasses must implement _build_ffmpeg_cmd")

    def encode(self, bgr_frame: NDArray) -> bytes:
        import cv2

        if bgr_frame.shape[1] != self._width or bgr_frame.shape[0] != self._height:
            bgr_frame = cv2.resize(bgr_frame, (self._width, self._height))

        self._frames.append(bgr_frame.copy())
        return b''

    def encode_all(self) -> bytes:
        if not self._frames:
            return b''

        cmd = self._build_ffmpeg_cmd()

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
            self._logging and self._logger.error("Encode error: {}".format(e))
            return b''

    def flush(self) -> bytes:
        return b''

    def close(self) -> None:
        self._logging and self._logger.debug("Closing {}".format(self.__class__.__name__))
        self._frames = []

    def get_compression_metadata(self) -> dict:
        return {
            "type": self.codec_type,
            "codec": self._codec,
            "width": self._width,
            "height": self._height,
        }