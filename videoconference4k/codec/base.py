import subprocess
import threading
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
        self._out_buffer = bytearray()
        self._out_lock = threading.Lock()
        self._stdout_thread = None
        self._stderr_thread = None

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
                chunk = stream.read1(65536)
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

    def encode(self, bgr_frame: NDArray) -> bytes:
        import cv2

        if self._process is None or self._process.poll() is not None:
            return b''

        if bgr_frame.shape[1] != self._width or bgr_frame.shape[0] != self._height:
            bgr_frame = cv2.resize(bgr_frame, (self._width, self._height))

        try:
            self._process.stdin.write(bgr_frame.tobytes())
            self._process.stdin.flush()
        except Exception as e:
            self._logging and self._logger.error("Encode error: {}".format(e))
            return b''

        with self._out_lock:
            encoded_data = bytes(self._out_buffer)
            self._out_buffer.clear()
        return encoded_data

    def flush(self) -> bytes:
        if self._process is None:
            return b''

        try:
            self._process.stdin.close()
        except Exception:
            pass

        try:
            self._process.wait(timeout=5)
        except Exception:
            pass

        if self._stdout_thread is not None:
            self._stdout_thread.join(timeout=5)

        with self._out_lock:
            remaining = bytes(self._out_buffer)
            self._out_buffer.clear()
        return remaining

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

        if self._stdout_thread is not None:
            self._stdout_thread.join(timeout=1)
            self._stdout_thread = None

        if self._stderr_thread is not None:
            self._stderr_thread.join(timeout=1)
            self._stderr_thread = None

        with self._out_lock:
            self._out_buffer.clear()

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