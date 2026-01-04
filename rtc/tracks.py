import asyncio
import fractions
import time
import numpy as np
from typing import Any, Union

from ..utils.common import import_dependency_safe

aiortc = import_dependency_safe("aiortc", error="silent")
if aiortc is not None:
    from aiortc import VideoStreamTrack, AudioStreamTrack
    from av import VideoFrame, AudioFrame

VIDEO_CLOCK_RATE = 90000
VIDEO_TIME_BASE = fractions.Fraction(1, VIDEO_CLOCK_RATE)
AUDIO_CLOCK_RATE = 48000
AUDIO_TIME_BASE = fractions.Fraction(1, AUDIO_CLOCK_RATE)
AUDIO_PTIME = 0.020

LocalVideoTrack = None
LocalAudioTrack = None

if aiortc is not None:

    class _LocalVideoTrack(VideoStreamTrack):

        kind = "video"

        def __init__(
            self,
            video_source: Any = None,
            framerate: Union[int, float] = 30,
            logging: bool = False,
        ):
            super().__init__()
            self.__logging = logging
            self.__framerate = framerate
            self.__video_ptime = 1 / self.__framerate
            self.__video_source = video_source
            self.__started = False
            self.__timestamp = 0
            self.__start_time = None

        async def recv(self):
            if not self.__started:
                self.__start_time = time.time()
                self.__started = True
                if self.__video_source is not None and hasattr(self.__video_source, "start"):
                    if hasattr(self.__video_source, "is_running"):
                        if not self.__video_source.is_running:
                            self.__video_source.start()
                    else:
                        self.__video_source.start()

            pts = self.__timestamp
            self.__timestamp += int(self.__video_ptime * VIDEO_CLOCK_RATE)

            wait = self.__start_time + (pts / VIDEO_CLOCK_RATE) - time.time()
            if wait > 0:
                await asyncio.sleep(wait)

            frame = None
            if self.__video_source is not None:
                frame = self.__video_source.read()

            if frame is None:
                frame = np.zeros((480, 640, 3), dtype=np.uint8)

            if frame.ndim == 2:
                video_frame = VideoFrame.from_ndarray(frame, format="gray")
            elif frame.ndim == 3 and frame.shape[2] == 4:
                video_frame = VideoFrame.from_ndarray(frame, format="bgra")
            else:
                video_frame = VideoFrame.from_ndarray(frame, format="bgr24")

            video_frame.pts = pts
            video_frame.time_base = VIDEO_TIME_BASE

            return video_frame

        def stop(self):
            super().stop()
            if self.__video_source is not None and hasattr(self.__video_source, "stop"):
                self.__video_source.stop()


    class _LocalAudioTrack(AudioStreamTrack):

        kind = "audio"

        def __init__(
            self,
            audio_source: Any = None,
            sample_rate: int = 48000,
            channels: int = 1,
            logging: bool = False,
        ):
            super().__init__()
            self.__logging = logging
            self.__audio_source = audio_source
            self.__sample_rate = sample_rate
            self.__channels = channels
            self.__samples_per_frame = int(self.__sample_rate * AUDIO_PTIME)
            self.__started = False
            self.__timestamp = 0
            self.__start_time = None

        async def recv(self):
            if not self.__started:
                self.__start_time = time.time()
                self.__started = True
                if self.__audio_source is not None and hasattr(self.__audio_source, "start"):
                    if not self.__audio_source.is_running:
                        self.__audio_source.start()

            pts = self.__timestamp
            self.__timestamp += self.__samples_per_frame

            wait = self.__start_time + (pts / self.__sample_rate) - time.time()
            if wait > 0:
                await asyncio.sleep(wait)

            audio_data = None
            if self.__audio_source is not None:
                audio_data = self.__audio_source.read(timeout=0.001)

            if audio_data is None:
                audio_data = np.zeros((self.__samples_per_frame, self.__channels), dtype=np.int16)

            if audio_data.ndim == 1:
                audio_data = audio_data.reshape(-1, self.__channels)

            if len(audio_data) < self.__samples_per_frame:
                padding = np.zeros((self.__samples_per_frame - len(audio_data), self.__channels), dtype=np.int16)
                audio_data = np.vstack([audio_data, padding])
            elif len(audio_data) > self.__samples_per_frame:
                audio_data = audio_data[:self.__samples_per_frame]

            audio_frame = AudioFrame.from_ndarray(
                audio_data.T if self.__channels > 1 else audio_data.reshape(1, -1),
                format="s16",
                layout="mono" if self.__channels == 1 else "stereo",
            )
            audio_frame.pts = pts
            audio_frame.sample_rate = self.__sample_rate
            audio_frame.time_base = AUDIO_TIME_BASE

            return audio_frame

        def stop(self):
            super().stop()
            if self.__audio_source is not None and hasattr(self.__audio_source, "stop"):
                self.__audio_source.stop()

    LocalVideoTrack = _LocalVideoTrack
    LocalAudioTrack = _LocalAudioTrack