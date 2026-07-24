import numpy as np
from numpy.typing import NDArray


class JitterBuffer:

    def __init__(
        self,
        sample_rate: int = 48000,
        channels: int = 1,
        target_ms: float = 80.0,
        max_ms: float = 200.0,
        dtype: str = "int16",
    ):
        self._sr = int(sample_rate)
        self._ch = int(channels)
        self._dtype = np.dtype(dtype)
        self._target = max(1, int(target_ms * self._sr / 1000))
        self._max = max(self._target + 1, int(max_ms * self._sr / 1000))
        self._discontinuity = 2 * self._sr
        self._base_ts = None
        self._play_idx = 0
        self._buf_start = 0
        self._buf = np.zeros((0, self._ch), self._dtype)
        self._primed = False

    @property
    def sample_rate(self) -> int:
        return self._sr

    @property
    def channels(self) -> int:
        return self._ch

    @property
    def depth(self) -> int:
        if self._base_ts is None:
            return 0
        return (self._buf_start + len(self._buf)) - self._play_idx

    @property
    def depth_ms(self) -> float:
        return self.depth * 1000.0 / self._sr

    @property
    def playout_pts_ns(self):
        if self._base_ts is None or not self._primed:
            return None
        return self._base_ts + int(self._play_idx * 1e9 / self._sr)

    def _shape(self, pcm: NDArray) -> NDArray:
        pcm = np.asarray(pcm)
        if pcm.dtype != self._dtype:
            pcm = pcm.astype(self._dtype)
        if pcm.ndim == 1:
            pcm = pcm.reshape(-1, 1)
        if pcm.shape[1] == self._ch:
            return pcm
        if pcm.shape[1] == 1 and self._ch > 1:
            return np.tile(pcm, (1, self._ch))
        if pcm.shape[1] > self._ch:
            return pcm[:, :self._ch].copy()
        pad = np.zeros((pcm.shape[0], self._ch - pcm.shape[1]), self._dtype)
        return np.concatenate([pcm, pad], axis=1)

    def _reset_to(self, pts_ns: int) -> None:
        self._base_ts = int(pts_ns)
        self._play_idx = 0
        self._buf_start = 0
        self._buf = np.zeros((0, self._ch), self._dtype)
        self._primed = False

    def _place(self, pcm: NDArray, idx: int) -> None:
        n = pcm.shape[0]
        end = idx + n
        if end <= self._play_idx:
            return

        if idx < self._play_idx:
            pcm = pcm[self._play_idx - idx:]
            idx = self._play_idx
            n = pcm.shape[0]
            end = idx + n

        buf_end = self._buf_start + len(self._buf)
        if end > buf_end:
            self._buf = np.concatenate(
                [self._buf, np.zeros((end - buf_end, self._ch), self._dtype)], axis=0
            )

        local = idx - self._buf_start
        self._buf[local:local + n] = pcm

    def insert(self, pcm: NDArray, pts_ns=None) -> None:
        pcm = self._shape(pcm)
        if pcm.shape[0] == 0:
            return

        if pts_ns is None:
            if self._base_ts is None:
                self._reset_to(0)
            self._place(pcm, max(self._play_idx, self._buf_start + len(self._buf)))
            return

        if self._base_ts is None:
            self._reset_to(pts_ns)

        idx = int(round((int(pts_ns) - self._base_ts) * self._sr / 1e9))

        if idx < self._play_idx - self._discontinuity or idx > self._play_idx + self._discontinuity:
            self._reset_to(pts_ns)
            idx = 0

        self._place(pcm, idx)

    def pop(self, n: int) -> NDArray:
        out = np.zeros((n, self._ch), self._dtype)
        if self._base_ts is None:
            return out

        if not self._primed:
            if self.depth < self._target:
                return out
            self._primed = True

        if self.depth > self._max:
            self._play_idx = (self._buf_start + len(self._buf)) - self._target

        local = self._play_idx - self._buf_start
        if local < 0:
            local = 0
            self._play_idx = self._buf_start

        available = len(self._buf) - local
        take = min(n, max(0, available))
        if take > 0:
            out[:take] = self._buf[local:local + take]

        self._play_idx += n

        drop = self._play_idx - self._buf_start
        if drop >= len(self._buf):
            self._buf = np.zeros((0, self._ch), self._dtype)
            self._buf_start = self._play_idx
        elif drop > 0:
            self._buf = self._buf[drop:]
            self._buf_start = self._play_idx

        return out

    def reset(self) -> None:
        self._base_ts = None
        self._play_idx = 0
        self._buf_start = 0
        self._buf = np.zeros((0, self._ch), self._dtype)
        self._primed = False
