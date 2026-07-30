import numpy as np
from typing import Optional
from numpy.typing import NDArray

from ..utils.common import get_logger

logger = get_logger("EchoCanceller")


class EchoCanceller:
    """Removes the far end's own voice from what the microphone picks up.

    Without this, anyone not wearing a headset sends the other person's speech
    straight back to them a fraction of a second later. The room is treated as
    an unknown filter applied to the audio just played, and that filter is
    learned continuously and subtracted from the microphone signal.

    The filter is partitioned and runs in the frequency domain, which is what
    makes an echo tail of tens of milliseconds affordable inside a realtime
    audio callback: each block costs three transforms and two array multiplies
    rather than a tap-by-tap convolution.

    The reference signal must be the audio actually handed to the speaker, and
    it must be given for every block, including the silent ones. Skipping
    silence would shift the reference against the microphone and the filter
    would be learning a delay that is not there.
    """

    def __init__(
        self,
        block_size: int,
        sample_rate: int = 48000,
        tail_ms: float = 200.0,
        step_size: float = 0.5,
        residual_suppression: bool = True,
        logging: bool = False,
    ):
        if block_size <= 0:
            raise ValueError("block_size must be positive")

        self._logging = logging
        self._n = int(block_size)
        self._sr = int(sample_rate)
        self._fft = 2 * self._n
        self._bins = self._fft // 2 + 1

        tail_samples = max(self._n, int(sample_rate * tail_ms / 1000.0))
        self._partitions = max(1, int(np.ceil(tail_samples / self._n)))

        self._weights = np.zeros((self._partitions, self._bins), dtype=np.complex128)
        self._ref_history = np.zeros((self._partitions, self._bins), dtype=np.complex128)
        self._ref_tail = np.zeros(self._n, dtype=np.float64)
        self._err_tail = np.zeros(self._n, dtype=np.float64)

        self._power = np.full(self._bins, 1e-6)
        self._step = float(step_size)
        self._eps = 1e-8
        self._power_floor = 1e-5
        self._quiet_far = 1e-7
        self._diverged_blocks = 0
        self._divergences = 0

        self._residual_suppression = residual_suppression
        self._echo_power = np.full(self._bins, 1e-9)
        self._err_power = np.full(self._bins, 1e-9)

        self._frozen_blocks = 0
        self._adapted_blocks = 0
        self._erle_db = 0.0
        self._near_energy = 1e-9
        self._echo_energy = 1e-9
        self._far_energy = 1e-9
        self._coupling_guess = 0.25
        self._converged = False
        self._inst_near = 1e-9
        self._inst_echo = 1e-9
        self._inst_far = 1e-9
        self._far_recent = np.full(self._partitions + 1, 1e-12)
        self._hangover = 0
        self._hangover_blocks = max(1, int(0.15 * sample_rate / self._n))

        self._logging and logger.debug(
            "EchoCanceller ready: {} Hz, {}-sample blocks, {:.0f} ms tail "
            "({} partitions).".format(sample_rate, self._n, tail_ms, self._partitions)
        )

    @property
    def block_size(self) -> int:
        return self._n

    @property
    def tail_ms(self) -> float:
        return self._partitions * self._n * 1000.0 / self._sr

    @property
    def erle_db(self) -> float:
        """How much echo is being removed, in decibels. Zero means none."""
        return round(self._erle_db, 1)

    @property
    def adapted_blocks(self) -> int:
        return self._adapted_blocks

    @property
    def frozen_blocks(self) -> int:
        """Blocks where adaptation was held because both people were talking."""
        return self._frozen_blocks

    def impulse_response(self) -> NDArray:
        """The echo path as currently learned, in the time domain.

        Useful for judging the filter on its own terms: convolve the far end
        with this and compare against the real echo, which separates how well
        the path is modelled from anything the suppressor does afterwards.
        """
        taps = np.fft.irfft(self._weights, n=self._fft, axis=1)[:, :self._n]
        return taps.reshape(-1)

    def reset(self) -> None:
        self._weights[:] = 0
        self._ref_history[:] = 0
        self._ref_tail[:] = 0
        self._err_tail[:] = 0
        self._power[:] = 1e-6
        self._echo_power[:] = 1e-9
        self._err_power[:] = 1e-9
        self._erle_db = 0.0
        self._converged = False
        self._hangover = 0

    def process(self, mic: NDArray, reference: NDArray) -> NDArray:
        """Subtract the echo of `reference` from `mic`, one block at a time.

        Both arrays must hold exactly block_size frames. Returns the cleaned
        microphone signal in the same dtype and shape it was given.
        """
        original_shape = mic.shape
        original_dtype = mic.dtype

        near = self._to_mono_float(mic)
        far = self._to_mono_float(reference)

        if near.shape[0] != self._n or far.shape[0] != self._n:
            return mic

        # Overlap-save: each transform sees the previous block and this one.
        block = np.concatenate((self._ref_tail, far))
        self._ref_tail = far.copy()
        spectrum = np.fft.rfft(block)

        self._ref_history = np.roll(self._ref_history, 1, axis=0)
        self._ref_history[0] = spectrum

        estimated = np.fft.irfft(
            (self._weights * self._ref_history).sum(axis=0), n=self._fft
        )[self._n:]

        error = near - estimated

        self._track_energies(near, far, estimated, error)

        # The reference power is tracked on every block, adapting or not, or a
        # long frozen stretch would leave the step size normalised by stale
        # numbers and the first update afterwards would overshoot wildly.
        self._power *= 0.9
        self._power += 0.1 * (np.abs(spectrum) ** 2)
        # A floor under the estimate. Without one it decays towards zero through
        # every quiet passage, and the next update divides by almost nothing and
        # throws the filter to infinity; the microphone then comes out louder
        # than it went in.
        np.maximum(self._power, self._power_floor, out=self._power)

        adapt = not self._is_double_talk()

        if self._check_divergence(near, error):
            adapt = False

        if adapt:
            self._adapted_blocks += 1

            padded_error = np.concatenate((np.zeros(self._n), error))
            error_spectrum = np.fft.rfft(padded_error)

            gradient = (
                np.conj(self._ref_history)
                * error_spectrum
                / (self._partitions * self._power + self._eps)
            )
            self._weights += self._step * gradient
            self._constrain()
        else:
            self._frozen_blocks += 1

        out = error
        if self._residual_suppression:
            out = self._suppress_residual(error, estimated, gentle=not adapt)

        return self._restore(out, original_shape, original_dtype)

    def _check_divergence(self, near, error) -> bool:
        """Notice the filter making things worse, and undo it.

        Subtracting an estimate that has gone wrong adds energy rather than
        removing it, and the person at the other end hears something worse than
        no cancellation at all. Persistently louder output than input is the
        symptom, and the only safe response is to throw the learned filter away
        and start again.
        """
        near_e = float(np.dot(near, near))
        err_e = float(np.dot(error, error))

        if err_e > near_e * 4.0 and near_e > 1e-12:
            self._diverged_blocks += 1
        else:
            self._diverged_blocks = max(0, self._diverged_blocks - 1)

        if self._diverged_blocks > 10:
            self._divergences += 1
            logger.warning(
                "Echo canceller diverged and was reset ({} so far).".format(self._divergences)
            )
            self.reset()
            self._diverged_blocks = 0
            return True
        return False

    @property
    def divergences(self) -> int:
        return self._divergences

    def _constrain(self) -> None:
        """Discard the part of each partition's response that cannot be real.

        Without this the filter slowly accumulates energy in the wrapped-around
        half of every block and the cancellation drifts apart.
        """
        taps = np.fft.irfft(self._weights, n=self._fft, axis=1)
        taps[:, self._n:] = 0.0
        self._weights = np.fft.rfft(taps, n=self._fft, axis=1)

    def _track_energies(self, near, far, estimated, error) -> None:
        near_e = float(np.dot(near, near)) + 1e-12
        far_e = float(np.dot(far, far)) + 1e-12
        err_e = float(np.dot(error, error)) + 1e-12
        echo_e = float(np.dot(estimated, estimated)) + 1e-12

        self._inst_near, self._inst_far, self._inst_echo = near_e, far_e, echo_e
        self._far_recent = np.roll(self._far_recent, 1)
        self._far_recent[0] = far_e
        self._near_energy = self._near_energy * 0.9 + near_e * 0.1
        self._far_energy = self._far_energy * 0.9 + far_e * 0.1
        self._echo_energy = self._echo_energy * 0.9 + echo_e * 0.1

        if near_e > 1e-9:
            instant = 10.0 * np.log10(near_e / err_e)
            self._erle_db = self._erle_db * 0.9 + instant * 0.1

        if not self._converged and self._erle_db > 6.0 and self._adapted_blocks > 200:
            self._converged = True

    def _is_double_talk(self) -> bool:
        """Whether the near speaker is talking over the far end.

        Adapting while both talk teaches the filter that the near voice is echo,
        and it starts cancelling the wrong person, so adaptation is held instead.

        Which evidence answers this depends on how far along the filter is. Once
        it cancels well its own estimate is the sharp test, but before then that
        estimate is near zero and using it would freeze adaptation permanently:
        the filter cannot converge because it never adapts, and never adapts
        because it has not converged. Until then the far end's own level bounds
        how loud the echo could possibly be, which needs no convergence at all.
        """
        # The loudest the far end has been anywhere inside the filter's reach.
        # Comparing against the current block instead would misjudge constantly:
        # the far end can be silent right now while the echo of what it said a
        # moment ago is still arriving, and that reads as the near speaker.
        peak_far = float(self._far_recent.max())

        if peak_far < self._quiet_far:
            return True

        if self._converged:
            talking = self._inst_near > self._inst_echo * 2.0
        else:
            talking = self._inst_near > peak_far * self._coupling_guess

        # Speech is full of short gaps. Judging each block on its own merits
        # would resume adaptation inside every pause between syllables, and the
        # filter would spend those gaps learning the near speaker's voice as if
        # it were echo. Once double talk is seen, adaptation stays held for a
        # while after it appears to stop.
        if talking:
            self._hangover = self._hangover_blocks
        elif self._hangover > 0:
            self._hangover -= 1

        return talking or self._hangover > 0

    def _suppress_residual(self, error, estimated, gentle: bool = False) -> NDArray:
        """Attenuate what the linear filter could not remove.

        A real room is not perfectly linear and the loudspeaker is not either,
        so a quiet remnant survives subtraction. Bins where the estimated echo
        still dominates the error get pulled down.

        When the near speaker is talking this has to be held back. The remnant
        and their voice occupy the same bins, so suppression aggressive enough
        to remove the one chews audible holes in the other, and the person the
        listener actually wants to hear comes out mangled.
        """
        error_spec = np.fft.rfft(error, n=self._fft)
        echo_spec = np.fft.rfft(estimated, n=self._fft)

        self._err_power = self._err_power * 0.7 + 0.3 * np.abs(error_spec) ** 2
        self._echo_power = self._echo_power * 0.7 + 0.3 * np.abs(echo_spec) ** 2

        weight = 0.05 if gentle else 0.3
        floor = 0.5 if gentle else 0.05

        gain = self._err_power / (self._err_power + weight * self._echo_power + self._eps)
        np.clip(gain, floor, 1.0, out=gain)

        cleaned = np.fft.irfft(error_spec * gain, n=self._fft)[:self._n]
        return cleaned

    def _to_mono_float(self, data: NDArray) -> NDArray:
        arr = np.asarray(data)
        # Read the dtype before mixing channels down. Averaging promotes
        # integers to float, so asking afterwards says float for what were
        # whole-numbered samples, the scaling is skipped, and everything
        # arrives here a factor of thirty thousand too large.
        dtype = arr.dtype
        if arr.ndim > 1:
            arr = arr.mean(axis=1)
        if np.issubdtype(dtype, np.integer):
            return arr.astype(np.float64) / float(np.iinfo(dtype).max)
        return arr.astype(np.float64, copy=False)

    def _restore(self, mono: NDArray, shape, dtype) -> NDArray:
        if np.issubdtype(dtype, np.integer):
            peak = float(np.iinfo(dtype).max)
            out = np.clip(mono * peak, -peak, peak).astype(dtype)
        else:
            out = np.clip(mono, -1.0, 1.0).astype(dtype)

        if len(shape) > 1:
            return np.repeat(out[:, None], shape[1], axis=1)
        return out
