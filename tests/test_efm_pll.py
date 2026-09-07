"""Tests for the EFM PLL (lddecode.efm_pll).

The EFM PLL converts the LaserDisc EFM signal (CD-format digital audio) into
run-length (T-value) data.  It is the only stage between the raw RF samples
and the .efm stream that ld-ldstoefm decodes, so a regression here silently
corrupts every digital audio decode.  These tests exercise the zero-crossing
detector, the PLL's run-length output, and the period/dropout handling.
"""

import numpy as np
import pytest

from lddecode.efm_pll import EFM_PLL, computeefmfilter


def make_signal(runs, period, amplitude=1000.0):
    """NRZI square wave: the level flips at the start of each run, and run k
    lasts runs[k] bit periods.  The trailing zero tail produces the final
    zero-crossing."""
    edges = np.cumsum([0.0] + [r * period for r in runs])
    total = int(np.ceil(edges[-1])) + 2
    sig = np.zeros(total, dtype=np.int16)
    level = amplitude
    for i in range(len(runs)):
        a = int(round(edges[i]))
        b = int(round(edges[i + 1]))
        sig[a:b] = level
        level = -level
    return sig


def pll_output(runs, period=None):
    """Run a fresh PLL over a synthetic signal and return the T-values.

    process() returns a view into an internal buffer that is reused on the
    next call, so copy it before returning."""
    pll = EFM_PLL()
    if period is None:
        period = pll.basePeriod
    return pll.process(make_signal(runs, period)).copy()


class TestComputeEfmFilter:
    def test_length_and_dtype(self):
        coeffs = computeefmfilter()
        assert len(coeffs) == 65536
        assert coeffs.dtype == np.complex128

    def test_dc_is_removed(self):
        # The first band is zero amplitude, so the DC bin must be zero.
        assert computeefmfilter()[0] == 0.0 + 0.0j

    def test_passband_is_nonzero(self):
        # Mid-band bins carry the equalisation gain.
        coeffs = computeefmfilter()
        assert np.all(np.abs(coeffs[100:1000]) > 0)


class TestEfmPll:
    def test_steady_run_lengths(self):
        # A long T=6 run must settle to exactly 6s (only the first edge
        # carries the PLL's startup transient).
        out = pll_output([6] * 40)
        assert np.all(out[1:] == 6)
        assert out[0] == 7

    def test_mixed_run_lengths(self):
        # Every run length from T3 to T11 must be recovered exactly after the
        # startup edge.
        runs = [3, 11, 4, 8, 5, 10, 3, 7, 6, 9]
        out = pll_output(runs)
        np.testing.assert_array_equal(out[1:], np.array(runs[1:], dtype=np.int8))

    def test_output_stays_in_valid_efm_range(self):
        runs = [3, 11, 4, 8, 5, 10, 3, 7, 6, 9, 3, 11, 4, 8]
        out = pll_output(runs)
        assert np.all(out >= 1)
        assert np.all(out <= 11)

    def test_processing_in_chunks_is_equivalent(self):
        # zcPreviousInput and delta must carry across process() calls, so
        # feeding a signal in two chunks gives the same result as one call.
        runs = [5, 7, 3, 9] * 4
        pll = EFM_PLL()
        period = pll.basePeriod
        sig = make_signal(runs, period)
        mid = len(sig) // 2

        chunked = np.concatenate([pll.process(sig[:mid]).copy(), pll.process(sig[mid:]).copy()])
        whole = pll_output(runs, period)
        np.testing.assert_array_equal(chunked, whole)

    def test_period_stays_bounded_under_frequency_offset(self):
        # A signal 15% off the nominal bit rate must not push currentPeriod
        # outside the PLL's hard limits.
        for factor in (0.85, 1.15):
            pll = EFM_PLL()
            out = pll.process(make_signal([3, 7, 5, 9] * 6, pll.basePeriod * factor))
            assert len(out) > 0
            assert pll.minimumPeriod <= pll.currentPeriod <= pll.maximumPeriod

    def test_dropout_does_not_break_lock(self):
        # A huge gap (200 bit periods) must clamp the run length at T11 rather
        # than emitting garbage, and the PLL must recover to the true run
        # length once the signal returns.
        runs = [4, 4, 200, 4, 4, 4, 4]
        out = pll_output(runs)
        assert np.all(out <= 11)
        # The post-dropout tail settles back to T4.
        assert np.all(out[-4:] == 4)

    def test_result_buffer_grows_with_input(self):
        # process() must reallocate its result buffer for inputs larger than
        # the initial 1<<16 samples.
        pll = EFM_PLL()
        big = make_signal([6] * 40000, pll.basePeriod)
        out = pll.process(big).copy()
        assert len(out) > 0
        assert np.all(out[1:] == 6)
