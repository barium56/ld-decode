"""Tests for the DSP and parsing helpers in lddecode.utils.

utils.py is the shared toolbox for the decoder core: it provides the
zero-crossing detection, pulse/peak finding, FM discriminator, FFT
overlap-save machinery and the file loader dispatch that core.py relies on.
It has only ~24% coverage, so these tests focus on the helpers that carry
real production logic and are cheap to exercise in isolation.
"""

import numpy as np
import pytest
import scipy.signal as sps

from lddecode import utils


class TestParseFrequency:
    def test_plain_mhz(self):
        assert utils.parse_frequency("40") == 40.0

    def test_suffixes(self):
        assert utils.parse_frequency("40MHz") == 40.0
        assert utils.parse_frequency("40mhz") == 40.0
        assert utils.parse_frequency("40khz") == pytest.approx(0.04)
        assert utils.parse_frequency("20000khz") == pytest.approx(20.0)
        assert utils.parse_frequency("0.5ghz") == pytest.approx(500.0)

    def test_fsc_suffixes(self):
        # fsc = 315/88 MHz (NTSC colour subcarrier); fscpal = (283.75*15625)+25 Hz.
        assert utils.parse_frequency("40fsc") == pytest.approx(40 * 315.0 / 88.0)
        assert utils.parse_frequency("40fscpal") == pytest.approx(
            40 * ((283.75 * 15625) + 25) / 1.0e6
        )


class TestMakeLoader:
    def test_dispatches_by_extension(self):
        assert utils.make_loader("x.lds") is utils.load_packed_data_4_40
        assert utils.make_loader("x.r30") is utils.load_packed_data_3_32
        assert utils.make_loader("x.rf") is utils.load_unpacked_data_float32
        assert utils.make_loader("x.s16") is utils.load_unpacked_data_s16
        assert utils.make_loader("x.u16") is utils.load_unpacked_data_u16
        assert utils.make_loader("x.u8") is utils.load_unpacked_data_u8
        assert isinstance(utils.make_loader("x.ldf"), utils.LoadLDF)

    def test_resampling_rejects_packed_formats(self):
        # .lds/.r30 cannot be resampled to 40 MHz; make_loader must refuse.
        with pytest.raises(ValueError):
            utils.make_loader("x.lds", inputfreq=8)
        with pytest.raises(ValueError):
            utils.make_loader("x.r30", inputfreq=8)


class TestCalczc:
    def test_rising_falling_and_either_edge(self):
        data = np.array([0.0, 1.0, 2.0, -1.0, -2.0, 1.0], dtype=np.float64)
        # Rising crossing of 0.5 between samples 0 and 1.
        assert utils.calczc(data, 0, 0.5, edge=1) == pytest.approx(0.5)
        # Falling crossing between samples 2 and 3.
        assert utils.calczc(data, 0, 0.5, edge=-1) == pytest.approx(2.5)
        # Either edge finds the first crossing.
        assert utils.calczc(data, 0, 0.5) == pytest.approx(0.5)

    def test_linear_interpolation_is_exact(self):
        data = np.array([10.0, 0.0, -10.0], dtype=np.float64)
        assert utils.calczc(data, 0, 0.0) == pytest.approx(1.0)

    def test_reverse_search(self):
        data = np.array([10.0, 0.0, -10.0], dtype=np.float64)
        # Searching backwards from the end finds the same crossing at 1.0.
        assert utils.calczc(data, 2, 0.0, reverse=True) == pytest.approx(1.0)

    def test_no_crossing_returns_none(self):
        data = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        assert utils.calczc(data, 0, 0.5) is None


class TestFindPulses:
    def test_detects_low_pulses(self):
        # Pulses are areas where the signal is <= high.  The first pulse is
        # suppressed (it starts at sample 0) and the trailing pulse is ignored.
        sig = np.zeros(100, dtype=np.float64)
        sig[10:15] = 0.9
        sig[30:33] = 1.1
        sig[50:52] = 0.8
        sig[70:90] = 1.2
        pulses = utils.findpulses(sig, None, 1.0)
        assert pulses == [utils.Pulse(33, 37)]

    def test_min_max_length_filter(self):
        # Two high excursions sandwich a low pulse of length 5.
        sig = np.zeros(100, dtype=np.float64)
        sig[10:15] = 1.1
        sig[20:25] = 1.1
        starts, lengths = utils.findpulses_numba_raw(sig, 1.0, min_synclen=4, max_synclen=6)
        np.testing.assert_array_equal(starts, np.array([15]))
        np.testing.assert_array_equal(lengths, np.array([5]))


class TestFindAreas:
    def test_areas_below_cross(self):
        arr = np.array([1.0, 0.3, 0.2, 0.3, 1.0, 0.4, 0.1, 1.0, 1.0], dtype=np.float64)
        assert utils.findareas(arr, 0.5) == [(0, 3, 3), (4, 6, 2)]


class TestEmphasisIir:
    def test_deemphasis_attenuates_highs(self):
        b, a = utils.emphasis_iir(75e-6, 3180e-6, 44100)
        # Stable one-pole filter with unity DC gain.
        assert abs(a[1]) < 1.0
        assert np.sum(b) / np.sum(a) == pytest.approx(1.0)
        gain = _gain_at(b, a, 44100, [1 / (2 * np.pi * 3180e-6), 1 / (2 * np.pi * 75e-6)])
        assert gain[1] < gain[0]  # high-frequency gain below low-frequency gain

    def test_preemphasis_boosts_highs(self):
        b, a = utils.emphasis_iir(3180e-6, 75e-6, 44100)
        assert abs(a[1]) < 1.0
        assert np.sum(b) / np.sum(a) == pytest.approx(1.0)
        gain = _gain_at(b, a, 44100, [1 / (2 * np.pi * 3180e-6), 1 / (2 * np.pi * 75e-6)])
        assert gain[1] > gain[0]  # high-frequency gain above low-frequency gain


def _gain_at(b, a, fs, freqs):
    w, h = sps.freqz(b, a, worN=2048)
    f = w * fs / (2 * np.pi)
    return [abs(h[np.argmin(abs(f - freq))]) for freq in freqs]


class TestUnwrapHilbert:
    def test_recovers_constant_frequency(self):
        fs = 40000.0
        t = np.arange(4096) / fs
        z = np.exp(2j * np.pi * 5000.0 * t)
        out = utils.unwrap_hilbert(z, fs)
        # Interior samples must equal the true instantaneous frequency.
        np.testing.assert_allclose(out[10:-10], 5000.0, atol=1e-6)

    def test_silence_is_zero(self):
        z = np.ones(64, dtype=np.complex128)
        out = utils.unwrap_hilbert(z, 40000.0)
        np.testing.assert_allclose(out, 0.0)


class TestBuildHilbert:
    def test_even_size(self):
        assert np.array_equal(utils.build_hilbert(8), [1, 2, 2, 2, 1, 0, 0, 0])

    def test_odd_size_raises(self):
        with pytest.raises(Exception):
            utils.build_hilbert(7)


class TestFftSlices:
    def test_determine_slices_returns_power_of_two(self):
        lowbin, nbins, cut_freq = utils.fft_determine_slices(10e6, 4e6, 40e6, 65536)
        assert nbins & (nbins - 1) == 0  # power of two

    def test_slice_preserves_bandwidth(self):
        lowbin, nbins, _ = utils.fft_determine_slices(10e6, 4e6, 40e6, 65536)
        fdomain = np.arange(65536, dtype=np.complex128)
        sliced = utils.fft_do_slice(fdomain, lowbin, nbins, 65536)
        assert len(sliced) == nbins


class TestOverlapSave:
    def test_round_trip_is_lossless(self):
        rng = np.random.default_rng(1)
        data = rng.integers(0, 255, 5000).astype(np.uint16)
        ffts = utils.overlap_save_fft(data)
        invf = utils.overlap_save_ifft(ffts, round=True).astype(np.uint16)
        np.testing.assert_array_equal(invf, data)


class TestHzToOutputArray:
    def test_rounds_to_nearest(self):
        inp = np.array([0.0, 1.4, 1.6, -1, 100.0], dtype=np.float64)
        out = utils.hz_to_output_array(inp, 0.0, 1.0, 0.0, 0.0, 1.0)
        np.testing.assert_array_equal(out, [0, 1, 2, 0, 100])


class TestGenwave:
    def test_constant_frequency_zero_crossings(self):
        # genwave(rate, fs) produces a sine at rate/2 Hz: 2000 Hz target at
        # 40 kHz over 4000 samples is 100 cycles, i.e. ~200 crossings.
        wave = utils.genwave(np.full(4000, 2000.0), 40000.0)
        zc = int(
            np.sum((wave[:-1] < 0) & (wave[1:] >= 0)) + np.sum((wave[:-1] >= 0) & (wave[1:] < 0))
        )
        assert 198 <= zc <= 202


class TestScalarHelpers:
    def test_rms(self):
        assert utils.rms(np.array([3.0, 4.0])) == pytest.approx(0.5)
        assert utils.rms(np.array([1.0, 1.0])) == 0.0

    def test_dsa_rescale_and_clip(self):
        assert utils.dsa_rescale_and_clip(-400000.0) == -32766
        assert utils.dsa_rescale_and_clip(0.0) == 0
        assert utils.dsa_rescale_and_clip(371081.0) == 32766
        assert utils.dsa_rescale_and_clip(400000.0) == 32766

    def test_db_lev_round_trip(self):
        assert utils.db_to_lev(-6) == pytest.approx(10 ** (-6 / 20))
        assert utils.lev_to_db(1.0) == pytest.approx(0.0)
        assert utils.lev_to_db(utils.db_to_lev(12)) == pytest.approx(12)

    def test_roundfloat(self):
        assert utils.roundfloat(1.23456) == pytest.approx(1.235)

    def test_distance_from_round(self):
        assert utils.distance_from_round(3.1) == pytest.approx(-0.1)
        assert utils.distance_from_round(3.9) == pytest.approx(0.1)


class TestLruUpdate:
    def test_moves_item_to_front(self):
        l = [1, 2, 3]
        utils.LRUupdate(l, 2)
        assert l == [2, 1, 3]

    def test_insert_new_item(self):
        l = [1, 2, 3]
        utils.LRUupdate(l, 9)
        assert l == [9, 1, 2, 3]


class TestStridedCollector:
    def test_accumulates_and_keeps_overlap(self):
        sc = utils.StridedCollector(blocklen=8, cut_begin=2, cut_end=0)
        assert not sc.have_block()
        assert not sc.add(np.arange(6, dtype=np.float64))
        assert sc.add(np.arange(6, dtype=np.float64) + 10)
        block = sc.get_block()
        np.testing.assert_array_equal(block, [0, 1, 2, 3, 4, 5, 10, 11])
        np.testing.assert_array_equal(sc.buffer, [10, 11, 12, 13, 14, 15])


class TestFieldInfo:
    def test_ring_buffer_negative_indexing(self):
        fi = utils.FieldInfo(field_history_size=3)
        for v in ("a", "b", "c", "d"):
            fi.append(v)
        assert fi[-1] == "d"
        assert fi[-2] == "c"

        with pytest.raises(AssertionError):
            fi[0]  # not yet written
        with pytest.raises(AssertionError):
            fi[-3]  # too old for the ring buffer

    def test_read_returns_unsent(self):
        fi = utils.FieldInfo(field_history_size=3)
        fi.append("a")
        fi.append("b")
        assert fi.read() == ["a", "b"]
        assert fi.read() == []


class TestClbFindbursts:
    def test_detects_zero_crossings_in_burst(self):
        # A clean sine burst: every half-cycle crossing is found, spaced by
        # the half period, and the rising/falling flags alternate.
        n = 64
        burst = np.sin(2 * np.pi * np.arange(n) / 8.0)
        isrising = np.zeros(16, dtype=np.bool_)
        zcs = np.zeros(16, dtype=np.float64)
        zc_count, phase_adjust, rising_count = utils.clb_findbursts(
            isrising, zcs, burst, 0, n, 0.1, 0.0, 0.0, 1.0, 0.0
        )
        assert zc_count == 15
        np.testing.assert_allclose(zcs[:zc_count], np.arange(4, 4 + 15 * 4, 4))
        assert phase_adjust == pytest.approx(0.0)
        assert rising_count == 7
