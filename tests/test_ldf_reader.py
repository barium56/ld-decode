"""Tests for the LDF reader (lddecode.ldf_reader, the ld-ldf-reader-py tool).

The reader demuxes and decodes the audio track of an LDF capture with PyAV
and writes mono s16 PCM to stdout.  It has no test coverage at all despite
being a shipped CLI tool, so these tests cover the file-open failure paths,
the start-offset skipping logic, and the argument validation in main().
"""

import io
import sys
import wave

import numpy as np
import pytest

from lddecode.ldf_reader import LdfReader, main


class FakeStdout:
    """Minimal stand-in for sys.stdout that only supports what the reader uses."""

    def __init__(self):
        self.buffer = io.BytesIO()

    def flush(self):
        pass


def run_reader(path, start_offset=0):
    """Run LdfReader.process() capturing its stdout writes.

    sys.stdout must be swapped inside the test body: pytest reinstalls its
    own capture object at call time, which would otherwise swallow the
    reader's binary output."""
    fake = FakeStdout()
    old = sys.stdout
    sys.stdout = fake
    try:
        reader = LdfReader(path, start_offset=start_offset, debug=False)
        ok = reader.process()
        return ok, fake.buffer.getvalue()
    finally:
        sys.stdout = old


def make_wav(path, rate=44100, samples=8000, freq=440.0):
    """Write a mono 16-bit sine WAV with the stdlib wave module."""
    t = np.arange(samples) / rate
    data = (np.sin(2 * np.pi * freq * t) * 1000).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(data.tobytes())
    return data


def test_missing_input_file_returns_false():
    ok, _ = run_reader("/nonexistent/file.ldf")
    assert ok is False


def test_decodes_wav_to_stdout(tmp_path):
    path = tmp_path / "test.wav"
    make_wav(path)

    ok, pcm = run_reader(str(path))
    assert ok is True

    samples = np.frombuffer(pcm, dtype=np.int16)
    # 8000 input samples decode to exactly 8000 mono s16 samples.
    assert len(samples) == 8000
    assert samples[0] == 0  # sin(0)
    assert samples[1] > 0  # rising edge of the sine


def test_start_offset_skips_exact_samples(tmp_path):
    path = tmp_path / "test.wav"
    data = make_wav(path)
    offset = 4000

    ok, pcm = run_reader(str(path), start_offset=offset)
    assert ok is True

    samples = np.frombuffer(pcm, dtype=np.int16)
    assert len(samples) == len(data) - offset
    # The first output sample must be the sine sample at the offset position.
    assert samples[0] == data[offset]


def test_start_offset_past_eof_writes_nothing(tmp_path):
    path = tmp_path / "test.wav"
    make_wav(path, samples=1000)

    ok, pcm = run_reader(str(path), start_offset=9000)
    assert ok is True
    assert pcm == b""


def test_unopenable_file_reports_failure(tmp_path):
    path = tmp_path / "not-audio.bin"
    path.write_bytes(b"this is not a media container")

    ok, _ = run_reader(str(path))
    assert ok is False


class TestMain:
    def test_version_exits_zero(self):
        with pytest.raises(SystemExit) as excinfo:
            main(["--version"])
        assert excinfo.value.code == 0

    def test_negative_start_offset_rejected(self):
        assert main(["-s", "-1", "anything.ldf"]) == 1

    def test_missing_input_returns_error(self):
        assert main(["/nonexistent/file.ldf"]) == 1

    def test_valid_input_processes(self, tmp_path):
        path = tmp_path / "test.wav"
        make_wav(path, samples=2000)
        assert main([str(path)]) == 0
