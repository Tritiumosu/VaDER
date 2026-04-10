"""
test_audio_passthrough.py

Unit tests for audio_passthrough.py.

All tests use mocked sounddevice so no physical audio hardware is required.
"""

from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import MagicMock, patch, call
import queue
import threading
import time

import numpy as np


# ---------------------------------------------------------------------------
# Stub sounddevice so the import does not require a real audio stack
# ---------------------------------------------------------------------------

def _make_sd_stub():
    sd = types.ModuleType("sounddevice")

    class FakeStream:
        """Minimal stand-in for sd.InputStream and sd.OutputStream."""
        def __init__(self, *args, **kwargs):
            self._callback = kwargs.get("callback")
            self.started  = False
            self.closed   = False

        def start(self):  self.started = True
        def stop(self):   self.started = False
        def close(self):  self.closed  = True

    sd.InputStream  = FakeStream
    sd.OutputStream = FakeStream
    return sd


# Inject the stub before importing our module
if "sounddevice" not in sys.modules:
    sys.modules["sounddevice"] = _make_sd_stub()

# Now it is safe to import
from audio_passthrough import (  # noqa: E402
    SoundCardAudioOutput,
    AudioPassthrough,
    AudioTxCapture,
)
from digi_input import SoundCardAudioSource  # noqa: E402


# ===========================================================================
# SoundCardAudioOutput tests
# ===========================================================================

class TestSoundCardAudioOutput(unittest.TestCase):

    def test_start_opens_output_stream(self):
        out = SoundCardAudioOutput(fs=48_000, block_size=512, device=0)
        out.start()
        self.assertIsNotNone(out._stream)
        self.assertTrue(out._stream.started)
        out.stop()

    def test_stop_closes_output_stream(self):
        out = SoundCardAudioOutput(fs=48_000, block_size=512, device=0)
        out.start()
        out.stop()
        self.assertFalse(out._running)
        self.assertIsNone(out._stream)

    def test_start_is_idempotent(self):
        """Calling start() twice should not open a second stream."""
        out = SoundCardAudioOutput(device=0)
        out.start()
        stream_id = id(out._stream)
        out.start()
        self.assertEqual(stream_id, id(out._stream))
        out.stop()

    def test_play_enqueues_samples(self):
        out = SoundCardAudioOutput(device=0)
        out.start()
        samples = np.zeros(100, dtype=np.float32)
        out.play(samples)
        self.assertEqual(out._q.qsize(), 1)
        out.stop()

    def test_play_drops_when_full(self):
        """play() must be non-blocking even when the queue is saturated."""
        out = SoundCardAudioOutput(device=0, max_queue_blocks=2)
        out.start()
        for _ in range(5):
            out.play(np.zeros(100, dtype=np.float32))
        # Queue size should not exceed maxsize
        self.assertLessEqual(out._q.qsize(), 2)
        out.stop()

    def test_callback_drains_queue_and_writes_output(self):
        """The output callback must copy queued samples into outdata."""
        out = SoundCardAudioOutput(fs=48_000, block_size=4, device=0)
        out.start()

        samples = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        out.play(samples)

        outdata = np.zeros((4, 1), dtype=np.float32)
        out._audio_callback(outdata, 4, None, None)

        np.testing.assert_array_almost_equal(outdata[:, 0], samples)
        out.stop()

    def test_callback_writes_silence_on_empty_queue(self):
        """The output callback must write silence when no data is queued."""
        out = SoundCardAudioOutput(fs=48_000, block_size=4, device=0)
        out.start()

        outdata = np.ones((4, 1), dtype=np.float32)
        out._audio_callback(outdata, 4, None, None)

        np.testing.assert_array_equal(outdata, 0.0)
        out.stop()

    def test_callback_pads_short_block_with_silence(self):
        """If the queued block is shorter than frames, the remainder is silence."""
        out = SoundCardAudioOutput(fs=48_000, block_size=8, device=0)
        out.start()

        samples = np.array([0.5, 0.6], dtype=np.float32)   # only 2 samples
        out.play(samples)

        outdata = np.ones((8, 1), dtype=np.float32)
        out._audio_callback(outdata, 8, None, None)

        # First 2 samples = data, rest = 0
        np.testing.assert_array_almost_equal(outdata[:2, 0], [0.5, 0.6])
        np.testing.assert_array_equal(outdata[2:], 0.0)
        out.stop()

    def test_missing_sounddevice_raises_runtime_error(self):
        """If sounddevice is not importable, start() must raise RuntimeError."""
        out = SoundCardAudioOutput(device=0)
        with patch.dict(sys.modules, {"sounddevice": None}):
            # remove the stub so the import inside start() fails
            orig = sys.modules.pop("sounddevice", None)
            try:
                with self.assertRaises(RuntimeError):
                    out.start()
            finally:
                if orig is not None:
                    sys.modules["sounddevice"] = orig


# ===========================================================================
# AudioPassthrough tests
# ===========================================================================

class TestAudioPassthrough(unittest.TestCase):

    def test_start_stop_lifecycle(self):
        pt = AudioPassthrough(input_device=0, output_device=1)
        pt.start()
        self.assertTrue(pt._running)
        self.assertIsNotNone(pt._src)
        pt.stop()
        self.assertFalse(pt._running)
        self.assertIsNone(pt._src)

    def test_start_is_idempotent(self):
        pt = AudioPassthrough(input_device=0, output_device=1)
        pt.start()
        src_id = id(pt._src)
        pt.start()                # second call should be a no-op
        self.assertEqual(src_id, id(pt._src))
        pt.stop()

    def test_rms_callback_is_invoked(self):
        """The rms_callback must be called each time a block is processed."""
        rms_values = []
        pt = AudioPassthrough(
            input_device=0, output_device=1,
            rms_callback=rms_values.append,
        )
        pt.start()

        # Inject one audio block directly into the internal queue
        samples = np.full(480, 0.5, dtype=np.float32)
        pt._in_q.put(samples)

        # Give the worker thread a moment to drain the queue
        for _ in range(20):
            if rms_values:
                break
            time.sleep(0.05)

        pt.stop()
        self.assertTrue(len(rms_values) >= 1)
        expected_rms = float(np.sqrt(np.mean(samples * samples)))
        self.assertAlmostEqual(rms_values[0], expected_rms, places=4)

    def test_audio_is_routed_to_output(self):
        """Samples fed into the internal queue must reach the output."""
        pt = AudioPassthrough(input_device=0, output_device=1)
        pt.start()

        samples = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        pt._in_q.put(samples)

        # Wait for worker thread to process
        for _ in range(20):
            if pt._out._q.qsize() > 0:
                break
            time.sleep(0.05)

        pt.stop()
        # The output queue should have received the samples
        result = pt._out._q.get_nowait()
        np.testing.assert_array_almost_equal(result, samples)

    def test_input_callback_feeds_internal_queue(self):
        """The sounddevice input callback must enqueue audio data."""
        pt = AudioPassthrough(input_device=0, output_device=1)
        pt.start()

        indata = np.array([[0.1], [0.2], [0.3]], dtype=np.float32)
        pt._in_callback(indata, 3, None, None)

        self.assertEqual(pt._in_q.qsize(), 1)
        received = pt._in_q.get_nowait()
        np.testing.assert_array_almost_equal(received, [0.1, 0.2, 0.3])
        pt.stop()

    def test_input_callback_drops_when_queue_full(self):
        """The input callback must not block when the queue is full."""
        pt = AudioPassthrough(input_device=0, output_device=1)
        # Fill the queue to capacity
        pt._in_q = queue.Queue(maxsize=2)
        pt.start()

        indata = np.zeros((4, 1), dtype=np.float32)
        for _ in range(5):
            pt._in_callback(indata, 4, None, None)  # must not raise

        self.assertLessEqual(pt._in_q.qsize(), 2)
        pt.stop()

    def test_stop_while_not_started_is_safe(self):
        """stop() on a never-started instance must not raise."""
        pt = AudioPassthrough(input_device=0, output_device=1)
        pt.stop()  # should not raise

    def test_missing_sounddevice_raises_runtime_error(self):
        pt = AudioPassthrough(input_device=0, output_device=1)
        with patch.dict(sys.modules, {"sounddevice": None}):
            orig = sys.modules.pop("sounddevice", None)
            try:
                with self.assertRaises(RuntimeError):
                    pt.start()
            finally:
                if orig is not None:
                    sys.modules["sounddevice"] = orig


# ===========================================================================
# AudioTxCapture tests
# ===========================================================================

class TestAudioTxCapture(unittest.TestCase):

    def test_start_stop_lifecycle(self):
        tx = AudioTxCapture(mic_device=2, radio_out_device=3)
        tx.start()
        self.assertTrue(tx._running)
        self.assertIsNotNone(tx._src)
        tx.stop()
        self.assertFalse(tx._running)
        self.assertIsNone(tx._src)

    def test_start_is_idempotent(self):
        tx = AudioTxCapture(mic_device=2, radio_out_device=3)
        tx.start()
        src_id = id(tx._src)
        tx.start()
        self.assertEqual(src_id, id(tx._src))
        tx.stop()

    def test_mic_audio_routed_to_radio_output(self):
        """Samples from mic must be forwarded to the radio output queue."""
        tx = AudioTxCapture(mic_device=2, radio_out_device=3)
        tx.start()

        samples = np.array([0.3, 0.4, 0.5], dtype=np.float32)
        tx._in_q.put(samples)

        for _ in range(20):
            if tx._out._q.qsize() > 0:
                break
            time.sleep(0.05)

        tx.stop()
        result = tx._out._q.get_nowait()
        np.testing.assert_array_almost_equal(result, samples)

    def test_rms_callback_invoked_during_tx(self):
        """rms_callback should fire for each processed block."""
        rms_vals = []
        tx = AudioTxCapture(
            mic_device=2, radio_out_device=3,
            rms_callback=rms_vals.append,
        )
        tx.start()

        samples = np.full(480, 0.8, dtype=np.float32)
        tx._in_q.put(samples)

        for _ in range(20):
            if rms_vals:
                break
            time.sleep(0.05)

        tx.stop()
        self.assertTrue(len(rms_vals) >= 1)
        self.assertAlmostEqual(rms_vals[0], 0.8, places=4)

    def test_input_callback_enqueues_data(self):
        tx = AudioTxCapture(mic_device=2, radio_out_device=3)
        tx.start()

        indata = np.array([[0.2], [0.4], [0.6]], dtype=np.float32)
        tx._in_callback(indata, 3, None, None)

        self.assertEqual(tx._in_q.qsize(), 1)
        received = tx._in_q.get_nowait()
        np.testing.assert_array_almost_equal(received, [0.2, 0.4, 0.6])
        tx.stop()

    def test_input_callback_drops_when_queue_full(self):
        tx = AudioTxCapture(mic_device=2, radio_out_device=3)
        tx._in_q = queue.Queue(maxsize=2)
        tx.start()

        indata = np.zeros((4, 1), dtype=np.float32)
        for _ in range(5):
            tx._in_callback(indata, 4, None, None)

        self.assertLessEqual(tx._in_q.qsize(), 2)
        tx.stop()

    def test_stop_while_not_started_is_safe(self):
        tx = AudioTxCapture(mic_device=2, radio_out_device=3)
        tx.stop()  # should not raise

    def test_none_device_indices_accepted(self):
        """None device indices should use the system default device."""
        tx = AudioTxCapture(mic_device=None, radio_out_device=None)
        tx.start()
        self.assertTrue(tx._running)
        tx.stop()

    def test_missing_sounddevice_raises_runtime_error(self):
        tx = AudioTxCapture(mic_device=2, radio_out_device=3)
        with patch.dict(sys.modules, {"sounddevice": None}):
            orig = sys.modules.pop("sounddevice", None)
            try:
                with self.assertRaises(RuntimeError):
                    tx.start()
            finally:
                if orig is not None:
                    sys.modules["sounddevice"] = orig



# ===========================================================================
# SoundCardAudioSource tests (digi_input.py)
# ===========================================================================
# digi_input imports sounddevice at call-time (inside start()), so the stub
# already injected above is sufficient.


class TestSoundCardAudioSourceStop(unittest.TestCase):
    """Tests for SoundCardAudioSource.stop() resource-management behaviour."""

    def _make_source(self):
        """Create a SoundCardAudioSource with a mocked internal stream."""
        src = SoundCardAudioSource(fs=48_000, block_size=4_800, device=0)
        mock_stream = MagicMock()
        src._stream = mock_stream
        src._running = True
        return src, mock_stream

    def test_stop_clears_stream_on_success(self):
        """After a clean stop(), _stream must be None."""
        src, _ = self._make_source()
        src.stop()
        self.assertIsNone(src._stream)
        self.assertFalse(src._running)

    def test_stop_clears_stream_when_close_raises(self):
        """_stream must be set to None even if close() raises (resource-leak regression)."""
        src, mock_stream = self._make_source()
        mock_stream.close.side_effect = RuntimeError("simulated close failure")
        with self.assertRaises(RuntimeError):
            src.stop()
        # Despite the exception, the dangling reference must have been cleared.
        self.assertIsNone(src._stream)

    def test_stop_clears_stream_when_stop_raises(self):
        """_stream must be set to None even if stream.stop() raises."""
        src, mock_stream = self._make_source()
        mock_stream.stop.side_effect = RuntimeError("simulated stop failure")
        with self.assertRaises(RuntimeError):
            src.stop()
        self.assertIsNone(src._stream)

    def test_stop_is_idempotent(self):
        """Calling stop() when already stopped must not raise."""
        src = SoundCardAudioSource(fs=48_000, block_size=4_800, device=0)
        src.stop()  # _stream is None — should be a no-op
        src.stop()  # second call should also be safe


if __name__ == "__main__":
    unittest.main(verbosity=2)
