"""
test_ft8_tx.py — pytest suite for ft8_tx.py (Milestone 4 TX orchestration).

Test groups
-----------
TestTxJob                      — construction and repr
TestValidateOperator           — validate_operator helper
TestFt8TxCoordinator_States    — state transitions from IDLE through COMPLETE
TestFt8TxCoordinator_Cancel    — cancel() before slot start
TestFt8TxCoordinator_ArmGuards — validation at arm() time (CAT, device, etc.)
TestFt8TxCoordinator_PTTOrder  — PTT key/unkey sequencing and guaranteed unkey
TestFt8TxCoordinator_MissedSlot — missed-slot detection and ERROR transition
TestFt8TxCoordinator_Reset     — reset() from terminal states
TestFt8TxCoordinator_Callbacks — on_state_change is invoked correctly
TestFt8TxCoordinator_AudioPlay — _play_audio skips gracefully without sounddevice
TestFt8TxCoordinator_ExceptionSafety — exception in audio still unkeys PTT
"""
from __future__ import annotations

import threading
import time
import types
import unittest
import unittest.mock as mock
from datetime import datetime, timezone

# Ensure ft8_tx module is importable from the same directory
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

from ft8_tx import (
    Ft8TxCoordinator,
    TxJob,
    TxState,
    validate_operator,
    PRE_KEY_S,
    POST_KEY_S,
    MISSED_SLOT_THRESHOLD_S,
    USB_AUDIO_SWITCH_DELAY_S,
    TX_OUTPUT_SAMPLE_RATE,
    TX_OUTPUT_DTYPE,
    TX_OUTPUT_BLOCKSIZE,
    _to_int16,
    _stream_play,
    _log_audio_diagnostics,
)
from ft8_ntp import Ft8SlotTimer, NtpTimeSync


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_radio(connected: bool = True):
    """Return a mock radio with ptt_on / ptt_off / is_connected."""
    r = mock.MagicMock()
    r.is_connected.return_value = connected
    return r


def _make_timer(seconds_to_next: float = 0.0):
    """Return a mock Ft8SlotTimer that always reports a fixed wait."""
    t = mock.MagicMock(spec=Ft8SlotTimer)
    t.seconds_to_next_slot.return_value = seconds_to_next
    t.next_slot_utc.return_value = datetime.now(tz=timezone.utc)
    return t


def _coord_with_fast_slot(
    radio=None,
    slot_wait: float = 0.01,
    post_key: float = 0.0,
    audio_device: int | None = 0,
) -> Ft8TxCoordinator:
    """
    Build a coordinator that fires almost immediately (slot_wait seconds),
    with audio playback replaced by a no-op.
    """
    if radio is None:
        radio = _make_radio()
    timer = _make_timer(seconds_to_next=slot_wait + PRE_KEY_S + 0.001)
    coord = Ft8TxCoordinator(
        radio=radio,
        slot_timer=timer,
        pre_key_s=PRE_KEY_S,
        post_key_s=post_key,
    )
    # Replace _play_audio with a no-op so tests don't need sounddevice
    coord._play_audio = mock.MagicMock()
    return coord


def _arm_and_wait(coord: Ft8TxCoordinator, msg: str = "CQ W4ABC EN52",
                  device: int | None = 0, timeout: float = 3.0) -> TxState:
    """Arm coordinator and block until it reaches a terminal state."""
    job = TxJob(msg, audio_device=device)
    coord.arm(job)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = coord.state
        if state in (TxState.COMPLETE, TxState.ERROR, TxState.CANCELED):
            return state
        time.sleep(0.02)
    return coord.state


# ═══════════════════════════════════════════════════════════════════════════════
# § 1  TxJob
# ═══════════════════════════════════════════════════════════════════════════════

class TestTxJob(unittest.TestCase):
    def test_defaults(self):
        job = TxJob("CQ W4ABC EN52")
        self.assertEqual(job.message, "CQ W4ABC EN52")
        self.assertAlmostEqual(job.f0_hz, 1500.0)
        self.assertIsNone(job.audio_device)
        self.assertAlmostEqual(job.amplitude, 0.5)

    def test_custom_values(self):
        job = TxJob("CQ K9XYZ EN52", f0_hz=700.0, audio_device=2, amplitude=0.3)
        self.assertAlmostEqual(job.f0_hz, 700.0)
        self.assertEqual(job.audio_device, 2)
        self.assertAlmostEqual(job.amplitude, 0.3)

    def test_message_stripped(self):
        job = TxJob("  CQ W4ABC EN52  ")
        self.assertEqual(job.message, "CQ W4ABC EN52")

    def test_repr(self):
        job = TxJob("CQ W4ABC EN52", audio_device=1)
        self.assertIn("CQ W4ABC EN52", repr(job))


# ═══════════════════════════════════════════════════════════════════════════════
# § 2  validate_operator
# ═══════════════════════════════════════════════════════════════════════════════

class TestValidateOperator(unittest.TestCase):
    def test_valid(self):
        ok, reason = validate_operator("W4ABC", "EN52")
        self.assertTrue(ok)
        self.assertEqual(reason, "")

    def test_invalid_callsign(self):
        ok, reason = validate_operator("BADCALL!!!", "EN52")
        self.assertFalse(ok)
        self.assertIn("callsign", reason.lower())

    def test_invalid_grid(self):
        ok, reason = validate_operator("W4ABC", "XX99XX")  # 6-char, not 4
        self.assertFalse(ok)
        self.assertIn("grid", reason.lower())

    def test_empty_callsign(self):
        ok, reason = validate_operator("", "EN52")
        self.assertFalse(ok)

    def test_empty_grid(self):
        ok, reason = validate_operator("W4ABC", "")
        self.assertFalse(ok)

    def test_lowercase_accepted(self):
        ok, reason = validate_operator("w4abc", "en52")
        self.assertTrue(ok)

    def test_valid_grid_variants(self):
        for grid in ("IO91", "QF56", "AA00", "RR99"):
            ok, _ = validate_operator("W4ABC", grid)
            self.assertTrue(ok, f"Grid {grid!r} should be valid")


# ═══════════════════════════════════════════════════════════════════════════════
# § 3  State transitions — happy path
# ═══════════════════════════════════════════════════════════════════════════════

class TestFt8TxCoordinatorStates(unittest.TestCase):
    """Verify the IDLE → ARMED → TX_PREP → TX_ACTIVE → COMPLETE path."""

    def test_initial_state_is_idle(self):
        coord = Ft8TxCoordinator()
        self.assertEqual(coord.state, TxState.IDLE)

    def test_arm_transitions_to_armed(self):
        coord = _coord_with_fast_slot()
        # Pause after arm() returns to check state before it fires
        # Use a large slot_wait so it doesn't fire during the assertion
        timer = _make_timer(seconds_to_next=10.0)
        coord._timer = timer
        job = TxJob("CQ W4ABC EN52", audio_device=0)
        coord.arm(job)
        self.assertEqual(coord.state, TxState.ARMED)
        coord.cancel()  # clean up

    def test_full_sequence_reaches_complete(self):
        coord = _coord_with_fast_slot(slot_wait=0.02)
        final = _arm_and_wait(coord)
        self.assertEqual(final, TxState.COMPLETE)

    def test_states_visited_during_sequence(self):
        """Record all states seen via callback."""
        visited = []
        coord = _coord_with_fast_slot(slot_wait=0.02)
        coord.on_state_change = lambda s, m: visited.append(s)
        _arm_and_wait(coord)
        self.assertIn(TxState.ARMED, visited)
        self.assertIn(TxState.TX_ACTIVE, visited)
        self.assertIn(TxState.COMPLETE, visited)

    def test_complete_has_no_error(self):
        coord = _coord_with_fast_slot(slot_wait=0.02)
        _arm_and_wait(coord)
        self.assertIsNone(coord.last_error)

    def test_current_job_available_while_armed(self):
        coord = _coord_with_fast_slot()
        timer = _make_timer(seconds_to_next=10.0)
        coord._timer = timer
        job = TxJob("CQ W4ABC EN52", audio_device=0)
        coord.arm(job)
        self.assertIsNotNone(coord.current_job)
        self.assertEqual(coord.current_job.message, "CQ W4ABC EN52")
        coord.cancel()


# ═══════════════════════════════════════════════════════════════════════════════
# § 4  cancel()
# ═══════════════════════════════════════════════════════════════════════════════

class TestFt8TxCoordinatorCancel(unittest.TestCase):
    def test_cancel_while_armed_transitions_canceled(self):
        coord = _coord_with_fast_slot()
        timer = _make_timer(seconds_to_next=10.0)  # far future slot
        coord._timer = timer
        job = TxJob("CQ W4ABC EN52", audio_device=0)
        coord.arm(job)
        accepted = coord.cancel()
        self.assertTrue(accepted)
        # Wait for worker thread to detect cancel
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            if coord.state == TxState.CANCELED:
                break
            time.sleep(0.02)
        self.assertEqual(coord.state, TxState.CANCELED)

    def test_cancel_returns_false_when_idle(self):
        coord = Ft8TxCoordinator()
        self.assertFalse(coord.cancel())

    def test_cancel_returns_false_when_complete(self):
        coord = _coord_with_fast_slot(slot_wait=0.02)
        _arm_and_wait(coord)
        coord.reset()
        self.assertFalse(coord.cancel())

    def test_no_ptt_after_cancel(self):
        radio = _make_radio()
        coord = _coord_with_fast_slot(radio=radio)
        timer = _make_timer(seconds_to_next=10.0)
        coord._timer = timer
        job = TxJob("CQ W4ABC EN52", audio_device=0)
        coord.arm(job)
        coord.cancel()
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            if coord.state == TxState.CANCELED:
                break
            time.sleep(0.02)
        radio.ptt_on.assert_not_called()
        radio.ptt_off.assert_not_called()


# ═══════════════════════════════════════════════════════════════════════════════
# § 5  arm() guardrails
# ═══════════════════════════════════════════════════════════════════════════════

class TestFt8TxCoordinatorArmGuards(unittest.TestCase):
    def test_raises_when_already_armed(self):
        coord = _coord_with_fast_slot()
        timer = _make_timer(seconds_to_next=10.0)
        coord._timer = timer
        job = TxJob("CQ W4ABC EN52", audio_device=0)
        coord.arm(job)
        with self.assertRaises(RuntimeError):
            coord.arm(TxJob("CQ W4ABC EN52", audio_device=0))
        coord.cancel()

    def test_raises_when_cat_disconnected(self):
        radio = _make_radio(connected=False)
        coord = Ft8TxCoordinator(radio=radio)
        coord._play_audio = mock.MagicMock()
        with self.assertRaises(RuntimeError) as ctx:
            coord.arm(TxJob("CQ W4ABC EN52", audio_device=0))
        self.assertIn("CAT", str(ctx.exception))

    def test_raises_when_message_empty(self):
        coord = _coord_with_fast_slot()
        with self.assertRaises(RuntimeError) as ctx:
            coord.arm(TxJob("", audio_device=0))
        self.assertIn("empty", str(ctx.exception))

    def test_raises_when_audio_device_negative(self):
        coord = _coord_with_fast_slot()
        with self.assertRaises(RuntimeError) as ctx:
            coord.arm(TxJob("CQ W4ABC EN52", audio_device=-1))
        self.assertIn("audio", str(ctx.exception).lower())

    def test_allows_none_audio_device(self):
        """audio_device=None means 'use default' — arm() should accept it."""
        coord = _coord_with_fast_slot(audio_device=None)
        timer = _make_timer(seconds_to_next=10.0)
        coord._timer = timer
        job = TxJob("CQ W4ABC EN52", audio_device=None)
        coord.arm(job)   # should not raise
        self.assertEqual(coord.state, TxState.ARMED)
        coord.cancel()

    def test_no_radio_is_allowed_for_testing(self):
        """radio=None should not block arm() — useful for unit tests."""
        coord = Ft8TxCoordinator(radio=None)
        coord._play_audio = mock.MagicMock()
        timer = _make_timer(seconds_to_next=0.02 + PRE_KEY_S + 0.001)
        coord._timer = timer
        final = _arm_and_wait(coord, device=0)
        self.assertEqual(final, TxState.COMPLETE)


# ═══════════════════════════════════════════════════════════════════════════════
# § 6  PTT key/unkey sequencing
# ═══════════════════════════════════════════════════════════════════════════════

class TestFt8TxCoordinatorPTTOrder(unittest.TestCase):
    def test_ptt_on_before_audio_and_off_after(self):
        """ptt_on must be called before _play_audio, ptt_off after."""
        radio = _make_radio()
        call_order = []
        radio.ptt_on.side_effect  = lambda: call_order.append("ptt_on")
        radio.ptt_off.side_effect = lambda: call_order.append("ptt_off")

        coord = _coord_with_fast_slot(radio=radio, slot_wait=0.02)
        coord._play_audio.side_effect = lambda *a, **kw: call_order.append("audio")
        _arm_and_wait(coord)

        self.assertEqual(call_order, ["ptt_on", "audio", "ptt_off"])

    def test_ptt_off_called_exactly_once(self):
        radio = _make_radio()
        coord = _coord_with_fast_slot(radio=radio, slot_wait=0.02)
        _arm_and_wait(coord)
        radio.ptt_off.assert_called_once()

    def test_ptt_on_called_exactly_once(self):
        radio = _make_radio()
        coord = _coord_with_fast_slot(radio=radio, slot_wait=0.02)
        _arm_and_wait(coord)
        radio.ptt_on.assert_called_once()


# ═══════════════════════════════════════════════════════════════════════════════
# § 7  Missed slot
# ═══════════════════════════════════════════════════════════════════════════════

class TestFt8TxCoordinatorMissedSlot(unittest.TestCase):
    """
    Simulate a missed-slot condition by making the timer report a slot boundary
    that was already passed by more than MISSED_SLOT_THRESHOLD_S.

    seconds_to_next_slot() returns a value close to FT8_SLOT_DURATION_S (15 s)
    when we are just past a boundary, so we can simulate this by returning
    15 - (MISSED_SLOT_THRESHOLD_S + 0.1) from seconds_to_next_slot() at the
    point where the check is performed (i.e. after the pre-key wait).
    """

    def test_missed_slot_transitions_error(self):
        radio = _make_radio()
        # The timer behaves normally for the initial wait calculation but then
        # reports that we are already 0.6 s past the boundary (> threshold).
        # We achieve this by making the timer return a very short initial wait
        # and then returning a value that triggers the overrun check.
        call_count = [0]
        def _seconds_side_effect():
            call_count[0] += 1
            if call_count[0] == 1:
                # Initial call: "only PRE_KEY_S until next slot" → wait ≈ 0
                return PRE_KEY_S + 0.001
            else:
                # Second call (overrun check): we're 0.6 s past the boundary
                return 15.0 - (MISSED_SLOT_THRESHOLD_S + 0.1)

        timer = mock.MagicMock(spec=Ft8SlotTimer)
        timer.seconds_to_next_slot.side_effect = _seconds_side_effect

        coord = Ft8TxCoordinator(radio=radio, slot_timer=timer)
        coord._play_audio = mock.MagicMock()

        final = _arm_and_wait(coord, device=0, timeout=3.0)
        self.assertEqual(final, TxState.ERROR)
        self.assertIsNotNone(coord.last_error)
        self.assertIn("Missed", coord.last_error)

    def test_missed_slot_no_ptt(self):
        """PTT must NOT be keyed if the slot was missed."""
        radio = _make_radio()
        call_count = [0]
        def _se():
            call_count[0] += 1
            if call_count[0] == 1:
                return PRE_KEY_S + 0.001
            return 15.0 - (MISSED_SLOT_THRESHOLD_S + 0.1)
        timer = mock.MagicMock(spec=Ft8SlotTimer)
        timer.seconds_to_next_slot.side_effect = _se
        coord = Ft8TxCoordinator(radio=radio, slot_timer=timer)
        coord._play_audio = mock.MagicMock()
        _arm_and_wait(coord, device=0, timeout=3.0)
        radio.ptt_on.assert_not_called()


# ═══════════════════════════════════════════════════════════════════════════════
# § 8  reset()
# ═══════════════════════════════════════════════════════════════════════════════

class TestFt8TxCoordinatorReset(unittest.TestCase):
    def test_reset_from_complete(self):
        coord = _coord_with_fast_slot(slot_wait=0.02)
        _arm_and_wait(coord)
        self.assertEqual(coord.state, TxState.COMPLETE)
        coord.reset()
        self.assertEqual(coord.state, TxState.IDLE)
        self.assertIsNone(coord.current_job)

    def test_reset_from_canceled(self):
        coord = _coord_with_fast_slot()
        timer = _make_timer(seconds_to_next=10.0)
        coord._timer = timer
        coord.arm(TxJob("CQ W4ABC EN52", audio_device=0))
        coord.cancel()
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and coord.state != TxState.CANCELED:
            time.sleep(0.02)
        coord.reset()
        self.assertEqual(coord.state, TxState.IDLE)

    def test_reset_from_error(self):
        coord = _coord_with_fast_slot(slot_wait=0.02)
        # Make play_audio raise to force ERROR
        coord._play_audio.side_effect = RuntimeError("device error")
        _arm_and_wait(coord)
        self.assertEqual(coord.state, TxState.ERROR)
        coord.reset()
        self.assertEqual(coord.state, TxState.IDLE)

    def test_reset_while_idle_is_noop(self):
        coord = Ft8TxCoordinator()
        coord.reset()  # should not raise
        self.assertEqual(coord.state, TxState.IDLE)


# ═══════════════════════════════════════════════════════════════════════════════
# § 9  on_state_change callbacks
# ═══════════════════════════════════════════════════════════════════════════════

class TestFt8TxCoordinatorCallbacks(unittest.TestCase):
    def test_callback_receives_all_terminal_states(self):
        states = []
        coord = _coord_with_fast_slot(slot_wait=0.02)
        coord.on_state_change = lambda s, m: states.append(s)
        _arm_and_wait(coord)
        self.assertIn(TxState.COMPLETE, states)

    def test_callback_exception_does_not_crash_coordinator(self):
        def _bad_cb(s, m):
            raise ValueError("callback boom")
        coord = _coord_with_fast_slot(slot_wait=0.02)
        coord.on_state_change = _bad_cb
        # Should still reach COMPLETE without raising
        final = _arm_and_wait(coord)
        self.assertEqual(final, TxState.COMPLETE)

    def test_no_callback_is_fine(self):
        coord = _coord_with_fast_slot(slot_wait=0.02)
        coord.on_state_change = None
        final = _arm_and_wait(coord)
        self.assertEqual(final, TxState.COMPLETE)


# ═══════════════════════════════════════════════════════════════════════════════
# § 10  _play_audio — no sounddevice
# ═══════════════════════════════════════════════════════════════════════════════

class TestFt8TxCoordinatorAudioPlay(unittest.TestCase):
    def test_play_audio_skips_without_sounddevice(self):
        """If sounddevice is not importable, _play_audio should log and return."""
        import numpy as np
        coord = Ft8TxCoordinator(radio=None)
        # Remove sounddevice from sys.modules and block import
        with mock.patch.dict("sys.modules", {"sounddevice": None}):
            # Should not raise
            coord._play_audio(np.zeros(10, dtype=np.float32), device=None)

    def _make_fake_sd(self, device_samplerate: int):
        """Return a minimal sounddevice stub with the given device sample rate."""
        import numpy as np

        fake_sd = mock.MagicMock()
        fake_sd.PortAudioError = Exception
        fake_sd.query_devices.return_value = {"default_samplerate": device_samplerate}
        fake_sd.default.device = (0, 1)
        return fake_sd

    def _make_stream_recorder(self):
        """Return a list and a _stream_play replacement that records calls."""
        stream_calls = []

        def _fake_stream_play(sd_mod, audio, fs, device, *, extra_settings=None):
            stream_calls.append({"audio": audio, "fs": fs, "device": device,
                                  "extra_settings": extra_settings})

        return stream_calls, _fake_stream_play

    def test_play_audio_always_outputs_at_tx_output_sample_rate(self):
        """
        _play_audio always resamples to TX_OUTPUT_SAMPLE_RATE (48 000 Hz) and
        calls _stream_play at that rate regardless of the device's reported
        native sample rate.
        """
        import numpy as np

        coord = Ft8TxCoordinator(radio=None)
        audio = np.zeros(100, dtype=np.float32)
        fake_sd = self._make_fake_sd(device_samplerate=48_000)
        stream_calls, fake_stream = self._make_stream_recorder()

        with mock.patch("ft8_tx._stream_play", side_effect=fake_stream), \
             mock.patch.dict("sys.modules", {"sounddevice": fake_sd}):
            coord._play_audio(audio, device=1)

        self.assertEqual(len(stream_calls), 1, "_stream_play should be called exactly once")
        self.assertEqual(stream_calls[0]["fs"], TX_OUTPUT_SAMPLE_RATE)
        # 48 000 / 12 000 = ×4 upsampling → output should be ≈ 4 × input length
        self.assertEqual(len(stream_calls[0]["audio"]), 400)
        # Audio must be float32 (same as voice-mode SoundCardAudioOutput)
        self.assertEqual(stream_calls[0]["audio"].dtype, np.float32)

    def test_play_audio_resamples_even_when_device_reports_ft8_native_rate(self):
        """
        Even when the device's default_samplerate equals FT8_FS (12 000 Hz),
        _play_audio resamples to TX_OUTPUT_SAMPLE_RATE (48 000 Hz) and
        keeps audio as float32.
        """
        import numpy as np
        from ft8_encode import FT8_FS

        coord = Ft8TxCoordinator(radio=None)
        audio = np.ones(200, dtype=np.float32) * 0.5
        fake_sd = self._make_fake_sd(device_samplerate=FT8_FS)
        stream_calls, fake_stream = self._make_stream_recorder()

        with mock.patch("ft8_tx._stream_play", side_effect=fake_stream), \
             mock.patch.dict("sys.modules", {"sounddevice": fake_sd}):
            coord._play_audio(audio, device=0)

        self.assertEqual(len(stream_calls), 1)
        self.assertEqual(stream_calls[0]["fs"], TX_OUTPUT_SAMPLE_RATE)
        self.assertEqual(stream_calls[0]["audio"].dtype, np.float32)

    def test_play_audio_resample_44100(self):
        """
        Verify that _play_audio always targets TX_OUTPUT_SAMPLE_RATE (48 000 Hz)
        regardless of the device's reported native rate (e.g. 44 100 Hz).
        """
        import numpy as np

        coord = Ft8TxCoordinator(radio=None)
        audio = np.zeros(120, dtype=np.float32)
        fake_sd = self._make_fake_sd(device_samplerate=44_100)
        stream_calls, fake_stream = self._make_stream_recorder()

        with mock.patch("ft8_tx._stream_play", side_effect=fake_stream), \
             mock.patch.dict("sys.modules", {"sounddevice": fake_sd}):
            coord._play_audio(audio, device=2)

        self.assertEqual(len(stream_calls), 1)
        # Always 48 000 Hz regardless of device native rate
        self.assertEqual(stream_calls[0]["fs"], TX_OUTPUT_SAMPLE_RATE)
        self.assertEqual(stream_calls[0]["audio"].dtype, np.float32)

    def test_play_audio_uses_default_output_when_device_none(self):
        """
        When device is None, _play_audio should still call _stream_play at
        TX_OUTPUT_SAMPLE_RATE with float32 audio without querying the device
        for its native sample rate.
        """
        import numpy as np

        coord = Ft8TxCoordinator(radio=None)
        audio = np.zeros(60, dtype=np.float32)
        fake_sd = self._make_fake_sd(device_samplerate=48_000)
        stream_calls, fake_stream = self._make_stream_recorder()

        with mock.patch("ft8_tx.platform.system", return_value="Linux"), \
             mock.patch("ft8_tx._stream_play", side_effect=fake_stream), \
             mock.patch.dict("sys.modules", {"sounddevice": fake_sd}):
            coord._play_audio(audio, device=None)

        # _stream_play must be called with the fixed TX_OUTPUT_SAMPLE_RATE,
        # regardless of what query_devices returns — the rate is not dynamically
        # obtained from the device (only diagnostic logging calls query_devices).
        self.assertEqual(stream_calls[0]["fs"], TX_OUTPUT_SAMPLE_RATE)
        self.assertEqual(stream_calls[0]["audio"].dtype, np.float32)

    def test_play_audio_falls_back_on_query_error(self):
        """
        _play_audio no longer queries the device native rate, so a
        query_devices failure has no effect — _stream_play is called at
        TX_OUTPUT_SAMPLE_RATE (48 000 Hz) with float32 audio.
        """
        import numpy as np

        coord = Ft8TxCoordinator(radio=None)
        audio = np.zeros(50, dtype=np.float32)

        fake_sd = mock.MagicMock()
        fake_sd.PortAudioError = Exception
        fake_sd.query_devices.side_effect = RuntimeError("device query failed")
        fake_sd.default.device = (0, 1)
        stream_calls, fake_stream = self._make_stream_recorder()

        with mock.patch("ft8_tx._stream_play", side_effect=fake_stream), \
             mock.patch.dict("sys.modules", {"sounddevice": fake_sd}):
            coord._play_audio(audio, device=None)

        self.assertEqual(len(stream_calls), 1)
        self.assertEqual(stream_calls[0]["fs"], TX_OUTPUT_SAMPLE_RATE)
        self.assertEqual(stream_calls[0]["audio"].dtype, np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# § 10b  Fixed TX audio format (16-bit, 48 000 Hz)
# ═══════════════════════════════════════════════════════════════════════════════

class TestTxAudioFixedFormat(unittest.TestCase):
    """
    Verify that _play_audio always outputs float32 / 48 000 Hz audio regardless
    of the input signal or device configuration, matching the voice-mode
    SoundCardAudioOutput / AudioTxCapture technique.
    """

    def _make_coord(self):
        return Ft8TxCoordinator(radio=None)

    def test_output_dtype_is_float32(self):
        """Audio passed to _stream_play must be a numpy float32 array (same as voice mode)."""
        import numpy as np

        coord = self._make_coord()
        audio = np.ones(100, dtype=np.float32) * 0.5

        fake_sd = mock.MagicMock()
        fake_sd.PortAudioError = Exception
        fake_sd.default.device = (0, 0)

        captured = []

        def _fake_stream_play(sd_mod, audio, fs, device, *, extra_settings=None):
            captured.append((audio, fs, device))

        with mock.patch("ft8_tx._stream_play", side_effect=_fake_stream_play), \
             mock.patch.dict("sys.modules", {"sounddevice": fake_sd}):
            coord._play_audio(audio, device=0)

        self.assertEqual(len(captured), 1)
        audio_out, sr, _ = captured[0]
        self.assertEqual(audio_out.dtype, np.float32, "Output array must be float32")
        self.assertEqual(sr, TX_OUTPUT_SAMPLE_RATE, "Sample rate must be TX_OUTPUT_SAMPLE_RATE")

    def test_output_sample_rate_is_48000(self):
        """Sample rate passed to _stream_play must always be TX_OUTPUT_SAMPLE_RATE (48 000)."""
        import numpy as np

        coord = self._make_coord()
        audio = np.zeros(50, dtype=np.float32)

        fake_sd = mock.MagicMock()
        fake_sd.PortAudioError = Exception
        fake_sd.default.device = (0, 0)

        captured_sr = []

        def _fake_stream_play(sd_mod, audio, fs, device, *, extra_settings=None):
            captured_sr.append(fs)

        with mock.patch("ft8_tx._stream_play", side_effect=_fake_stream_play), \
             mock.patch.dict("sys.modules", {"sounddevice": fake_sd}):
            coord._play_audio(audio, device=0)

        self.assertEqual(captured_sr[0], TX_OUTPUT_SAMPLE_RATE)

    def test_int16_amplitude_within_range(self):
        """int16 values must stay within [-32767, 32767] for a unit-amplitude signal."""
        import numpy as np

        audio = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)
        result = _to_int16(audio)
        self.assertEqual(result.dtype, np.int16)
        self.assertGreaterEqual(int(result.min()), -32767)
        self.assertLessEqual(int(result.max()), 32767)

    def test_int16_clips_above_unity(self):
        """Samples exceeding 1.0 in float must be clipped, not wrap around."""
        import numpy as np

        audio = np.array([1.5, -1.5, 2.0], dtype=np.float32)
        result = _to_int16(audio)
        # All values clipped to the [-32767, 32767] range; no wrap-around
        self.assertTrue(np.all(result >= -32767))
        self.assertTrue(np.all(result <= 32767))

    def test_device_native_rate_not_queried(self):
        """
        _play_audio must NOT use sd.query_devices to determine the output
        rate — the rate is fixed at TX_OUTPUT_SAMPLE_RATE.  Diagnostic logging
        may call query_devices for driver info, but the sample rate passed to
        _stream_play must always be TX_OUTPUT_SAMPLE_RATE regardless.
        """
        import numpy as np

        coord = self._make_coord()
        audio = np.zeros(60, dtype=np.float32)

        fake_sd = mock.MagicMock()
        fake_sd.PortAudioError = Exception
        fake_sd.default.device = (0, 0)

        stream_calls = []

        def fake_stream(sd_mod, aud, fs, device, *, extra_settings=None):
            stream_calls.append({"fs": fs, "audio": aud})

        with mock.patch("ft8_tx.platform.system", return_value="Linux"), \
             mock.patch("ft8_tx._stream_play", side_effect=fake_stream), \
             mock.patch.dict("sys.modules", {"sounddevice": fake_sd}):
            coord._play_audio(audio, device=1)

        self.assertEqual(len(stream_calls), 1)
        self.assertEqual(stream_calls[0]["fs"], TX_OUTPUT_SAMPLE_RATE,
                         "_stream_play must always receive TX_OUTPUT_SAMPLE_RATE")

    def test_play_audio_raises_if_resample_fails(self):
        """
        If _resample_audio returns the original FT8_FS rate instead of
        TX_OUTPUT_SAMPLE_RATE (e.g. scipy unavailable), _play_audio must raise
        RuntimeError immediately rather than passing 12 kHz audio to the driver.
        """
        import numpy as np
        from ft8_tx import TX_OUTPUT_SAMPLE_RATE

        coord = self._make_coord()
        audio = np.zeros(50, dtype=np.float32)

        fake_sd = mock.MagicMock()
        fake_sd.PortAudioError = Exception
        fake_sd.default.device = (0, 0)

        with mock.patch("ft8_tx._resample_audio", return_value=(audio, 12_000)), \
             mock.patch("ft8_tx._stream_play") as mock_stream, \
             mock.patch.dict("sys.modules", {"sounddevice": fake_sd}):
            with self.assertRaises(RuntimeError) as ctx:
                coord._play_audio(audio, device=0)

        self.assertIn(str(TX_OUTPUT_SAMPLE_RATE), str(ctx.exception))
        # _stream_play must NOT have been called
        mock_stream.assert_not_called()

class TestFt8TxCoordinatorExceptionSafety(unittest.TestCase):
    def test_ptt_unkeyed_if_audio_raises(self):
        """Even if _play_audio raises, PTT must be unkeyed."""
        radio = _make_radio()
        coord = _coord_with_fast_slot(radio=radio, slot_wait=0.02)
        coord._play_audio.side_effect = RuntimeError("soundcard gone")
        final = _arm_and_wait(coord)
        self.assertEqual(final, TxState.ERROR)
        radio.ptt_off.assert_called_once()

    def test_state_is_error_if_audio_raises(self):
        coord = _coord_with_fast_slot(slot_wait=0.02)
        coord._play_audio.side_effect = RuntimeError("boom")
        final = _arm_and_wait(coord)
        self.assertEqual(final, TxState.ERROR)
        self.assertIsNotNone(coord.last_error)

    def test_ptt_unkeyed_if_ptt_on_raises(self):
        """
        If ptt_on raises, ptt_off is still attempted in the finally path
        (though in this case ptt_keyed is False so ptt_off is skipped — the
        important thing is no exception propagates out of the thread).
        """
        radio = _make_radio()
        radio.ptt_on.side_effect = RuntimeError("serial error")
        coord = _coord_with_fast_slot(radio=radio, slot_wait=0.02)
        final = _arm_and_wait(coord)
        self.assertEqual(final, TxState.ERROR)

    def test_second_arm_works_after_reset_from_error(self):
        coord = _coord_with_fast_slot(slot_wait=0.02)
        coord._play_audio.side_effect = RuntimeError("first attempt fails")
        _arm_and_wait(coord)
        coord.reset()
        # Fix the mock and try again
        coord._play_audio.side_effect = None
        coord._play_audio.return_value = None
        final = _arm_and_wait(coord)
        self.assertEqual(final, TxState.COMPLETE)


# ═══════════════════════════════════════════════════════════════════════════════
# § 12  Integration: slot scheduling math
# ═══════════════════════════════════════════════════════════════════════════════

class TestFt8TxCoordinatorSlotIntegration(unittest.TestCase):
    """
    Verify that the coordinator actually waits for approximately the right
    amount of time before firing TX.
    """

    def test_tx_fires_approximately_at_slot_boundary(self):
        """
        If seconds_to_next_slot() returns T, TX should fire approximately
        T - PRE_KEY_S seconds later (within ±50 ms tolerance for scheduler jitter).
        """
        SLOT_WAIT = 0.10   # 100 ms
        timer = _make_timer(seconds_to_next=SLOT_WAIT + PRE_KEY_S)
        radio = _make_radio()
        coord = Ft8TxCoordinator(
            radio=radio, slot_timer=timer,
            pre_key_s=PRE_KEY_S, post_key_s=0.0,
        )
        coord._play_audio = mock.MagicMock()

        t0 = time.monotonic()
        job = TxJob("CQ W4ABC EN52", audio_device=0)
        coord.arm(job)
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            if coord.state in (TxState.COMPLETE, TxState.ERROR, TxState.CANCELED):
                break
            time.sleep(0.01)
        elapsed = time.monotonic() - t0

        self.assertGreaterEqual(elapsed, SLOT_WAIT - 0.05)
        self.assertLessEqual(elapsed, SLOT_WAIT + 0.30)


# ═══════════════════════════════════════════════════════════════════════════════
# § 13  WASAPI fallback helper and _play_audio Windows fallback
# ═══════════════════════════════════════════════════════════════════════════════

from ft8_tx import _find_wasapi_output_device, _find_mme_output_device, _is_wdm_ks_device


class TestFindWasapiOutputDevice(unittest.TestCase):
    """Unit tests for _find_wasapi_output_device()."""

    def _make_sd(self, host_apis, devices, default_device=(0, 1)):
        """Build a minimal sounddevice stub."""
        sd = mock.MagicMock()
        sd.query_hostapis.side_effect = lambda idx=None: (
            host_apis[idx] if idx is not None else host_apis
        )
        sd.query_devices.side_effect = lambda idx=None: (
            devices[idx] if idx is not None else devices
        )
        sd.default.device = default_device
        return sd

    def test_returns_none_when_no_wasapi_host_api(self):
        """If no WASAPI host API is present, return None."""
        host_apis = [{"index": 0, "name": "MME", "default_output_device": 0}]
        devices = [{"index": 0, "name": "Speaker", "hostapi": 0, "max_output_channels": 2}]
        sd = self._make_sd(host_apis, devices)
        result = _find_wasapi_output_device(sd, 0)
        self.assertIsNone(result)

    def test_returns_exact_name_match_in_wasapi(self):
        """If the device name matches a WASAPI device exactly, return that index."""
        host_apis = [
            {"index": 0, "name": "MME", "default_output_device": 0},
            {"index": 1, "name": "Windows WASAPI", "default_output_device": 2},
        ]
        devices = [
            {"index": 0, "name": "Speaker", "hostapi": 0, "max_output_channels": 2},
            {"index": 1, "name": "Microphone", "hostapi": 0, "max_output_channels": 0},
            {"index": 2, "name": "Speaker", "hostapi": 1, "max_output_channels": 2},
        ]
        sd = self._make_sd(host_apis, devices)
        result = _find_wasapi_output_device(sd, 0)
        self.assertEqual(result, 2)

    def test_returns_partial_name_match_in_wasapi(self):
        """If full name doesn't match but a partial (stripped) name does, return it."""
        host_apis = [
            {"index": 0, "name": "WDM-KS", "default_output_device": 0},
            {"index": 1, "name": "Windows WASAPI", "default_output_device": 2},
        ]
        devices = [
            {"index": 0, "name": "USB Audio Device (WDM-KS)", "hostapi": 0, "max_output_channels": 2},
            {"index": 2, "name": "USB Audio Device (WASAPI)", "hostapi": 1, "max_output_channels": 2},
        ]
        sd = self._make_sd(host_apis, devices)
        # The WDM-KS device name is "USB Audio Device (WDM-KS)"; stripped → "USB Audio Device"
        result = _find_wasapi_output_device(sd, 0)
        self.assertEqual(result, 2)

    def test_returns_wasapi_default_when_no_name_match(self):
        """When no name match is found, fall back to WASAPI default output."""
        host_apis = [
            {"index": 0, "name": "WDM-KS", "default_output_device": 0},
            {"index": 1, "name": "Windows WASAPI", "default_output_device": 5},
        ]
        devices = [
            {"index": 0, "name": "Foo", "hostapi": 0, "max_output_channels": 2},
            {"index": 5, "name": "Bar", "hostapi": 1, "max_output_channels": 2},
        ]
        sd = self._make_sd(host_apis, devices)
        result = _find_wasapi_output_device(sd, 0)
        self.assertEqual(result, 5)

    def test_returns_none_when_sd_raises(self):
        """If sounddevice raises unexpectedly, return None (no propagation)."""
        sd = mock.MagicMock()
        sd.query_hostapis.side_effect = RuntimeError("boom")
        result = _find_wasapi_output_device(sd, 0)
        self.assertIsNone(result)

    def test_returns_none_for_no_wasapi_output_devices(self):
        """WASAPI host API exists but has no output devices → return None."""
        host_apis = [
            {"index": 0, "name": "Windows WASAPI", "default_output_device": -1},
        ]
        devices = [
            {"index": 0, "name": "Mic", "hostapi": 0, "max_output_channels": 0},
        ]
        sd = self._make_sd(host_apis, devices)
        result = _find_wasapi_output_device(sd, None)
        self.assertIsNone(result)


class TestPlayAudioWasapiFallback(unittest.TestCase):
    """Tests for the WASAPI fallback path inside _play_audio."""

    def _make_coord(self):
        return Ft8TxCoordinator(radio=None)

    def _make_fake_sd(self):
        """Return a minimal sounddevice stub."""
        fake_sd = mock.MagicMock()
        fake_sd.PortAudioError = RuntimeError
        fake_sd.query_devices.return_value = {"default_samplerate": 48_000, "name": "USB Speaker"}
        fake_sd.default.device = (0, 0)
        return fake_sd

    def test_wasapi_fallback_triggered_on_windows_portaudio_error(self):
        """
        When _stream_play raises PortAudioError on Windows and a WASAPI device is
        found, _play_audio retries with the WASAPI device index and succeeds.
        """
        import numpy as np

        coord = self._make_coord()
        audio = np.zeros(100, dtype=np.float32)
        fake_sd = self._make_fake_sd()

        stream_calls = []

        def _fake_stream(sd_mod, aud, fs, device, *, extra_settings=None):
            stream_calls.append(device)
            if len(stream_calls) == 1:
                raise RuntimeError("WDM-KS error")

        # _find_wasapi_output_device will be called — patch it to return 3
        with mock.patch("ft8_tx.platform.system", return_value="Windows"), \
             mock.patch("ft8_tx._find_wasapi_output_device", return_value=3), \
             mock.patch("ft8_tx._stream_play", side_effect=_fake_stream), \
             mock.patch.dict("sys.modules", {"sounddevice": fake_sd}):
            coord._play_audio(audio, device=0)

        self.assertEqual(len(stream_calls), 2, "_stream_play should be called twice (original + fallback)")
        self.assertEqual(stream_calls[1], 3, "Second call should use WASAPI device 3")

    def test_no_wasapi_fallback_on_non_windows(self):
        """
        On non-Windows platforms, a PortAudioError should NOT trigger the
        WASAPI fallback; it should propagate as RuntimeError immediately.
        """
        import numpy as np

        coord = self._make_coord()
        audio = np.zeros(100, dtype=np.float32)
        fake_sd = self._make_fake_sd()

        def _fake_stream(sd_mod, aud, fs, device, *, extra_settings=None):
            raise RuntimeError("PA error")

        with mock.patch("ft8_tx.platform.system", return_value="Linux"), \
             mock.patch("ft8_tx._stream_play", side_effect=_fake_stream), \
             mock.patch.dict("sys.modules", {"sounddevice": fake_sd}):
            with self.assertRaises(RuntimeError):
                coord._play_audio(audio, device=0)

    def test_wasapi_fallback_raises_if_no_wasapi_device_found(self):
        """
        On Windows, if no WASAPI device is found the code performs a single
        retry after a 200 ms pause before giving up.  When the retry also
        fails, a RuntimeError is raised.
        """
        import numpy as np

        coord = self._make_coord()
        audio = np.zeros(100, dtype=np.float32)
        fake_sd = self._make_fake_sd()

        def _fake_stream(sd_mod, aud, fs, device, *, extra_settings=None):
            raise RuntimeError("WDM-KS error")

        with mock.patch("ft8_tx.platform.system", return_value="Windows"), \
             mock.patch("ft8_tx._find_wasapi_output_device", return_value=None), \
             mock.patch("ft8_tx.time.sleep"), \
             mock.patch("ft8_tx._stream_play", side_effect=_fake_stream) as mock_stream, \
             mock.patch.dict("sys.modules", {"sounddevice": fake_sd}):
            with self.assertRaises(RuntimeError):
                coord._play_audio(audio, device=0)

        # Two _stream_play attempts: initial try + one retry after the 200 ms pause
        self.assertEqual(mock_stream.call_count, 2,
                         "One retry expected after transient failure when no WASAPI found")

    def test_wasapi_fallback_raises_if_fallback_also_fails(self):
        """
        On Windows, when the original attempt, the reactive WASAPI attempt, and
        the WASAPI shared-mode retry all fail, the code falls through to the
        delay-retry.  When that also fails, a RuntimeError is raised.
        """
        import numpy as np

        coord = self._make_coord()
        audio = np.zeros(100, dtype=np.float32)
        fake_sd = self._make_fake_sd()

        def _fake_stream(sd_mod, aud, fs, device, *, extra_settings=None):
            raise RuntimeError("audio error")

        with mock.patch("ft8_tx.platform.system", return_value="Windows"), \
             mock.patch("ft8_tx._find_wasapi_output_device", return_value=7), \
             mock.patch("ft8_tx._find_mme_output_device", return_value=None), \
             mock.patch("ft8_tx.time.sleep"), \
             mock.patch("ft8_tx._stream_play", side_effect=_fake_stream) as mock_stream, \
             mock.patch.dict("sys.modules", {"sounddevice": fake_sd}):
            with self.assertRaises(RuntimeError):
                coord._play_audio(audio, device=0)

        # original + WASAPI first attempt + WASAPI shared mode + delay-retry = 4
        self.assertEqual(mock_stream.call_count, 4,
                         "Expected original + WASAPI + WASAPI shared + delay-retry attempts")

    def test_retry_after_delay_succeeds_when_no_wasapi_device(self):
        """
        On Windows, when no WASAPI device is found and the first _stream_play call
        raises PortAudioError, the code waits 200 ms and retries.  If the
        retry succeeds, _play_audio returns without raising.
        """
        import numpy as np

        coord = self._make_coord()
        audio = np.zeros(100, dtype=np.float32)
        fake_sd = self._make_fake_sd()

        call_count = [0]
        sleep_calls = []

        def _fake_stream(sd_mod, aud, fs, device, *, extra_settings=None):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("WDM-KS transient error")
            # Second attempt succeeds (no raise)

        with mock.patch("ft8_tx.platform.system", return_value="Windows"), \
             mock.patch("ft8_tx._find_wasapi_output_device", return_value=None), \
             mock.patch("ft8_tx.time.sleep",
                        side_effect=lambda t: sleep_calls.append(t)), \
             mock.patch("ft8_tx._stream_play", side_effect=_fake_stream), \
             mock.patch.dict("sys.modules", {"sounddevice": fake_sd}):
            # Should NOT raise — second attempt succeeds
            coord._play_audio(audio, device=0)

        self.assertEqual(call_count[0], 2, "Retry should be attempted")
        self.assertTrue(any(t >= USB_AUDIO_SWITCH_DELAY_S for t in sleep_calls),
                        "A ≥200 ms sleep must precede the retry")


# ═══════════════════════════════════════════════════════════════════════════════
# § 14  _is_wdm_ks_device() helper
# ═══════════════════════════════════════════════════════════════════════════════

class TestIsWdmKsDevice(unittest.TestCase):
    """Unit tests for _is_wdm_ks_device()."""

    def _make_sd(self, device_hostapi: int, api_name: str):
        """Return a minimal sounddevice stub for a given host API name."""
        sd = mock.MagicMock()
        sd.query_devices.return_value = {"hostapi": device_hostapi, "name": "Test Device"}
        sd.query_hostapis.return_value = {"name": api_name}
        return sd

    def test_returns_true_for_wdm_ks_device(self):
        """A device under 'Windows WDM-KS' host API is detected as WDM-KS."""
        sd = self._make_sd(device_hostapi=2, api_name="Windows WDM-KS")
        self.assertTrue(_is_wdm_ks_device(sd, 0))

    def test_returns_true_for_wdm_ks_lowercase(self):
        """Case-insensitive match: 'windows wdm-ks' is still WDM-KS."""
        sd = self._make_sd(device_hostapi=0, api_name="windows wdm-ks")
        self.assertTrue(_is_wdm_ks_device(sd, 1))

    def test_returns_false_for_wasapi_device(self):
        """A device under WASAPI is NOT a WDM-KS device."""
        sd = self._make_sd(device_hostapi=1, api_name="Windows WASAPI")
        self.assertFalse(_is_wdm_ks_device(sd, 0))

    def test_returns_false_for_mme_device(self):
        """A device under MME is NOT a WDM-KS device."""
        sd = self._make_sd(device_hostapi=0, api_name="MME")
        self.assertFalse(_is_wdm_ks_device(sd, 0))

    def test_returns_false_for_directsound_device(self):
        """A device under DirectSound is NOT a WDM-KS device."""
        sd = self._make_sd(device_hostapi=0, api_name="Windows DirectSound")
        self.assertFalse(_is_wdm_ks_device(sd, 0))

    def test_returns_false_when_device_index_is_none(self):
        """device_index=None returns False without calling sounddevice APIs."""
        sd = mock.MagicMock()
        self.assertFalse(_is_wdm_ks_device(sd, None))
        sd.query_devices.assert_not_called()

    def test_returns_false_when_device_index_is_negative(self):
        """device_index < 0 returns False without calling sounddevice APIs."""
        sd = mock.MagicMock()
        self.assertFalse(_is_wdm_ks_device(sd, -1))
        sd.query_devices.assert_not_called()

    def test_returns_false_when_no_hostapi_key(self):
        """If device info lacks 'hostapi' key, returns False gracefully."""
        sd = mock.MagicMock()
        sd.query_devices.return_value = {"name": "Some Device"}
        self.assertFalse(_is_wdm_ks_device(sd, 0))

    def test_returns_false_when_query_devices_raises(self):
        """If query_devices raises, returns False without propagating."""
        sd = mock.MagicMock()
        sd.query_devices.side_effect = RuntimeError("no device")
        self.assertFalse(_is_wdm_ks_device(sd, 0))

    def test_returns_false_when_query_hostapis_raises(self):
        """If query_hostapis raises, returns False without propagating."""
        sd = mock.MagicMock()
        sd.query_devices.return_value = {"hostapi": 0, "name": "Dev"}
        sd.query_hostapis.side_effect = KeyError("unknown api")
        self.assertFalse(_is_wdm_ks_device(sd, 0))


# ═══════════════════════════════════════════════════════════════════════════════
# § 15  Proactive WDM-KS → WASAPI swap in _play_audio
# ═══════════════════════════════════════════════════════════════════════════════

class TestPlayAudioProactiveWdmKsSwap(unittest.TestCase):
    """
    Verify that _play_audio proactively replaces a WDM-KS device with its
    WASAPI equivalent on Windows before any stream is opened, so that
    KSPROPERTY_AUDIO_SAMPLING_FREQ / WDM-KS PortAudio errors are avoided
    entirely rather than discovered on the first failed _stream_play() call.
    """

    def _make_coord(self):
        return Ft8TxCoordinator(radio=None)

    def _make_wdm_ks_fake_sd(self):
        """Return a fake sounddevice that looks like a WDM-KS device."""
        fake_sd = mock.MagicMock()
        fake_sd.PortAudioError = RuntimeError
        fake_sd.query_devices.return_value = {
            "default_samplerate": 48_000,
            "name": "USB Audio Device",
            "hostapi": 2,
        }
        fake_sd.query_hostapis.return_value = {"name": "Windows WDM-KS"}
        fake_sd.default.device = (0, 0)
        return fake_sd

    def test_proactive_swap_uses_wasapi_device_on_first_play(self):
        """
        When the configured device is WDM-KS on Windows, _stream_play should be
        called with the WASAPI device index on the *first* attempt — no
        WDM-KS attempt is made at all.
        """
        import numpy as np

        coord = self._make_coord()
        audio = np.zeros(100, dtype=np.float32)
        fake_sd = self._make_wdm_ks_fake_sd()

        stream_calls = []

        def _fake_stream(sd_mod, aud, fs, device, *, extra_settings=None):
            stream_calls.append(device)

        # _is_wdm_ks_device will detect WDM-KS; _find_wasapi_output_device
        # is patched to return 5 (the WASAPI equivalent device index).
        with mock.patch("ft8_tx.platform.system", return_value="Windows"), \
             mock.patch("ft8_tx._find_wasapi_output_device", return_value=5), \
             mock.patch("ft8_tx._stream_play", side_effect=_fake_stream), \
             mock.patch.dict("sys.modules", {"sounddevice": fake_sd}):
            coord._play_audio(audio, device=0)

        self.assertEqual(len(stream_calls), 1,
                         "_stream_play should be called exactly once (no WDM-KS attempt)")
        self.assertEqual(stream_calls[0], 5,
                         "The WASAPI device (5) should be used on the first call")

    def test_proactive_swap_skipped_for_non_wdm_ks_device(self):
        """
        When the configured device is NOT WDM-KS, no proactive swap occurs
        and _stream_play is called with the original device index.
        """
        import numpy as np

        coord = self._make_coord()
        audio = np.zeros(100, dtype=np.float32)

        fake_sd = mock.MagicMock()
        fake_sd.PortAudioError = RuntimeError
        fake_sd.query_devices.return_value = {
            "default_samplerate": 48_000,
            "name": "USB Speaker",
            "hostapi": 1,
        }
        fake_sd.query_hostapis.return_value = {"name": "Windows WASAPI"}
        fake_sd.default.device = (0, 0)

        stream_calls = []

        def _fake_stream(sd_mod, aud, fs, device, *, extra_settings=None):
            stream_calls.append(device)

        with mock.patch("ft8_tx.platform.system", return_value="Windows"), \
             mock.patch("ft8_tx._stream_play", side_effect=_fake_stream), \
             mock.patch.dict("sys.modules", {"sounddevice": fake_sd}):
            coord._play_audio(audio, device=2)

        self.assertEqual(len(stream_calls), 1)
        self.assertEqual(stream_calls[0], 2,
                         "Original (non-WDM-KS) device should be used unchanged")

    def test_proactive_swap_skipped_on_non_windows(self):
        """
        On non-Windows platforms, no proactive WDM-KS swap should occur
        even if the device would be WDM-KS on Windows.
        """
        import numpy as np

        coord = self._make_coord()
        audio = np.zeros(100, dtype=np.float32)

        fake_sd = mock.MagicMock()
        fake_sd.PortAudioError = RuntimeError
        fake_sd.query_devices.return_value = {
            "default_samplerate": 48_000,
            "name": "Device",
        }
        fake_sd.default.device = (0, 0)

        stream_calls = []

        def _fake_stream(sd_mod, aud, fs, device, *, extra_settings=None):
            stream_calls.append(device)

        wasapi_called = []

        with mock.patch("ft8_tx.platform.system", return_value="Linux"), \
             mock.patch("ft8_tx._is_wdm_ks_device",
                        side_effect=lambda *a, **kw: wasapi_called.append(1) or True), \
             mock.patch("ft8_tx._stream_play", side_effect=_fake_stream), \
             mock.patch.dict("sys.modules", {"sounddevice": fake_sd}):
            coord._play_audio(audio, device=3)

        self.assertEqual(len(wasapi_called), 0,
                         "_is_wdm_ks_device must not be called on non-Windows")
        self.assertEqual(stream_calls[0], 3)

    def test_proactive_swap_falls_back_to_original_when_no_wasapi_found(self):
        """
        If _find_wasapi_output_device returns None during the proactive check,
        _find_mme_output_device is tried next.  When both return None the
        original device is used unchanged (no swap, no crash).
        """
        import numpy as np

        coord = self._make_coord()
        audio = np.zeros(100, dtype=np.float32)
        fake_sd = self._make_wdm_ks_fake_sd()

        stream_calls = []

        def _fake_stream(sd_mod, aud, fs, device, *, extra_settings=None):
            stream_calls.append(device)

        # Neither WASAPI nor MME equivalent found — original device must be used unchanged
        with mock.patch("ft8_tx.platform.system", return_value="Windows"), \
             mock.patch("ft8_tx._find_wasapi_output_device", return_value=None), \
             mock.patch("ft8_tx._find_mme_output_device", return_value=None), \
             mock.patch("ft8_tx._stream_play", side_effect=_fake_stream), \
             mock.patch.dict("sys.modules", {"sounddevice": fake_sd}):
            coord._play_audio(audio, device=4)

        self.assertEqual(len(stream_calls), 1)
        self.assertEqual(stream_calls[0], 4,
                         "Original device used when no WASAPI or MME equivalent found")

    def test_proactive_swap_retry_after_delay_if_wasapi_fails(self):
        """
        If the proactively-selected WASAPI device raises PortAudioError on the
        first attempt, the reactive path tries WASAPI shared mode next, then
        falls back to a 200 ms delay-retry on the same device.  When all three
        attempts fail the error is raised as RuntimeError.
        """
        import numpy as np

        coord = self._make_coord()
        audio = np.zeros(100, dtype=np.float32)
        fake_sd = self._make_wdm_ks_fake_sd()

        sleep_calls = []

        def _fake_stream(sd_mod, aud, fs, device, *, extra_settings=None):
            raise RuntimeError("stream error")

        # Both proactive and reactive helpers return the same WASAPI device (5)
        with mock.patch("ft8_tx.platform.system", return_value="Windows"), \
             mock.patch("ft8_tx._find_wasapi_output_device", return_value=5), \
             mock.patch("ft8_tx.time.sleep",
                        side_effect=lambda t: sleep_calls.append(t)), \
             mock.patch("ft8_tx._stream_play", side_effect=_fake_stream) as mock_stream, \
             mock.patch.dict("sys.modules", {"sounddevice": fake_sd}):
            with self.assertRaises(RuntimeError):
                coord._play_audio(audio, device=0)

        # proactive first WASAPI attempt + WASAPI shared mode + delay-retry = 3
        self.assertEqual(mock_stream.call_count, 3,
                         "Expected proactive first attempt + shared mode + delay-retry")
        # A 200 ms sleep must have been inserted before the final delay-retry
        self.assertTrue(any(t >= USB_AUDIO_SWITCH_DELAY_S for t in sleep_calls),
                        "A ≥200 ms sleep must occur before the delay-retry")

    def test_proactive_swap_retry_succeeds_after_transient_error(self):
        """
        If the proactively-selected WASAPI device fails on the first attempt
        (transient error), the reactive path tries WASAPI shared mode next.
        When shared mode succeeds, _play_audio returns without raising and
        without inserting a 200 ms delay (shared mode is tried immediately).
        """
        import numpy as np

        coord = self._make_coord()
        audio = np.zeros(100, dtype=np.float32)
        fake_sd = self._make_wdm_ks_fake_sd()

        call_count = [0]
        sleep_calls = []

        def _fake_stream(sd_mod, aud, fs, device, *, extra_settings=None):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("stream error")
            # Second attempt (WASAPI shared mode) succeeds

        # Both proactive and reactive helpers return the same WASAPI device (5)
        with mock.patch("ft8_tx.platform.system", return_value="Windows"), \
             mock.patch("ft8_tx._find_wasapi_output_device", return_value=5), \
             mock.patch("ft8_tx.time.sleep",
                        side_effect=lambda t: sleep_calls.append(t)), \
             mock.patch("ft8_tx._stream_play", side_effect=_fake_stream), \
             mock.patch.dict("sys.modules", {"sounddevice": fake_sd}):
            # Must not raise — WASAPI shared mode retry succeeds
            coord._play_audio(audio, device=0)

        self.assertEqual(call_count[0], 2,
                         "First WASAPI attempt + WASAPI shared mode retry = 2 calls")
        # WASAPI shared mode is tried immediately — no sleep before it
        self.assertFalse(any(t >= USB_AUDIO_SWITCH_DELAY_S for t in sleep_calls),
                         "No 200 ms sleep should occur when shared mode succeeds")

    def test_proactive_swap_default_output_wdm_ks(self):
        """
        When device=None (default output) and the system default output is a
        WDM-KS device, _play_audio must proactively swap to WASAPI — using
        sd.default.device[1] as the source index for WDM-KS detection.
        """
        import numpy as np

        coord = self._make_coord()
        audio = np.zeros(100, dtype=np.float32)

        fake_sd = mock.MagicMock()
        fake_sd.PortAudioError = RuntimeError
        # sd.default.device[1] == 3 is the default output
        fake_sd.default.device = (0, 3)
        fake_sd.query_devices.return_value = {
            "default_samplerate": 48_000,
            "name": "Default Speaker (WDM-KS)",
            "hostapi": 2,
        }
        fake_sd.query_hostapis.return_value = {"name": "Windows WDM-KS"}

        stream_calls = []

        def _fake_stream(sd_mod, aud, fs, device, *, extra_settings=None):
            stream_calls.append(device)

        # _find_wasapi_output_device should be called with the resolved index (3)
        # and returns WASAPI device 7.
        with mock.patch("ft8_tx.platform.system", return_value="Windows"), \
             mock.patch("ft8_tx._find_wasapi_output_device", return_value=7) as mock_fwod, \
             mock.patch("ft8_tx._stream_play", side_effect=_fake_stream), \
             mock.patch.dict("sys.modules", {"sounddevice": fake_sd}):
            coord._play_audio(audio, device=None)

        # Should have looked up the WASAPI counterpart for the default device (3)
        mock_fwod.assert_called_once_with(fake_sd, 3)
        self.assertEqual(len(stream_calls), 1,
                         "_stream_play should be called exactly once")
        self.assertEqual(stream_calls[0], 7,
                         "WASAPI device (7) should be used for the default WDM-KS output")

    def test_proactive_swap_negative_device_default_wdm_ks(self):
        """
        When device=-1 (sentinel for 'use default'), the same default-output
        WDM-KS detection path applies and the swap fires correctly.
        """
        import numpy as np

        coord = self._make_coord()
        audio = np.zeros(100, dtype=np.float32)

        fake_sd = mock.MagicMock()
        fake_sd.PortAudioError = RuntimeError
        fake_sd.default.device = (0, 4)
        fake_sd.query_devices.return_value = {
            "default_samplerate": 48_000,
            "name": "USB Out (WDM-KS)",
            "hostapi": 2,
        }
        fake_sd.query_hostapis.return_value = {"name": "Windows WDM-KS"}

        stream_calls = []

        def _fake_stream(sd_mod, aud, fs, device, *, extra_settings=None):
            stream_calls.append(device)

        with mock.patch("ft8_tx.platform.system", return_value="Windows"), \
             mock.patch("ft8_tx._find_wasapi_output_device", return_value=8) as mock_fwod, \
             mock.patch("ft8_tx._stream_play", side_effect=_fake_stream), \
             mock.patch.dict("sys.modules", {"sounddevice": fake_sd}):
            coord._play_audio(audio, device=-1)

        # Resolved index should be sd.default.device[1] == 4
        mock_fwod.assert_called_once_with(fake_sd, 4)
        self.assertEqual(stream_calls[0], 8)


# ═══════════════════════════════════════════════════════════════════════════════
# § 15b  Proactive WDM-KS → WASAPI shared-mode fallback
# ═══════════════════════════════════════════════════════════════════════════════

class TestPlayAudioProactiveWdmKsSharedModeFallback(unittest.TestCase):
    """
    Verify that when the proactive WDM-KS swap selects a WASAPI device and
    the first WASAPI play attempt fails, _play_audio retries in WASAPI shared
    mode before falling back to MME or the delay-retry.

    This covers the specific bug where sounddevice's default WASAPI mode
    (exclusive) propagates the same WdmSyncIoctl / KSPROPERTY_AUDIO_SAMPLING_FREQ
    error that caused the original WDM-KS failure.  WASAPI shared mode routes
    audio through the Windows Audio Session API mixer and avoids those IOCTLs.
    """

    def _make_coord(self):
        return Ft8TxCoordinator(radio=None)

    def _make_fake_sd_wdm_ks(self):
        """Return a fake sounddevice that looks like a WDM-KS device."""
        fake_sd = mock.MagicMock()
        fake_sd.PortAudioError = RuntimeError
        fake_sd.query_devices.return_value = {
            "default_samplerate": 48_000,
            "name": "USB Audio CODEC",
            "hostapi": 2,
        }
        fake_sd.query_hostapis.return_value = {"name": "Windows WDM-KS"}
        fake_sd.default.device = (0, 0)
        return fake_sd

    def test_proactive_wasapi_exclusive_fails_shared_mode_succeeds(self):
        """
        When the proactive WASAPI swap selects device 5 and the first _stream_play
        attempt fails (exclusive WASAPI mode), _play_audio must retry in shared
        mode (WasapiSettings via extra_settings) and succeed — without inserting
        a 200 ms delay.

        Stream call sequence: first WASAPI attempt (fails) → shared mode (succeeds).
        """
        import numpy as np

        coord = self._make_coord()
        audio = np.zeros(100, dtype=np.float32)
        fake_sd = self._make_fake_sd_wdm_ks()

        call_count = [0]
        stream_calls = []

        def _fake_stream(sd_mod, aud, fs, device, *, extra_settings=None):
            call_count[0] += 1
            stream_calls.append({"device": device, "extra_settings": extra_settings})
            if call_count[0] == 1:
                raise RuntimeError(
                    "WdmSyncIoctl: DeviceIoControl GLE = 0x00000490 "
                    "(prop_set = {8C134960-51AD-11CF-878A-94F801C10000}, prop_id = 10)"
                )
            # Second attempt (shared mode) succeeds

        sleep_calls = []

        # Proactive & reactive both return WASAPI device 5 (same device).
        with mock.patch("ft8_tx.platform.system", return_value="Windows"), \
             mock.patch("ft8_tx._find_wasapi_output_device", return_value=5), \
             mock.patch("ft8_tx.time.sleep",
                        side_effect=lambda t: sleep_calls.append(t)), \
             mock.patch("ft8_tx._stream_play", side_effect=_fake_stream), \
             mock.patch.dict("sys.modules", {"sounddevice": fake_sd}):
            coord._play_audio(audio, device=0)   # Must not raise

        self.assertEqual(call_count[0], 2,
                         "first WASAPI attempt (fails) + WASAPI shared (succeeds) = 2 calls")
        # Shared mode is tried immediately — no 200 ms sleep needed
        self.assertFalse(any(t >= USB_AUDIO_SWITCH_DELAY_S for t in sleep_calls),
                         "No delay sleep should occur when WASAPI shared mode succeeds")
        # The second call must use extra_settings for WasapiSettings
        self.assertIsNotNone(stream_calls[1]["extra_settings"],
                             "Shared-mode retry must pass WasapiSettings via extra_settings")
        self.assertEqual(stream_calls[1]["device"], 5,
                         "Shared-mode retry must use the same WASAPI device index")

    def test_proactive_wasapi_shared_mode_fails_falls_through_to_delay_retry(self):
        """
        When the proactive first WASAPI attempt fails AND WASAPI shared mode
        also fails, the code falls through to the delay-retry on the same device.
        If the delay-retry succeeds, _play_audio returns without raising.

        Stream call sequence: first WASAPI attempt (fails) → shared (fails) → delay-retry (succeeds).
        """
        import numpy as np

        coord = self._make_coord()
        audio = np.zeros(100, dtype=np.float32)
        fake_sd = self._make_fake_sd_wdm_ks()

        call_count = [0]

        def _fake_stream(sd_mod, aud, fs, device, *, extra_settings=None):
            call_count[0] += 1
            if call_count[0] <= 2:
                raise RuntimeError("WDM-KS / WASAPI error")
            # Third attempt (delay-retry) succeeds

        sleep_calls = []

        with mock.patch("ft8_tx.platform.system", return_value="Windows"), \
             mock.patch("ft8_tx._find_wasapi_output_device", return_value=5), \
             mock.patch("ft8_tx._find_mme_output_device", return_value=None), \
             mock.patch("ft8_tx.time.sleep",
                        side_effect=lambda t: sleep_calls.append(t)), \
             mock.patch("ft8_tx._stream_play", side_effect=_fake_stream), \
             mock.patch.dict("sys.modules", {"sounddevice": fake_sd}):
            coord._play_audio(audio, device=0)   # Must not raise

        self.assertEqual(call_count[0], 3,
                         "first WASAPI attempt + shared mode + delay-retry = 3 calls")
        # The 200 ms sleep must occur before the delay-retry (3rd attempt)
        self.assertTrue(any(t >= USB_AUDIO_SWITCH_DELAY_S for t in sleep_calls),
                        "A ≥200 ms sleep must precede the delay-retry")

    def test_proactive_wasapi_shared_mode_not_tried_without_wasapi_settings(self):
        """
        When sounddevice lacks WasapiSettings (old version), the shared-mode
        step is skipped and the code falls straight through to MME / delay-retry.
        """
        import numpy as np

        coord = self._make_coord()
        audio = np.zeros(100, dtype=np.float32)

        # MagicMock with spec: no WasapiSettings attribute
        fake_sd = mock.MagicMock(
            spec=["PortAudioError", "play", "wait", "default",
                  "query_devices", "query_hostapis", "OutputStream",
                  "CallbackStop"]
        )
        fake_sd.PortAudioError = RuntimeError
        fake_sd.query_devices.return_value = {
            "default_samplerate": 48_000,
            "name": "USB CODEC",
            "hostapi": 2,
        }
        fake_sd.query_hostapis.return_value = {"name": "Windows WDM-KS"}
        fake_sd.default.device = (0, 0)

        call_count = [0]

        def _fake_stream(sd_mod, aud, fs, device, *, extra_settings=None):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("WDM-KS error")
            # Second attempt (delay-retry) succeeds

        with mock.patch("ft8_tx.platform.system", return_value="Windows"), \
             mock.patch("ft8_tx._find_wasapi_output_device", return_value=5), \
             mock.patch("ft8_tx._find_mme_output_device", return_value=None), \
             mock.patch("ft8_tx.time.sleep"), \
             mock.patch("ft8_tx._stream_play", side_effect=_fake_stream), \
             mock.patch.dict("sys.modules", {"sounddevice": fake_sd}):
            coord._play_audio(audio, device=0)   # Must not raise

        # No shared mode tried → first WASAPI attempt (fails) + delay-retry (succeeds) = 2
        self.assertEqual(call_count[0], 2,
                         "Without WasapiSettings: first WASAPI attempt + delay-retry = 2 calls")


# ═══════════════════════════════════════════════════════════════════════════════
# § 15d  WASAPI shared-mode fallback for directly-selected WASAPI device
# ═══════════════════════════════════════════════════════════════════════════════

class TestPlayAudioDirectWasapiSharedModeFallback(unittest.TestCase):
    """
    Verify that when the user directly configures a WASAPI device (no WDM-KS
    proactive swap) and the first play attempt fails with a PortAudioError,
    _play_audio retries in WASAPI shared mode before falling back to
    MME or the delay-retry.

    This reproduces the specific bug where a user-selected WASAPI device fails
    with 'WdmSyncIoctl: DeviceIoControl GLE = 0x00000490 (prop_id = 10)'
    (PaErrorCode -9999) — the same KSPROPERTY_AUDIO_SAMPLING_FREQ error that
    occurs on WDM-KS devices.  Without the fix, the code skipped shared-mode
    and went straight to the useless 200 ms delay-retry on the same device,
    which also failed, producing a RuntimeError.
    """

    def _make_coord(self):
        return Ft8TxCoordinator(radio=None)

    def _make_wasapi_fake_sd(self):
        """Return a fake sounddevice that looks like a WASAPI device."""
        fake_sd = mock.MagicMock()
        fake_sd.PortAudioError = RuntimeError
        # query_devices returns WASAPI device info
        fake_sd.query_devices.return_value = {
            "default_samplerate": 48_000,
            "name": "USB Audio CODEC",
            "hostapi": 1,
        }
        fake_sd.query_hostapis.return_value = {"name": "Windows WASAPI"}
        fake_sd.default.device = (0, 5)
        return fake_sd

    def test_direct_wasapi_exclusive_fails_shared_mode_tried(self):
        """
        When the user configures a WASAPI device directly (no WDM-KS proactive
        swap), and the first play attempt fails, _play_audio must retry in
        WASAPI shared mode before the delay-retry.

        This is the core bug: the old code required _proactive_swap_occurred to
        be True before trying shared mode; a directly-selected WASAPI device
        that fails in exclusive mode would skip shared mode and go straight to
        the useless delay-retry.
        """
        import numpy as np

        coord = self._make_coord()
        audio = np.zeros(100, dtype=np.float32)
        fake_sd = self._make_wasapi_fake_sd()

        call_count = [0]
        stream_calls = []

        def _fake_stream(sd_mod, aud, fs, device, *, extra_settings=None):
            call_count[0] += 1
            stream_calls.append({"device": device, "extra_settings": extra_settings})
            if call_count[0] == 1:
                raise RuntimeError(
                    "WdmSyncIoctl: DeviceIoControl GLE = 0x00000490 "
                    "(prop_set = {8C134960-51AD-11CF-878A-94F801C10000}, prop_id = 10)"
                )
            # Second attempt (WASAPI shared mode) succeeds

        sleep_calls = []

        # _find_wasapi_output_device returns the SAME device (already WASAPI),
        # so _proactive_swap_occurred is False and wasapi_dev == effective_device
        with mock.patch("ft8_tx.platform.system", return_value="Windows"), \
             mock.patch("ft8_tx._is_wdm_ks_device", return_value=False), \
             mock.patch("ft8_tx._find_wasapi_output_device", return_value=5), \
             mock.patch("ft8_tx.time.sleep",
                        side_effect=lambda t: sleep_calls.append(t)), \
             mock.patch("ft8_tx._stream_play", side_effect=_fake_stream), \
             mock.patch.dict("sys.modules", {"sounddevice": fake_sd}):
            coord._play_audio(audio, device=5)   # Must not raise

        self.assertEqual(call_count[0], 2,
                         "first WASAPI attempt (fails) + WASAPI shared mode (succeeds) = 2 calls")
        # Shared mode must be tried immediately — no 200 ms delay
        self.assertFalse(any(t >= USB_AUDIO_SWITCH_DELAY_S for t in sleep_calls),
                         "No delay sleep should occur when WASAPI shared mode succeeds")
        # The second call must use WasapiSettings (exclusive=False)
        self.assertIsNotNone(stream_calls[1]["extra_settings"],
                             "Shared-mode retry must pass WasapiSettings via extra_settings")
        self.assertEqual(stream_calls[1]["device"], 5,
                         "Shared-mode retry must use the same WASAPI device index")

    def test_direct_wasapi_shared_mode_fails_falls_to_delay_retry(self):
        """
        When the directly-selected WASAPI device fails in both exclusive AND
        shared mode, the code falls through to the 200 ms delay-retry.
        If delay-retry succeeds, _play_audio returns without raising.
        """
        import numpy as np

        coord = self._make_coord()
        audio = np.zeros(100, dtype=np.float32)
        fake_sd = self._make_wasapi_fake_sd()

        call_count = [0]

        def _fake_stream(sd_mod, aud, fs, device, *, extra_settings=None):
            call_count[0] += 1
            if call_count[0] <= 2:
                raise RuntimeError("WDM-KS / WASAPI error")
            # Third attempt (delay-retry) succeeds

        sleep_calls = []

        with mock.patch("ft8_tx.platform.system", return_value="Windows"), \
             mock.patch("ft8_tx._is_wdm_ks_device", return_value=False), \
             mock.patch("ft8_tx._find_wasapi_output_device", return_value=5), \
             mock.patch("ft8_tx._find_mme_output_device", return_value=None), \
             mock.patch("ft8_tx.time.sleep",
                        side_effect=lambda t: sleep_calls.append(t)), \
             mock.patch("ft8_tx._stream_play", side_effect=_fake_stream), \
             mock.patch.dict("sys.modules", {"sounddevice": fake_sd}):
            coord._play_audio(audio, device=5)   # Must not raise

        self.assertEqual(call_count[0], 3,
                         "first WASAPI attempt + shared mode + delay-retry = 3 calls")
        self.assertTrue(any(t >= USB_AUDIO_SWITCH_DELAY_S for t in sleep_calls),
                        "A ≥200 ms sleep must precede the delay-retry")

    def test_direct_wasapi_all_fail_raises_runtime_error(self):
        """
        When all attempts for a directly-selected WASAPI device fail
        (exclusive, shared mode, delay-retry), RuntimeError is raised.
        """
        import numpy as np

        coord = self._make_coord()
        audio = np.zeros(100, dtype=np.float32)
        fake_sd = self._make_wasapi_fake_sd()

        def _fake_stream(sd_mod, aud, fs, device, *, extra_settings=None):
            raise RuntimeError("WDM-KS / WASAPI error")

        with mock.patch("ft8_tx.platform.system", return_value="Windows"), \
             mock.patch("ft8_tx._is_wdm_ks_device", return_value=False), \
             mock.patch("ft8_tx._find_wasapi_output_device", return_value=5), \
             mock.patch("ft8_tx._find_mme_output_device", return_value=None), \
             mock.patch("ft8_tx.time.sleep"), \
             mock.patch("ft8_tx._stream_play", side_effect=_fake_stream) as mock_stream, \
             mock.patch.dict("sys.modules", {"sounddevice": fake_sd}):
            with self.assertRaises(RuntimeError):
                coord._play_audio(audio, device=5)

        # exclusive + shared mode + delay-retry = 3 attempts
        self.assertEqual(mock_stream.call_count, 3,
                         "exclusive + shared mode + delay-retry = 3 attempts before giving up")



class TestStreamPlay(unittest.TestCase):
    """
    Unit tests for the _stream_play() module-level helper.

    _stream_play() uses sd.OutputStream with a float32 callback to play audio —
    the same technique as SoundCardAudioOutput / AudioTxCapture in voice mode.
    These tests mock sd.OutputStream to verify the interface contract without
    requiring actual audio hardware.
    """

    def _make_fake_sd(self, raise_on_stream_open=False):
        """
        Build a minimal sounddevice stub whose OutputStream context manager
        immediately calls `finished_callback` (simulating instant playback).
        """
        import contextlib

        fake_sd = mock.MagicMock()
        fake_sd.PortAudioError = RuntimeError
        fake_sd.CallbackStop = StopIteration

        # Track all OutputStream() calls
        stream_opens = []

        @contextlib.contextmanager
        def _fake_output_stream(**kwargs):
            if raise_on_stream_open:
                raise RuntimeError("PortAudio error opening stream")
            stream_opens.append(kwargs)
            # Simulate instant playback completion by calling finished_callback
            cb = kwargs.get("finished_callback")
            if cb:
                cb()
            yield mock.MagicMock()

        fake_sd.OutputStream = _fake_output_stream
        return fake_sd, stream_opens

    def test_opens_output_stream_with_float32_and_48khz(self):
        """
        _stream_play must open sd.OutputStream with dtype='float32' and
        samplerate=TX_OUTPUT_SAMPLE_RATE (48 000 Hz) — same as SoundCardAudioOutput.
        """
        import numpy as np

        fake_sd, stream_opens = self._make_fake_sd()
        audio = np.zeros(100, dtype=np.float32)

        _stream_play(fake_sd, audio, TX_OUTPUT_SAMPLE_RATE, device=1)

        self.assertEqual(len(stream_opens), 1)
        kw = stream_opens[0]
        self.assertEqual(kw.get("dtype"), TX_OUTPUT_DTYPE,
                         "OutputStream must use TX_OUTPUT_DTYPE (same as voice mode)")
        self.assertEqual(kw.get("samplerate"), TX_OUTPUT_SAMPLE_RATE)
        self.assertEqual(kw.get("blocksize"), TX_OUTPUT_BLOCKSIZE,
                         "OutputStream must use TX_OUTPUT_BLOCKSIZE")
        self.assertEqual(kw.get("channels"), 1)

    def test_passes_device_kwarg_when_device_valid(self):
        """When device >= 0, it must be passed as a keyword arg to OutputStream."""
        import numpy as np

        fake_sd, stream_opens = self._make_fake_sd()
        audio = np.zeros(50, dtype=np.float32)

        _stream_play(fake_sd, audio, TX_OUTPUT_SAMPLE_RATE, device=3)

        self.assertEqual(stream_opens[0].get("device"), 3)

    def test_omits_device_kwarg_when_device_none(self):
        """When device is None, no 'device' kwarg should be passed to OutputStream."""
        import numpy as np

        fake_sd, stream_opens = self._make_fake_sd()
        audio = np.zeros(50, dtype=np.float32)

        _stream_play(fake_sd, audio, TX_OUTPUT_SAMPLE_RATE, device=None)

        self.assertNotIn("device", stream_opens[0])

    def test_omits_device_kwarg_when_device_negative(self):
        """When device < 0, no 'device' kwarg should be passed to OutputStream."""
        import numpy as np

        fake_sd, stream_opens = self._make_fake_sd()
        audio = np.zeros(50, dtype=np.float32)

        _stream_play(fake_sd, audio, TX_OUTPUT_SAMPLE_RATE, device=-1)

        self.assertNotIn("device", stream_opens[0])

    def test_passes_extra_settings_when_provided(self):
        """When extra_settings is given, it must be forwarded to OutputStream."""
        import numpy as np

        fake_sd, stream_opens = self._make_fake_sd()
        audio = np.zeros(50, dtype=np.float32)
        ws = mock.MagicMock(name="WasapiSettings")

        _stream_play(fake_sd, audio, TX_OUTPUT_SAMPLE_RATE, device=0,
                     extra_settings=ws)

        self.assertIs(stream_opens[0].get("extra_settings"), ws)

    def test_propagates_portaudio_error_from_stream_open(self):
        """
        If sd.OutputStream raises PortAudioError on open, _stream_play must
        propagate it without catching it.
        """
        import numpy as np

        fake_sd, _ = self._make_fake_sd(raise_on_stream_open=True)
        audio = np.zeros(50, dtype=np.float32)

        with self.assertRaises(RuntimeError):
            _stream_play(fake_sd, audio, TX_OUTPUT_SAMPLE_RATE, device=0)

    def test_audio_reshaped_to_column_vector(self):
        """
        _stream_play must reshape a 1-D audio array to (N, 1) before feeding
        it to OutputStream, and the callback must correctly copy audio data
        into the outdata buffer.
        """
        import numpy as np

        fake_sd = mock.MagicMock()
        fake_sd.PortAudioError = RuntimeError
        fake_sd.CallbackStop = StopIteration

        callback_outdata_snapshot = []
        import contextlib

        @contextlib.contextmanager
        def _fake_output_stream(**kwargs):
            cb = kwargs.get("callback")
            # Call the callback with a dummy outdata to verify audio shape
            # and that audio data is correctly written into the buffer.
            outdata = np.zeros((200, 1), dtype=np.float32)
            if cb:
                try:
                    cb(outdata, 200, None, None)
                except StopIteration:
                    pass
                callback_outdata_snapshot.append(outdata.copy())
            fcb = kwargs.get("finished_callback")
            if fcb:
                fcb()
            yield mock.MagicMock()

        fake_sd.OutputStream = _fake_output_stream

        # Use non-zero audio so we can verify the data was actually copied
        audio = np.linspace(0.1, 0.9, 100, dtype=np.float32)
        _stream_play(fake_sd, audio, TX_OUTPUT_SAMPLE_RATE, device=0)

        # outdata passed to callback must be 2-D (n_frames, channels)
        self.assertTrue(len(callback_outdata_snapshot) > 0)
        result = callback_outdata_snapshot[0]
        self.assertEqual(result.ndim, 2, "outdata must be 2-D (n_frames, 1)")
        # Audio data must have been copied into the first 100 frames
        np.testing.assert_allclose(result[:100, 0], audio,
                                   err_msg="Callback must copy audio samples into outdata")


# ═══════════════════════════════════════════════════════════════════════════════
# § 16  _find_mme_output_device() helper
# ═══════════════════════════════════════════════════════════════════════════════

class TestFindMmeOutputDevice(unittest.TestCase):
    """Unit tests for _find_mme_output_device()."""

    def _make_sd(self, host_apis, devices, default_device=(0, 1)):
        """Build a minimal sounddevice stub."""
        sd = mock.MagicMock()
        sd.query_hostapis.side_effect = lambda idx=None: (
            host_apis[idx] if idx is not None else host_apis
        )
        sd.query_devices.side_effect = lambda idx=None: (
            devices[idx] if idx is not None else devices
        )
        sd.default.device = default_device
        return sd

    def test_returns_none_when_no_mme_host_api(self):
        """If no MME host API is present, return None."""
        host_apis = [{"index": 0, "name": "Windows WASAPI", "default_output_device": 0}]
        devices = [{"index": 0, "name": "Speaker", "hostapi": 0, "max_output_channels": 2}]
        sd = self._make_sd(host_apis, devices)
        result = _find_mme_output_device(sd, 0)
        self.assertIsNone(result)

    def test_returns_exact_name_match_in_mme(self):
        """If the device name matches an MME device exactly, return that index."""
        host_apis = [
            {"index": 0, "name": "MME", "default_output_device": 0},
            {"index": 1, "name": "Windows WASAPI", "default_output_device": 2},
        ]
        devices = [
            {"index": 0, "name": "Speaker", "hostapi": 0, "max_output_channels": 2},
            {"index": 1, "name": "Microphone", "hostapi": 0, "max_output_channels": 0},
            {"index": 2, "name": "Speaker", "hostapi": 1, "max_output_channels": 2},
        ]
        sd = self._make_sd(host_apis, devices)
        # Request device 2 (WASAPI "Speaker") → match MME "Speaker" at index 0
        result = _find_mme_output_device(sd, 2)
        self.assertEqual(result, 0)

    def test_returns_partial_name_match_in_mme(self):
        """If full name doesn't match but a partial (stripped) name does, return it."""
        host_apis = [
            {"index": 0, "name": "Windows WDM-KS", "default_output_device": 0},
            {"index": 1, "name": "MME", "default_output_device": 2},
        ]
        devices = [
            {"index": 0, "name": "USB Audio Device (WDM-KS)", "hostapi": 0, "max_output_channels": 2},
            {"index": 2, "name": "USB Audio Device", "hostapi": 1, "max_output_channels": 2},
        ]
        sd = self._make_sd(host_apis, devices)
        result = _find_mme_output_device(sd, 0)
        self.assertEqual(result, 2)

    def test_returns_mme_default_when_no_name_match(self):
        """When no name match is found, fall back to MME default output."""
        host_apis = [
            {"index": 0, "name": "Windows WDM-KS", "default_output_device": 0},
            {"index": 1, "name": "MME", "default_output_device": 5},
        ]
        devices = [
            {"index": 0, "name": "Foo", "hostapi": 0, "max_output_channels": 2},
            {"index": 5, "name": "Bar", "hostapi": 1, "max_output_channels": 2},
        ]
        sd = self._make_sd(host_apis, devices)
        result = _find_mme_output_device(sd, 0)
        self.assertEqual(result, 5)

    def test_returns_none_when_sd_raises(self):
        """If sounddevice raises unexpectedly, return None (no propagation)."""
        sd = mock.MagicMock()
        sd.query_hostapis.side_effect = RuntimeError("boom")
        result = _find_mme_output_device(sd, 0)
        self.assertIsNone(result)

    def test_returns_none_for_no_mme_output_devices(self):
        """MME host API exists but has no output devices → return None."""
        host_apis = [
            {"index": 0, "name": "MME", "default_output_device": -1},
        ]
        devices = [
            {"index": 0, "name": "Mic", "hostapi": 0, "max_output_channels": 0},
        ]
        sd = self._make_sd(host_apis, devices)
        result = _find_mme_output_device(sd, None)
        self.assertIsNone(result)


# ═══════════════════════════════════════════════════════════════════════════════
# § 17  MME fallback paths in _play_audio
# ═══════════════════════════════════════════════════════════════════════════════

class TestPlayAudioMmeFallback(unittest.TestCase):
    """
    Verify that _play_audio uses MME as a secondary fallback (after WASAPI)
    both proactively (when device is WDM-KS) and reactively (after a
    PortAudioError).
    """

    def _make_coord(self):
        return Ft8TxCoordinator(radio=None)

    def _make_fake_sd(self):
        fake_sd = mock.MagicMock()
        fake_sd.PortAudioError = RuntimeError
        fake_sd.query_devices.return_value = {"default_samplerate": 48_000, "name": "USB Dev"}
        fake_sd.default.device = (0, 0)
        return fake_sd

    def test_proactive_swap_uses_mme_when_no_wasapi_found(self):
        """
        When device is WDM-KS and _find_wasapi_output_device returns None,
        _find_mme_output_device is tried and _stream_play should be called with
        the MME device on the FIRST attempt (no WDM-KS attempt).
        """
        import numpy as np

        coord = self._make_coord()
        audio = np.zeros(100, dtype=np.float32)

        fake_sd = mock.MagicMock()
        fake_sd.PortAudioError = RuntimeError
        fake_sd.query_devices.return_value = {
            "default_samplerate": 48_000,
            "name": "USB Audio CODEC",
            "hostapi": 2,
        }
        fake_sd.query_hostapis.return_value = {"name": "Windows WDM-KS"}
        fake_sd.default.device = (0, 0)

        stream_calls = []

        def _fake_stream(sd_mod, aud, fs, device, *, extra_settings=None):
            stream_calls.append(device)

        with mock.patch("ft8_tx.platform.system", return_value="Windows"), \
             mock.patch("ft8_tx._find_wasapi_output_device", return_value=None), \
             mock.patch("ft8_tx._find_mme_output_device", return_value=6), \
             mock.patch("ft8_tx._stream_play", side_effect=_fake_stream), \
             mock.patch.dict("sys.modules", {"sounddevice": fake_sd}):
            coord._play_audio(audio, device=0)

        self.assertEqual(len(stream_calls), 1,
                         "_stream_play should be called exactly once (MME, no WDM-KS attempt)")
        self.assertEqual(stream_calls[0], 6,
                         "The MME device (6) should be used on the first call")

    def test_reactive_mme_fallback_triggered_when_no_wasapi(self):
        """
        When _stream_play raises PortAudioError on Windows, no WASAPI device is
        found, but an MME device is found, _play_audio retries with the MME
        device and succeeds.
        """
        import numpy as np

        coord = self._make_coord()
        audio = np.zeros(100, dtype=np.float32)
        fake_sd = self._make_fake_sd()

        stream_calls = []

        def _fake_stream(sd_mod, aud, fs, device, *, extra_settings=None):
            stream_calls.append(device)
            if len(stream_calls) == 1:
                raise RuntimeError("WDM-KS error")
            # Second attempt on MME device succeeds

        with mock.patch("ft8_tx.platform.system", return_value="Windows"), \
             mock.patch("ft8_tx._find_wasapi_output_device", return_value=None), \
             mock.patch("ft8_tx._find_mme_output_device", return_value=9), \
             mock.patch("ft8_tx._stream_play", side_effect=_fake_stream), \
             mock.patch.dict("sys.modules", {"sounddevice": fake_sd}):
            coord._play_audio(audio, device=0)

        self.assertEqual(len(stream_calls), 2, "original attempt + MME retry")
        self.assertEqual(stream_calls[1], 9, "Second call should use MME device 9")

    def test_reactive_mme_fallback_raises_if_mme_also_fails(self):
        """
        When both the original device and the MME fallback raise PortAudioError,
        a RuntimeError is raised.
        """
        import numpy as np

        coord = self._make_coord()
        audio = np.zeros(100, dtype=np.float32)
        fake_sd = self._make_fake_sd()

        def _fake_stream(sd_mod, aud, fs, device, *, extra_settings=None):
            raise RuntimeError("audio error")

        with mock.patch("ft8_tx.platform.system", return_value="Windows"), \
             mock.patch("ft8_tx._find_wasapi_output_device", return_value=None), \
             mock.patch("ft8_tx._find_mme_output_device", return_value=9), \
             mock.patch("ft8_tx._stream_play", side_effect=_fake_stream) as mock_stream, \
             mock.patch.dict("sys.modules", {"sounddevice": fake_sd}):
            with self.assertRaises(RuntimeError):
                coord._play_audio(audio, device=0)

        self.assertEqual(mock_stream.call_count, 2,
                         "original attempt + MME fallback attempt")

    def test_mme_fallback_skipped_when_same_as_effective_device(self):
        """
        If _find_mme_output_device returns the same index as effective_device,
        the MME retry is skipped and the 200 ms delay path is taken instead.
        """
        import numpy as np

        coord = self._make_coord()
        audio = np.zeros(100, dtype=np.float32)
        fake_sd = self._make_fake_sd()

        call_count = [0]

        def _fake_stream(sd_mod, aud, fs, device, *, extra_settings=None):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("transient error")
            # Second attempt (delay retry) succeeds

        with mock.patch("ft8_tx.platform.system", return_value="Windows"), \
             mock.patch("ft8_tx._find_wasapi_output_device", return_value=None), \
             mock.patch("ft8_tx._find_mme_output_device", return_value=0), \
             mock.patch("ft8_tx.time.sleep"), \
             mock.patch("ft8_tx._stream_play", side_effect=_fake_stream), \
             mock.patch.dict("sys.modules", {"sounddevice": fake_sd}):
            # device=0, mme returns 0 → same device → skip MME → delay retry
            coord._play_audio(audio, device=0)

        self.assertEqual(call_count[0], 2,
                         "original attempt + delay-retry (MME skipped — same device)")

    def test_wasapi_preferred_over_mme_in_proactive_swap(self):
        """
        When both WASAPI and MME alternatives exist, WASAPI should be used
        (higher priority) and _find_mme_output_device should NOT be called.
        """
        import numpy as np

        coord = self._make_coord()
        audio = np.zeros(100, dtype=np.float32)

        fake_sd = mock.MagicMock()
        fake_sd.PortAudioError = RuntimeError
        fake_sd.query_devices.return_value = {
            "default_samplerate": 48_000,
            "name": "USB Audio",
            "hostapi": 2,
        }
        fake_sd.query_hostapis.return_value = {"name": "Windows WDM-KS"}
        fake_sd.default.device = (0, 0)

        stream_calls = []

        def _fake_stream(sd_mod, aud, fs, device, *, extra_settings=None):
            stream_calls.append(device)

        mme_called = []

        with mock.patch("ft8_tx.platform.system", return_value="Windows"), \
             mock.patch("ft8_tx._find_wasapi_output_device", return_value=5), \
             mock.patch("ft8_tx._find_mme_output_device",
                        side_effect=lambda *a, **kw: mme_called.append(1) or 10), \
             mock.patch("ft8_tx._stream_play", side_effect=_fake_stream), \
             mock.patch.dict("sys.modules", {"sounddevice": fake_sd}):
            coord._play_audio(audio, device=0)

        self.assertEqual(len(mme_called), 0,
                         "_find_mme_output_device must not be called when WASAPI is available")
        self.assertEqual(stream_calls[0], 5, "WASAPI device used (not MME)")


# ═══════════════════════════════════════════════════════════════════════════════
# § 20  WASAPI shared-mode fallback
# ═══════════════════════════════════════════════════════════════════════════════

class TestWasapiSharedModeFallback(unittest.TestCase):
    """
    Verify that _play_audio tries WASAPI shared mode (WasapiSettings(exclusive=False))
    when the first WASAPI attempt fails, before falling through to MME/delay-retry.
    """

    def _make_coord(self):
        return Ft8TxCoordinator(radio=None)

    def test_wasapi_shared_mode_succeeds_after_exclusive_fails(self):
        """
        When the first WASAPI attempt raises PortAudioError, _play_audio should
        retry with WasapiSettings(exclusive=False) (via extra_settings) and succeed.
        Sequence: original → first WASAPI attempt (fails) → WASAPI shared (succeeds).
        """
        import numpy as np

        coord = self._make_coord()
        audio = np.zeros(100, dtype=np.float32)

        fake_sd = mock.MagicMock()
        fake_sd.PortAudioError = RuntimeError
        fake_sd.query_devices.return_value = {"default_samplerate": 48_000, "name": "USB Speaker"}
        fake_sd.default.device = (0, 0)

        # Calls 1 and 2 raise; call 3 (WASAPI shared mode) succeeds.
        call_count = [0]
        stream_calls = []

        def _fake_stream(sd_mod, aud, fs, device, *, extra_settings=None):
            call_count[0] += 1
            stream_calls.append({"device": device, "extra_settings": extra_settings})
            if call_count[0] <= 2:
                raise RuntimeError("stream error")
            # Third attempt (WASAPI shared mode) succeeds

        with mock.patch("ft8_tx.platform.system", return_value="Windows"), \
             mock.patch("ft8_tx._find_wasapi_output_device", return_value=7), \
             mock.patch("ft8_tx._stream_play", side_effect=_fake_stream), \
             mock.patch.dict("sys.modules", {"sounddevice": fake_sd}):
            coord._play_audio(audio, device=0)

        # original (fails) + first WASAPI attempt (fails) + WASAPI shared (succeeds) = 3
        self.assertEqual(call_count[0], 3)
        # The third call must include extra_settings for shared mode
        self.assertIsNotNone(stream_calls[2]["extra_settings"],
                             "Third call must pass extra_settings for WASAPI shared mode")

    def test_wasapi_shared_mode_raises_if_all_fail(self):
        """
        When the first WASAPI attempt AND shared mode both fail, the code falls
        through to the delay-retry.  When that also fails, RuntimeError is raised.
        """
        import numpy as np

        coord = self._make_coord()
        audio = np.zeros(100, dtype=np.float32)

        fake_sd = mock.MagicMock()
        fake_sd.PortAudioError = RuntimeError
        fake_sd.query_devices.return_value = {"default_samplerate": 48_000, "name": "USB Speaker"}
        fake_sd.default.device = (0, 0)

        def _fake_stream(sd_mod, aud, fs, device, *, extra_settings=None):
            raise RuntimeError("stream error")

        with mock.patch("ft8_tx.platform.system", return_value="Windows"), \
             mock.patch("ft8_tx._find_wasapi_output_device", return_value=7), \
             mock.patch("ft8_tx._find_mme_output_device", return_value=None), \
             mock.patch("ft8_tx.time.sleep"), \
             mock.patch("ft8_tx._stream_play", side_effect=_fake_stream) as mock_stream, \
             mock.patch.dict("sys.modules", {"sounddevice": fake_sd}):
            with self.assertRaises(RuntimeError):
                coord._play_audio(audio, device=0)

        # original + first WASAPI attempt + WASAPI shared + delay-retry = 4 total
        self.assertEqual(mock_stream.call_count, 4)

    def test_wasapi_shared_mode_skipped_when_wasapi_settings_unavailable(self):
        """
        If sounddevice lacks WasapiSettings (older version), the shared-mode
        step is skipped.  The code falls through to the delay-retry on the
        original device instead of raising after the first WASAPI attempt.
        """
        import numpy as np

        coord = self._make_coord()
        audio = np.zeros(100, dtype=np.float32)

        fake_sd = mock.MagicMock(spec=["PortAudioError", "default",
                                        "query_devices", "query_hostapis"])
        fake_sd.PortAudioError = RuntimeError
        fake_sd.default.device = (0, 0)

        call_count = [0]

        def _fake_stream(sd_mod, aud, fs, device, *, extra_settings=None):
            call_count[0] += 1
            raise RuntimeError("stream error")

        with mock.patch("ft8_tx.platform.system", return_value="Windows"), \
             mock.patch("ft8_tx._find_wasapi_output_device", return_value=7), \
             mock.patch("ft8_tx._find_mme_output_device", return_value=None), \
             mock.patch("ft8_tx.time.sleep"), \
             mock.patch("ft8_tx._stream_play", side_effect=_fake_stream), \
             mock.patch.dict("sys.modules", {"sounddevice": fake_sd}):
            with self.assertRaises(RuntimeError):
                coord._play_audio(audio, device=0)

        # original + first WASAPI attempt + delay-retry (no shared-mode step) = 3
        self.assertEqual(call_count[0], 3)


# ═══════════════════════════════════════════════════════════════════════════════
# § 21  Bidirectional WASAPI name matching
# ═══════════════════════════════════════════════════════════════════════════════

class TestFindWasapiOutputDeviceBidirectional(unittest.TestCase):
    """
    Verify that _find_wasapi_output_device matches device names bidirectionally:
    the WASAPI name can be a substring of the WDM-KS name, or vice versa.
    """

    def _make_sd(self, host_apis, devices):
        sd = mock.MagicMock()
        sd.query_hostapis.side_effect = lambda idx=None: (
            host_apis[idx] if idx is not None else host_apis
        )
        sd.query_devices.side_effect = lambda idx=None: (
            devices[idx] if idx is not None else devices
        )
        sd.default.device = (0, 1)
        return sd

    def test_wasapi_name_shorter_than_wdm_ks_name(self):
        """
        WASAPI device name is a prefix of the WDM-KS device name.
        e.g. WDM-KS: "USB Audio Device" → WASAPI: "USB Audio"
        The bidirectional check must catch this case.
        """
        from ft8_tx import _find_wasapi_output_device

        host_apis = [
            {"index": 0, "name": "Windows WDM-KS", "default_output_device": 0},
            {"index": 1, "name": "Windows WASAPI", "default_output_device": -1},
        ]
        devices = [
            {"index": 0, "name": "USB Audio Device", "hostapi": 0, "max_output_channels": 2},
            {"index": 1, "name": "USB Audio",         "hostapi": 1, "max_output_channels": 2},
        ]
        sd = self._make_sd(host_apis, devices)
        result = _find_wasapi_output_device(sd, 0)
        self.assertEqual(result, 1,
                         "Should match 'USB Audio' (WASAPI) against 'USB Audio Device' (WDM-KS)")

    def test_wdm_ks_name_shorter_than_wasapi_name(self):
        """
        WDM-KS device name is a prefix of the WASAPI device name.
        e.g. WDM-KS: "USB Audio" → WASAPI: "USB Audio Device (WASAPI)"
        The original one-directional check already handles this; confirm it still works.
        """
        from ft8_tx import _find_wasapi_output_device

        host_apis = [
            {"index": 0, "name": "Windows WDM-KS", "default_output_device": 0},
            {"index": 1, "name": "Windows WASAPI",  "default_output_device": -1},
        ]
        devices = [
            {"index": 0, "name": "USB Audio",                  "hostapi": 0, "max_output_channels": 2},
            {"index": 1, "name": "USB Audio Device (WASAPI)",  "hostapi": 1, "max_output_channels": 2},
        ]
        sd = self._make_sd(host_apis, devices)
        result = _find_wasapi_output_device(sd, 0)
        self.assertEqual(result, 1)


# ═══════════════════════════════════════════════════════════════════════════════
# § 22  Audio pre-generation during ARMED phase
# ═══════════════════════════════════════════════════════════════════════════════

class TestAudioPregeneration(unittest.TestCase):
    """
    Verify that FT8 audio is pre-generated during the ARMED phase
    (before the slot-wait loop) so that encoding errors surface early
    and encoding latency is eliminated from the critical TX window.
    """

    def test_audio_pregenerated_before_ptt_on(self):
        """
        ft8_encode_message must be called before ptt_on() so that an encoding
        failure puts the coordinator into ERROR without keying PTT.
        """
        import ft8_tx as _ft8_tx

        radio = _make_radio()
        timer = _make_timer(seconds_to_next=0.05 + PRE_KEY_S)
        coord = Ft8TxCoordinator(radio=radio, slot_timer=timer)

        original_encode = _ft8_tx.ft8_encode_message
        coord._play_audio = mock.MagicMock()

        call_order: list[str] = []

        with mock.patch.object(
            _ft8_tx, "ft8_encode_message",
            side_effect=lambda *a, **kw: (
                call_order.append("encode")
                or original_encode(*a, **kw)
            ),
        ):
            # Wrap ptt_on to record the call order
            original_ptt_on = coord._ptt_on
            def _ptt_on_recording():
                call_order.append("ptt_on")
                return original_ptt_on()
            coord._ptt_on = _ptt_on_recording

            final = _arm_and_wait(coord, msg="CQ W4ABC EN52")

        self.assertEqual(final, TxState.COMPLETE)
        # Encoding should have been called exactly once
        self.assertIn("encode", call_order)
        self.assertIn("ptt_on", call_order)
        # encode must appear before ptt_on in the call sequence
        self.assertLess(
            call_order.index("encode"),
            call_order.index("ptt_on"),
            "ft8_encode_message must be called before _ptt_on",
        )

    def test_encoding_error_before_slot_transitions_to_error_without_ptt(self):
        """
        If ft8_encode_message raises during the ARMED phase (before the slot
        boundary), the coordinator must transition to ERROR without keying PTT.
        """
        import ft8_tx as _ft8_tx

        radio = _make_radio()
        timer = _make_timer(seconds_to_next=0.05 + PRE_KEY_S)
        coord = Ft8TxCoordinator(radio=radio, slot_timer=timer)
        coord._play_audio = mock.MagicMock()

        ptt_keyed = []
        coord._ptt_on = mock.MagicMock(side_effect=lambda: ptt_keyed.append(True))

        with mock.patch.object(_ft8_tx, "ft8_encode_message",
                                side_effect=RuntimeError("encoding failed")):
            final = _arm_and_wait(coord, msg="CQ W4ABC EN52")

        self.assertEqual(final, TxState.ERROR,
                         "Encoding failure must put coordinator into ERROR")
        self.assertEqual(len(ptt_keyed), 0,
                         "PTT must NOT be keyed when encoding fails")


# ═══════════════════════════════════════════════════════════════════════════════
# § N  _log_audio_diagnostics
# ═══════════════════════════════════════════════════════════════════════════════

class TestLogAudioDiagnostics(unittest.TestCase):
    """
    Verify that _log_audio_diagnostics emits an INFO-level log message that
    includes host-API and device information, and that _play_audio calls it
    before attempting to open any audio stream.
    """

    def _make_fake_sd(self):
        """Return a minimal sounddevice stub sufficient for diagnostics."""
        fake_sd = mock.MagicMock()
        fake_sd.PortAudioError = Exception
        fake_sd.query_hostapis.return_value = [
            {
                "index": 0,
                "name": "MME",
                "default_output_device": 0,
                "device_count": 1,
            }
        ]
        fake_sd.query_devices.return_value = [
            {
                "index": 0,
                "name": "Fake Output",
                "hostapi": 0,
                "max_output_channels": 2,
                "default_samplerate": 48000.0,
            }
        ]
        fake_sd.default.device = (0, 0)
        return fake_sd

    def test_diagnostics_logs_at_info_level(self):
        """_log_audio_diagnostics should emit at least one INFO record."""
        import logging

        fake_sd = self._make_fake_sd()
        with self.assertLogs("ft8_tx", level=logging.INFO) as log_ctx:
            _log_audio_diagnostics(fake_sd, device_index=0)

        # At least one INFO record should have been emitted
        info_records = [r for r in log_ctx.records if r.levelno == logging.INFO]
        self.assertTrue(info_records, "Expected at least one INFO log from _log_audio_diagnostics")

    def test_diagnostics_includes_host_api_info(self):
        """The logged message should mention host API and device names."""
        import logging

        fake_sd = self._make_fake_sd()
        with self.assertLogs("ft8_tx", level=logging.INFO) as log_ctx:
            _log_audio_diagnostics(fake_sd, device_index=0)

        combined = "\n".join(log_ctx.output)
        self.assertIn("MME", combined, "Host API name should appear in diagnostics log")
        self.assertIn("Fake Output", combined, "Device name should appear in diagnostics log")

    def test_diagnostics_includes_selected_device(self):
        """When a valid device_index is given, the log should identify it."""
        import logging

        fake_sd = self._make_fake_sd()
        # Make query_devices(0) return the single-device info when called with an index
        fake_sd.query_devices.side_effect = lambda *args, **kw: (
            {
                "index": 0,
                "name": "Fake Output",
                "hostapi": 0,
                "max_output_channels": 2,
                "default_samplerate": 48000.0,
            }
            if args
            else [
                {
                    "index": 0,
                    "name": "Fake Output",
                    "hostapi": 0,
                    "max_output_channels": 2,
                    "default_samplerate": 48000.0,
                }
            ]
        )
        with self.assertLogs("ft8_tx", level=logging.INFO) as log_ctx:
            _log_audio_diagnostics(fake_sd, device_index=0)

        combined = "\n".join(log_ctx.output)
        self.assertIn("Selected device [0]", combined)

    def test_diagnostics_called_before_stream_open(self):
        """
        _play_audio must call _log_audio_diagnostics before attempting to
        open any output stream.
        """
        import numpy as np
        import logging

        coord = Ft8TxCoordinator(radio=None)
        audio = np.zeros(100, dtype=np.float32)
        fake_sd = self._make_fake_sd()
        call_order: list[str] = []

        def _fake_diagnostics(sd_mod, device_index):
            call_order.append("diagnostics")

        def _fake_stream_play(sd_mod, audio, fs, device, *, extra_settings=None):
            call_order.append("stream_play")

        with mock.patch("ft8_tx._log_audio_diagnostics", side_effect=_fake_diagnostics), \
             mock.patch("ft8_tx._stream_play", side_effect=_fake_stream_play), \
             mock.patch.dict("sys.modules", {"sounddevice": fake_sd}):
            coord._play_audio(audio, device=None)

        self.assertIn("diagnostics", call_order,
                      "_log_audio_diagnostics should be called during _play_audio")
        self.assertIn("stream_play", call_order,
                      "_stream_play should be called during _play_audio")
        self.assertLess(
            call_order.index("diagnostics"),
            call_order.index("stream_play"),
            "_log_audio_diagnostics must be called before _stream_play",
        )

    def test_diagnostics_survives_query_failures(self):
        """
        If sounddevice query methods raise, _log_audio_diagnostics must not
        propagate exceptions (diagnostics must never crash the TX path).
        """
        import logging

        fake_sd = mock.MagicMock()
        fake_sd.query_hostapis.side_effect = RuntimeError("no PortAudio")
        fake_sd.query_devices.side_effect = RuntimeError("no PortAudio")
        fake_sd.default.device = (None, None)

        # Should not raise regardless of whether any log records are emitted.
        # _log_audio_diagnostics is designed to absorb all internal failures so
        # it can never crash the TX path.
        try:
            _log_audio_diagnostics(fake_sd, device_index=None)
        except Exception as exc:  # broad catch is intentional: any raise is a test failure
            self.fail(f"_log_audio_diagnostics raised unexpectedly: {exc}")


if __name__ == "__main__":
    unittest.main()
