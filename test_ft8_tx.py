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

        played_calls = []

        def _fake_play(data, samplerate, **kwargs):
            played_calls.append({"data": data, "samplerate": samplerate})

        fake_sd.play.side_effect = _fake_play
        fake_sd.wait = mock.MagicMock()
        return fake_sd, played_calls

    def test_play_audio_resamples_to_device_native_rate(self):
        """
        When the device's default_samplerate (48 000 Hz) differs from FT8_FS
        (12 000 Hz), _play_audio must resample and call sd.play at the
        native rate so that paInvalidSampleRate is avoided.
        """
        import numpy as np
        from ft8_encode import FT8_FS

        coord = Ft8TxCoordinator(radio=None)
        audio = np.zeros(100, dtype=np.float32)
        fake_sd, played_calls = self._make_fake_sd(device_samplerate=48_000)

        with mock.patch.dict("sys.modules", {"sounddevice": fake_sd}):
            coord._play_audio(audio, device=1)

        self.assertEqual(len(played_calls), 1, "sd.play should be called exactly once")
        self.assertEqual(played_calls[0]["samplerate"], 48_000)
        # 48 000 / 12 000 = ×4 upsampling → output should be ≈ 4 × input length
        self.assertEqual(len(played_calls[0]["data"]), 400)

    def test_play_audio_no_resample_when_rates_match(self):
        """
        When the device's default_samplerate already equals FT8_FS,
        _play_audio should call sd.play with the original audio unchanged.
        """
        import numpy as np
        from ft8_encode import FT8_FS

        coord = Ft8TxCoordinator(radio=None)
        audio = np.ones(200, dtype=np.float32) * 0.5
        fake_sd, played_calls = self._make_fake_sd(device_samplerate=FT8_FS)

        with mock.patch.dict("sys.modules", {"sounddevice": fake_sd}):
            coord._play_audio(audio, device=0)

        self.assertEqual(len(played_calls), 1)
        self.assertEqual(played_calls[0]["samplerate"], FT8_FS)
        self.assertEqual(len(played_calls[0]["data"]), 200)

    def test_play_audio_resample_44100(self):
        """
        Verify resampling works for a 44 100 Hz device (gcd=300 with 12 000).
        """
        import numpy as np
        from ft8_encode import FT8_FS
        from math import gcd

        coord = Ft8TxCoordinator(radio=None)
        audio = np.zeros(120, dtype=np.float32)
        fake_sd, played_calls = self._make_fake_sd(device_samplerate=44_100)

        with mock.patch.dict("sys.modules", {"sounddevice": fake_sd}):
            coord._play_audio(audio, device=2)

        self.assertEqual(len(played_calls), 1)
        self.assertEqual(played_calls[0]["samplerate"], 44_100)
        # Expected length: 120 × 44100 / 12000 = 120 × 3.675 = 441
        g = gcd(44_100, FT8_FS)
        expected_len = 120 * (44_100 // g) // (FT8_FS // g)
        self.assertEqual(len(played_calls[0]["data"]), expected_len)

    def test_play_audio_uses_default_output_when_device_none(self):
        """
        When device is None, _play_audio queries the default output device
        (sd.default.device[1]) to determine the native sample rate.
        """
        import numpy as np
        from ft8_encode import FT8_FS

        coord = Ft8TxCoordinator(radio=None)
        audio = np.zeros(60, dtype=np.float32)
        fake_sd, played_calls = self._make_fake_sd(device_samplerate=48_000)

        with mock.patch.dict("sys.modules", {"sounddevice": fake_sd}):
            coord._play_audio(audio, device=None)

        # Should have queried sd.default.device[1] = 1
        fake_sd.query_devices.assert_called_once_with(1)
        self.assertEqual(played_calls[0]["samplerate"], 48_000)

    def test_play_audio_falls_back_on_query_error(self):
        """
        If sd.query_devices raises, _play_audio falls back to FT8_FS and
        still calls sd.play (no exception propagated from the query failure).
        """
        import numpy as np
        from ft8_encode import FT8_FS

        coord = Ft8TxCoordinator(radio=None)
        audio = np.zeros(50, dtype=np.float32)

        fake_sd = mock.MagicMock()
        fake_sd.PortAudioError = Exception
        fake_sd.query_devices.side_effect = RuntimeError("device query failed")
        fake_sd.default.device = (0, 1)
        played_calls = []

        def _fake_play(data, samplerate, **kwargs):
            played_calls.append(samplerate)

        fake_sd.play.side_effect = _fake_play
        fake_sd.wait = mock.MagicMock()

        with mock.patch.dict("sys.modules", {"sounddevice": fake_sd}):
            coord._play_audio(audio, device=None)

        self.assertEqual(len(played_calls), 1)
        self.assertEqual(played_calls[0], FT8_FS)


# ═══════════════════════════════════════════════════════════════════════════════
# § 11  Exception safety — PTT unkey on error
# ═══════════════════════════════════════════════════════════════════════════════

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

from ft8_tx import _find_wasapi_output_device, _is_wdm_ks_device


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

    def test_wasapi_fallback_triggered_on_windows_portaudio_error(self):
        """
        When sd.play raises PortAudioError on Windows and a WASAPI device is
        found, _play_audio retries with the WASAPI device index and succeeds.
        """
        import numpy as np

        coord = self._make_coord()
        audio = np.zeros(100, dtype=np.float32)

        # Fake sounddevice: first play raises, second succeeds
        fake_sd = mock.MagicMock()
        fake_sd.PortAudioError = RuntimeError
        fake_sd.query_devices.return_value = {"default_samplerate": 48_000, "name": "USB Speaker"}
        fake_sd.default.device = (0, 0)

        play_calls = []

        def _play(data, samplerate, **kwargs):
            play_calls.append(kwargs.get("device"))
            if len(play_calls) == 1:
                raise RuntimeError("WDM-KS error")

        fake_sd.play.side_effect = _play
        fake_sd.wait = mock.MagicMock()

        # _find_wasapi_output_device will be called — patch it to return 3
        with mock.patch("ft8_tx.platform.system", return_value="Windows"), \
             mock.patch("ft8_tx._find_wasapi_output_device", return_value=3), \
             mock.patch.dict("sys.modules", {"sounddevice": fake_sd}):
            coord._play_audio(audio, device=0)

        self.assertEqual(len(play_calls), 2, "sd.play should be called twice (original + fallback)")
        self.assertEqual(play_calls[1], 3, "Second call should use WASAPI device 3")

    def test_no_wasapi_fallback_on_non_windows(self):
        """
        On non-Windows platforms, a PortAudioError should NOT trigger the
        WASAPI fallback; it should propagate as RuntimeError immediately.
        """
        import numpy as np

        coord = self._make_coord()
        audio = np.zeros(100, dtype=np.float32)

        fake_sd = mock.MagicMock()
        fake_sd.PortAudioError = RuntimeError
        fake_sd.query_devices.return_value = {"default_samplerate": 48_000, "name": "Speaker"}
        fake_sd.default.device = (0, 0)
        fake_sd.play.side_effect = RuntimeError("PA error")
        fake_sd.wait = mock.MagicMock()

        with mock.patch("ft8_tx.platform.system", return_value="Linux"), \
             mock.patch.dict("sys.modules", {"sounddevice": fake_sd}):
            with self.assertRaises(RuntimeError):
                coord._play_audio(audio, device=0)

        # play should only be called once — no retry on Linux
        self.assertEqual(fake_sd.play.call_count, 1)

    def test_wasapi_fallback_raises_if_no_wasapi_device_found(self):
        """
        On Windows, if no WASAPI device is found, the original PortAudioError
        is re-raised as RuntimeError without a retry.
        """
        import numpy as np

        coord = self._make_coord()
        audio = np.zeros(100, dtype=np.float32)

        fake_sd = mock.MagicMock()
        fake_sd.PortAudioError = RuntimeError
        fake_sd.query_devices.return_value = {"default_samplerate": 48_000, "name": "Speaker"}
        fake_sd.default.device = (0, 0)
        fake_sd.play.side_effect = RuntimeError("WDM-KS error")
        fake_sd.wait = mock.MagicMock()

        with mock.patch("ft8_tx.platform.system", return_value="Windows"), \
             mock.patch("ft8_tx._find_wasapi_output_device", return_value=None), \
             mock.patch.dict("sys.modules", {"sounddevice": fake_sd}):
            with self.assertRaises(RuntimeError):
                coord._play_audio(audio, device=0)

        self.assertEqual(fake_sd.play.call_count, 1, "No retry when WASAPI device not found")

    def test_wasapi_fallback_raises_if_fallback_also_fails(self):
        """
        On Windows, if WASAPI retry also raises PortAudioError, a RuntimeError
        is raised (not a bare PortAudioError propagation).
        """
        import numpy as np

        coord = self._make_coord()
        audio = np.zeros(100, dtype=np.float32)

        fake_sd = mock.MagicMock()
        fake_sd.PortAudioError = RuntimeError
        fake_sd.query_devices.return_value = {"default_samplerate": 48_000, "name": "Speaker"}
        fake_sd.default.device = (0, 0)
        fake_sd.play.side_effect = RuntimeError("audio error")
        fake_sd.wait = mock.MagicMock()

        with mock.patch("ft8_tx.platform.system", return_value="Windows"), \
             mock.patch("ft8_tx._find_wasapi_output_device", return_value=7), \
             mock.patch.dict("sys.modules", {"sounddevice": fake_sd}):
            with self.assertRaises(RuntimeError):
                coord._play_audio(audio, device=0)

        self.assertEqual(fake_sd.play.call_count, 2, "Both original and fallback attempts made")


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
    entirely rather than discovered on the first failed sd.play() call.
    """

    def _make_coord(self):
        return Ft8TxCoordinator(radio=None)

    def test_proactive_swap_uses_wasapi_device_on_first_play(self):
        """
        When the configured device is WDM-KS on Windows, sd.play should be
        called with the WASAPI device index on the *first* attempt — no
        WDM-KS attempt is made at all.
        """
        import numpy as np

        coord = self._make_coord()
        audio = np.zeros(100, dtype=np.float32)

        fake_sd = mock.MagicMock()
        fake_sd.PortAudioError = RuntimeError
        fake_sd.query_devices.return_value = {
            "default_samplerate": 48_000,
            "name": "USB Audio Device",
            "hostapi": 2,
        }
        fake_sd.query_hostapis.return_value = {"name": "Windows WDM-KS"}
        fake_sd.default.device = (0, 0)

        play_calls = []

        def _play(data, samplerate, **kwargs):
            play_calls.append(kwargs.get("device"))

        fake_sd.play.side_effect = _play
        fake_sd.wait = mock.MagicMock()

        # _is_wdm_ks_device will detect WDM-KS; _find_wasapi_output_device
        # is patched to return 5 (the WASAPI equivalent device index).
        with mock.patch("ft8_tx.platform.system", return_value="Windows"), \
             mock.patch("ft8_tx._find_wasapi_output_device", return_value=5), \
             mock.patch.dict("sys.modules", {"sounddevice": fake_sd}):
            coord._play_audio(audio, device=0)

        self.assertEqual(len(play_calls), 1,
                         "sd.play should be called exactly once (no WDM-KS attempt)")
        self.assertEqual(play_calls[0], 5,
                         "The WASAPI device (5) should be used on the first call")

    def test_proactive_swap_skipped_for_non_wdm_ks_device(self):
        """
        When the configured device is NOT WDM-KS, no proactive swap occurs
        and sd.play is called with the original device index.
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

        play_calls = []

        def _play(data, samplerate, **kwargs):
            play_calls.append(kwargs.get("device"))

        fake_sd.play.side_effect = _play
        fake_sd.wait = mock.MagicMock()

        with mock.patch("ft8_tx.platform.system", return_value="Windows"), \
             mock.patch.dict("sys.modules", {"sounddevice": fake_sd}):
            coord._play_audio(audio, device=2)

        self.assertEqual(len(play_calls), 1)
        self.assertEqual(play_calls[0], 2,
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

        play_calls = []

        def _play(data, samplerate, **kwargs):
            play_calls.append(kwargs.get("device"))

        fake_sd.play.side_effect = _play
        fake_sd.wait = mock.MagicMock()

        wasapi_called = []

        with mock.patch("ft8_tx.platform.system", return_value="Linux"), \
             mock.patch("ft8_tx._is_wdm_ks_device",
                        side_effect=lambda *a, **kw: wasapi_called.append(1) or True), \
             mock.patch.dict("sys.modules", {"sounddevice": fake_sd}):
            coord._play_audio(audio, device=3)

        self.assertEqual(len(wasapi_called), 0,
                         "_is_wdm_ks_device must not be called on non-Windows")
        self.assertEqual(play_calls[0], 3)

    def test_proactive_swap_falls_back_to_original_when_no_wasapi_found(self):
        """
        If _find_wasapi_output_device returns None during the proactive check,
        the original device is used unchanged (no swap, no crash).
        """
        import numpy as np

        coord = self._make_coord()
        audio = np.zeros(100, dtype=np.float32)

        fake_sd = mock.MagicMock()
        fake_sd.PortAudioError = RuntimeError
        fake_sd.query_devices.return_value = {
            "default_samplerate": 48_000,
            "name": "USB Dev",
            "hostapi": 2,
        }
        fake_sd.query_hostapis.return_value = {"name": "Windows WDM-KS"}
        fake_sd.default.device = (0, 0)

        play_calls = []

        def _play(data, samplerate, **kwargs):
            play_calls.append(kwargs.get("device"))

        fake_sd.play.side_effect = _play
        fake_sd.wait = mock.MagicMock()

        # No WASAPI equivalent found
        with mock.patch("ft8_tx.platform.system", return_value="Windows"), \
             mock.patch("ft8_tx._find_wasapi_output_device", return_value=None), \
             mock.patch.dict("sys.modules", {"sounddevice": fake_sd}):
            coord._play_audio(audio, device=4)

        self.assertEqual(len(play_calls), 1)
        self.assertEqual(play_calls[0], 4,
                         "Original device used when no WASAPI equivalent found")

    def test_proactive_swap_no_double_retry_if_wasapi_fails(self):
        """
        If the proactively-selected WASAPI device raises PortAudioError, the
        reactive fallback must NOT retry with the same WASAPI device again
        (it would be a pointless duplicate attempt).
        """
        import numpy as np

        coord = self._make_coord()
        audio = np.zeros(100, dtype=np.float32)

        fake_sd = mock.MagicMock()
        fake_sd.PortAudioError = RuntimeError
        fake_sd.query_devices.return_value = {
            "default_samplerate": 48_000,
            "name": "USB Dev",
            "hostapi": 2,
        }
        fake_sd.query_hostapis.return_value = {"name": "Windows WDM-KS"}
        fake_sd.default.device = (0, 0)
        # All play attempts fail
        fake_sd.play.side_effect = RuntimeError("stream error")
        fake_sd.wait = mock.MagicMock()

        # Both proactive and reactive helpers return the same WASAPI device (5)
        with mock.patch("ft8_tx.platform.system", return_value="Windows"), \
             mock.patch("ft8_tx._find_wasapi_output_device", return_value=5), \
             mock.patch.dict("sys.modules", {"sounddevice": fake_sd}):
            with self.assertRaises(RuntimeError):
                coord._play_audio(audio, device=0)

        # Only one play attempt — the proactive WASAPI attempt; no duplicate retry
        self.assertEqual(fake_sd.play.call_count, 1,
                         "Must not retry with the same WASAPI device twice")

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

        play_calls = []

        def _play(data, samplerate, **kwargs):
            play_calls.append(kwargs.get("device"))

        fake_sd.play.side_effect = _play
        fake_sd.wait = mock.MagicMock()

        # _find_wasapi_output_device should be called with the resolved index (3)
        # and returns WASAPI device 7.
        with mock.patch("ft8_tx.platform.system", return_value="Windows"), \
             mock.patch("ft8_tx._find_wasapi_output_device", return_value=7) as mock_fwod, \
             mock.patch.dict("sys.modules", {"sounddevice": fake_sd}):
            coord._play_audio(audio, device=None)

        # Should have looked up the WASAPI counterpart for the default device (3)
        mock_fwod.assert_called_once_with(fake_sd, 3)
        self.assertEqual(len(play_calls), 1,
                         "sd.play should be called exactly once")
        self.assertEqual(play_calls[0], 7,
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

        play_calls = []

        def _play(data, samplerate, **kwargs):
            play_calls.append(kwargs.get("device"))

        fake_sd.play.side_effect = _play
        fake_sd.wait = mock.MagicMock()

        with mock.patch("ft8_tx.platform.system", return_value="Windows"), \
             mock.patch("ft8_tx._find_wasapi_output_device", return_value=8) as mock_fwod, \
             mock.patch.dict("sys.modules", {"sounddevice": fake_sd}):
            coord._play_audio(audio, device=-1)

        # Resolved index should be sd.default.device[1] == 4
        mock_fwod.assert_called_once_with(fake_sd, 4)
        self.assertEqual(play_calls[0], 8)


if __name__ == "__main__":
    unittest.main()
