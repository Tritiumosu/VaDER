"""
test_ft8_qso_assist.py — Unit tests for the CQ-initiated QSO assist layer.

Tests the ft8_qso.py state machine and parser components that underpin the
automated CQ response pre-fill feature introduced in Milestone 4.

Test groups
-----------
TestReceivedMessageParsing
    — Classify FT8 message strings into their typed fields and flags.

TestFt8QsoManagerCqFlow
    — Full CQ-initiated exchange from start_cq() through the final 73,
      verifying state transitions, message composition, and state machine
      guards.

TestFt8QsoManagerMultipleCallers
    — First-valid-responder semantics: second caller is ignored once the
      state machine has locked onto the first.

TestFt8QsoManagerBuildRecord
    — QsoRecord construction from a completed QSO session.

TestQsoRecord
    — QsoRecord dataclass helpers (ADIF date/time formatting).

Run:  pytest test_ft8_qso_assist.py -v
"""
from __future__ import annotations

import unittest
from datetime import datetime, timezone

from ft8_qso import (
    Ft8QsoManager,
    OperatorConfig,
    QsoRecord,
    QsoState,
    ReceivedMessage,
    compose_cq,
    compose_exchange,
    compose_rrr,
    compose_rr73,
    compose_73,
)
from ft8_ntp import Ft8SlotTimer
import unittest.mock as mock


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_timer() -> Ft8SlotTimer:
    """Return a mock Ft8SlotTimer that always reports 0 s to next slot."""
    t = mock.MagicMock(spec=Ft8SlotTimer)
    t.seconds_to_next_slot.return_value = 0.0
    t.current_slot_parity.return_value = 0
    t.next_slot_utc.return_value = datetime.now(tz=timezone.utc)
    return t


def _make_manager(callsign: str = "W4ABC", grid: str = "EN52") -> Ft8QsoManager:
    op = OperatorConfig(callsign=callsign, grid=grid)
    return Ft8QsoManager(operator=op, slot_timer=_make_timer())


# ═══════════════════════════════════════════════════════════════════════════════
# § 1  ReceivedMessage parsing
# ═══════════════════════════════════════════════════════════════════════════════

class TestReceivedMessageParsing(unittest.TestCase):

    def test_cq_message(self):
        rx = ReceivedMessage("CQ W4ABC EN52")
        self.assertTrue(rx.is_cq)
        self.assertEqual(rx.call1, "CQ")
        self.assertEqual(rx.call2, "W4ABC")
        self.assertEqual(rx.extra, "EN52")

    def test_qrz_treated_as_cq(self):
        rx = ReceivedMessage("QRZ W4ABC EN52")
        self.assertTrue(rx.is_cq)

    def test_snr_report_positive(self):
        rx = ReceivedMessage("W4ABC K9XYZ +05")
        self.assertFalse(rx.is_cq)
        self.assertFalse(rx.is_rrr)
        self.assertFalse(rx.is_rr73)
        self.assertFalse(rx.is_r_report)
        self.assertEqual(rx.snr_db, 5)

    def test_snr_report_negative(self):
        rx = ReceivedMessage("W4ABC K9XYZ -07")
        self.assertEqual(rx.snr_db, -7)
        self.assertFalse(rx.is_r_report)

    def test_r_prefixed_report(self):
        rx = ReceivedMessage("K9XYZ W4ABC R-07")
        self.assertTrue(rx.is_r_report)
        self.assertEqual(rx.snr_db, -7)
        self.assertFalse(rx.is_rrr)
        self.assertFalse(rx.is_rr73)

    def test_rrr_flag(self):
        rx = ReceivedMessage("W4ABC K9XYZ RRR")
        self.assertTrue(rx.is_rrr)
        self.assertFalse(rx.is_rr73)
        self.assertFalse(rx.is_73)

    def test_rr73_flag(self):
        rx = ReceivedMessage("W4ABC K9XYZ RR73")
        self.assertTrue(rx.is_rr73)
        self.assertFalse(rx.is_rrr)
        self.assertFalse(rx.is_73)

    def test_73_flag(self):
        rx = ReceivedMessage("K9XYZ W4ABC 73")
        self.assertTrue(rx.is_73)
        self.assertFalse(rx.is_rrr)
        self.assertFalse(rx.is_rr73)

    def test_is_addressed_to(self):
        rx = ReceivedMessage("W4ABC K9XYZ -05")
        self.assertTrue(rx.is_addressed_to("W4ABC"))
        self.assertTrue(rx.is_addressed_to("w4abc"))   # case insensitive
        self.assertFalse(rx.is_addressed_to("K9XYZ"))

    def test_is_from(self):
        rx = ReceivedMessage("W4ABC K9XYZ -05")
        self.assertTrue(rx.is_from("K9XYZ"))
        self.assertTrue(rx.is_from("k9xyz"))
        self.assertFalse(rx.is_from("W4ABC"))

    def test_empty_message_is_safe(self):
        rx = ReceivedMessage("")
        self.assertFalse(rx.is_cq)
        self.assertFalse(rx.is_rrr)
        self.assertIsNone(rx.snr_db)
        self.assertEqual(rx.call1, "")
        self.assertEqual(rx.call2, "")
        self.assertEqual(rx.extra, "")

    def test_two_field_message(self):
        rx = ReceivedMessage("W4ABC K9XYZ")
        self.assertEqual(rx.call1, "W4ABC")
        self.assertEqual(rx.call2, "K9XYZ")
        self.assertEqual(rx.extra, "")
        self.assertFalse(rx.is_rrr)
        self.assertFalse(rx.is_rr73)

    def test_lowercase_input_normalized(self):
        rx = ReceivedMessage("cq w4abc en52")
        self.assertTrue(rx.is_cq)
        self.assertEqual(rx.call2, "W4ABC")


# ═══════════════════════════════════════════════════════════════════════════════
# § 2  CQ-initiated exchange flow
# ═══════════════════════════════════════════════════════════════════════════════

class TestFt8QsoManagerCqFlow(unittest.TestCase):

    # ── start_cq ──────────────────────────────────────────────────────────

    def test_start_cq_returns_cq_message(self):
        mgr = _make_manager()
        msg = mgr.start_cq()
        self.assertEqual(msg, "CQ W4ABC EN52")

    def test_start_cq_sets_state_to_cq_sent(self):
        mgr = _make_manager()
        mgr.start_cq()
        self.assertEqual(mgr.state, QsoState.CQ_SENT)

    def test_start_cq_queues_tx(self):
        mgr = _make_manager()
        mgr.start_cq()
        self.assertEqual(mgr.get_queued_tx(), "CQ W4ABC EN52")

    def test_start_cq_requires_grid(self):
        op = OperatorConfig(callsign="W4ABC")
        mgr = Ft8QsoManager(operator=op, slot_timer=_make_timer())
        with self.assertRaises(RuntimeError):
            mgr.start_cq()

    # ── advance: CQ_SENT → EXCHANGE_SENT ──────────────────────────────────

    def test_advance_cq_reply_locks_dx_call(self):
        mgr = _make_manager()
        mgr.start_cq()
        mgr.advance("W4ABC K9XYZ -05")
        self.assertEqual(mgr.dx_call, "K9XYZ")

    def test_advance_cq_reply_returns_exchange_message(self):
        mgr = _make_manager()
        mgr.start_cq()
        msg = mgr.advance("W4ABC K9XYZ -05", snr_db=-7)
        self.assertIsNotNone(msg)
        # Exchange format: DX_CALL OUR_CALL R±SNR
        self.assertIn("K9XYZ", msg)
        self.assertIn("W4ABC", msg)
        self.assertTrue(msg.split()[-1].startswith("R"))

    def test_advance_cq_reply_transitions_to_exchange_sent(self):
        mgr = _make_manager()
        mgr.start_cq()
        mgr.advance("W4ABC K9XYZ -05")
        self.assertEqual(mgr.state, QsoState.EXCHANGE_SENT)

    def test_advance_cq_reply_queues_exchange_tx(self):
        mgr = _make_manager()
        mgr.start_cq()
        expected = mgr.advance("W4ABC K9XYZ -05", snr_db=0)
        self.assertEqual(mgr.get_queued_tx(), expected)

    def test_advance_unaddressed_message_ignored_in_cq_state(self):
        mgr = _make_manager()
        mgr.start_cq()
        result = mgr.advance("VK2TIM K9XYZ -05")  # not addressed to us
        self.assertIsNone(result)
        self.assertEqual(mgr.state, QsoState.CQ_SENT)  # state unchanged

    def test_advance_cq_message_not_a_reply(self):
        """Decoding another CQ while in CQ_SENT should be ignored."""
        mgr = _make_manager()
        mgr.start_cq()
        result = mgr.advance("CQ K9XYZ FN43")
        self.assertIsNone(result)
        self.assertEqual(mgr.state, QsoState.CQ_SENT)

    # ── advance: EXCHANGE_SENT → COMPLETE (via RR73) ─────────────────────

    def test_advance_rr73_transitions_to_complete(self):
        mgr = _make_manager()
        mgr.start_cq()
        mgr.advance("W4ABC K9XYZ -05")
        mgr.advance("W4ABC K9XYZ RR73")
        self.assertEqual(mgr.state, QsoState.COMPLETE)

    def test_advance_rr73_returns_73_message(self):
        mgr = _make_manager()
        mgr.start_cq()
        mgr.advance("W4ABC K9XYZ -05")
        msg = mgr.advance("W4ABC K9XYZ RR73")
        self.assertIsNotNone(msg)
        self.assertIn("73", msg)
        self.assertIn("K9XYZ", msg)
        self.assertIn("W4ABC", msg)

    def test_advance_rrr_also_transitions_to_complete(self):
        mgr = _make_manager()
        mgr.start_cq()
        mgr.advance("W4ABC K9XYZ -05")
        msg = mgr.advance("W4ABC K9XYZ RRR")
        self.assertEqual(mgr.state, QsoState.COMPLETE)
        self.assertIsNotNone(msg)

    def test_advance_after_complete_returns_none(self):
        """Once COMPLETE, further advance() calls return None."""
        mgr = _make_manager()
        mgr.start_cq()
        mgr.advance("W4ABC K9XYZ -05")
        mgr.advance("W4ABC K9XYZ RR73")
        self.assertEqual(mgr.state, QsoState.COMPLETE)
        result = mgr.advance("W4ABC K9XYZ RR73")  # duplicate RR73
        self.assertIsNone(result)

    # ── advance: wrong sender ignored in EXCHANGE_SENT ────────────────────

    def test_advance_wrong_sender_ignored_in_exchange_sent(self):
        mgr = _make_manager()
        mgr.start_cq()
        mgr.advance("W4ABC K9XYZ -05")  # K9XYZ locked as dx
        # Another station sends RR73 — should be ignored
        result = mgr.advance("W4ABC VK2TIM RR73")
        self.assertIsNone(result)
        self.assertEqual(mgr.state, QsoState.EXCHANGE_SENT)

    def test_advance_r_report_not_rrr_rr73_ignored(self):
        """An R-prefixed report in EXCHANGE_SENT does not advance state."""
        mgr = _make_manager()
        mgr.start_cq()
        mgr.advance("W4ABC K9XYZ -05")
        result = mgr.advance("W4ABC K9XYZ R-07")
        self.assertIsNone(result)
        self.assertEqual(mgr.state, QsoState.EXCHANGE_SENT)

    # ── is_active / is_complete ───────────────────────────────────────────

    def test_is_active_in_cq_sent(self):
        mgr = _make_manager()
        mgr.start_cq()
        self.assertTrue(mgr.is_active)

    def test_is_active_false_after_complete(self):
        mgr = _make_manager()
        mgr.start_cq()
        mgr.advance("W4ABC K9XYZ -05")
        mgr.advance("W4ABC K9XYZ RR73")
        self.assertFalse(mgr.is_active)

    # ── abort / reset ─────────────────────────────────────────────────────

    def test_abort_sets_aborted_state(self):
        mgr = _make_manager()
        mgr.start_cq()
        mgr.abort()
        self.assertEqual(mgr.state, QsoState.ABORTED)

    def test_abort_clears_queued_tx(self):
        mgr = _make_manager()
        mgr.start_cq()
        mgr.abort()
        self.assertIsNone(mgr.get_queued_tx())

    def test_abort_clears_dx_call(self):
        mgr = _make_manager()
        mgr.start_cq()
        mgr.advance("W4ABC K9XYZ -05")
        mgr.abort()
        self.assertIsNone(mgr.dx_call)

    def test_reset_returns_to_idle(self):
        mgr = _make_manager()
        mgr.start_cq()
        mgr.advance("W4ABC K9XYZ -05")
        mgr.reset()
        self.assertEqual(mgr.state, QsoState.IDLE)
        self.assertIsNone(mgr.get_queued_tx())


# ═══════════════════════════════════════════════════════════════════════════════
# § 3  Multiple callers — first valid responder wins
# ═══════════════════════════════════════════════════════════════════════════════

class TestFt8QsoManagerMultipleCallers(unittest.TestCase):

    def test_first_caller_locks_in(self):
        """The first station to reply to the CQ becomes the DX contact."""
        mgr = _make_manager()
        mgr.start_cq()
        mgr.advance("W4ABC K9XYZ -05")  # first caller
        self.assertEqual(mgr.dx_call, "K9XYZ")
        self.assertEqual(mgr.state, QsoState.EXCHANGE_SENT)

    def test_second_caller_ignored_after_first(self):
        """A second caller's reply is silently ignored."""
        mgr = _make_manager()
        mgr.start_cq()
        mgr.advance("W4ABC K9XYZ -05")  # first caller
        # Second caller — should NOT change dx_call or state
        result = mgr.advance("W4ABC VK2TIM -03")
        self.assertIsNone(result)
        self.assertEqual(mgr.dx_call, "K9XYZ")
        self.assertEqual(mgr.state, QsoState.EXCHANGE_SENT)

    def test_repeated_reply_from_first_caller_ignored(self):
        """Same caller retransmitting their reply is a no-op after lock-in."""
        mgr = _make_manager()
        mgr.start_cq()
        first = mgr.advance("W4ABC K9XYZ -05")
        second = mgr.advance("W4ABC K9XYZ -05")  # duplicate decode
        self.assertIsNone(second)
        self.assertEqual(mgr.state, QsoState.EXCHANGE_SENT)
        self.assertEqual(mgr.get_queued_tx(), first)

    def test_multiple_rrr_from_dx_only_completes_once(self):
        """Two copies of RR73 from the DX station complete the QSO once."""
        mgr = _make_manager()
        mgr.start_cq()
        mgr.advance("W4ABC K9XYZ -05")
        mgr.advance("W4ABC K9XYZ RR73")   # completes
        self.assertEqual(mgr.state, QsoState.COMPLETE)
        result = mgr.advance("W4ABC K9XYZ RR73")  # duplicate
        self.assertIsNone(result)
        self.assertEqual(mgr.state, QsoState.COMPLETE)


# ═══════════════════════════════════════════════════════════════════════════════
# § 4  build_record()
# ═══════════════════════════════════════════════════════════════════════════════

class TestFt8QsoManagerBuildRecord(unittest.TestCase):

    def _complete_qso(self) -> Ft8QsoManager:
        mgr = _make_manager("W4ABC", "EN52")
        mgr.start_cq()
        mgr.advance("W4ABC K9XYZ -05", snr_db=0)
        mgr.advance("W4ABC K9XYZ RR73")
        return mgr

    def test_build_record_on_complete_qso(self):
        mgr = self._complete_qso()
        rec = mgr.build_record(freq_mhz=14.074, band="20m")
        self.assertEqual(rec.our_call, "W4ABC")
        self.assertEqual(rec.dx_call, "K9XYZ")
        self.assertEqual(rec.freq_mhz, 14.074)
        self.assertEqual(rec.band, "20m")
        self.assertEqual(rec.mode, "FT8")
        self.assertEqual(rec.initiated, "CQ")

    def test_build_record_records_received_snr(self):
        mgr = _make_manager()
        mgr.start_cq()
        mgr.advance("W4ABC K9XYZ -05", snr_db=0)
        mgr.advance("W4ABC K9XYZ RR73")
        rec = mgr.build_record()
        # K9XYZ reported -05 to us
        self.assertEqual(rec.rst_rcvd, "-05")

    def test_build_record_raises_if_not_complete(self):
        mgr = _make_manager()
        mgr.start_cq()
        with self.assertRaises(RuntimeError):
            mgr.build_record()

    def test_build_record_raises_without_dx_call(self):
        mgr = _make_manager()
        # Manually force COMPLETE without a dx_call (edge case)
        mgr.state = QsoState.COMPLETE
        with self.assertRaises(RuntimeError):
            mgr.build_record()

    def test_build_record_time_on_is_utc(self):
        mgr = self._complete_qso()
        rec = mgr.build_record()
        self.assertIsNotNone(rec.time_on.tzinfo)
        self.assertEqual(rec.time_on.tzinfo, timezone.utc)

    def test_build_record_time_on_captured_at_session_start(self):
        """time_on must reflect when start_cq() was called, not when build_record() was called."""
        import time as _time
        mgr = _make_manager()
        t_before = datetime.now(timezone.utc)
        mgr.start_cq()
        t_after_start = datetime.now(timezone.utc)

        mgr.advance("W4ABC K9XYZ -05", snr_db=0)
        mgr.advance("W4ABC K9XYZ RR73")

        # Small sleep so build_record() wall time is measurably later
        _time.sleep(0.05)
        rec = mgr.build_record()

        # time_on must be between t_before and t_after_start (the session window),
        # not close to the build_record() call time (which is ~50 ms later).
        self.assertGreaterEqual(rec.time_on, t_before)
        self.assertLessEqual(rec.time_on, t_after_start)

    def test_build_record_time_on_reset_after_reset(self):
        """After reset(), a new session captures a fresh time_on."""
        import time as _time
        mgr = _make_manager()
        mgr.start_cq()
        first_time_on = mgr._time_on_utc

        mgr.reset()
        _time.sleep(0.05)
        mgr.start_cq()
        second_time_on = mgr._time_on_utc

        self.assertGreater(second_time_on, first_time_on)


# ═══════════════════════════════════════════════════════════════════════════════
# § 5  QsoRecord helpers
# ═══════════════════════════════════════════════════════════════════════════════

class TestQsoRecord(unittest.TestCase):

    def _make_record(self, **kw) -> QsoRecord:
        defaults = dict(
            our_call="W4ABC",
            dx_call="K9XYZ",
            time_on=datetime(2025, 7, 4, 18, 30, 0, tzinfo=timezone.utc),
        )
        defaults.update(kw)
        return QsoRecord(**defaults)

    def test_adif_date(self):
        rec = self._make_record()
        self.assertEqual(rec.adif_date(), "20250704")

    def test_adif_time(self):
        rec = self._make_record()
        self.assertEqual(rec.adif_time(), "183000")

    def test_default_mode_is_ft8(self):
        rec = self._make_record()
        self.assertEqual(rec.mode, "FT8")

    def test_default_freq_is_zero(self):
        rec = self._make_record()
        self.assertEqual(rec.freq_mhz, 0.0)

    def test_default_initiated_is_cq(self):
        rec = self._make_record()
        self.assertEqual(rec.initiated, "CQ")

    def test_custom_rst(self):
        rec = self._make_record(rst_sent="-03", rst_rcvd="+01")
        self.assertEqual(rec.rst_sent, "-03")
        self.assertEqual(rec.rst_rcvd, "+01")

    def test_record_is_immutable(self):
        """QsoRecord must be frozen — mutation must raise FrozenInstanceError."""
        from dataclasses import FrozenInstanceError
        rec = self._make_record()
        with self.assertRaises(FrozenInstanceError):
            rec.dx_call = "W1AW"  # type: ignore[misc]


if __name__ == "__main__":
    unittest.main(verbosity=2)
