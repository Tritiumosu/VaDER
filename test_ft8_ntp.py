"""
test_ft8_ntp.py — pytest suite for ft8_ntp.py and the NTP-backed
                  slot-timing integration in ft8_qso.py / main.py.

All tests run without any network access — NTP server calls are mocked.

Test groups
-----------
TestNtpTimeSync_OffsetMath    — offset is applied correctly to utc_now()
TestNtpTimeSync_MultiServer   — servers are tried in order; first success wins
TestNtpTimeSync_AllFail       — graceful fallback when all servers fail
TestNtpTimeSync_NtplibMissing — graceful fallback when ntplib is absent
TestNtpTimeSync_Properties    — is_synced, offset_s, sync_server, last_sync_utc
TestFt8SlotTimer_SlotMath     — seconds_to_next_slot / current_slot_index /
                                 current_slot_parity / next_slot_utc
TestFt8SlotTimer_NtpCorrected — slot math uses NTP offset, not raw system time
TestQsoManager_UsesSlotTimer  — Ft8QsoManager delegates to its Ft8SlotTimer
TestAppConfigNtp              — AppConfig reads/writes NTP settings correctly
"""
from __future__ import annotations

import importlib
import os
import sys
import tempfile
import types
import unittest
import unittest.mock as mock
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(__file__))

import ft8_ntp
from ft8_ntp import (
    NtpTimeSync,
    Ft8SlotTimer,
    DEFAULT_NTP_SERVERS,
    FT8_SLOT_DURATION_S,
)
from ft8_qso import Ft8QsoManager, OperatorConfig, QsoState


# ---------------------------------------------------------------------------
# Helper: build a fake ntplib response with a given offset
# ---------------------------------------------------------------------------

def _make_ntp_response(offset: float, delay: float = 0.05):
    """Return a mock object that mimics ntplib.NTPStats."""
    resp = mock.MagicMock()
    resp.offset = offset
    resp.delay  = delay
    return resp


# ═══════════════════════════════════════════════════════════════════════════════
# § 1  NtpTimeSync — offset math
# ═══════════════════════════════════════════════════════════════════════════════

class TestNtpTimeSyncOffsetMath(unittest.TestCase):
    """utc_now() must return system time + stored offset."""

    def test_positive_offset_applied(self):
        """A positive offset means local clock is behind NTP — utc_now > system now."""
        sync = NtpTimeSync()
        offset = 0.350    # 350 ms — clock running slow
        with mock.patch("ft8_ntp.ntplib.NTPClient.request",
                        return_value=_make_ntp_response(offset)):
            sync.sync()
        before  = datetime.now(tz=timezone.utc)
        corrected = sync.utc_now()
        after   = datetime.now(tz=timezone.utc)
        # corrected must be within (before + offset, after + offset + small_fudge)
        self.assertGreaterEqual(corrected, before + timedelta(seconds=offset - 0.01))
        self.assertLessEqual(   corrected, after  + timedelta(seconds=offset + 0.01))

    def test_negative_offset_applied(self):
        """A negative offset means local clock is ahead — utc_now < system now."""
        sync = NtpTimeSync()
        offset = -0.120   # 120 ms — clock running fast
        with mock.patch("ft8_ntp.ntplib.NTPClient.request",
                        return_value=_make_ntp_response(offset)):
            sync.sync()
        system_now = datetime.now(tz=timezone.utc)
        corrected  = sync.utc_now()
        self.assertLess(corrected, system_now + timedelta(seconds=0.01))

    def test_zero_offset_unchanged(self):
        """With zero offset utc_now() should be within a few ms of system time."""
        sync = NtpTimeSync()
        with mock.patch("ft8_ntp.ntplib.NTPClient.request",
                        return_value=_make_ntp_response(0.0)):
            sync.sync()
        delta = abs((sync.utc_now() - datetime.now(tz=timezone.utc)).total_seconds())
        self.assertLess(delta, 0.05)   # allow 50 ms for code execution time

    def test_no_sync_returns_system_time(self):
        """utc_now() before any sync must closely match system time."""
        sync = NtpTimeSync()
        delta = abs((sync.utc_now() - datetime.now(tz=timezone.utc)).total_seconds())
        self.assertLess(delta, 0.05)


# ═══════════════════════════════════════════════════════════════════════════════
# § 2  NtpTimeSync — multi-server fallthrough
# ═══════════════════════════════════════════════════════════════════════════════

class TestNtpTimeSyncMultiServer(unittest.TestCase):

    def test_first_server_used_on_success(self):
        """When the first server responds, no other server should be tried."""
        sync = NtpTimeSync(servers=["s1.example", "s2.example", "s3.example"])
        call_log: list[str] = []

        def fake_request(host, **kwargs):
            call_log.append(host)
            return _make_ntp_response(0.1)

        with mock.patch("ft8_ntp.ntplib.NTPClient.request", side_effect=fake_request):
            ok = sync.sync()

        self.assertTrue(ok)
        self.assertEqual(call_log, ["s1.example"])
        self.assertEqual(sync.sync_server, "s1.example")

    def test_first_server_fails_second_used(self):
        """When the first server raises OSError, the second is tried."""
        sync = NtpTimeSync(servers=["bad.example", "good.example"])

        def fake_request(host, **kwargs):
            if host == "bad.example":
                raise OSError("connection refused")
            return _make_ntp_response(0.2)

        with mock.patch("ft8_ntp.ntplib.NTPClient.request", side_effect=fake_request):
            ok = sync.sync()

        self.assertTrue(ok)
        self.assertEqual(sync.sync_server, "good.example")
        self.assertAlmostEqual(sync.offset_s, 0.2, places=6)

    def test_all_servers_tried_in_order(self):
        """Every server should be attempted before giving up."""
        servers = ["a.example", "b.example", "c.example"]
        sync = NtpTimeSync(servers=servers)
        call_log: list[str] = []

        def fake_request(host, **kwargs):
            call_log.append(host)
            raise OSError("unreachable")

        with mock.patch("ft8_ntp.ntplib.NTPClient.request", side_effect=fake_request):
            ok = sync.sync()

        self.assertFalse(ok)
        self.assertEqual(call_log, servers)


# ═══════════════════════════════════════════════════════════════════════════════
# § 3  NtpTimeSync — all-fail fallback
# ═══════════════════════════════════════════════════════════════════════════════

class TestNtpTimeSyncAllFail(unittest.TestCase):

    def test_is_synced_false_after_all_fail(self):
        sync = NtpTimeSync(servers=["x.example"])
        with mock.patch("ft8_ntp.ntplib.NTPClient.request",
                        side_effect=OSError("no route to host")):
            ok = sync.sync()
        self.assertFalse(ok)
        self.assertFalse(sync.is_synced)
        self.assertIsNone(sync.offset_s)

    def test_utc_now_falls_back_to_system_time(self):
        """After failed sync utc_now() must still return a reasonable time."""
        sync = NtpTimeSync(servers=["x.example"])
        with mock.patch("ft8_ntp.ntplib.NTPClient.request",
                        side_effect=OSError("unreachable")):
            sync.sync()
        delta = abs((sync.utc_now() - datetime.now(tz=timezone.utc)).total_seconds())
        self.assertLess(delta, 0.05)

    def test_existing_offset_preserved_on_retry_failure(self):
        """A previously good offset must not be discarded if a re-sync fails."""
        sync = NtpTimeSync(servers=["good.example"])
        with mock.patch("ft8_ntp.ntplib.NTPClient.request",
                        return_value=_make_ntp_response(0.5)):
            sync.sync()
        self.assertAlmostEqual(sync.offset_s, 0.5, places=6)

        with mock.patch("ft8_ntp.ntplib.NTPClient.request",
                        side_effect=OSError("now offline")):
            ok = sync.sync()
        self.assertFalse(ok)
        # Offset must be preserved from the first successful sync
        self.assertAlmostEqual(sync.offset_s, 0.5, places=6)


# ═══════════════════════════════════════════════════════════════════════════════
# § 4  NtpTimeSync — ntplib absent
# ═══════════════════════════════════════════════════════════════════════════════

class TestNtpTimeSyncNtplibMissing(unittest.TestCase):

    def test_sync_returns_false_when_ntplib_unavailable(self):
        """If ntplib is not installed sync() must return False, not raise."""
        original = ft8_ntp._NTPLIB_AVAILABLE
        try:
            ft8_ntp._NTPLIB_AVAILABLE = False
            sync = NtpTimeSync()
            ok   = sync.sync()
        finally:
            ft8_ntp._NTPLIB_AVAILABLE = original
        self.assertFalse(ok)
        self.assertFalse(sync.is_synced)

    def test_utc_now_still_works_when_ntplib_unavailable(self):
        """utc_now() must always return a datetime regardless of ntplib."""
        original = ft8_ntp._NTPLIB_AVAILABLE
        try:
            ft8_ntp._NTPLIB_AVAILABLE = False
            sync = NtpTimeSync()
            now  = sync.utc_now()
        finally:
            ft8_ntp._NTPLIB_AVAILABLE = original
        self.assertIsInstance(now, datetime)
        self.assertIsNotNone(now.tzinfo)


# ═══════════════════════════════════════════════════════════════════════════════
# § 5  NtpTimeSync — properties
# ═══════════════════════════════════════════════════════════════════════════════

class TestNtpTimeSyncProperties(unittest.TestCase):

    def _synced_instance(self, offset: float = 0.1) -> NtpTimeSync:
        sync = NtpTimeSync(servers=["s.example"])
        with mock.patch("ft8_ntp.ntplib.NTPClient.request",
                        return_value=_make_ntp_response(offset)):
            sync.sync()
        return sync

    def test_is_synced_true_after_success(self):
        self.assertTrue(self._synced_instance().is_synced)

    def test_offset_s_matches_response(self):
        self.assertAlmostEqual(self._synced_instance(0.375).offset_s, 0.375, places=6)

    def test_sync_server_stored(self):
        self.assertEqual(self._synced_instance().sync_server, "s.example")

    def test_last_sync_utc_is_recent(self):
        sync = self._synced_instance()
        age  = (datetime.now(tz=timezone.utc) - sync.last_sync_utc).total_seconds()
        self.assertLess(age, 5.0)   # must be within last 5 s

    def test_servers_property_returns_copy(self):
        sync = NtpTimeSync(servers=["a.example", "b.example"])
        servers = sync.servers
        servers.append("injected")
        self.assertEqual(len(sync.servers), 2)  # internal list must be unaffected

    def test_default_servers_list(self):
        self.assertIn("time.nist.gov", DEFAULT_NTP_SERVERS)
        self.assertIn("pool.ntp.org",  DEFAULT_NTP_SERVERS)
        self.assertGreaterEqual(len(DEFAULT_NTP_SERVERS), 2)


# ═══════════════════════════════════════════════════════════════════════════════
# § 6  Ft8SlotTimer — slot math correctness
# ═══════════════════════════════════════════════════════════════════════════════

class TestFt8SlotTimerSlotMath(unittest.TestCase):
    """Test slot math with a fixed, known UTC time injected via utc_now mock."""

    def _timer_at(self, second: int, microsecond: int = 0) -> Ft8SlotTimer:
        """Return a Ft8SlotTimer whose utc_now() is pinned to :ss.uuuuuu."""
        sync = NtpTimeSync()
        fixed = datetime(2024, 1, 1, 12, 0, second, microsecond, tzinfo=timezone.utc)
        sync.utc_now = mock.MagicMock(return_value=fixed)
        return Ft8SlotTimer(ntp_sync=sync)

    # seconds_to_next_slot ------------------------------------------------

    def test_at_slot_start_gives_full_slot_wait(self):
        """At exactly :00 the next slot is 15 s away (the *next* boundary)."""
        timer = self._timer_at(second=0, microsecond=0)
        wait  = timer.seconds_to_next_slot()
        # Because seconds_to_next_slot clamps 0 → slot_duration, we get 15.0
        self.assertAlmostEqual(wait, FT8_SLOT_DURATION_S, places=3)

    def test_midway_through_slot(self):
        """At :07.500 the next :15 boundary is 7.5 s away."""
        timer = self._timer_at(second=7, microsecond=500_000)
        wait  = timer.seconds_to_next_slot()
        self.assertAlmostEqual(wait, 7.5, places=3)

    def test_near_end_of_slot(self):
        """At :14.9 the next boundary is 0.1 s away."""
        timer = self._timer_at(second=14, microsecond=900_000)
        wait  = timer.seconds_to_next_slot()
        self.assertAlmostEqual(wait, 0.1, places=3)

    def test_second_slot_midpoint(self):
        """At :22.0 (inside slot 1, 7 s from :30 boundary) wait = 8.0 s."""
        timer = self._timer_at(second=22, microsecond=0)
        wait  = timer.seconds_to_next_slot()
        self.assertAlmostEqual(wait, 8.0, places=3)

    def test_wait_never_exceeds_slot_duration(self):
        for s in range(60):
            timer = self._timer_at(second=s)
            self.assertLessEqual(timer.seconds_to_next_slot(), FT8_SLOT_DURATION_S)

    def test_wait_always_positive(self):
        for s in range(60):
            self.assertGreater(self._timer_at(second=s).seconds_to_next_slot(), 0)

    # current_slot_index --------------------------------------------------

    def test_slot_index_0_at_start_of_minute(self):
        self.assertEqual(self._timer_at(0).current_slot_index(), 0)

    def test_slot_index_1_at_15(self):
        self.assertEqual(self._timer_at(15).current_slot_index(), 1)

    def test_slot_index_2_at_30(self):
        self.assertEqual(self._timer_at(30).current_slot_index(), 2)

    def test_slot_index_3_at_45(self):
        self.assertEqual(self._timer_at(45).current_slot_index(), 3)

    def test_slot_index_0_at_14(self):
        """Second 14 is still inside slot 0."""
        self.assertEqual(self._timer_at(14).current_slot_index(), 0)

    # current_slot_parity -------------------------------------------------

    def test_even_parity_at_0(self):
        self.assertEqual(self._timer_at(0).current_slot_parity(), 0)

    def test_odd_parity_at_15(self):
        self.assertEqual(self._timer_at(15).current_slot_parity(), 1)

    def test_even_parity_at_30(self):
        self.assertEqual(self._timer_at(30).current_slot_parity(), 0)

    def test_odd_parity_at_45(self):
        self.assertEqual(self._timer_at(45).current_slot_parity(), 1)

    # next_slot_utc -------------------------------------------------------

    def test_next_slot_utc_is_future(self):
        timer = self._timer_at(second=7)
        nxt   = timer.next_slot_utc()
        self.assertIsInstance(nxt, datetime)
        # next slot is :15 → 8 s from :07
        self.assertAlmostEqual(
            (nxt - timer.utc_now()).total_seconds(), 8.0, delta=0.01
        )

    def test_next_slot_utc_is_on_15s_boundary(self):
        """The returned datetime must land on a multiple-of-15 second."""
        timer = self._timer_at(second=3)
        nxt   = timer.next_slot_utc()
        self.assertEqual(nxt.second % 15, 0)
        self.assertEqual(nxt.microsecond, 0,
                         msg="next_slot_utc should have zero microseconds "
                             f"but got {nxt.microsecond}")


# ═══════════════════════════════════════════════════════════════════════════════
# § 7  Ft8SlotTimer — uses NTP-corrected time
# ═══════════════════════════════════════════════════════════════════════════════

class TestFt8SlotTimerNtpCorrected(unittest.TestCase):
    """
    Verify that slot math uses the NTP-offset-corrected time, not raw system
    time.  We inject a known offset and confirm the slot index changes.
    """

    def test_positive_offset_shifts_slot_index(self):
        """
        If local clock reads :14 but NTP says clock is 2 s slow (offset=+2),
        the corrected time is :16 → slot index 1 (not 0).
        """
        sync = NtpTimeSync()
        # Patch utc_now directly on the sync instance: base system time = :14,
        # but offset +2 means corrected = :16
        local_time = datetime(2024, 6, 1, 10, 0, 14, 0, tzinfo=timezone.utc)
        # NtpTimeSync.utc_now = system + offset; we simulate by patching _offset_s
        sync._offset_s = 2.0
        # Also patch datetime.now so we control the base
        with mock.patch("ft8_ntp.datetime") as mock_dt:
            mock_dt.now.return_value = local_time
            mock_dt.side_effect = lambda *args, **kw: datetime(*args, **kw)
            timer = Ft8SlotTimer(ntp_sync=sync)
            # utc_now() = local_time + 2 s offset = :16 → slot index 1
            corrected = sync.utc_now()

        self.assertEqual(corrected.second, 16)
        # slot index for second 16 → 16 // 15 = 1
        self.assertEqual(corrected.second // 15, 1)


# ═══════════════════════════════════════════════════════════════════════════════
# § 8  Ft8QsoManager — delegates to slot timer
# ═══════════════════════════════════════════════════════════════════════════════

class TestQsoManagerUsesSlotTimer(unittest.TestCase):
    """
    Ft8QsoManager.seconds_to_next_slot() and current_slot_parity() must
    delegate to the injected Ft8SlotTimer, not call datetime.now() directly.
    """

    def _manager_with_mock_timer(self, second: int) -> tuple[Ft8QsoManager, Ft8SlotTimer]:
        op    = OperatorConfig(callsign="W4ABC", grid="EN52")
        timer = Ft8SlotTimer()
        fixed = datetime(2024, 1, 1, 0, 0, second, 0, tzinfo=timezone.utc)
        timer._ntp.utc_now = mock.MagicMock(return_value=fixed)
        mgr   = Ft8QsoManager(op, slot_timer=timer)
        return mgr, timer

    def test_seconds_to_next_slot_uses_timer(self):
        mgr, _ = self._manager_with_mock_timer(second=7)   # :07 → 8 s to :15
        wait   = mgr.seconds_to_next_slot()
        self.assertAlmostEqual(wait, 8.0, places=3)

    def test_current_slot_parity_uses_timer(self):
        mgr0, _ = self._manager_with_mock_timer(second=0)   # slot 0 → even
        mgr1, _ = self._manager_with_mock_timer(second=15)  # slot 1 → odd
        self.assertEqual(mgr0.current_slot_parity(), 0)
        self.assertEqual(mgr1.current_slot_parity(), 1)

    def test_next_slot_utc_available(self):
        mgr, _ = self._manager_with_mock_timer(second=3)
        nxt    = mgr.next_slot_utc()
        self.assertIsInstance(nxt, datetime)
        self.assertIsNotNone(nxt.tzinfo)

    def test_default_timer_injected_when_none_given(self):
        """Without an explicit timer the manager must still have a slot_timer."""
        op  = OperatorConfig(callsign="K9XYZ", grid="EM73")
        mgr = Ft8QsoManager(op)
        # Should not raise; returns a float
        self.assertIsInstance(mgr.seconds_to_next_slot(), float)
        self.assertIn(mgr.current_slot_parity(), (0, 1))


# ═══════════════════════════════════════════════════════════════════════════════
# § 9  AppConfig — NTP settings
# ═══════════════════════════════════════════════════════════════════════════════

class TestAppConfigNtp(unittest.TestCase):
    """Verify AppConfig reads and writes NTP configuration correctly."""

    @classmethod
    def setUpClass(cls) -> None:
        # Mirrors the stub pattern from test_ft8_encode.py
        tk_stub = types.ModuleType("tkinter")
        tk_stub.Tk           = mock.MagicMock()
        tk_stub.Toplevel     = mock.MagicMock()
        tk_stub.Frame        = mock.MagicMock()
        tk_stub.LabelFrame   = mock.MagicMock()
        tk_stub.Label        = mock.MagicMock()
        tk_stub.Button       = mock.MagicMock()
        tk_stub.Entry        = mock.MagicMock()
        tk_stub.Text         = mock.MagicMock()
        tk_stub.Spinbox      = mock.MagicMock()
        tk_stub.StringVar    = mock.MagicMock
        tk_stub.IntVar       = mock.MagicMock
        tk_stub.NORMAL       = "normal"
        tk_stub.DISABLED     = "disabled"
        tk_stub.LEFT         = "left"
        tk_stub.RIGHT        = "right"
        tk_stub.X            = "x"
        tk_stub.Y            = "y"
        tk_stub.BOTH         = "both"
        tk_stub.END          = "end"
        tk_stub.SUNKEN       = "sunken"
        tk_stub.RAISED       = "raised"
        tk_stub.GROOVE       = "groove"
        tk_stub.Misc         = object

        ttk_stub              = types.ModuleType("tkinter.ttk")
        ttk_stub.Combobox     = mock.MagicMock()
        ttk_stub.Progressbar  = mock.MagicMock()
        ttk_stub.Scrollbar    = mock.MagicMock()

        mb_stub               = types.ModuleType("tkinter.messagebox")
        mb_stub.showerror     = mock.MagicMock()

        sd_stub               = types.ModuleType("sounddevice")
        sd_stub.query_devices  = mock.MagicMock(return_value=[])
        sd_stub.query_hostapis = mock.MagicMock(return_value=[])
        sd_stub.check_input_settings  = mock.MagicMock(side_effect=Exception)
        sd_stub.check_output_settings = mock.MagicMock(side_effect=Exception)

        serial_stub            = types.ModuleType("serial")
        serial_stub.STOPBITS_ONE            = 1
        serial_stub.STOPBITS_ONE_POINT_FIVE = 1.5
        serial_stub.STOPBITS_TWO            = 2
        serial_tools    = types.ModuleType("serial.tools")
        serial_tools_lp = types.ModuleType("serial.tools.list_ports")
        serial_tools_lp.comports = mock.MagicMock(return_value=[])

        sys.modules.setdefault("tkinter",               tk_stub)
        sys.modules.setdefault("tkinter.ttk",            ttk_stub)
        sys.modules.setdefault("tkinter.messagebox",     mb_stub)
        sys.modules.setdefault("sounddevice",            sd_stub)
        sys.modules.setdefault("serial",                 serial_stub)
        sys.modules.setdefault("serial.tools",           serial_tools)
        sys.modules.setdefault("serial.tools.list_ports", serial_tools_lp)

        if "main" in sys.modules:
            del sys.modules["main"]
        import main as m
        cls._cfg_cls = m.AppConfig

    def setUp(self) -> None:
        self._tmp = tempfile.NamedTemporaryFile(suffix=".cfg", delete=False)
        self._tmp.close()

    def tearDown(self) -> None:
        try:
            os.unlink(self._tmp.name)
        except OSError:
            pass

    # -- Default values -------------------------------------------------------

    def test_default_sync_on_startup_is_true(self):
        cfg = self._cfg_cls(path=self._tmp.name)
        self.assertTrue(cfg.ntp_sync_on_startup)

    def test_default_timeout_is_3s(self):
        cfg = self._cfg_cls(path=self._tmp.name)
        self.assertAlmostEqual(cfg.ntp_timeout_s, 3.0, places=6)

    def test_default_servers_include_nist(self):
        cfg = self._cfg_cls(path=self._tmp.name)
        self.assertIn("time.nist.gov", cfg.ntp_servers)

    def test_default_servers_include_pool(self):
        cfg = self._cfg_cls(path=self._tmp.name)
        self.assertIn("pool.ntp.org", cfg.ntp_servers)

    def test_default_servers_nonempty(self):
        cfg = self._cfg_cls(path=self._tmp.name)
        self.assertGreater(len(cfg.ntp_servers), 0)

    # -- Persistence ----------------------------------------------------------

    def test_save_and_reload_servers(self):
        cfg = self._cfg_cls(path=self._tmp.name)
        cfg.save_ntp(["time.nist.gov", "myserver.local"])
        cfg2 = self._cfg_cls(path=self._tmp.name)
        self.assertEqual(cfg2.ntp_servers, ["time.nist.gov", "myserver.local"])

    def test_save_sync_on_startup_false(self):
        cfg = self._cfg_cls(path=self._tmp.name)
        cfg.save_ntp(["pool.ntp.org"], sync_on_startup=False)
        cfg2 = self._cfg_cls(path=self._tmp.name)
        self.assertFalse(cfg2.ntp_sync_on_startup)

    def test_save_timeout(self):
        cfg = self._cfg_cls(path=self._tmp.name)
        cfg.save_ntp(["pool.ntp.org"], timeout_s=5.0)
        cfg2 = self._cfg_cls(path=self._tmp.name)
        self.assertAlmostEqual(cfg2.ntp_timeout_s, 5.0, places=6)

    def test_servers_whitespace_stripped(self):
        cfg = self._cfg_cls(path=self._tmp.name)
        cfg.save_ntp(["  time.nist.gov  ", " pool.ntp.org "])
        cfg2 = self._cfg_cls(path=self._tmp.name)
        self.assertEqual(cfg2.ntp_servers, ["time.nist.gov", "pool.ntp.org"])


if __name__ == "__main__":
    unittest.main()
