"""
test_qso_logging.py — Unit tests for Milestone 5 QSO logging.

Tests cover:
  Group A — ADIF module (adif_log.py)
    1.  _adif_field — correct tag formatting
    2.  _adif_field — empty value returns empty string
    3.  AdifContact.to_adif_record — all non-empty fields included
    4.  AdifContact.to_adif_record — zero freq_mhz omitted
    5.  AdifContact.to_adif_record — empty optional fields omitted
    6.  append_adif_contact — creates file with header then record
    7.  append_adif_contact — second append does not repeat header
    8.  append_adif_contact — thread-safe concurrent appends
    9.  qso_record_to_adif_contact — converts QsoRecord fields correctly

  Group B — ft8_qso.py Milestone 5 additions
   10.  QsoRecord — new optional fields have correct defaults
   11.  QsoRecord — new fields round-trip through frozen dataclass
   12.  Ft8QsoManager — _tx_snr tracked in CQ path (advance from CQ_SENT)
   13.  Ft8QsoManager — _tx_snr tracked in REPLY path (start_from_received)
   14.  Ft8QsoManager — _dx_grid extracted from CQ message extra field
   15.  Ft8QsoManager — _dx_grid is None when extra field is not a grid
   16.  Ft8QsoManager — build_record uses _tx_snr when snr_sent not supplied
   17.  Ft8QsoManager — build_record snr_sent param overrides _tx_snr
   18.  Ft8QsoManager — _tx_snr cleared on reset()
   19.  Ft8QsoManager — _tx_snr cleared on abort()
   20.  Ft8QsoManager — build_record includes my_grid
   21.  Ft8QsoManager — dx_grid property exposed correctly
   22.  AppConfig — operator_name defaults to empty string
   23.  AppConfig — save_operator persists name
   24.  AppConfig — save_operator name param has safe default (backward compat)

  Group C — Voice QSO log form (main.py GUI helpers, tested without display)
   25.  _on_log_voice_qso — writes ADIF file and appends to log_box
   26.  _on_log_voice_qso — shows error when DX callsign is empty
   27.  _on_log_voice_qso — resets form fields after successful log
   28.  _on_log_voice_qso — uses _current_freq and _current_mode
   29.  _on_log_voice_qso — RST defaults to 59 when field is blank

  Group D — FT8 auto-logging (main.py, GUI thread logic)
   30.  _log_ft8_qso — does nothing when QSO not COMPLETE
   31.  _log_ft8_qso — does nothing when _qso_mgr is None
   32.  _log_ft8_qso — writes ADIF on COMPLETE QSO
   33.  _log_ft8_qso — _ft8_qso_logged guard prevents double-write
   34.  _maybe_assist_prefill — logs QSO when advance returns None+COMPLETE (Station B)
   35.  _apply_tx_state_update — logs QSO on TxState.COMPLETE + QsoState.COMPLETE (Station A)

Run:  pytest test_qso_logging.py -v
"""
from __future__ import annotations

import os
import sys
import tempfile
import threading
import time
import types
import unittest
import unittest.mock as mock
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(__file__))

# ---------------------------------------------------------------------------
# Non-GUI imports (safe on all platforms)
# ---------------------------------------------------------------------------

import adif_log
from adif_log import (
    AdifContact,
    _adif_field,
    _ensure_header,
    append_adif_contact,
    qso_record_to_adif_contact,
)
from ft8_qso import (
    Ft8QsoManager,
    OperatorConfig,
    QsoRecord,
    QsoState,
)
from ft8_ntp import Ft8SlotTimer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_timer() -> Ft8SlotTimer:
    """Mock Ft8SlotTimer that always reports 0 s to next slot."""
    t = mock.MagicMock(spec=Ft8SlotTimer)
    t.seconds_to_next_slot.return_value = 0.0
    t.current_slot_parity.return_value = 0
    t.next_slot_utc.return_value = datetime.now(tz=timezone.utc)
    return t


def _make_manager(callsign: str = "W4ABC", grid: str = "EN52") -> Ft8QsoManager:
    op = OperatorConfig(callsign=callsign, grid=grid)
    return Ft8QsoManager(operator=op, slot_timer=_make_timer())


def _import_main_no_gui():
    """Import main.py with all hardware/display stubs."""
    tk_stub = types.ModuleType("tkinter")
    tk_stub.Tk         = mock.MagicMock()
    tk_stub.Toplevel   = mock.MagicMock()
    tk_stub.Frame      = mock.MagicMock()
    tk_stub.LabelFrame = mock.MagicMock()
    tk_stub.Label      = mock.MagicMock()
    tk_stub.Button     = mock.MagicMock()
    tk_stub.Entry      = mock.MagicMock()
    tk_stub.Text       = mock.MagicMock()
    tk_stub.Spinbox    = mock.MagicMock()
    tk_stub.StringVar  = mock.MagicMock
    tk_stub.IntVar     = mock.MagicMock
    tk_stub.BooleanVar = mock.MagicMock
    tk_stub.Scale      = mock.MagicMock
    tk_stub.NORMAL     = "normal"
    tk_stub.DISABLED   = "disabled"
    tk_stub.LEFT       = "left"
    tk_stub.RIGHT      = "right"
    tk_stub.X          = "x"
    tk_stub.Y          = "y"
    tk_stub.BOTH       = "both"
    tk_stub.END        = "end"
    tk_stub.SUNKEN     = "sunken"
    tk_stub.RAISED     = "raised"
    tk_stub.GROOVE     = "groove"
    tk_stub.Misc       = object

    ttk_stub = types.ModuleType("tkinter.ttk")
    ttk_stub.Combobox    = mock.MagicMock()
    ttk_stub.Progressbar = mock.MagicMock()
    ttk_stub.Scrollbar   = mock.MagicMock()

    mb_stub = types.ModuleType("tkinter.messagebox")
    mb_stub.showerror = mock.MagicMock()

    sd_stub = types.ModuleType("sounddevice")
    sd_stub.query_devices         = mock.MagicMock(return_value=[])
    sd_stub.query_hostapis        = mock.MagicMock(return_value=[])
    sd_stub.check_input_settings  = mock.MagicMock(side_effect=Exception("not available"))
    sd_stub.check_output_settings = mock.MagicMock(side_effect=Exception("not available"))

    serial_stub   = types.ModuleType("serial")
    serial_stub.STOPBITS_ONE            = 1
    serial_stub.STOPBITS_ONE_POINT_FIVE = 1.5
    serial_stub.STOPBITS_TWO           = 2
    serial_tools    = types.ModuleType("serial.tools")
    serial_tools_lp = types.ModuleType("serial.tools.list_ports")
    serial_tools_lp.comports = mock.MagicMock(return_value=[])

    sys.modules.setdefault("tkinter",          tk_stub)
    sys.modules.setdefault("tkinter.ttk",      ttk_stub)
    sys.modules.setdefault("tkinter.messagebox", mb_stub)
    sys.modules.setdefault("sounddevice",      sd_stub)
    sys.modules.setdefault("serial",           serial_stub)
    sys.modules.setdefault("serial.tools",     serial_tools)
    sys.modules.setdefault("serial.tools.list_ports", serial_tools_lp)

    import importlib
    if "main" in sys.modules:
        del sys.modules["main"]
    import main as m
    return m


def _make_gui(main_mod, adif_path: str):
    """Construct a minimal RadioGUI with all hardware stubs."""
    fake_radio = mock.MagicMock()
    fake_radio.is_connected.return_value = False
    fake_radio.get_frequency.return_value = 14.074
    fake_radio.get_s_meter.return_value   = 0
    fake_radio.port = "COM1"
    fake_radio.baud = 38400

    with tempfile.NamedTemporaryFile(suffix=".cfg", delete=False) as tf:
        cfg_path = tf.name
    cfg = main_mod.AppConfig(path=cfg_path)

    fake_root = mock.MagicMock()
    fake_root.after.return_value = "after_handle"
    fake_root.focus_get.return_value = None

    # Patch ADIF_LOG_PATH to temp file
    main_mod.ADIF_LOG_PATH = adif_path

    gui = main_mod.RadioGUI.__new__(main_mod.RadioGUI)
    # Minimal bootstrap — only the attributes we test against
    gui.root   = fake_root
    gui.radio  = fake_radio
    gui._config = cfg

    gui._ui_queue   = __import__("queue").Queue()
    gui._shutdown   = threading.Event()
    gui._poll_thread = None
    gui._op_mode    = "voice"
    gui._freq_step  = 0.001
    gui.audio_device_index       = None
    gui.audio_output_device_index = None
    gui.tx_mic_device_index      = None
    gui.tx_radio_out_device_index = None
    gui._audio_src    = None
    gui._audio_thread = None
    gui._audio_stop   = threading.Event()
    gui._audio_passthrough = None
    gui._tx_capture   = None

    from ft8_decode import FT8ConsoleDecoder
    gui._ft8 = mock.MagicMock(spec=FT8ConsoleDecoder)

    from ft8_tx import Ft8TxCoordinator, TxState
    gui._tx_coord = mock.MagicMock(spec=Ft8TxCoordinator)
    gui._tx_coord.state = TxState.IDLE
    gui._tx_coord.cancel.return_value = True

    from ft8_ntp import default_slot_timer
    gui._slot_timer = default_slot_timer
    gui._tx_countdown_after = None
    gui._qso_mgr              = None
    gui._qso_assist_active    = False
    gui._qso_assist_prefilled = ""
    gui._ft8_qso_logged       = False
    gui._ft8_qso_initiated    = "CQ"
    gui._auto_arm_var         = mock.MagicMock()
    gui._auto_arm_var.get.return_value = False
    gui._cq_retry_after = None
    gui._current_freq     = 14.074
    gui._current_mode     = "USB"
    gui._current_rf_power = 10

    # Stub all widget attributes used in methods under test
    gui._tx_msg_var        = mock.MagicMock()
    gui._tx_callsign_var   = mock.MagicMock()
    gui._tx_callsign_var.get.return_value = "W4ABC"
    gui._tx_grid_var       = mock.MagicMock()
    gui._tx_grid_var.get.return_value = "EN52"
    gui._tx_status_var     = mock.MagicMock()
    gui._tx_status_lbl     = mock.MagicMock()
    gui._arm_btn           = mock.MagicMock()
    gui._cancel_btn        = mock.MagicMock()
    gui._cq_session_btn    = mock.MagicMock()
    gui._stop_session_btn  = mock.MagicMock()
    gui.log_box            = mock.MagicMock()
    gui.ft8_log            = mock.MagicMock()

    # Voice QSO form vars
    gui._qso_dx_call_var  = mock.MagicMock()
    gui._qso_dx_grid_var  = mock.MagicMock()
    gui._qso_rst_sent_var = mock.MagicMock()
    gui._qso_rst_rcvd_var = mock.MagicMock()
    gui._qso_comment_var  = mock.MagicMock()

    # Default values for QSO form
    gui._qso_dx_call_var.get.return_value  = "K9XYZ"
    gui._qso_dx_grid_var.get.return_value  = "FN20"
    gui._qso_rst_sent_var.get.return_value = "59"
    gui._qso_rst_rcvd_var.get.return_value = "57"
    gui._qso_comment_var.get.return_value  = "First contact"

    return gui, cfg_path


# =============================================================================
# Group A — ADIF module
# =============================================================================

class TestAdifField(unittest.TestCase):

    def test_non_empty_field(self):
        result = _adif_field("CALL", "K9XYZ")
        self.assertEqual(result, "<CALL:5>K9XYZ")

    def test_empty_value_returns_empty_string(self):
        result = _adif_field("CALL", "")
        self.assertEqual(result, "")

    def test_field_name_uppercased(self):
        result = _adif_field("call", "K9XYZ")
        self.assertTrue(result.startswith("<CALL:"))

    def test_length_counts_characters(self):
        result = _adif_field("BAND", "20m")
        self.assertEqual(result, "<BAND:3>20m")


class TestAdifContact(unittest.TestCase):

    def _make_full_contact(self) -> AdifContact:
        return AdifContact(
            call="K9XYZ",
            qso_date="20240115",
            time_on="120000",
            freq_mhz=14.074,
            band="20m",
            mode="FT8",
            rst_sent="-05",
            rst_rcvd="+07",
            station_callsign="W4ABC",
            my_gridsquare="EN52",
            gridsquare="FN20",
            tx_pwr="10",
            comment="Test QSO",
        )

    def test_to_adif_record_contains_required_fields(self):
        c = self._make_full_contact()
        rec = c.to_adif_record()
        self.assertIn("<CALL:5>K9XYZ", rec)
        self.assertIn("<QSO_DATE:8>20240115", rec)
        self.assertIn("<TIME_ON:6>120000", rec)
        self.assertIn("<EOR>", rec)

    def test_to_adif_record_contains_optional_fields(self):
        c = self._make_full_contact()
        rec = c.to_adif_record()
        self.assertIn("<BAND:3>20m", rec)
        self.assertIn("<MODE:3>FT8", rec)
        self.assertIn("<RST_SENT:3>-05", rec)
        self.assertIn("<RST_RCVD:3>+07", rec)
        self.assertIn("<STATION_CALLSIGN:5>W4ABC", rec)
        self.assertIn("<MY_GRIDSQUARE:4>EN52", rec)
        self.assertIn("<GRIDSQUARE:4>FN20", rec)
        self.assertIn("<TX_PWR:2>10", rec)
        self.assertIn("<COMMENT:8>Test QSO", rec)

    def test_zero_freq_omitted(self):
        c = AdifContact(call="K9XYZ", qso_date="20240115", time_on="120000")
        rec = c.to_adif_record()
        self.assertNotIn("FREQ", rec)

    def test_nonzero_freq_included(self):
        c = AdifContact(
            call="K9XYZ", qso_date="20240115", time_on="120000", freq_mhz=14.074
        )
        rec = c.to_adif_record()
        self.assertIn("<FREQ:", rec)
        self.assertIn("14.074", rec)

    def test_empty_optional_fields_omitted(self):
        c = AdifContact(call="K9XYZ", qso_date="20240115", time_on="120000")
        rec = c.to_adif_record()
        self.assertNotIn("BAND", rec)
        self.assertNotIn("MODE", rec)
        self.assertNotIn("COMMENT", rec)

    def test_record_ends_with_eor(self):
        c = AdifContact(call="K9XYZ", qso_date="20240115", time_on="120000")
        rec = c.to_adif_record()
        self.assertTrue(rec.strip().endswith("<EOR>"))


class TestAppendAdifContact(unittest.TestCase):

    def setUp(self):
        tf = tempfile.NamedTemporaryFile(suffix=".adi", delete=False)
        tf.close()
        self.path = tf.name
        os.unlink(self.path)  # Remove so append_adif_contact creates fresh

    def tearDown(self):
        if os.path.exists(self.path):
            os.unlink(self.path)

    def _contact(self, call: str = "K9XYZ") -> AdifContact:
        return AdifContact(
            call=call,
            qso_date="20240115",
            time_on="120000",
            band="20m",
            mode="FT8",
        )

    def test_creates_file_with_header(self):
        append_adif_contact(self.path, self._contact())
        with open(self.path, encoding="utf-8") as fh:
            content = fh.read()
        self.assertIn("<EOH>", content)
        self.assertIn("<CALL:5>K9XYZ", content)

    def test_second_append_no_duplicate_header(self):
        append_adif_contact(self.path, self._contact("K9XYZ"))
        append_adif_contact(self.path, self._contact("N1ABD"))
        with open(self.path, encoding="utf-8") as fh:
            content = fh.read()
        self.assertEqual(content.count("<EOH>"), 1, "Header must appear exactly once")
        self.assertIn("<CALL:5>K9XYZ", content)
        self.assertIn("<CALL:5>N1ABD", content)

    def test_thread_safe_concurrent_appends(self):
        """Multiple threads appending simultaneously must not corrupt the file."""
        calls = [f"AB{i}CDE" for i in range(10)]

        def worker(call):
            append_adif_contact(self.path, self._contact(call))

        threads = [threading.Thread(target=worker, args=(c,)) for c in calls]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        with open(self.path, encoding="utf-8") as fh:
            content = fh.read()
        self.assertEqual(content.count("<EOR>"), len(calls))


class TestQsoRecordToAdifContact(unittest.TestCase):

    def _make_record(self) -> QsoRecord:
        return QsoRecord(
            our_call="W4ABC",
            dx_call="K9XYZ",
            freq_mhz=14.074,
            band="20m",
            mode="FT8",
            time_on=datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc),
            rst_sent="-05",
            rst_rcvd="+07",
        )

    def test_basic_fields_mapped(self):
        rec = self._make_record()
        contact = qso_record_to_adif_contact(rec)
        self.assertEqual(contact.call, "K9XYZ")
        self.assertEqual(contact.station_callsign, "W4ABC")
        self.assertEqual(contact.freq_mhz, 14.074)
        self.assertEqual(contact.band, "20m")
        self.assertEqual(contact.mode, "FT8")
        self.assertEqual(contact.rst_sent, "-05")
        self.assertEqual(contact.rst_rcvd, "+07")
        self.assertEqual(contact.qso_date, "20240115")
        self.assertEqual(contact.time_on, "120000")

    def test_optional_kwargs_passed_through(self):
        rec = self._make_record()
        contact = qso_record_to_adif_contact(
            rec,
            my_grid="EN52",
            dx_grid="FN20",
            tx_pwr="10",
            comment="Contest",
        )
        self.assertEqual(contact.my_gridsquare, "EN52")
        self.assertEqual(contact.gridsquare, "FN20")
        self.assertEqual(contact.tx_pwr, "10")
        self.assertEqual(contact.comment, "Contest")


# =============================================================================
# Group B — ft8_qso.py Milestone 5 additions
# =============================================================================

class TestQsoRecordNewFields(unittest.TestCase):

    def test_optional_fields_default_empty(self):
        rec = QsoRecord(our_call="W4ABC", dx_call="K9XYZ")
        self.assertEqual(rec.dx_grid, "")
        self.assertEqual(rec.my_grid, "")
        self.assertEqual(rec.tx_pwr_w, 0.0)
        self.assertEqual(rec.comment, "")

    def test_optional_fields_round_trip(self):
        rec = QsoRecord(
            our_call="W4ABC",
            dx_call="K9XYZ",
            dx_grid="FN20",
            my_grid="EN52",
            tx_pwr_w=50.0,
            comment="Milestone 5 test",
        )
        self.assertEqual(rec.dx_grid, "FN20")
        self.assertEqual(rec.my_grid, "EN52")
        self.assertEqual(rec.tx_pwr_w, 50.0)
        self.assertEqual(rec.comment, "Milestone 5 test")

    def test_frozen_includes_new_fields(self):
        """Mutation of new fields must also raise FrozenInstanceError."""
        from dataclasses import FrozenInstanceError
        rec = QsoRecord(our_call="W4ABC", dx_call="K9XYZ")
        with self.assertRaises((FrozenInstanceError, TypeError, AttributeError)):
            rec.dx_grid = "XX00"  # type: ignore[misc]


class TestFt8QsoManagerTxSnr(unittest.TestCase):

    def test_tx_snr_tracked_in_cq_path(self):
        mgr = _make_manager()
        mgr.start_cq()
        # Simulate receiving a reply with SNR -7
        mgr.advance("W4ABC K9XYZ -07", snr_db=-7)
        self.assertEqual(mgr._tx_snr, -7)

    def test_tx_snr_tracked_in_reply_path(self):
        mgr = _make_manager()
        # We answer a CQ; snr_db is the signal strength we measured
        mgr.start_from_received("CQ K9XYZ FN20", snr_db=-5)
        self.assertEqual(mgr._tx_snr, -5)

    def test_tx_snr_cleared_on_reset(self):
        mgr = _make_manager()
        mgr.start_cq()
        mgr.advance("W4ABC K9XYZ -07", snr_db=-7)
        mgr.reset()
        self.assertIsNone(mgr._tx_snr)

    def test_tx_snr_cleared_on_abort(self):
        mgr = _make_manager()
        mgr.start_cq()
        mgr.advance("W4ABC K9XYZ -07", snr_db=-7)
        mgr.abort()
        self.assertIsNone(mgr._tx_snr)

    def test_build_record_uses_tx_snr_by_default(self):
        """build_record() should use internally tracked _tx_snr when snr_sent not provided."""
        mgr = _make_manager()
        mgr.start_cq()
        mgr.advance("W4ABC K9XYZ -07", snr_db=-7)   # K9XYZ replies to our CQ → EXCHANGE_SENT
        mgr.advance("W4ABC K9XYZ RR73")              # K9XYZ confirms → COMPLETE
        rec = mgr.build_record(freq_mhz=14.074, band="20m")
        self.assertEqual(rec.rst_sent, "-07")

    def test_build_record_snr_sent_param_overrides_tx_snr(self):
        """Explicit snr_sent parameter overrides the internally tracked value."""
        mgr = _make_manager()
        mgr.start_cq()
        mgr.advance("W4ABC K9XYZ -07", snr_db=-7)
        mgr.advance("W4ABC K9XYZ RR73")   # → COMPLETE
        rec = mgr.build_record(freq_mhz=14.074, band="20m", snr_sent=-3)
        self.assertEqual(rec.rst_sent, "-03")


class TestFt8QsoManagerDxGrid(unittest.TestCase):

    def test_dx_grid_extracted_from_cq_message(self):
        mgr = _make_manager()
        mgr.start_from_received("CQ K9XYZ FN20", snr_db=-5)
        self.assertEqual(mgr._dx_grid, "FN20")
        self.assertEqual(mgr.dx_grid, "FN20")

    def test_dx_grid_none_when_extra_is_not_grid(self):
        """A CQ that includes a sub-band designator instead of grid."""
        mgr = _make_manager()
        mgr.start_from_received("CQ K9XYZ DX", snr_db=-5)
        self.assertIsNone(mgr._dx_grid)

    def test_dx_grid_none_for_non_cq_message(self):
        mgr = _make_manager()
        mgr.start_cq()
        mgr.advance("W4ABC K9XYZ -07", snr_db=-7)
        # No CQ was decoded; _dx_grid should remain None
        self.assertIsNone(mgr._dx_grid)

    def test_dx_grid_in_build_record(self):
        mgr = _make_manager()
        mgr.start_from_received("CQ K9XYZ FN20", snr_db=-5)
        mgr.advance("W4ABC K9XYZ R-05")   # REPLY_SENT → RRR_SENT (K9XYZ sends exchange to W4ABC)
        # Simulate confirming 73 from K9XYZ
        mgr.advance("W4ABC K9XYZ 73")     # RRR_SENT → COMPLETE
        rec = mgr.build_record(freq_mhz=14.074, band="20m", my_grid="EN52")
        self.assertEqual(rec.dx_grid, "FN20")
        self.assertEqual(rec.my_grid, "EN52")

    def test_dx_grid_cleared_on_reset(self):
        mgr = _make_manager()
        mgr.start_from_received("CQ K9XYZ FN20", snr_db=-5)
        mgr.reset()
        self.assertIsNone(mgr._dx_grid)


class TestAppConfigOperatorName(unittest.TestCase):

    def test_operator_name_defaults_to_empty(self):
        with tempfile.NamedTemporaryFile(suffix=".cfg", delete=False) as tf:
            path = tf.name
        try:
            # Import AppConfig via main stub
            m = _import_main_no_gui()
            cfg = m.AppConfig(path=path)
            self.assertEqual(cfg.operator_name, "")
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_save_operator_persists_name(self):
        with tempfile.NamedTemporaryFile(suffix=".cfg", delete=False) as tf:
            path = tf.name
        try:
            m = _import_main_no_gui()
            cfg = m.AppConfig(path=path)
            cfg.save_operator("W4ABC", "EN52", name="John")
            cfg2 = m.AppConfig(path=path)
            self.assertEqual(cfg2.operator_name, "John")
            self.assertEqual(cfg2.operator_callsign, "W4ABC")
            self.assertEqual(cfg2.operator_grid, "EN52")
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_save_operator_name_has_safe_default(self):
        """save_operator(call, grid) without name= must not raise."""
        with tempfile.NamedTemporaryFile(suffix=".cfg", delete=False) as tf:
            path = tf.name
        try:
            m = _import_main_no_gui()
            cfg = m.AppConfig(path=path)
            cfg.save_operator("W4ABC", "EN52")  # name defaults to ""
            cfg2 = m.AppConfig(path=path)
            self.assertEqual(cfg2.operator_name, "")
        finally:
            if os.path.exists(path):
                os.unlink(path)


# =============================================================================
# Group C — Voice QSO log form
# =============================================================================

class TestVoiceQsoLogForm(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.main = _import_main_no_gui()

    def setUp(self):
        tf = tempfile.NamedTemporaryFile(suffix=".adi", delete=False)
        tf.close()
        self.adif_path = tf.name
        os.unlink(self.adif_path)
        self.gui, self.cfg_path = _make_gui(self.main, self.adif_path)

    def tearDown(self):
        for p in (self.adif_path, self.cfg_path):
            if os.path.exists(p):
                os.unlink(p)

    def test_log_voice_qso_writes_adif(self):
        self.gui._on_log_voice_qso()
        self.assertTrue(os.path.exists(self.adif_path), "ADIF file must be created")
        with open(self.adif_path, encoding="utf-8") as fh:
            content = fh.read()
        self.assertIn("<CALL:5>K9XYZ", content)
        self.assertIn("<EOR>", content)

    def test_log_voice_qso_shows_error_when_dx_call_empty(self):
        self.gui._qso_dx_call_var.get.return_value = ""
        # Patch the messagebox.showerror in main module (the actual import used)
        with mock.patch.object(self.main.messagebox, "showerror") as mock_err:
            self.gui._on_log_voice_qso()
            mock_err.assert_called_once()
        self.assertFalse(os.path.exists(self.adif_path), "No ADIF file when DX call empty")

    def test_log_voice_qso_resets_form_fields(self):
        self.gui._on_log_voice_qso()
        self.gui._qso_dx_call_var.set.assert_called_with("")
        self.gui._qso_rst_sent_var.set.assert_called_with("59")
        self.gui._qso_rst_rcvd_var.set.assert_called_with("59")

    def test_log_voice_qso_uses_current_freq_and_mode(self):
        self.gui._current_freq = 14.225
        self.gui._current_mode = "USB"
        self.gui._on_log_voice_qso()
        with open(self.adif_path, encoding="utf-8") as fh:
            content = fh.read()
        self.assertIn("14.225", content)
        self.assertIn("<MODE:3>USB", content)

    def test_log_voice_qso_rst_defaults_to_59_when_blank(self):
        self.gui._qso_rst_sent_var.get.return_value = ""
        self.gui._qso_rst_rcvd_var.get.return_value = ""
        self.gui._on_log_voice_qso()
        with open(self.adif_path, encoding="utf-8") as fh:
            content = fh.read()
        self.assertIn("<RST_SENT:2>59", content)
        self.assertIn("<RST_RCVD:2>59", content)

    def test_log_voice_qso_includes_grid_fields(self):
        self.gui._on_log_voice_qso()
        with open(self.adif_path, encoding="utf-8") as fh:
            content = fh.read()
        self.assertIn("FN20", content)   # DX grid
        self.assertIn("EN52", content)   # our grid

    def test_log_voice_qso_appends_to_log_box(self):
        self.gui._on_log_voice_qso()
        self.gui.log_box.insert.assert_called()


# =============================================================================
# Group D — FT8 auto-logging
# =============================================================================

class TestFt8AutoLogging(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.main = _import_main_no_gui()

    def setUp(self):
        tf = tempfile.NamedTemporaryFile(suffix=".adi", delete=False)
        tf.close()
        self.adif_path = tf.name
        os.unlink(self.adif_path)
        self.gui, self.cfg_path = _make_gui(self.main, self.adif_path)

    def tearDown(self):
        for p in (self.adif_path, self.cfg_path):
            if os.path.exists(p):
                os.unlink(p)

    def _complete_ft8_qso(self):
        """
        Drive a FT8 QSO to COMPLETE state via the state machine and attach
        the manager to the GUI.
        """
        op  = OperatorConfig(callsign="W4ABC", grid="EN52")
        mgr = Ft8QsoManager(operator=op, slot_timer=_make_timer())
        mgr.start_cq()
        mgr.advance("W4ABC K9XYZ -07", snr_db=-7)  # K9XYZ replies: TO=W4ABC FROM=K9XYZ → EXCHANGE_SENT
        mgr.advance("W4ABC K9XYZ RR73")             # K9XYZ confirms: TO=W4ABC FROM=K9XYZ → COMPLETE
        self.gui._qso_mgr           = mgr
        self.gui._qso_assist_active = True

    def test_log_ft8_qso_does_nothing_when_not_complete(self):
        op  = OperatorConfig(callsign="W4ABC", grid="EN52")
        mgr = Ft8QsoManager(operator=op, slot_timer=_make_timer())
        mgr.start_cq()
        self.gui._qso_mgr = mgr
        self.gui._log_ft8_qso()
        self.assertFalse(os.path.exists(self.adif_path))

    def test_log_ft8_qso_does_nothing_when_mgr_is_none(self):
        self.gui._qso_mgr = None
        self.gui._log_ft8_qso()
        self.assertFalse(os.path.exists(self.adif_path))

    def test_log_ft8_qso_writes_adif_on_complete(self):
        self._complete_ft8_qso()
        self.gui._log_ft8_qso()
        self.assertTrue(os.path.exists(self.adif_path))
        with open(self.adif_path, encoding="utf-8") as fh:
            content = fh.read()
        self.assertIn("<CALL:5>K9XYZ", content)
        self.assertIn("<MODE:3>FT8", content)
        self.assertIn("<EOR>", content)

    def test_log_ft8_qso_dedup_guard(self):
        """Calling _log_ft8_qso twice must produce only one EOR."""
        self._complete_ft8_qso()
        self.gui._log_ft8_qso()
        self.gui._log_ft8_qso()  # second call — should be a no-op
        with open(self.adif_path, encoding="utf-8") as fh:
            content = fh.read()
        self.assertEqual(content.count("<EOR>"), 1, "Only one record should be written")

    def test_maybe_assist_prefill_logs_on_station_b_complete(self):
        """
        Station B: after receiving confirming 73, advance() returns None and
        QsoState becomes COMPLETE.  _log_ft8_qso must be called.
        """
        op  = OperatorConfig(callsign="W4ABC", grid="EN52")
        mgr = Ft8QsoManager(operator=op, slot_timer=_make_timer())
        # Station B flow: reply to CQ → REPLY_SENT → RRR_SENT
        mgr.start_from_received("CQ K9XYZ FN20", snr_db=-5)
        mgr.advance("W4ABC K9XYZ R-05")   # K9XYZ sends exchange TO=W4ABC → RRR_SENT

        self.gui._qso_mgr           = mgr
        self.gui._qso_assist_active = True

        from ft8_tx import TxState
        self.gui._tx_coord.state = TxState.IDLE

        # Receive the confirming 73 from K9XYZ — advance() returns None, state → COMPLETE
        self.gui._maybe_assist_prefill("W4ABC K9XYZ 73", snr_db=-5.0)

        self.assertTrue(os.path.exists(self.adif_path), "ADIF file must be created")
        with open(self.adif_path, encoding="utf-8") as fh:
            content = fh.read()
        self.assertIn("<CALL:5>K9XYZ", content)

    def test_apply_tx_state_update_logs_on_station_a_complete(self):
        """
        Station A: after TxState.COMPLETE fires and QsoState is COMPLETE,
        the ADIF record must be written.
        """
        self._complete_ft8_qso()
        from ft8_tx import TxState
        self.gui._apply_tx_state_update(TxState.COMPLETE, "TX complete")
        self.assertTrue(os.path.exists(self.adif_path))
        with open(self.adif_path, encoding="utf-8") as fh:
            content = fh.read()
        self.assertIn("<CALL:5>K9XYZ", content)


if __name__ == "__main__":
    unittest.main(verbosity=2)
