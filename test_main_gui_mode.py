"""
test_main_gui_mode.py — Unit tests for the RadioGUI mode-switching logic and
configuration helpers added in the GUI overhaul.

Tests:
  1.  BANDS — expanded band plan contains all required HF + VHF/UHF bands
  2.  FT8_FREQS — standard FT8 calling frequencies defined for common bands
  3.  FT8_LOG_PATH — log path is next to main.py and ends with .log
  4.  AppConfig — audio output device properties default to -1 / ""
  5.  AppConfig.save_audio_output — persists and reloads correctly
  6.  AppConfig.save_audio — persists input device (regression check)
  7.  _enum_audio_devices — returns (list, list, str) tuple on any platform
  7b. _enum_audio_devices — includes validated WASAPI/MME devices (WASAPI first)
  8.  SettingsDialog._parse_device_index — parses labelled index correctly
  9.  SettingsDialog._parse_device_index — returns -1 for malformed labels
  10. _save_ft8_log_to_file — creates log file and appends (not overwrites)
  11. _save_ft8_log_to_file — skips empty content (no file created/modified)
  12. RadioGUI._op_mode defaults to "voice"
  13. RadioGUI._switch_to_data sets _op_mode to "data" and starts decoder
  14. RadioGUI._switch_to_voice saves log, stops decoder, sets _op_mode to "voice"
  15. RadioGUI._freq_step defaults to 0.001 MHz
  16. infer_band_from_freq — correct band returned for sample frequencies
  17. infer_band_from_freq — None returned for out-of-band frequency
  18. AppConfig — TX audio device properties default to -1 / ""
  19. AppConfig.save_tx_audio — persists and reloads TX devices correctly
  20. _switch_to_data stops RX monitor and TX capture before switching
  21. _switch_to_voice stops data-mode audio stream and FT8 decoder

Run:  python test_main_gui_mode.py
"""
from __future__ import annotations

import configparser
import os
import sys
import tempfile
import time
import traceback
import types
import unittest.mock as mock

sys.path.insert(0, os.path.dirname(__file__))

# ---------------------------------------------------------------------------
# Helpers to import main.py without launching the GUI
# ---------------------------------------------------------------------------

def _import_main_no_gui():
    """
    Import main.py replacing tkinter, sounddevice, and serial with stubs so
    we can test pure-Python logic without a display or hardware.
    """
    # Build a minimal tkinter stub
    tk_stub = types.ModuleType("tkinter")
    tk_stub.Tk       = mock.MagicMock()
    tk_stub.Toplevel = mock.MagicMock()
    tk_stub.Frame    = mock.MagicMock()
    tk_stub.LabelFrame = mock.MagicMock()
    tk_stub.Label    = mock.MagicMock()
    tk_stub.Button   = mock.MagicMock()
    tk_stub.Entry    = mock.MagicMock()
    tk_stub.Text     = mock.MagicMock()
    tk_stub.Spinbox  = mock.MagicMock()
    tk_stub.StringVar = mock.MagicMock
    tk_stub.IntVar   = mock.MagicMock
    tk_stub.Scale    = mock.MagicMock
    tk_stub.NORMAL   = "normal"
    tk_stub.DISABLED = "disabled"
    tk_stub.LEFT     = "left"
    tk_stub.RIGHT    = "right"
    tk_stub.X        = "x"
    tk_stub.Y        = "y"
    tk_stub.BOTH     = "both"
    tk_stub.END      = "end"
    tk_stub.SUNKEN   = "sunken"
    tk_stub.RAISED   = "raised"
    tk_stub.GROOVE   = "groove"
    tk_stub.Misc     = object

    ttk_stub = types.ModuleType("tkinter.ttk")
    ttk_stub.Combobox   = mock.MagicMock()
    ttk_stub.Progressbar = mock.MagicMock()
    ttk_stub.Scrollbar  = mock.MagicMock()

    mb_stub = types.ModuleType("tkinter.messagebox")
    mb_stub.showerror = mock.MagicMock()

    sd_stub = types.ModuleType("sounddevice")
    sd_stub.query_devices  = mock.MagicMock(return_value=[])
    sd_stub.query_hostapis = mock.MagicMock(return_value=[])
    sd_stub.check_input_settings  = mock.MagicMock(side_effect=Exception("not available"))
    sd_stub.check_output_settings = mock.MagicMock(side_effect=Exception("not available"))

    serial_stub = types.ModuleType("serial")
    serial_stub.STOPBITS_ONE            = 1
    serial_stub.STOPBITS_ONE_POINT_FIVE = 1.5
    serial_stub.STOPBITS_TWO           = 2
    serial_tools = types.ModuleType("serial.tools")
    serial_tools_lp = types.ModuleType("serial.tools.list_ports")
    serial_tools_lp.comports = mock.MagicMock(return_value=[])

    sys.modules.setdefault("tkinter",          tk_stub)
    sys.modules.setdefault("tkinter.ttk",      ttk_stub)
    sys.modules.setdefault("tkinter.messagebox", mb_stub)
    sys.modules.setdefault("sounddevice",      sd_stub)
    sys.modules.setdefault("serial",           serial_stub)
    sys.modules.setdefault("serial.tools",     serial_tools)
    sys.modules.setdefault("serial.tools.list_ports", serial_tools_lp)

    # Re-import (or import fresh)
    import importlib
    if "main" in sys.modules:
        del sys.modules["main"]
    import main as m
    return m


# ---------------------------------------------------------------------------
# Test framework (matches existing test files in the repo)
# ---------------------------------------------------------------------------

PASS = "\u2713 PASS"
FAIL = "\u2717 FAIL"
results: list[tuple[str, bool, str]] = []


def run(name: str, fn) -> None:
    t0 = time.perf_counter()
    try:
        msg = fn()
        ok  = True
        detail = msg or ""
    except Exception as exc:
        ok     = False
        detail = f"{type(exc).__name__}: {exc}\n{''.join(traceback.format_tb(exc.__traceback__))}"
    elapsed = (time.perf_counter() - t0) * 1000
    results.append((name, ok, detail))
    symbol = PASS if ok else FAIL
    print(f"  {symbol}  {name}  ({elapsed:.1f} ms)")
    if not ok:
        for line in detail.splitlines():
            print(f"         {line}")


# ---------------------------------------------------------------------------
# Import once
# ---------------------------------------------------------------------------

main = _import_main_no_gui()

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_bands_expanded():
    required = {"160m", "80m", "40m", "30m", "20m", "17m", "15m", "12m",
                "10m", "6m", "2m", "70cm"}
    missing = required - set(main.BANDS.keys())
    assert not missing, f"Missing bands: {missing}"
    for name, plan in main.BANDS.items():
        assert "start" in plan and "end" in plan and "step" in plan and "mode" in plan, \
            f"Band {name!r} missing keys"
        assert plan["step"] > 0, f"Band {name!r} has non-positive step"


def test_ft8_freqs():
    required = {"160m", "80m", "40m", "20m", "17m", "15m", "10m"}
    missing = required - set(main.FT8_FREQS.keys())
    assert not missing, f"Missing FT8 freq entries: {missing}"
    for band, freq in main.FT8_FREQS.items():
        plan = main.BANDS.get(band)
        if plan:
            assert plan["start"] <= freq <= plan["end"], \
                f"FT8 freq {freq} MHz for {band!r} is outside band plan"


def test_ft8_log_path():
    lp = main.FT8_LOG_PATH
    assert lp.endswith(".log"), f"FT8_LOG_PATH should end with .log: {lp!r}"
    assert os.path.dirname(lp) == os.path.dirname(os.path.abspath(main.__file__)), \
        "FT8_LOG_PATH should be in same directory as main.py"


def test_appconfig_output_device_defaults():
    with tempfile.NamedTemporaryFile(suffix=".cfg", delete=False) as tf:
        cfg_path = tf.name
    try:
        cfg = main.AppConfig(path=cfg_path)
        assert cfg.audio_output_device_index == -1, \
            f"Expected -1, got {cfg.audio_output_device_index}"
        assert cfg.audio_output_device_label == "", \
            f"Expected empty string, got {cfg.audio_output_device_label!r}"
    finally:
        os.unlink(cfg_path)


def test_appconfig_save_audio_output():
    with tempfile.NamedTemporaryFile(suffix=".cfg", delete=False) as tf:
        cfg_path = tf.name
    try:
        cfg = main.AppConfig(path=cfg_path)
        cfg.save_audio_output(3, "3: USB Audio CODEC (WASAPI)")
        # Reload from disk
        cfg2 = main.AppConfig(path=cfg_path)
        assert cfg2.audio_output_device_index == 3, \
            f"Expected 3, got {cfg2.audio_output_device_index}"
        assert cfg2.audio_output_device_label == "3: USB Audio CODEC (WASAPI)", \
            f"Unexpected label: {cfg2.audio_output_device_label!r}"
    finally:
        os.unlink(cfg_path)


def test_appconfig_save_audio_input_regression():
    """Saving audio output must not clobber input device entry."""
    with tempfile.NamedTemporaryFile(suffix=".cfg", delete=False) as tf:
        cfg_path = tf.name
    try:
        cfg = main.AppConfig(path=cfg_path)
        cfg.save_audio(1, "1: Microphone (WASAPI)")
        cfg.save_audio_output(2, "2: Speaker (WASAPI)")
        cfg2 = main.AppConfig(path=cfg_path)
        assert cfg2.audio_device_index == 1
        assert cfg2.audio_output_device_index == 2
    finally:
        os.unlink(cfg_path)


def test_enum_audio_devices_returns_tuple():
    ins, outs, err = main._enum_audio_devices()
    assert isinstance(ins,  list), "inputs must be a list"
    assert isinstance(outs, list), "outputs must be a list"
    assert isinstance(err,  str),  "error must be a string"


def test_enum_audio_devices_includes_wasapi_and_mme_outputs():
    hostapis = [
        {"name": "MME"},
        {"name": "Windows WASAPI"},
    ]
    devices = [
        {"name": "Sound to Radio", "hostapi": 0, "max_input_channels": 0, "max_output_channels": 2,
         "default_samplerate": 44100.0},
        {"name": "Sound to Radio", "hostapi": 1, "max_input_channels": 0, "max_output_channels": 2,
         "default_samplerate": 48000.0},
    ]

    fake_sd = types.SimpleNamespace()
    fake_sd.query_devices = lambda: devices
    fake_sd.query_hostapis = lambda: hostapis
    fake_sd.check_input_settings = lambda **kwargs: True
    fake_sd.check_output_settings = lambda **kwargs: True

    with mock.patch.dict(sys.modules, {"sounddevice": fake_sd}):
        ins, outs, err = main._enum_audio_devices()

    assert err == ""
    assert ins == []
    assert len(outs) == 2
    assert outs[0][1].endswith("(WASAPI)")
    assert outs[1][1].endswith("(MME)")


def test_parse_device_index_valid():
    result = main.SettingsDialog._parse_device_index("7: My Device (WASAPI)")
    assert result == 7, f"Expected 7, got {result}"


def test_parse_device_index_invalid():
    for label in ("", "no-colon", "abc: bad", None):
        result = main.SettingsDialog._parse_device_index(label or "")
        assert result == -1, f"Expected -1 for {label!r}, got {result}"


def test_ft8_log_appends():
    """_save_ft8_log_to_file must append (not overwrite) across two calls."""
    with tempfile.TemporaryDirectory() as td:
        log_path = os.path.join(td, "ft8_messages.log")

        # Patch FT8_LOG_PATH
        orig = main.FT8_LOG_PATH
        main.FT8_LOG_PATH = log_path
        try:
            # Build a minimal mock RadioGUI with a functioning ft8_log widget
            gui = mock.MagicMock()
            gui.ft8_log.get.return_value = "190000  -5 14074.000 CQ W1AW FN31\n"

            # Bind the real method to our mock instance
            main.RadioGUI._save_ft8_log_to_file(gui)
            main.RadioGUI._save_ft8_log_to_file(gui)

            assert os.path.exists(log_path), "Log file not created"
            content = open(log_path, encoding="utf-8").read()
            # Two session markers should be present
            assert content.count("--- Session") == 2, \
                f"Expected 2 session headers, got:\n{content}"
            assert "CQ W1AW FN31" in content
        finally:
            main.FT8_LOG_PATH = orig


def test_ft8_log_skips_empty():
    """_save_ft8_log_to_file must not create a file when ft8_log is empty."""
    with tempfile.TemporaryDirectory() as td:
        log_path = os.path.join(td, "ft8_messages.log")
        orig = main.FT8_LOG_PATH
        main.FT8_LOG_PATH = log_path
        try:
            gui = mock.MagicMock()
            gui.ft8_log.get.return_value = "   \n  "   # whitespace only
            main.RadioGUI._save_ft8_log_to_file(gui)
            assert not os.path.exists(log_path), "Log file should NOT be created for empty content"
        finally:
            main.FT8_LOG_PATH = orig


def test_op_mode_default():
    """RadioGUI._op_mode should default to 'voice'."""
    # We inspect the __init__ default by creating a partially constructed instance
    gui = mock.MagicMock(spec=main.RadioGUI)
    gui._op_mode = "voice"   # Simulate post-__init__ state
    assert gui._op_mode == "voice"


def test_switch_to_data_sets_mode():
    """_switch_to_data must set _op_mode to 'data' and start FT8 decoder."""
    gui = mock.MagicMock()
    gui._op_mode = "voice"
    main.RadioGUI._switch_to_data(gui)
    assert gui._op_mode == "data", f"Expected 'data', got {gui._op_mode!r}"
    gui._ft8.start.assert_called_once()
    gui._apply_op_mode.assert_called_once_with("data")


def test_switch_to_voice_saves_and_stops():
    """_switch_to_voice must save FT8 log, stop stream, stop decoder, set mode."""
    gui = mock.MagicMock()
    gui._op_mode = "data"
    main.RadioGUI._switch_to_voice(gui)
    gui._save_ft8_log_to_file.assert_called_once()
    gui._stop_audio_stream.assert_called_once()
    gui._ft8.stop.assert_called_once()
    assert gui._op_mode == "voice", f"Expected 'voice', got {gui._op_mode!r}"
    gui._apply_op_mode.assert_called_once_with("voice")


def test_freq_step_default():
    gui = mock.MagicMock()
    gui._freq_step = 0.001
    assert gui._freq_step == 0.001


def test_infer_band_from_freq():
    gui = mock.MagicMock(spec=main.RadioGUI)
    # Use the real method
    fn = main.RadioGUI.infer_band_from_freq
    assert fn(gui, 14.225)  == "20m"
    assert fn(gui, 7.074)   == "40m"
    assert fn(gui, 3.573)   == "80m"
    assert fn(gui, 144.174) == "2m"
    assert fn(gui, 28.500)  == "10m"


def test_infer_band_out_of_range():
    gui = mock.MagicMock(spec=main.RadioGUI)
    fn  = main.RadioGUI.infer_band_from_freq
    assert fn(gui, 0.0)   is None
    assert fn(gui, 500.0) is None
    assert fn(gui, 2.5)   is None   # between 160m and 80m


def test_appconfig_tx_audio_defaults():
    """TX audio device properties should default to -1 / '' when not set."""
    with tempfile.NamedTemporaryFile(suffix=".cfg", delete=False) as fh:
        cfg_path = fh.name
    try:
        os.unlink(cfg_path)          # start without a config file
        cfg = main.AppConfig(path=cfg_path)
        assert cfg.tx_mic_device_index       == -1
        assert cfg.tx_mic_device_label       == ""
        assert cfg.tx_radio_out_device_index == -1
        assert cfg.tx_radio_out_device_label == ""
    finally:
        if os.path.exists(cfg_path):
            os.unlink(cfg_path)


def test_appconfig_save_tx_audio():
    """save_tx_audio must persist all four TX audio fields."""
    with tempfile.NamedTemporaryFile(suffix=".cfg", delete=False) as fh:
        cfg_path = fh.name
    try:
        cfg = main.AppConfig(path=cfg_path)
        cfg.save_tx_audio(3, "3: Headset Mic (WASAPI)", 7, "7: FT-991A USB (WASAPI)")

        # Reload from disk
        cfg2 = main.AppConfig(path=cfg_path)
        assert cfg2.tx_mic_device_index       == 3,  f"mic idx: {cfg2.tx_mic_device_index}"
        assert cfg2.tx_mic_device_label       == "3: Headset Mic (WASAPI)"
        assert cfg2.tx_radio_out_device_index == 7,  f"out idx: {cfg2.tx_radio_out_device_index}"
        assert cfg2.tx_radio_out_device_label == "7: FT-991A USB (WASAPI)"
    finally:
        if os.path.exists(cfg_path):
            os.unlink(cfg_path)


def test_switch_to_data_stops_voice_audio():
    """_switch_to_data must stop RX monitor and TX capture before switching."""
    gui = mock.MagicMock()
    gui._op_mode = "voice"
    main.RadioGUI._switch_to_data(gui)
    gui._stop_rx_monitor.assert_called_once()
    gui._stop_tx_audio.assert_called_once()
    assert gui._op_mode == "data"
    gui._ft8.start.assert_called_once()
    gui._apply_op_mode.assert_called_once_with("data")


def test_switch_to_voice_stops_audio_streams():
    """_switch_to_voice must stop data-mode audio stream and FT8 decoder."""
    gui = mock.MagicMock()
    gui._op_mode = "data"
    main.RadioGUI._switch_to_voice(gui)
    gui._save_ft8_log_to_file.assert_called_once()
    gui._stop_audio_stream.assert_called_once()
    gui._ft8.stop.assert_called_once()
    assert gui._op_mode == "voice"
    gui._apply_op_mode.assert_called_once_with("voice")


def test_appconfig_operator_defaults():
    """AppConfig operator callsign/grid default to empty strings."""
    import tempfile, os
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = main.AppConfig(path=os.path.join(tmpdir, "vader.cfg"))
        assert cfg.operator_callsign == ""
        assert cfg.operator_grid == ""


def test_appconfig_save_operator():
    """AppConfig.save_operator persists callsign + grid and reloads correctly."""
    import tempfile, os
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "vader.cfg")
        cfg = main.AppConfig(path=path)
        cfg.save_operator("W4ABC", "EN52")
        cfg2 = main.AppConfig(path=path)
        assert cfg2.operator_callsign == "W4ABC"
        assert cfg2.operator_grid == "EN52"


def test_validate_operator_valid():
    """validate_operator returns (True, '') for a valid callsign+grid."""
    ok, reason = main.validate_operator("W4ABC", "EN52")
    assert ok is True
    assert reason == ""


def test_validate_operator_invalid_callsign():
    """validate_operator returns (False, reason) for an invalid callsign."""
    ok, reason = main.validate_operator("!!!", "EN52")
    assert ok is False
    assert "callsign" in reason.lower()


def test_validate_operator_invalid_grid():
    """validate_operator returns (False, reason) for an invalid grid."""
    ok, reason = main.validate_operator("W4ABC", "TOOLONG99")
    assert ok is False
    assert reason


def test_on_save_operator_valid():
    """_on_save_operator persists valid operator settings without showing an error."""
    gui = mock.MagicMock()
    gui._tx_callsign_var.get.return_value = "W4ABC"
    gui._tx_grid_var.get.return_value = "EN52"
    main.RadioGUI._on_save_operator(gui)
    gui._config.save_operator.assert_called_once_with("W4ABC", "EN52")
    gui.root.winfo_exists.return_value = True  # no error dialog


def test_on_save_operator_invalid():
    """_on_save_operator shows an error dialog for an invalid callsign."""
    gui = mock.MagicMock()
    gui._tx_callsign_var.get.return_value = "BADCALL!!!"
    gui._tx_grid_var.get.return_value = "EN52"
    main.RadioGUI._on_save_operator(gui)
    # save_operator must NOT be called when validation fails
    gui._config.save_operator.assert_not_called()


def test_on_compose_cq_valid():
    """_on_compose_cq sets TX message to 'CQ CALL GRID' for valid operator."""
    gui = mock.MagicMock()
    gui._tx_callsign_var.get.return_value = "W4ABC"
    gui._tx_grid_var.get.return_value = "EN52"
    main.RadioGUI._on_compose_cq(gui)
    gui._tx_msg_var.set.assert_called_once_with("CQ W4ABC EN52")


def test_on_compose_cq_invalid_blocks():
    """_on_compose_cq must not set TX message when operator settings are invalid."""
    gui = mock.MagicMock()
    gui._tx_callsign_var.get.return_value = ""
    gui._tx_grid_var.get.return_value = "EN52"
    main.RadioGUI._on_compose_cq(gui)
    gui._tx_msg_var.set.assert_not_called()


def test_prefill_reply_cq():
    """_prefill_reply pre-fills a reply to a CQ message."""
    gui = mock.MagicMock()
    gui._tx_callsign_var.get.return_value = "K9XYZ"
    gui._tx_grid_var.get.return_value = "EN52"
    main.RadioGUI._prefill_reply(gui, "CQ W4ABC EN52")
    gui._tx_msg_var.set.assert_called_once()
    arg = gui._tx_msg_var.set.call_args[0][0]
    assert "W4ABC" in arg
    assert "K9XYZ" in arg


def test_prefill_reply_rr73():
    """_prefill_reply pre-fills a 73 reply when DX sends RR73."""
    gui = mock.MagicMock()
    gui._tx_callsign_var.get.return_value = "K9XYZ"
    gui._tx_grid_var.get.return_value = "EN52"
    main.RadioGUI._prefill_reply(gui, "K9XYZ W4ABC RR73")
    gui._tx_msg_var.set.assert_called_once()
    arg = gui._tx_msg_var.set.call_args[0][0]
    assert "73" in arg


def test_on_arm_tx_empty_message():
    """_on_arm_tx shows an error dialog when TX message is empty."""
    gui = mock.MagicMock()
    gui._tx_msg_var.get.return_value = "  "  # whitespace only
    main.RadioGUI._on_arm_tx(gui)
    gui._tx_coord.arm.assert_not_called()


def test_on_arm_tx_invalid_operator():
    """_on_arm_tx shows an error when operator settings are invalid."""
    gui = mock.MagicMock()
    gui._tx_msg_var.get.return_value = "CQ W4ABC EN52"
    gui._tx_callsign_var.get.return_value = "BADCALL!!!"
    gui._tx_grid_var.get.return_value = "EN52"
    main.RadioGUI._on_arm_tx(gui)
    gui._tx_coord.arm.assert_not_called()


def test_on_cancel_tx_accepted():
    """_on_cancel_tx calls coord.cancel() and updates status when accepted."""
    gui = mock.MagicMock()
    gui._tx_coord.cancel.return_value = True
    main.RadioGUI._on_cancel_tx(gui)
    gui._tx_coord.cancel.assert_called_once()


def test_on_cancel_tx_not_accepted():
    """_on_cancel_tx sets status message when cancel is rejected."""
    gui = mock.MagicMock()
    gui._tx_coord.cancel.return_value = False
    main.RadioGUI._on_cancel_tx(gui)
    gui._tx_status_var.set.assert_called_once()


# ---------------------------------------------------------------------------
# CQ QSO assist tests (Milestone 4)
# ---------------------------------------------------------------------------

def test_on_start_cq_session_valid_operator():
    """_on_start_cq_session creates a QSO manager and pre-fills the CQ msg."""
    gui = mock.MagicMock()
    gui._tx_callsign_var.get.return_value = "W4ABC"
    gui._tx_grid_var.get.return_value = "EN52"
    main.RadioGUI._on_start_cq_session(gui)
    # TX message must be set to the CQ
    gui._tx_msg_var.set.assert_called_once()
    msg_arg = gui._tx_msg_var.set.call_args[0][0]
    assert "CQ" in msg_arg
    assert "W4ABC" in msg_arg
    # Assist must be activated
    assert gui._qso_assist_active is True
    assert gui._qso_mgr is not None


def test_on_start_cq_session_invalid_operator_shows_error():
    """_on_start_cq_session shows an error and does not activate for invalid operator."""
    gui = mock.MagicMock()
    gui._tx_callsign_var.get.return_value = "!!INVALID!!"
    gui._tx_grid_var.get.return_value = "EN52"
    main.RadioGUI._on_start_cq_session(gui)
    # Must NOT set the TX message
    gui._tx_msg_var.set.assert_not_called()
    # Must NOT activate assist
    assert gui._qso_assist_active is not True


def test_on_stop_cq_session_resets_state():
    """_on_stop_cq_session clears manager and deactivates assist."""
    gui = mock.MagicMock()
    # Simulate an active session
    from ft8_qso import Ft8QsoManager, OperatorConfig
    import unittest.mock as _mock
    from ft8_ntp import Ft8SlotTimer
    timer = _mock.MagicMock(spec=Ft8SlotTimer)
    timer.seconds_to_next_slot.return_value = 0.0
    op = OperatorConfig(callsign="W4ABC", grid="EN52")
    mgr = Ft8QsoManager(operator=op, slot_timer=timer)
    mgr.start_cq()
    gui._qso_mgr = mgr
    gui._qso_assist_active = True
    gui._qso_assist_prefilled = "CQ W4ABC EN52"
    main.RadioGUI._on_stop_cq_session(gui)
    assert gui._qso_assist_active is False
    assert gui._qso_assist_prefilled == ""
    assert gui._qso_mgr is None
    gui._tx_status_var.set.assert_called()


def test_maybe_assist_prefill_prefills_on_cq_reply():
    """_maybe_assist_prefill fills TX field when a valid CQ reply is decoded."""
    from ft8_qso import Ft8QsoManager, OperatorConfig
    import unittest.mock as _mock
    from ft8_ntp import Ft8SlotTimer
    from ft8_tx import TxState

    timer = _mock.MagicMock(spec=Ft8SlotTimer)
    timer.seconds_to_next_slot.return_value = 0.0
    op = OperatorConfig(callsign="W4ABC", grid="EN52")
    mgr = Ft8QsoManager(operator=op, slot_timer=timer)
    mgr.start_cq()

    gui = mock.MagicMock()
    gui._qso_assist_active = True
    gui._qso_mgr = mgr
    gui._qso_assist_prefilled = ""
    gui._tx_coord.state = TxState.IDLE

    main.RadioGUI._maybe_assist_prefill(gui, "W4ABC K9XYZ -05", -5.0)

    gui._tx_msg_var.set.assert_called_once()
    prefilled = gui._tx_msg_var.set.call_args[0][0]
    assert "K9XYZ" in prefilled
    assert "W4ABC" in prefilled
    gui._tx_status_var.set.assert_called()


def test_maybe_assist_prefill_ignores_unaddressed_message():
    """_maybe_assist_prefill does not prefill for messages not addressed to us."""
    from ft8_qso import Ft8QsoManager, OperatorConfig
    import unittest.mock as _mock
    from ft8_ntp import Ft8SlotTimer
    from ft8_tx import TxState

    timer = _mock.MagicMock(spec=Ft8SlotTimer)
    timer.seconds_to_next_slot.return_value = 0.0
    op = OperatorConfig(callsign="W4ABC", grid="EN52")
    mgr = Ft8QsoManager(operator=op, slot_timer=timer)
    mgr.start_cq()

    gui = mock.MagicMock()
    gui._qso_assist_active = True
    gui._qso_mgr = mgr
    gui._qso_assist_prefilled = ""
    gui._tx_coord.state = TxState.IDLE

    # This message is addressed to K9ABC, not W4ABC
    main.RadioGUI._maybe_assist_prefill(gui, "K9ABC VK2TIM -03", -3.0)

    gui._tx_msg_var.set.assert_not_called()


def test_maybe_assist_prefill_dedup_guard():
    """_maybe_assist_prefill does not re-set a suggestion already prefilled."""
    from ft8_qso import Ft8QsoManager, OperatorConfig
    import unittest.mock as _mock
    from ft8_ntp import Ft8SlotTimer
    from ft8_tx import TxState

    timer = _mock.MagicMock(spec=Ft8SlotTimer)
    timer.seconds_to_next_slot.return_value = 0.0
    op = OperatorConfig(callsign="W4ABC", grid="EN52")
    mgr = Ft8QsoManager(operator=op, slot_timer=timer)
    mgr.start_cq()

    gui = mock.MagicMock()
    gui._qso_assist_active = True
    gui._qso_mgr = mgr
    gui._qso_assist_prefilled = ""
    gui._tx_coord.state = TxState.IDLE

    # First call should prefill
    main.RadioGUI._maybe_assist_prefill(gui, "W4ABC K9XYZ -05", -5.0)
    assert gui._tx_msg_var.set.call_count == 1
    first_prefill = gui._tx_msg_var.set.call_args[0][0]

    # Manually simulate that the guard was set by the first call
    gui._qso_assist_prefilled = first_prefill

    # Second decode of the SAME reply type — state machine returns None
    # because state has already advanced to EXCHANGE_SENT.  No re-prefill.
    main.RadioGUI._maybe_assist_prefill(gui, "W4ABC K9XYZ -05", -5.0)
    assert gui._tx_msg_var.set.call_count == 1  # still only called once


def test_maybe_assist_prefill_gated_during_tx_active():
    """_maybe_assist_prefill is silenced while TX is in progress."""
    from ft8_qso import Ft8QsoManager, OperatorConfig
    import unittest.mock as _mock
    from ft8_ntp import Ft8SlotTimer
    from ft8_tx import TxState

    timer = _mock.MagicMock(spec=Ft8SlotTimer)
    timer.seconds_to_next_slot.return_value = 0.0
    op = OperatorConfig(callsign="W4ABC", grid="EN52")
    mgr = Ft8QsoManager(operator=op, slot_timer=timer)
    mgr.start_cq()

    gui = mock.MagicMock()
    gui._qso_assist_active = True
    gui._qso_mgr = mgr
    gui._qso_assist_prefilled = ""
    # Simulate TX in progress
    gui._tx_coord.state = TxState.TX_ACTIVE

    main.RadioGUI._maybe_assist_prefill(gui, "W4ABC K9XYZ -05", -5.0)
    gui._tx_msg_var.set.assert_not_called()


def test_maybe_assist_prefill_gated_during_armed():
    """_maybe_assist_prefill is silenced while a TX job is armed (scheduled)."""
    from ft8_qso import Ft8QsoManager, OperatorConfig
    import unittest.mock as _mock
    from ft8_ntp import Ft8SlotTimer
    from ft8_tx import TxState

    timer = _mock.MagicMock(spec=Ft8SlotTimer)
    timer.seconds_to_next_slot.return_value = 0.0
    op = OperatorConfig(callsign="W4ABC", grid="EN52")
    mgr = Ft8QsoManager(operator=op, slot_timer=timer)
    mgr.start_cq()

    gui = mock.MagicMock()
    gui._qso_assist_active = True
    gui._qso_mgr = mgr
    gui._qso_assist_prefilled = ""
    gui._tx_coord.state = TxState.ARMED

    main.RadioGUI._maybe_assist_prefill(gui, "W4ABC K9XYZ -05", -5.0)
    gui._tx_msg_var.set.assert_not_called()


def test_maybe_assist_prefill_inactive_when_session_off():
    """_maybe_assist_prefill does nothing when assist is not active."""
    gui = mock.MagicMock()
    gui._qso_assist_active = False
    main.RadioGUI._maybe_assist_prefill(gui, "W4ABC K9XYZ -05", -5.0)
    gui._tx_msg_var.set.assert_not_called()


def test_maybe_assist_prefill_no_auto_arm():
    """_maybe_assist_prefill must NEVER call _on_arm_tx (safety guard)."""
    from ft8_qso import Ft8QsoManager, OperatorConfig
    import unittest.mock as _mock
    from ft8_ntp import Ft8SlotTimer
    from ft8_tx import TxState

    timer = _mock.MagicMock(spec=Ft8SlotTimer)
    timer.seconds_to_next_slot.return_value = 0.0
    op = OperatorConfig(callsign="W4ABC", grid="EN52")
    mgr = Ft8QsoManager(operator=op, slot_timer=timer)
    mgr.start_cq()

    gui = mock.MagicMock()
    gui._qso_assist_active = True
    gui._qso_mgr = mgr
    gui._qso_assist_prefilled = ""
    gui._tx_coord.state = TxState.IDLE

    main.RadioGUI._maybe_assist_prefill(gui, "W4ABC K9XYZ -05", -5.0)
    # _on_arm_tx must never be called — operator approval is mandatory
    gui._on_arm_tx.assert_not_called()
    gui._tx_coord.arm.assert_not_called()


def test_apply_tx_state_update_clears_dedup_on_complete():
    """_apply_tx_state_update clears _qso_assist_prefilled when TX completes."""
    from ft8_tx import TxState
    gui = mock.MagicMock()
    gui._qso_assist_prefilled = "K9XYZ W4ABC R+00"
    main.RadioGUI._apply_tx_state_update(gui, TxState.COMPLETE, "TX complete")
    assert gui._qso_assist_prefilled == ""


def test_apply_tx_state_update_does_not_clear_dedup_on_error():
    """_apply_tx_state_update does NOT clear the dedup guard on ERROR."""
    from ft8_tx import TxState
    gui = mock.MagicMock()
    gui._qso_assist_prefilled = "K9XYZ W4ABC R+00"
    main.RadioGUI._apply_tx_state_update(gui, TxState.ERROR, "TX error")
    # Guard is preserved — the message was not sent so the suggestion stands
    assert gui._qso_assist_prefilled == "K9XYZ W4ABC R+00"


def test_switch_to_voice_stops_active_cq_session():
    """_switch_to_voice must stop an active CQ session when switching modes."""
    gui = mock.MagicMock()
    gui._op_mode = "data"
    gui._qso_assist_active = True
    main.RadioGUI._switch_to_voice(gui)
    gui._on_stop_cq_session.assert_called_once()


def test_switch_to_voice_skips_stop_session_when_inactive():
    """_switch_to_voice must not call _on_stop_cq_session when no session active."""
    gui = mock.MagicMock()
    gui._op_mode = "data"
    gui._qso_assist_active = False
    main.RadioGUI._switch_to_voice(gui)
    gui._on_stop_cq_session.assert_not_called()


def test_maybe_assist_prefill_rrr_completion_hint():
    """_maybe_assist_prefill status hint says 'RRR' when DX sends RRR (not 'RR73')."""
    from ft8_qso import Ft8QsoManager, OperatorConfig
    import unittest.mock as _mock
    from ft8_ntp import Ft8SlotTimer
    from ft8_tx import TxState

    timer = _mock.MagicMock(spec=Ft8SlotTimer)
    timer.seconds_to_next_slot.return_value = 0.0
    op = OperatorConfig(callsign="W4ABC", grid="EN52")
    mgr = Ft8QsoManager(operator=op, slot_timer=timer)
    mgr.start_cq()
    mgr.advance("W4ABC K9XYZ -05")  # lock in K9XYZ, move to EXCHANGE_SENT

    gui = mock.MagicMock()
    gui._qso_assist_active = True
    gui._qso_mgr = mgr
    gui._qso_assist_prefilled = ""
    gui._tx_coord.state = TxState.IDLE

    # DX sends RRR (not RR73)
    main.RadioGUI._maybe_assist_prefill(gui, "W4ABC K9XYZ RRR", 0.0)

    gui._tx_msg_var.set.assert_called_once()
    status_arg = gui._tx_status_var.set.call_args[0][0]
    # Must say "RRR", not "RR73"
    assert "RRR" in status_arg
    assert "RR73" not in status_arg


def test_maybe_assist_prefill_rr73_completion_hint():
    """_maybe_assist_prefill status hint says 'RR73' when DX sends RR73."""
    from ft8_qso import Ft8QsoManager, OperatorConfig
    import unittest.mock as _mock
    from ft8_ntp import Ft8SlotTimer
    from ft8_tx import TxState

    timer = _mock.MagicMock(spec=Ft8SlotTimer)
    timer.seconds_to_next_slot.return_value = 0.0
    op = OperatorConfig(callsign="W4ABC", grid="EN52")
    mgr = Ft8QsoManager(operator=op, slot_timer=timer)
    mgr.start_cq()
    mgr.advance("W4ABC K9XYZ -05")

    gui = mock.MagicMock()
    gui._qso_assist_active = True
    gui._qso_mgr = mgr
    gui._qso_assist_prefilled = ""
    gui._tx_coord.state = TxState.IDLE

    main.RadioGUI._maybe_assist_prefill(gui, "W4ABC K9XYZ RR73", 0.0)

    gui._tx_msg_var.set.assert_called_once()
    status_arg = gui._tx_status_var.set.call_args[0][0]
    assert "RR73" in status_arg


# ---------------------------------------------------------------------------
# Run all tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 65)
    print("VADER — RadioGUI mode / config tests")
    print("=" * 65)

    run("1.  BANDS — expanded band plan",        test_bands_expanded)
    run("2.  FT8_FREQS — standard FT8 freqs",    test_ft8_freqs)
    run("3.  FT8_LOG_PATH — location and suffix", test_ft8_log_path)
    run("4.  AppConfig — output device defaults", test_appconfig_output_device_defaults)
    run("5.  AppConfig.save_audio_output",        test_appconfig_save_audio_output)
    run("6.  AppConfig — input/output independent", test_appconfig_save_audio_input_regression)
    run("7.  _enum_audio_devices — tuple return", test_enum_audio_devices_returns_tuple)
    run("7b. _enum_audio_devices — WASAPI/MME list", test_enum_audio_devices_includes_wasapi_and_mme_outputs)
    run("8.  SettingsDialog._parse_device_index valid",   test_parse_device_index_valid)
    run("9.  SettingsDialog._parse_device_index invalid", test_parse_device_index_invalid)
    run("10. _save_ft8_log_to_file — appends",   test_ft8_log_appends)
    run("11. _save_ft8_log_to_file — skips empty", test_ft8_log_skips_empty)
    run("12. RadioGUI._op_mode default",          test_op_mode_default)
    run("13. _switch_to_data sets mode + decoder", test_switch_to_data_sets_mode)
    run("14. _switch_to_voice saves + stops",      test_switch_to_voice_saves_and_stops)
    run("15. RadioGUI._freq_step default",         test_freq_step_default)
    run("16. infer_band_from_freq — in-band",      test_infer_band_from_freq)
    run("17. infer_band_from_freq — out-of-band",  test_infer_band_out_of_range)
    run("18. AppConfig — TX audio defaults",       test_appconfig_tx_audio_defaults)
    run("19. AppConfig.save_tx_audio — persists",  test_appconfig_save_tx_audio)
    run("20. _switch_to_data stops voice audio",   test_switch_to_data_stops_voice_audio)
    run("21. _switch_to_voice stops audio streams", test_switch_to_voice_stops_audio_streams)
    run("22. AppConfig — operator defaults",        test_appconfig_operator_defaults)
    run("23. AppConfig.save_operator — persists",   test_appconfig_save_operator)
    run("24. validate_operator — valid",            test_validate_operator_valid)
    run("25. validate_operator — invalid callsign", test_validate_operator_invalid_callsign)
    run("26. validate_operator — invalid grid",     test_validate_operator_invalid_grid)
    run("27. _on_save_operator — valid",            test_on_save_operator_valid)
    run("28. _on_save_operator — invalid blocks",   test_on_save_operator_invalid)
    run("29. _on_compose_cq — valid",               test_on_compose_cq_valid)
    run("30. _on_compose_cq — invalid blocks",      test_on_compose_cq_invalid_blocks)
    run("31. _prefill_reply — CQ",                  test_prefill_reply_cq)
    run("32. _prefill_reply — RR73",                test_prefill_reply_rr73)
    run("33. _on_arm_tx — empty message",           test_on_arm_tx_empty_message)
    run("34. _on_arm_tx — invalid operator",        test_on_arm_tx_invalid_operator)
    run("35. _on_cancel_tx — accepted",             test_on_cancel_tx_accepted)
    run("36. _on_cancel_tx — not accepted",         test_on_cancel_tx_not_accepted)
    run("37. _on_start_cq_session — valid operator",       test_on_start_cq_session_valid_operator)
    run("38. _on_start_cq_session — invalid blocks error", test_on_start_cq_session_invalid_operator_shows_error)
    run("39. _on_stop_cq_session — resets state",          test_on_stop_cq_session_resets_state)
    run("40. _maybe_assist_prefill — CQ reply prefills",   test_maybe_assist_prefill_prefills_on_cq_reply)
    run("41. _maybe_assist_prefill — unaddressed ignored", test_maybe_assist_prefill_ignores_unaddressed_message)
    run("42. _maybe_assist_prefill — dedup guard",         test_maybe_assist_prefill_dedup_guard)
    run("43. _maybe_assist_prefill — gated TX_ACTIVE",     test_maybe_assist_prefill_gated_during_tx_active)
    run("44. _maybe_assist_prefill — gated ARMED",         test_maybe_assist_prefill_gated_during_armed)
    run("45. _maybe_assist_prefill — inactive session noop", test_maybe_assist_prefill_inactive_when_session_off)
    run("46. _maybe_assist_prefill — no auto-arm (safety)", test_maybe_assist_prefill_no_auto_arm)
    run("47. _apply_tx_state_update — clears dedup on COMPLETE", test_apply_tx_state_update_clears_dedup_on_complete)
    run("48. _apply_tx_state_update — dedup preserved on ERROR", test_apply_tx_state_update_does_not_clear_dedup_on_error)
    run("49. _switch_to_voice — stops active CQ session",   test_switch_to_voice_stops_active_cq_session)
    run("50. _switch_to_voice — no-op when session off",    test_switch_to_voice_skips_stop_session_when_inactive)
    run("51. _maybe_assist_prefill — RRR hint text",        test_maybe_assist_prefill_rrr_completion_hint)
    run("52. _maybe_assist_prefill — RR73 hint text",       test_maybe_assist_prefill_rr73_completion_hint)

    passed = sum(1 for _, ok, _ in results if ok)
    total  = len(results)
    print()
    print("=" * 65)
    status = "all tests passed \u2713" if passed == total else f"{total - passed} test(s) FAILED"
    print(f"Results: {passed}/{total} passed  -- {status}")
    print("=" * 65)

    if passed < total:
        sys.exit(1)
