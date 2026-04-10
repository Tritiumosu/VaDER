"""
test_ft991a_cat.py — Unit tests for the expanded Yaesu FT-991A CAT library.

All tests mock the serial connection so no physical radio is required.
Each test verifies the exact CAT command string sent to the radio (or the
correct parsing of a simulated radio response).

Run with:  python -m pytest test_ft991a_cat.py -v
"""
from __future__ import annotations

import io
import sys
import types
import unittest
from unittest.mock import MagicMock, patch, call

# ---------------------------------------------------------------------------
# Minimal serial stub so the import does not require pyserial to be installed
# ---------------------------------------------------------------------------
if "serial" not in sys.modules:
    _serial_stub = types.ModuleType("serial")
    _serial_stub.Serial = MagicMock
    _serial_stub.STOPBITS_ONE = 1
    _serial_stub.STOPBITS_ONE_POINT_FIVE = 1.5
    _serial_stub.STOPBITS_TWO = 2
    sys.modules["serial"] = _serial_stub

from ft991a_cat import Yaesu991AControl  # noqa: E402


# ---------------------------------------------------------------------------
# Helper: build a pre-connected controller whose serial port is mocked
# ---------------------------------------------------------------------------
def _make_ctrl() -> Yaesu991AControl:
    ctrl = Yaesu991AControl()
    mock_conn = MagicMock()
    mock_conn.is_open = True
    ctrl.conn = mock_conn
    return ctrl


def _last_write(ctrl) -> str:
    """Return the most recently written CAT command string (without trailing ;)."""
    raw = ctrl.conn.write.call_args[0][0]
    return raw.decode("ascii").rstrip(";")


def _set_response(ctrl, text: str):
    """Prime the mock to return *text* (without the trailing ;) on read_until."""
    ctrl.conn.read_until.return_value = f"{text};".encode("ascii")


# ===========================================================================
# Existing command regression tests
# ===========================================================================

class TestExistingCommands(unittest.TestCase):

    def test_set_frequency(self):
        ctrl = _make_ctrl()
        ctrl.set_frequency(14.225)
        self.assertEqual(_last_write(ctrl), "FA014225000")

    def test_set_frequency_no_stdout(self):
        """set_frequency must not print debug output (regression for removed debug print)."""
        ctrl = _make_ctrl()
        buf = io.StringIO()
        _orig, sys.stdout = sys.stdout, buf
        try:
            ctrl.set_frequency(14.074)
        finally:
            sys.stdout = _orig
        self.assertEqual(buf.getvalue(), "", "set_frequency must not print to stdout")

    def test_get_frequency(self):
        ctrl = _make_ctrl()
        _set_response(ctrl, "FA014250000")
        self.assertAlmostEqual(ctrl.get_frequency(), 14.25)

    def test_set_mode(self):
        ctrl = _make_ctrl()
        ctrl.set_mode("USB")
        self.assertEqual(_last_write(ctrl), "MD02")

    def test_set_mode_c4fm(self):
        ctrl = _make_ctrl()
        ctrl.set_mode("C4FM")
        self.assertEqual(_last_write(ctrl), "MD0E")

    def test_get_mode(self):
        ctrl = _make_ctrl()
        _set_response(ctrl, "MD02")
        self.assertEqual(ctrl.get_mode(), "USB")

    def test_set_rf_power_clamp(self):
        ctrl = _make_ctrl()
        ctrl.set_rf_power(200)
        self.assertEqual(_last_write(ctrl), "PC100")

    def test_set_rf_power_min_clamp(self):
        ctrl = _make_ctrl()
        ctrl.set_rf_power(1)
        self.assertEqual(_last_write(ctrl), "PC005")

    def test_ptt_on(self):
        ctrl = _make_ctrl()
        ctrl.ptt_on()
        self.assertEqual(_last_write(ctrl), "TX1")

    def test_ptt_off(self):
        ctrl = _make_ctrl()
        ctrl.ptt_off()
        self.assertEqual(_last_write(ctrl), "TX0")


# ===========================================================================
# AB – VFO-A to VFO-B
# ===========================================================================

class TestAB(unittest.TestCase):

    def test_vfo_a_to_b(self):
        ctrl = _make_ctrl()
        ctrl.vfo_a_to_b()
        self.assertEqual(_last_write(ctrl), "AB")


# ===========================================================================
# AC – Antenna Tuner Control
# ===========================================================================

class TestAC(unittest.TestCase):

    def test_set_tuner_off(self):
        ctrl = _make_ctrl()
        ctrl.set_antenna_tuner(0)
        self.assertEqual(_last_write(ctrl), "AC000")

    def test_set_tuner_on(self):
        ctrl = _make_ctrl()
        ctrl.set_antenna_tuner(1)
        self.assertEqual(_last_write(ctrl), "AC001")

    def test_set_tuner_start(self):
        ctrl = _make_ctrl()
        ctrl.set_antenna_tuner(2)
        self.assertEqual(_last_write(ctrl), "AC002")

    def test_set_tuner_invalid(self):
        ctrl = _make_ctrl()
        result = ctrl.set_antenna_tuner(5)
        self.assertFalse(result)

    def test_get_tuner(self):
        ctrl = _make_ctrl()
        _set_response(ctrl, "AC001")
        self.assertEqual(ctrl.get_antenna_tuner(), 1)

    def test_get_tuner_none_on_bad_response(self):
        ctrl = _make_ctrl()
        _set_response(ctrl, "??")
        self.assertIsNone(ctrl.get_antenna_tuner())


# ===========================================================================
# AG – AF Gain
# ===========================================================================

class TestAG(unittest.TestCase):

    def test_set_af_gain(self):
        ctrl = _make_ctrl()
        ctrl.set_af_gain(128)
        self.assertEqual(_last_write(ctrl), "AG0128")

    def test_set_af_gain_clamp_max(self):
        ctrl = _make_ctrl()
        ctrl.set_af_gain(300)
        self.assertEqual(_last_write(ctrl), "AG0255")

    def test_set_af_gain_clamp_min(self):
        ctrl = _make_ctrl()
        ctrl.set_af_gain(-5)
        self.assertEqual(_last_write(ctrl), "AG0000")

    def test_get_af_gain(self):
        ctrl = _make_ctrl()
        _set_response(ctrl, "AG0200")
        self.assertEqual(ctrl.get_af_gain(), 200)


# ===========================================================================
# AI – Auto Information
# ===========================================================================

class TestAI(unittest.TestCase):

    def test_set_ai_on(self):
        ctrl = _make_ctrl()
        ctrl.set_auto_information(True)
        self.assertEqual(_last_write(ctrl), "AI1")

    def test_set_ai_off(self):
        ctrl = _make_ctrl()
        ctrl.set_auto_information(False)
        self.assertEqual(_last_write(ctrl), "AI0")

    def test_get_ai(self):
        ctrl = _make_ctrl()
        _set_response(ctrl, "AI1")
        self.assertEqual(ctrl.get_auto_information(), 1)


# ===========================================================================
# AM – VFO-A to Memory
# ===========================================================================

class TestAM(unittest.TestCase):

    def test_vfo_a_to_memory(self):
        ctrl = _make_ctrl()
        ctrl.vfo_a_to_memory()
        self.assertEqual(_last_write(ctrl), "AM")


# ===========================================================================
# BA – VFO-B to VFO-A
# ===========================================================================

class TestBA(unittest.TestCase):

    def test_vfo_b_to_a(self):
        ctrl = _make_ctrl()
        ctrl.vfo_b_to_a()
        self.assertEqual(_last_write(ctrl), "BA")


# ===========================================================================
# BC – Auto Notch
# ===========================================================================

class TestBC(unittest.TestCase):

    def test_set_auto_notch_on(self):
        ctrl = _make_ctrl()
        ctrl.set_auto_notch(True)
        self.assertEqual(_last_write(ctrl), "BC01")

    def test_set_auto_notch_off(self):
        ctrl = _make_ctrl()
        ctrl.set_auto_notch(False)
        self.assertEqual(_last_write(ctrl), "BC00")

    def test_get_auto_notch(self):
        ctrl = _make_ctrl()
        _set_response(ctrl, "BC01")
        self.assertEqual(ctrl.get_auto_notch(), 1)


# ===========================================================================
# BD – Band Down
# ===========================================================================

class TestBD(unittest.TestCase):

    def test_band_down(self):
        ctrl = _make_ctrl()
        ctrl.band_down()
        self.assertEqual(_last_write(ctrl), "BD0")


# ===========================================================================
# BI – Break-In
# ===========================================================================

class TestBI(unittest.TestCase):

    def test_set_break_in_on(self):
        ctrl = _make_ctrl()
        ctrl.set_break_in(True)
        self.assertEqual(_last_write(ctrl), "BI1")

    def test_get_break_in(self):
        ctrl = _make_ctrl()
        _set_response(ctrl, "BI0")
        self.assertEqual(ctrl.get_break_in(), 0)


# ===========================================================================
# BP – Manual Notch
# ===========================================================================

class TestBP(unittest.TestCase):

    def test_set_manual_notch_on(self):
        ctrl = _make_ctrl()
        ctrl.set_manual_notch(0, 1)
        self.assertEqual(_last_write(ctrl), "BP00001")

    def test_set_manual_notch_freq(self):
        ctrl = _make_ctrl()
        ctrl.set_manual_notch(1, 100)
        self.assertEqual(_last_write(ctrl), "BP01100")

    def test_set_manual_notch_freq_clamp(self):
        ctrl = _make_ctrl()
        ctrl.set_manual_notch(1, 999)
        self.assertEqual(_last_write(ctrl), "BP01320")

    def test_get_manual_notch(self):
        ctrl = _make_ctrl()
        _set_response(ctrl, "BP01100")
        self.assertEqual(ctrl.get_manual_notch(1), 100)

    def test_set_manual_notch_invalid_sub(self):
        ctrl = _make_ctrl()
        result = ctrl.set_manual_notch(5, 100)
        self.assertFalse(result)


# ===========================================================================
# BS – Band Select
# ===========================================================================

class TestBS(unittest.TestCase):

    def test_band_select_14(self):
        ctrl = _make_ctrl()
        ctrl.band_select("14")
        self.assertEqual(_last_write(ctrl), "BS05")

    def test_band_select_144(self):
        ctrl = _make_ctrl()
        ctrl.band_select("144")
        self.assertEqual(_last_write(ctrl), "BS15")

    def test_band_select_gen(self):
        ctrl = _make_ctrl()
        ctrl.band_select("GEN")
        self.assertEqual(_last_write(ctrl), "BS11")

    def test_band_select_invalid(self):
        ctrl = _make_ctrl()
        result = ctrl.band_select("999")
        self.assertFalse(result)


# ===========================================================================
# BU – Band Up
# ===========================================================================

class TestBU(unittest.TestCase):

    def test_band_up(self):
        ctrl = _make_ctrl()
        ctrl.band_up()
        self.assertEqual(_last_write(ctrl), "BU0")


# ===========================================================================
# BY – Busy (read-only)
# ===========================================================================

class TestBY(unittest.TestCase):

    def test_get_busy_true(self):
        ctrl = _make_ctrl()
        _set_response(ctrl, "BY10")
        self.assertTrue(ctrl.get_busy())

    def test_get_busy_false(self):
        ctrl = _make_ctrl()
        _set_response(ctrl, "BY00")
        self.assertFalse(ctrl.get_busy())


# ===========================================================================
# CH – Channel Up/Down
# ===========================================================================

class TestCH(unittest.TestCase):

    def test_channel_up(self):
        ctrl = _make_ctrl()
        ctrl.channel_up()
        self.assertEqual(_last_write(ctrl), "CH0")

    def test_channel_down(self):
        ctrl = _make_ctrl()
        ctrl.channel_down()
        self.assertEqual(_last_write(ctrl), "CH1")


# ===========================================================================
# CN – CTCSS/DCS Number
# ===========================================================================

class TestCN(unittest.TestCase):

    def test_set_ctcss_number(self):
        ctrl = _make_ctrl()
        ctrl.set_ctcss_dcs_number(0, 12)
        self.assertEqual(_last_write(ctrl), "CN00012")

    def test_set_dcs_number(self):
        ctrl = _make_ctrl()
        ctrl.set_ctcss_dcs_number(1, 5)
        self.assertEqual(_last_write(ctrl), "CN01005")

    def test_set_ctcss_out_of_range(self):
        ctrl = _make_ctrl()
        result = ctrl.set_ctcss_dcs_number(0, 50)
        self.assertFalse(result)

    def test_get_ctcss_number(self):
        ctrl = _make_ctrl()
        _set_response(ctrl, "CN00012")
        self.assertEqual(ctrl.get_ctcss_dcs_number(0), 12)


# ===========================================================================
# CO – Contour / APF
# ===========================================================================

class TestCO(unittest.TestCase):

    def test_set_contour_on(self):
        ctrl = _make_ctrl()
        ctrl.set_contour(0, 1)
        self.assertEqual(_last_write(ctrl), "CO000001")

    def test_set_contour_freq(self):
        ctrl = _make_ctrl()
        ctrl.set_contour(1, 1000)
        self.assertEqual(_last_write(ctrl), "CO011000")

    def test_get_contour(self):
        ctrl = _make_ctrl()
        _set_response(ctrl, "CO001000")
        self.assertEqual(ctrl.get_contour(1), 1000)

    def test_set_contour_invalid_sub(self):
        ctrl = _make_ctrl()
        result = ctrl.set_contour(9, 0)
        self.assertFalse(result)


# ===========================================================================
# CS – CW Spot
# ===========================================================================

class TestCS(unittest.TestCase):

    def test_set_cw_spot_on(self):
        ctrl = _make_ctrl()
        ctrl.set_cw_spot(True)
        self.assertEqual(_last_write(ctrl), "CS1")

    def test_get_cw_spot(self):
        ctrl = _make_ctrl()
        _set_response(ctrl, "CS0")
        self.assertEqual(ctrl.get_cw_spot(), 0)


# ===========================================================================
# CT – CTCSS Mode
# ===========================================================================

class TestCT(unittest.TestCase):

    def test_set_ctcss_mode_enc_dec(self):
        ctrl = _make_ctrl()
        ctrl.set_ctcss_mode(1)
        self.assertEqual(_last_write(ctrl), "CT01")

    def test_set_ctcss_mode_invalid(self):
        ctrl = _make_ctrl()
        result = ctrl.set_ctcss_mode(5)
        self.assertFalse(result)

    def test_get_ctcss_mode(self):
        ctrl = _make_ctrl()
        _set_response(ctrl, "CT02")
        self.assertEqual(ctrl.get_ctcss_mode(), 2)


# ===========================================================================
# DA – Dimmer
# ===========================================================================

class TestDA(unittest.TestCase):

    def test_set_dimmer(self):
        ctrl = _make_ctrl()
        ctrl.set_dimmer(2, 8)
        # Format: DA + P1(00=fixed,2 digits) + P2(led,2 digits) + P3(tft,2 digits)
        self.assertEqual(_last_write(ctrl), "DA000208")

    def test_get_dimmer(self):
        ctrl = _make_ctrl()
        _set_response(ctrl, "DA000208")
        result = ctrl.get_dimmer()
        self.assertIsNotNone(result)
        self.assertIn("led", result)
        self.assertIn("tft", result)


# ===========================================================================
# DN – Mic Down
# ===========================================================================

class TestDN(unittest.TestCase):

    def test_mic_down(self):
        ctrl = _make_ctrl()
        ctrl.mic_down()
        self.assertEqual(_last_write(ctrl), "DN")


# ===========================================================================
# DT – Date and Time
# ===========================================================================

class TestDT(unittest.TestCase):

    def test_set_date(self):
        ctrl = _make_ctrl()
        ctrl.set_date("20251231")
        self.assertEqual(_last_write(ctrl), "DT020251231")

    def test_set_date_invalid_length(self):
        ctrl = _make_ctrl()
        result = ctrl.set_date("2025123")
        self.assertFalse(result)

    def test_set_time(self):
        ctrl = _make_ctrl()
        ctrl.set_time("143000")
        self.assertEqual(_last_write(ctrl), "DT1143000")

    def test_set_timezone_positive(self):
        ctrl = _make_ctrl()
        ctrl.set_timezone("+0530")
        self.assertEqual(_last_write(ctrl), "DT2+0530")

    def test_set_timezone_negative(self):
        ctrl = _make_ctrl()
        ctrl.set_timezone("-0500")
        self.assertEqual(_last_write(ctrl), "DT2-0500")

    def test_set_timezone_invalid(self):
        ctrl = _make_ctrl()
        result = ctrl.set_timezone("0530")
        self.assertFalse(result)


# ===========================================================================
# ED / EU – Encoder Down / Up
# ===========================================================================

class TestED_EU(unittest.TestCase):

    def test_encoder_down_main(self):
        ctrl = _make_ctrl()
        ctrl.encoder_down(0, 5)
        self.assertEqual(_last_write(ctrl), "ED005")

    def test_encoder_up_sub(self):
        ctrl = _make_ctrl()
        ctrl.encoder_up(1, 10)
        self.assertEqual(_last_write(ctrl), "EU110")

    def test_encoder_down_multi(self):
        ctrl = _make_ctrl()
        ctrl.encoder_down(8, 1)
        self.assertEqual(_last_write(ctrl), "ED801")

    def test_encoder_invalid_encoder(self):
        ctrl = _make_ctrl()
        result = ctrl.encoder_down(5, 1)
        self.assertFalse(result)

    def test_encoder_steps_clamp(self):
        ctrl = _make_ctrl()
        ctrl.encoder_up(0, 200)
        self.assertEqual(_last_write(ctrl), "EU099")


# ===========================================================================
# EK – ENT Key
# ===========================================================================

class TestEK(unittest.TestCase):

    def test_ent_key(self):
        ctrl = _make_ctrl()
        ctrl.ent_key()
        self.assertEqual(_last_write(ctrl), "EK")


# ===========================================================================
# EX – Menu
# ===========================================================================

class TestEX(unittest.TestCase):

    def test_set_menu(self):
        ctrl = _make_ctrl()
        ctrl.set_menu(31, 3)
        self.assertEqual(_last_write(ctrl), "EX0313")

    def test_set_menu_invalid(self):
        ctrl = _make_ctrl()
        result = ctrl.set_menu(0, 1)
        self.assertFalse(result)

    def test_set_menu_over_max(self):
        ctrl = _make_ctrl()
        result = ctrl.set_menu(154, 0)
        self.assertFalse(result)

    def test_get_menu(self):
        ctrl = _make_ctrl()
        _set_response(ctrl, "EX0313")
        val = ctrl.get_menu(31)
        self.assertEqual(val, "3")


# ===========================================================================
# FB – Frequency VFO-B
# ===========================================================================

class TestFB(unittest.TestCase):

    def test_set_frequency_b(self):
        ctrl = _make_ctrl()
        ctrl.set_frequency_b(14.074)
        self.assertEqual(_last_write(ctrl), "FB014074000")

    def test_get_frequency_b(self):
        ctrl = _make_ctrl()
        _set_response(ctrl, "FB014074000")
        self.assertAlmostEqual(ctrl.get_frequency_b(), 14.074)

    def test_get_frequency_b_bad_response(self):
        ctrl = _make_ctrl()
        _set_response(ctrl, "??")
        self.assertEqual(ctrl.get_frequency_b(), 0.0)


# ===========================================================================
# FS – Fast Step
# ===========================================================================

class TestFS(unittest.TestCase):

    def test_set_fast_step_on(self):
        ctrl = _make_ctrl()
        ctrl.set_fast_step(True)
        self.assertEqual(_last_write(ctrl), "FS1")

    def test_get_fast_step(self):
        ctrl = _make_ctrl()
        _set_response(ctrl, "FS0")
        self.assertEqual(ctrl.get_fast_step(), 0)


# ===========================================================================
# FT – Function TX (split VFO)
# ===========================================================================

class TestFT(unittest.TestCase):

    def test_set_tx_vfo_a(self):
        ctrl = _make_ctrl()
        ctrl.set_tx_vfo(2)
        self.assertEqual(_last_write(ctrl), "FT2")

    def test_set_tx_vfo_b(self):
        ctrl = _make_ctrl()
        ctrl.set_tx_vfo(3)
        self.assertEqual(_last_write(ctrl), "FT3")

    def test_set_tx_vfo_invalid(self):
        ctrl = _make_ctrl()
        result = ctrl.set_tx_vfo(1)
        self.assertFalse(result)

    def test_get_tx_vfo(self):
        ctrl = _make_ctrl()
        _set_response(ctrl, "FT0")
        self.assertEqual(ctrl.get_tx_vfo(), 0)


# ===========================================================================
# GT – AGC Function
# ===========================================================================

class TestGT(unittest.TestCase):

    def test_set_agc_fast(self):
        ctrl = _make_ctrl()
        ctrl.set_agc(1)
        self.assertEqual(_last_write(ctrl), "GT01")

    def test_set_agc_invalid(self):
        ctrl = _make_ctrl()
        result = ctrl.set_agc(5)
        self.assertFalse(result)

    def test_get_agc(self):
        ctrl = _make_ctrl()
        _set_response(ctrl, "GT02")
        self.assertEqual(ctrl.get_agc(), 2)


# ===========================================================================
# ID – Identification (read-only)
# ===========================================================================

class TestID(unittest.TestCase):

    def test_get_id(self):
        ctrl = _make_ctrl()
        _set_response(ctrl, "ID0670")
        self.assertEqual(ctrl.get_id(), "0670")


# ===========================================================================
# IS – IF Shift
# ===========================================================================

class TestIS(unittest.TestCase):

    def test_set_if_shift_positive(self):
        ctrl = _make_ctrl()
        ctrl.set_if_shift(200)
        self.assertEqual(_last_write(ctrl), "IS0+0200")

    def test_set_if_shift_negative(self):
        ctrl = _make_ctrl()
        ctrl.set_if_shift(-400)
        self.assertEqual(_last_write(ctrl), "IS0-0400")

    def test_set_if_shift_zero(self):
        ctrl = _make_ctrl()
        ctrl.set_if_shift(0)
        self.assertEqual(_last_write(ctrl), "IS0+0000")

    def test_set_if_shift_clamp_max(self):
        ctrl = _make_ctrl()
        ctrl.set_if_shift(2000)
        self.assertEqual(_last_write(ctrl), "IS0+1200")

    def test_set_if_shift_clamp_min(self):
        ctrl = _make_ctrl()
        ctrl.set_if_shift(-2000)
        self.assertEqual(_last_write(ctrl), "IS0-1200")

    def test_get_if_shift_positive(self):
        ctrl = _make_ctrl()
        _set_response(ctrl, "IS0+0200")
        self.assertEqual(ctrl.get_if_shift(), 200)

    def test_get_if_shift_negative(self):
        ctrl = _make_ctrl()
        _set_response(ctrl, "IS0-0400")
        self.assertEqual(ctrl.get_if_shift(), -400)


# ===========================================================================
# KP – Key Pitch
# ===========================================================================

class TestKP(unittest.TestCase):

    def test_set_key_pitch(self):
        ctrl = _make_ctrl()
        ctrl.set_key_pitch(25)
        self.assertEqual(_last_write(ctrl), "KP25")

    def test_set_key_pitch_clamp(self):
        ctrl = _make_ctrl()
        ctrl.set_key_pitch(100)
        self.assertEqual(_last_write(ctrl), "KP75")

    def test_get_key_pitch(self):
        ctrl = _make_ctrl()
        _set_response(ctrl, "KP25")
        self.assertEqual(ctrl.get_key_pitch(), 25)


# ===========================================================================
# KR – Keyer
# ===========================================================================

class TestKR(unittest.TestCase):

    def test_set_keyer_on(self):
        ctrl = _make_ctrl()
        ctrl.set_keyer(True)
        self.assertEqual(_last_write(ctrl), "KR1")

    def test_get_keyer(self):
        ctrl = _make_ctrl()
        _set_response(ctrl, "KR0")
        self.assertEqual(ctrl.get_keyer(), 0)


# ===========================================================================
# KS – Key Speed
# ===========================================================================

class TestKS(unittest.TestCase):

    def test_set_key_speed(self):
        ctrl = _make_ctrl()
        ctrl.set_key_speed(25)
        self.assertEqual(_last_write(ctrl), "KS025")

    def test_set_key_speed_clamp_min(self):
        ctrl = _make_ctrl()
        ctrl.set_key_speed(1)
        self.assertEqual(_last_write(ctrl), "KS004")

    def test_set_key_speed_clamp_max(self):
        ctrl = _make_ctrl()
        ctrl.set_key_speed(100)
        self.assertEqual(_last_write(ctrl), "KS060")

    def test_get_key_speed(self):
        ctrl = _make_ctrl()
        _set_response(ctrl, "KS025")
        self.assertEqual(ctrl.get_key_speed(), 25)


# ===========================================================================
# LK – Lock
# ===========================================================================

class TestLK(unittest.TestCase):

    def test_set_lock_on(self):
        ctrl = _make_ctrl()
        ctrl.set_lock(True)
        self.assertEqual(_last_write(ctrl), "LK1")

    def test_set_lock_off(self):
        ctrl = _make_ctrl()
        ctrl.set_lock(False)
        self.assertEqual(_last_write(ctrl), "LK0")

    def test_get_lock(self):
        ctrl = _make_ctrl()
        _set_response(ctrl, "LK0")
        self.assertEqual(ctrl.get_lock(), 0)


# ===========================================================================
# MA – Memory to VFO-A
# ===========================================================================

class TestMA(unittest.TestCase):

    def test_memory_to_vfo_a(self):
        ctrl = _make_ctrl()
        ctrl.memory_to_vfo_a()
        self.assertEqual(_last_write(ctrl), "MA")


# ===========================================================================
# MC – Memory Channel
# ===========================================================================

class TestMC(unittest.TestCase):

    def test_set_memory_channel(self):
        ctrl = _make_ctrl()
        ctrl.set_memory_channel(5)
        self.assertEqual(_last_write(ctrl), "MC005")

    def test_set_memory_channel_clamp(self):
        ctrl = _make_ctrl()
        ctrl.set_memory_channel(200)
        self.assertEqual(_last_write(ctrl), "MC117")

    def test_get_memory_channel(self):
        ctrl = _make_ctrl()
        _set_response(ctrl, "MC005")
        self.assertEqual(ctrl.get_memory_channel(), 5)


# ===========================================================================
# MG – Mic Gain
# ===========================================================================

class TestMG(unittest.TestCase):

    def test_set_mic_gain(self):
        ctrl = _make_ctrl()
        ctrl.set_mic_gain(50)
        self.assertEqual(_last_write(ctrl), "MG050")

    def test_set_mic_gain_clamp(self):
        ctrl = _make_ctrl()
        ctrl.set_mic_gain(200)
        self.assertEqual(_last_write(ctrl), "MG100")

    def test_get_mic_gain(self):
        ctrl = _make_ctrl()
        _set_response(ctrl, "MG050")
        self.assertEqual(ctrl.get_mic_gain(), 50)


# ===========================================================================
# MS – Meter SW
# ===========================================================================

class TestMS(unittest.TestCase):

    def test_set_meter_swr(self):
        ctrl = _make_ctrl()
        ctrl.set_meter(3)
        self.assertEqual(_last_write(ctrl), "MS3")

    def test_set_meter_invalid(self):
        ctrl = _make_ctrl()
        result = ctrl.set_meter(6)
        self.assertFalse(result)

    def test_get_meter_type(self):
        ctrl = _make_ctrl()
        _set_response(ctrl, "MS3")
        self.assertEqual(ctrl.get_meter_type(), 3)


# ===========================================================================
# MX – MOX Set
# ===========================================================================

class TestMX(unittest.TestCase):

    def test_set_mox_on(self):
        ctrl = _make_ctrl()
        ctrl.set_mox(True)
        self.assertEqual(_last_write(ctrl), "MX1")

    def test_set_mox_off(self):
        ctrl = _make_ctrl()
        ctrl.set_mox(False)
        self.assertEqual(_last_write(ctrl), "MX0")

    def test_get_mox(self):
        ctrl = _make_ctrl()
        _set_response(ctrl, "MX0")
        self.assertEqual(ctrl.get_mox(), 0)


# ===========================================================================
# NA – Narrow
# ===========================================================================

class TestNA(unittest.TestCase):

    def test_set_narrow_on(self):
        ctrl = _make_ctrl()
        ctrl.set_narrow(True)
        self.assertEqual(_last_write(ctrl), "NA01")

    def test_get_narrow(self):
        ctrl = _make_ctrl()
        _set_response(ctrl, "NA00")
        self.assertEqual(ctrl.get_narrow(), 0)


# ===========================================================================
# NB – Noise Blanker
# ===========================================================================

class TestNB(unittest.TestCase):

    def test_set_noise_blanker_on(self):
        ctrl = _make_ctrl()
        ctrl.set_noise_blanker(True)
        self.assertEqual(_last_write(ctrl), "NB01")

    def test_get_noise_blanker(self):
        ctrl = _make_ctrl()
        _set_response(ctrl, "NB00")
        self.assertEqual(ctrl.get_noise_blanker(), 0)


# ===========================================================================
# NL – Noise Blanker Level
# ===========================================================================

class TestNL(unittest.TestCase):

    def test_set_nb_level(self):
        ctrl = _make_ctrl()
        ctrl.set_noise_blanker_level(5)
        self.assertEqual(_last_write(ctrl), "NL0005")

    def test_set_nb_level_clamp_max(self):
        ctrl = _make_ctrl()
        ctrl.set_noise_blanker_level(15)
        self.assertEqual(_last_write(ctrl), "NL0010")

    def test_get_nb_level(self):
        ctrl = _make_ctrl()
        _set_response(ctrl, "NL0005")
        self.assertEqual(ctrl.get_noise_blanker_level(), 5)


# ===========================================================================
# NR – Noise Reduction
# ===========================================================================

class TestNR(unittest.TestCase):

    def test_set_nr_on(self):
        ctrl = _make_ctrl()
        ctrl.set_noise_reduction(True)
        self.assertEqual(_last_write(ctrl), "NR01")

    def test_get_nr(self):
        ctrl = _make_ctrl()
        _set_response(ctrl, "NR00")
        self.assertEqual(ctrl.get_noise_reduction(), 0)


# ===========================================================================
# OS – Offset / Repeater Shift
# ===========================================================================

class TestOS(unittest.TestCase):

    def test_set_repeater_shift_plus(self):
        ctrl = _make_ctrl()
        ctrl.set_repeater_shift(1)
        self.assertEqual(_last_write(ctrl), "OS01")

    def test_set_repeater_shift_invalid(self):
        ctrl = _make_ctrl()
        result = ctrl.set_repeater_shift(3)
        self.assertFalse(result)

    def test_get_repeater_shift(self):
        ctrl = _make_ctrl()
        _set_response(ctrl, "OS00")
        self.assertEqual(ctrl.get_repeater_shift(), 0)


# ===========================================================================
# PA – Pre-Amp
# ===========================================================================

class TestPA(unittest.TestCase):

    def test_set_preamp_amp1(self):
        ctrl = _make_ctrl()
        ctrl.set_preamp(1)
        self.assertEqual(_last_write(ctrl), "PA01")

    def test_set_preamp_ipo(self):
        ctrl = _make_ctrl()
        ctrl.set_preamp(0)
        self.assertEqual(_last_write(ctrl), "PA00")

    def test_set_preamp_invalid(self):
        ctrl = _make_ctrl()
        result = ctrl.set_preamp(3)
        self.assertFalse(result)

    def test_get_preamp(self):
        ctrl = _make_ctrl()
        _set_response(ctrl, "PA01")
        self.assertEqual(ctrl.get_preamp(), 1)


# ===========================================================================
# PL – Speech Processor Level
# ===========================================================================

class TestPL(unittest.TestCase):

    def test_set_spl(self):
        ctrl = _make_ctrl()
        ctrl.set_speech_processor_level(75)
        self.assertEqual(_last_write(ctrl), "PL075")

    def test_get_spl(self):
        ctrl = _make_ctrl()
        _set_response(ctrl, "PL075")
        self.assertEqual(ctrl.get_speech_processor_level(), 75)


# ===========================================================================
# PR – Speech Processor
# ===========================================================================

class TestPR(unittest.TestCase):

    def test_set_speech_processor_on(self):
        ctrl = _make_ctrl()
        ctrl.set_speech_processor(0, True)
        self.assertEqual(_last_write(ctrl), "PR02")

    def test_set_speech_processor_off(self):
        ctrl = _make_ctrl()
        ctrl.set_speech_processor(0, False)
        self.assertEqual(_last_write(ctrl), "PR01")

    def test_set_speech_processor_invalid(self):
        ctrl = _make_ctrl()
        result = ctrl.set_speech_processor(2, True)
        self.assertFalse(result)

    def test_get_speech_processor(self):
        ctrl = _make_ctrl()
        _set_response(ctrl, "PR02")
        self.assertEqual(ctrl.get_speech_processor(0), 2)


# ===========================================================================
# PS – Power Switch
# ===========================================================================

class TestPS(unittest.TestCase):

    def test_set_power_on(self):
        ctrl = _make_ctrl()
        ctrl.set_power_switch(True)
        self.assertEqual(_last_write(ctrl), "PS1")

    def test_get_power_switch(self):
        ctrl = _make_ctrl()
        _set_response(ctrl, "PS1")
        self.assertEqual(ctrl.get_power_switch(), 1)


# ===========================================================================
# QI / QR / QS
# ===========================================================================

class TestQMB(unittest.TestCase):

    def test_qmb_store(self):
        ctrl = _make_ctrl()
        ctrl.qmb_store()
        self.assertEqual(_last_write(ctrl), "QI")

    def test_qmb_recall(self):
        ctrl = _make_ctrl()
        ctrl.qmb_recall()
        self.assertEqual(_last_write(ctrl), "QR")

    def test_quick_split(self):
        ctrl = _make_ctrl()
        ctrl.quick_split()
        self.assertEqual(_last_write(ctrl), "QS")


# ===========================================================================
# RA – RF Attenuator
# ===========================================================================

class TestRA(unittest.TestCase):

    def test_set_rf_attenuator_on(self):
        ctrl = _make_ctrl()
        ctrl.set_rf_attenuator(True)
        self.assertEqual(_last_write(ctrl), "RA01")

    def test_get_rf_attenuator(self):
        ctrl = _make_ctrl()
        _set_response(ctrl, "RA00")
        self.assertEqual(ctrl.get_rf_attenuator(), 0)


# ===========================================================================
# RC / RD / RU – Clarifier
# ===========================================================================

class TestClarifier(unittest.TestCase):

    def test_clar_clear(self):
        ctrl = _make_ctrl()
        ctrl.clar_clear()
        self.assertEqual(_last_write(ctrl), "RC")

    def test_clar_down(self):
        ctrl = _make_ctrl()
        ctrl.clar_down(500)
        self.assertEqual(_last_write(ctrl), "RD0500")

    def test_clar_up(self):
        ctrl = _make_ctrl()
        ctrl.clar_up(100)
        self.assertEqual(_last_write(ctrl), "RU0100")

    def test_clar_down_clamp(self):
        ctrl = _make_ctrl()
        ctrl.clar_down(10000)
        self.assertEqual(_last_write(ctrl), "RD9999")


# ===========================================================================
# RG – RF Gain
# ===========================================================================

class TestRG(unittest.TestCase):

    def test_set_rf_gain(self):
        ctrl = _make_ctrl()
        ctrl.set_rf_gain(200)
        self.assertEqual(_last_write(ctrl), "RG0200")

    def test_get_rf_gain(self):
        ctrl = _make_ctrl()
        _set_response(ctrl, "RG0200")
        self.assertEqual(ctrl.get_rf_gain(), 200)


# ===========================================================================
# RL – Noise Reduction Level
# ===========================================================================

class TestRL(unittest.TestCase):

    def test_set_nr_level(self):
        ctrl = _make_ctrl()
        ctrl.set_noise_reduction_level(8)
        self.assertEqual(_last_write(ctrl), "RL008")

    def test_set_nr_level_clamp_min(self):
        ctrl = _make_ctrl()
        ctrl.set_noise_reduction_level(0)
        self.assertEqual(_last_write(ctrl), "RL001")

    def test_get_nr_level(self):
        ctrl = _make_ctrl()
        _set_response(ctrl, "RL008")
        self.assertEqual(ctrl.get_noise_reduction_level(), 8)


# ===========================================================================
# RM – Read Meter
# ===========================================================================

class TestRM(unittest.TestCase):

    def test_read_meter_s(self):
        ctrl = _make_ctrl()
        _set_response(ctrl, "RM1120")
        self.assertEqual(ctrl.read_meter(1), 120)

    def test_read_meter_swr(self):
        ctrl = _make_ctrl()
        _set_response(ctrl, "RM6025")
        self.assertEqual(ctrl.read_meter(6), 25)


# ===========================================================================
# RS – Radio Status
# ===========================================================================

class TestRS(unittest.TestCase):

    def test_get_radio_status(self):
        ctrl = _make_ctrl()
        _set_response(ctrl, "RS0")
        self.assertEqual(ctrl.get_radio_status(), 0)


# ===========================================================================
# RT / XT – RX / TX Clarifier
# ===========================================================================

class TestClarifierOnOff(unittest.TestCase):

    def test_set_rx_clar_on(self):
        ctrl = _make_ctrl()
        ctrl.set_rx_clarifier(True)
        self.assertEqual(_last_write(ctrl), "RT1")

    def test_get_rx_clar(self):
        ctrl = _make_ctrl()
        _set_response(ctrl, "RT0")
        self.assertEqual(ctrl.get_rx_clarifier(), 0)

    def test_set_tx_clar_on(self):
        ctrl = _make_ctrl()
        ctrl.set_tx_clarifier(True)
        self.assertEqual(_last_write(ctrl), "XT1")

    def test_get_tx_clar(self):
        ctrl = _make_ctrl()
        _set_response(ctrl, "XT0")
        self.assertEqual(ctrl.get_tx_clarifier(), 0)


# ===========================================================================
# SC – Scan
# ===========================================================================

class TestSC(unittest.TestCase):

    def test_set_scan_up(self):
        ctrl = _make_ctrl()
        ctrl.set_scan(1)
        self.assertEqual(_last_write(ctrl), "SC1")

    def test_set_scan_off(self):
        ctrl = _make_ctrl()
        ctrl.set_scan(0)
        self.assertEqual(_last_write(ctrl), "SC0")

    def test_set_scan_invalid(self):
        ctrl = _make_ctrl()
        result = ctrl.set_scan(3)
        self.assertFalse(result)

    def test_get_scan(self):
        ctrl = _make_ctrl()
        _set_response(ctrl, "SC2")
        self.assertEqual(ctrl.get_scan(), 2)


# ===========================================================================
# SD – CW Break-in Delay
# ===========================================================================

class TestSD(unittest.TestCase):

    def test_set_break_in_delay(self):
        ctrl = _make_ctrl()
        ctrl.set_break_in_delay(500)
        self.assertEqual(_last_write(ctrl), "SD0500")

    def test_set_break_in_delay_clamp_min(self):
        ctrl = _make_ctrl()
        ctrl.set_break_in_delay(10)
        self.assertEqual(_last_write(ctrl), "SD0030")

    def test_get_break_in_delay(self):
        ctrl = _make_ctrl()
        _set_response(ctrl, "SD0500")
        self.assertEqual(ctrl.get_break_in_delay(), 500)


# ===========================================================================
# SH – Width (Passband)
# ===========================================================================

class TestSH(unittest.TestCase):

    def test_set_width(self):
        ctrl = _make_ctrl()
        ctrl.set_width(7)
        self.assertEqual(_last_write(ctrl), "SH007")

    def test_set_width_default(self):
        ctrl = _make_ctrl()
        ctrl.set_width(0)
        self.assertEqual(_last_write(ctrl), "SH000")

    def test_get_width(self):
        ctrl = _make_ctrl()
        _set_response(ctrl, "SH007")
        self.assertEqual(ctrl.get_width(), 7)


# ===========================================================================
# SQ – Squelch
# ===========================================================================

class TestSQ(unittest.TestCase):

    def test_set_squelch(self):
        ctrl = _make_ctrl()
        ctrl.set_squelch(40)
        self.assertEqual(_last_write(ctrl), "SQ0040")

    def test_set_squelch_clamp(self):
        ctrl = _make_ctrl()
        ctrl.set_squelch(200)
        self.assertEqual(_last_write(ctrl), "SQ0100")

    def test_get_squelch(self):
        ctrl = _make_ctrl()
        _set_response(ctrl, "SQ0040")
        self.assertEqual(ctrl.get_squelch(), 40)


# ===========================================================================
# SV – Swap VFO
# ===========================================================================

class TestSV(unittest.TestCase):

    def test_swap_vfo(self):
        ctrl = _make_ctrl()
        ctrl.swap_vfo()
        self.assertEqual(_last_write(ctrl), "SV")


# ===========================================================================
# TS – TXW
# ===========================================================================

class TestTS(unittest.TestCase):

    def test_set_txw_on(self):
        ctrl = _make_ctrl()
        ctrl.set_txw(True)
        self.assertEqual(_last_write(ctrl), "TS1")

    def test_get_txw(self):
        ctrl = _make_ctrl()
        _set_response(ctrl, "TS0")
        self.assertEqual(ctrl.get_txw(), 0)


# ===========================================================================
# TX – TX State read
# ===========================================================================

class TestTX(unittest.TestCase):

    def test_get_tx_state(self):
        ctrl = _make_ctrl()
        _set_response(ctrl, "TX0")
        self.assertEqual(ctrl.get_tx_state(), 0)


# ===========================================================================
# UL – PLL Lock
# ===========================================================================

class TestUL(unittest.TestCase):

    def test_get_pll_locked(self):
        ctrl = _make_ctrl()
        _set_response(ctrl, "UL0")
        self.assertTrue(ctrl.get_pll_lock())

    def test_get_pll_unlocked(self):
        ctrl = _make_ctrl()
        _set_response(ctrl, "UL1")
        self.assertFalse(ctrl.get_pll_lock())


# ===========================================================================
# UP – Up key
# ===========================================================================

class TestUP(unittest.TestCase):

    def test_up(self):
        ctrl = _make_ctrl()
        ctrl.up()
        self.assertEqual(_last_write(ctrl), "UP")


# ===========================================================================
# VD – VOX Delay
# ===========================================================================

class TestVD(unittest.TestCase):

    def test_set_vox_delay(self):
        ctrl = _make_ctrl()
        ctrl.set_vox_delay(500)
        self.assertEqual(_last_write(ctrl), "VD0500")

    def test_set_vox_delay_clamp_min(self):
        ctrl = _make_ctrl()
        ctrl.set_vox_delay(10)
        self.assertEqual(_last_write(ctrl), "VD0030")

    def test_get_vox_delay(self):
        ctrl = _make_ctrl()
        _set_response(ctrl, "VD0500")
        self.assertEqual(ctrl.get_vox_delay(), 500)


# ===========================================================================
# VG – VOX Gain
# ===========================================================================

class TestVG(unittest.TestCase):

    def test_set_vox_gain(self):
        ctrl = _make_ctrl()
        ctrl.set_vox_gain(80)
        self.assertEqual(_last_write(ctrl), "VG080")

    def test_get_vox_gain(self):
        ctrl = _make_ctrl()
        _set_response(ctrl, "VG080")
        self.assertEqual(ctrl.get_vox_gain(), 80)


# ===========================================================================
# VM – V/M Key
# ===========================================================================

class TestVM(unittest.TestCase):

    def test_vm_key(self):
        ctrl = _make_ctrl()
        ctrl.vm_key()
        self.assertEqual(_last_write(ctrl), "VM")


# ===========================================================================
# VX – VOX Status
# ===========================================================================

class TestVX(unittest.TestCase):

    def test_set_vox_on(self):
        ctrl = _make_ctrl()
        ctrl.set_vox(True)
        self.assertEqual(_last_write(ctrl), "VX1")

    def test_get_vox(self):
        ctrl = _make_ctrl()
        _set_response(ctrl, "VX0")
        self.assertEqual(ctrl.get_vox(), 0)


# ===========================================================================
# ZI – Zero In
# ===========================================================================

class TestZI(unittest.TestCase):

    def test_zero_in(self):
        ctrl = _make_ctrl()
        ctrl.zero_in()
        self.assertEqual(_last_write(ctrl), "ZI")


# ===========================================================================
# Disconnected guard – commands should not raise when not connected
# ===========================================================================

class TestDisconnectedGuard(unittest.TestCase):

    def test_no_exception_when_disconnected(self):
        ctrl = Yaesu991AControl()
        # These should all return None / False / 0.0 gracefully without raising
        self.assertIsNone(ctrl.get_frequency() or None)
        self.assertIsNone(ctrl.get_mode())
        self.assertIsNone(ctrl.get_auto_information())
        self.assertIsNone(ctrl.get_rf_gain())


if __name__ == "__main__":
    unittest.main()
