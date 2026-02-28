import threading
import serial
from decimal import Decimal, ROUND_HALF_UP


class Yaesu991AControl:
    """
    Minimal CAT controller for Yaesu FT-991A using Yaesu ASCII CAT over serial.

    This module intentionally contains NO GUI code.
    """
    def __init__(self, port="/dev/ttyUSB0", baud=38400, timeout=1, stopbits=1):
        self.port = port
        self.baud = baud
        self.timeout = timeout
        self.conn = None  # don't auto-connect on launch
        self._io_lock = threading.Lock()
        self._stopbits = float(stopbits)
        self._stopbits_serial = None  # resolved in connect(); also settable by GUI

        # CTCSS Tone Mapping (Index 001-050) - unused by GUI for now, but kept here.
        self.tone_map = {
            67.0: "001", 69.3: "002", 71.9: "003", 74.4: "004", 77.0: "005",
            79.7: "006", 82.5: "007", 85.4: "008", 88.5: "009", 91.5: "010",
            100.0: "013", 103.5: "014", 123.0: "019", 141.3: "023", 151.4: "025"
        }

    def is_connected(self):
        return self.conn is not None and getattr(self.conn, "is_open", False)

    def connect(self):
        """Open serial connection. Returns (True, None) or (False, error_message)."""
        if self.is_connected():
            return True, None
        try:
            # Resolve stop bits: prefer the pre-mapped constant set by the GUI
            # (_stopbits_serial), fall back to converting the float _stopbits value.
            if self._stopbits_serial is not None:
                sb = self._stopbits_serial
            else:
                _sb_map = {
                    1: serial.STOPBITS_ONE,
                    1.5: serial.STOPBITS_ONE_POINT_FIVE,
                    2: serial.STOPBITS_TWO,
                }
                sb = _sb_map.get(float(self._stopbits), serial.STOPBITS_ONE)

            self.conn = serial.Serial(
                port=self.port,
                baudrate=self.baud,
                timeout=self.timeout,
                stopbits=sb,
            )
            return True, None
        except Exception as e:
            self.conn = None
            return False, str(e)

    def disconnect(self):
        """Close serial connection. Safe to call even if already disconnected."""
        try:
            if self.conn is not None:
                try:
                    self.conn.close()
                finally:
                    self.conn = None
        except Exception:
            self.conn = None

    def _execute(self, cmd, read=False):
        """Standard Yaesu ASCII CAT execution: send f'{cmd};' and optionally read until ';'."""
        if not self.is_connected():
            return None

        try:
            with self._io_lock:
                self.conn.write(f"{cmd};".encode("ascii"))
                if read:
                    resp = self.conn.read_until(b";").decode(errors="replace")
                    return resp.strip().replace(";", "")
        except Exception as e:
            # Keep this lightweight; GUI can surface errors if desired.
            print(f"Serial Error: {e}")
        return None

    def set_frequency(self, mhz):
        # Avoid float truncation errors (e.g. 13.9993 instead of 14.0000).
        hz = int(
            (Decimal(str(mhz)) * Decimal("1000000"))
            .quantize(Decimal("1"), rounding=ROUND_HALF_UP)
        )
        self._execute(f"FA{hz:09d}")
        print("mhz=", mhz, "hz_float=", float(mhz) * 1_000_000) # Temporary code to confirm frequency output via CAT

    def get_frequency(self):
        resp = self._execute("FA", read=True)
        # Expected something like: "FA014250000"
        if not resp or not resp.startswith("FA") or len(resp) <= 2:
            return 0.0
        digits = resp[2:]
        if not digits.isdigit():
            return 0.0
        return float(Decimal(digits) / Decimal("1000000"))

    def set_mode(self, mode_str):
        modes = {"LSB": "1", "USB": "2", "CW": "3", "FM": "4", "AM": "5", "C4FM": "E"}
        mode_str = (mode_str or "").strip().upper()
        if mode_str in modes:
            self._execute(f"MD0{modes[mode_str]}")

    def get_mode(self):
        """
        Read current mode using the Yaesu CAT 'MD' command.
        Returns currently active mode or None if unknown/unavailable.
        """
        code_to_mode = {
            "1": "LSB", "2": "USB", "3": "CW-U", "4": "FM", "5": "AM",
            "6": "RTTY-L", "7": "CW-L", "8": "DATA-L", "9": "RTTY-U",
            "A": "DATA-FM", "C": "DATA-U", "E": "C4FM"
        }

        resp = self._execute("MD0", read=True)
        if not resp:
            resp = self._execute("MD", read=True)
        if not resp:
            return None

        if resp.startswith("MD") and len(resp) >= 4:
            code = resp[-1].upper()
            return code_to_mode.get(code)

        return None

    def get_swr_meter(self):
        resp = self._execute("RM6", read=True)
        return int(resp[3:6]) if resp and len(resp) >= 6 and resp[3:6].isdigit() else 0

    def get_s_meter(self):
        """Query SM0, expect SM0xxx where xxx is 000-255."""
        resp = self._execute("SM0", read=True)
        if not resp:
            return 0
        if resp.startswith("SM") and len(resp) >= 6:
            digits = resp[-3:]
            if digits.isdigit():
                return int(digits)
        return 0

    def get_rf_power(self):
        """Query PC, expect PCxxx where xxx is typically 005-100."""
        resp = self._execute("PC", read=True)
        if not resp:
            return 0
        if resp.startswith("PC") and len(resp) >= 5:
            digits = resp[-3:]
            if digits.isdigit():
                return int(digits)
        return 0

    def set_rf_power(self, level):
        """Clamp 5..100 and send PCxxx."""
        try:
            level_int = int(level)
        except (TypeError, ValueError):
            return False

        level_int = max(5, min(100, level_int))
        self._execute(f"PC{level_int:03d}")
        return True

    def ptt_on(self):
        self._execute("TX1")

    def ptt_off(self):
        self._execute("TX0")

    # ------------------------------------------------------------------ #
    # AB – VFO-A TO VFO-B  (set only)
    # ------------------------------------------------------------------ #
    def vfo_a_to_b(self):
        """Copy VFO-A frequency/mode to VFO-B."""
        self._execute("AB")

    # ------------------------------------------------------------------ #
    # AC – ANTENNA TUNER CONTROL
    # ------------------------------------------------------------------ #
    def set_antenna_tuner(self, state):
        """
        Set antenna tuner state.

        Parameters
        ----------
        state : int
            0 = Tuner OFF, 1 = Tuner ON, 2 = Start/Stop tuning.
        """
        if state not in (0, 1, 2):
            return False
        self._execute(f"AC00{state}")
        return True

    def get_antenna_tuner(self):
        """
        Read antenna tuner state.

        Returns
        -------
        int or None
            0 = OFF, 1 = ON, 2 = Tuning in progress; None on error.
        """
        resp = self._execute("AC", read=True)
        if resp and resp.startswith("AC") and len(resp) >= 5:
            ch = resp[-1]
            if ch.isdigit():
                return int(ch)
        return None

    # ------------------------------------------------------------------ #
    # AG – AF GAIN
    # ------------------------------------------------------------------ #
    def set_af_gain(self, level):
        """
        Set AF (audio) gain.

        Parameters
        ----------
        level : int
            0–255.
        """
        try:
            level = max(0, min(255, int(level)))
        except (TypeError, ValueError):
            return False
        self._execute(f"AG0{level:03d}")
        return True

    def get_af_gain(self):
        """
        Read AF gain.

        Returns
        -------
        int or None
            0–255; None on error.
        """
        resp = self._execute("AG0", read=True)
        if resp and resp.startswith("AG") and len(resp) >= 6:
            digits = resp[-3:]
            if digits.isdigit():
                return int(digits)
        return None

    # ------------------------------------------------------------------ #
    # AI – AUTO INFORMATION
    # ------------------------------------------------------------------ #
    def set_auto_information(self, enabled):
        """
        Enable or disable auto-information mode.

        Parameters
        ----------
        enabled : bool or int
            0/False = OFF, 1/True = ON.
        """
        val = 1 if enabled else 0
        self._execute(f"AI{val}")

    def get_auto_information(self):
        """
        Read auto-information state.

        Returns
        -------
        int or None
            0 = OFF, 1 = ON; None on error.
        """
        resp = self._execute("AI", read=True)
        if resp and resp.startswith("AI") and len(resp) >= 3:
            ch = resp[-1]
            if ch.isdigit():
                return int(ch)
        return None

    # ------------------------------------------------------------------ #
    # AM – VFO-A TO MEMORY CHANNEL  (set only)
    # ------------------------------------------------------------------ #
    def vfo_a_to_memory(self):
        """Write current VFO-A data to the current memory channel."""
        self._execute("AM")

    # ------------------------------------------------------------------ #
    # BA – VFO-B TO VFO-A  (set only)
    # ------------------------------------------------------------------ #
    def vfo_b_to_a(self):
        """Copy VFO-B frequency/mode to VFO-A."""
        self._execute("BA")

    # ------------------------------------------------------------------ #
    # BC – AUTO NOTCH
    # ------------------------------------------------------------------ #
    def set_auto_notch(self, enabled):
        """
        Enable or disable the automatic notch filter.

        Parameters
        ----------
        enabled : bool or int
            0/False = OFF, 1/True = ON.
        """
        val = 1 if enabled else 0
        self._execute(f"BC0{val}")

    def get_auto_notch(self):
        """
        Read auto-notch state.

        Returns
        -------
        int or None
            0 = OFF, 1 = ON; None on error.
        """
        resp = self._execute("BC0", read=True)
        if resp and resp.startswith("BC") and len(resp) >= 4:
            ch = resp[-1]
            if ch.isdigit():
                return int(ch)
        return None

    # ------------------------------------------------------------------ #
    # BD – BAND DOWN  (set only)
    # ------------------------------------------------------------------ #
    def band_down(self):
        """Step down one band."""
        self._execute("BD0")

    # ------------------------------------------------------------------ #
    # BI – BREAK-IN (CW)
    # ------------------------------------------------------------------ #
    def set_break_in(self, enabled):
        """
        Enable or disable CW break-in.

        Parameters
        ----------
        enabled : bool or int
            0/False = OFF, 1/True = ON.
        """
        val = 1 if enabled else 0
        self._execute(f"BI{val}")

    def get_break_in(self):
        """
        Read CW break-in state.

        Returns
        -------
        int or None
            0 = OFF, 1 = ON; None on error.
        """
        resp = self._execute("BI", read=True)
        if resp and resp.startswith("BI") and len(resp) >= 3:
            ch = resp[-1]
            if ch.isdigit():
                return int(ch)
        return None

    # ------------------------------------------------------------------ #
    # BP – MANUAL NOTCH
    # ------------------------------------------------------------------ #
    def set_manual_notch(self, sub, value):
        """
        Set manual notch on/off or notch frequency.

        Parameters
        ----------
        sub : int
            0 = ON/OFF control, 1 = Notch frequency level.
        value : int
            sub=0: 0 (OFF) or 1 (ON).
            sub=1: 1–320 (notch frequency × 10 Hz).
        """
        if sub not in (0, 1):
            return False
        if sub == 0:
            value = 1 if value else 0
        else:
            value = max(1, min(320, int(value)))
        self._execute(f"BP0{sub}{value:03d}")
        return True

    def get_manual_notch(self, sub):
        """
        Read manual notch state or frequency.

        Parameters
        ----------
        sub : int
            0 = ON/OFF state, 1 = Notch frequency.

        Returns
        -------
        int or None
        """
        if sub not in (0, 1):
            return None
        resp = self._execute(f"BP0{sub}", read=True)
        if resp and resp.startswith("BP") and len(resp) >= 6:
            digits = resp[-3:]
            if digits.isdigit():
                return int(digits)
        return None

    # ------------------------------------------------------------------ #
    # BS – BAND SELECT  (set only)
    # ------------------------------------------------------------------ #
    # Band codes per the manual:
    # 00=1.8 MHz, 01=3.5 MHz, 02=5 MHz, 03=7 MHz, 04=10 MHz, 05=14 MHz,
    # 06=18 MHz, 07=21 MHz, 08=24.5 MHz, 09=28 MHz, 10=50 MHz, 11=GEN,
    # 12=MW, 13=-, 14=AIR, 15=144 MHz, 16=430 MHz
    BAND_SELECT_MAP = {
        "1.8": "00", "3.5": "01", "5": "02", "7": "03",
        "10": "04", "14": "05", "18": "06", "21": "07",
        "24.5": "08", "28": "09", "50": "10", "GEN": "11",
        "MW": "12", "AIR": "14", "144": "15", "430": "16",
    }

    def band_select(self, band):
        """
        Select an operating band.

        Parameters
        ----------
        band : str or int
            Band designator string (e.g. "14", "144") or two-digit code (0–16).
        """
        key = str(band).strip().upper()
        code = self.BAND_SELECT_MAP.get(key)
        if code is None:
            # Accept a raw two-digit code like "05"
            if key.isdigit() and 0 <= int(key) <= 16:
                code = f"{int(key):02d}"
            else:
                return False
        self._execute(f"BS{code}")
        return True

    # ------------------------------------------------------------------ #
    # BU – BAND UP  (set only)
    # ------------------------------------------------------------------ #
    def band_up(self):
        """Step up one band."""
        self._execute("BU0")

    # ------------------------------------------------------------------ #
    # BY – BUSY  (read only)
    # ------------------------------------------------------------------ #
    def get_busy(self):
        """
        Read RX busy (squelch open) state.

        Returns
        -------
        bool or None
            True if busy, False if not, None on error.
        """
        resp = self._execute("BY", read=True)
        if resp and resp.startswith("BY") and len(resp) >= 3:
            ch = resp[2]
            if ch.isdigit():
                return ch == "1"
        return None

    # ------------------------------------------------------------------ #
    # CH – CHANNEL UP / DOWN  (set only)
    # ------------------------------------------------------------------ #
    def channel_up(self):
        """Step to the next memory channel."""
        self._execute("CH0")

    def channel_down(self):
        """Step to the previous memory channel."""
        self._execute("CH1")

    # ------------------------------------------------------------------ #
    # CN – CTCSS / DCS TONE NUMBER
    # ------------------------------------------------------------------ #
    def set_ctcss_dcs_number(self, tone_type, number):
        """
        Set the CTCSS tone or DCS code number.

        Parameters
        ----------
        tone_type : int
            0 = CTCSS (0–49), 1 = DCS (0–103).
        number : int
            Index into the tone/code table.
        """
        tone_type = int(tone_type)
        number = int(number)
        if tone_type == 0 and not (0 <= number <= 49):
            return False
        if tone_type == 1 and not (0 <= number <= 103):
            return False
        self._execute(f"CN0{tone_type}{number:03d}")
        return True

    def get_ctcss_dcs_number(self, tone_type):
        """
        Read the current CTCSS tone or DCS code index.

        Parameters
        ----------
        tone_type : int
            0 = CTCSS, 1 = DCS.

        Returns
        -------
        int or None
        """
        resp = self._execute(f"CN0{tone_type}", read=True)
        if resp and resp.startswith("CN") and len(resp) >= 7:
            digits = resp[-3:]
            if digits.isdigit():
                return int(digits)
        return None

    # ------------------------------------------------------------------ #
    # CO – CONTOUR / APF
    # ------------------------------------------------------------------ #
    def set_contour(self, sub, value):
        """
        Control the contour or APF filter.

        Parameters
        ----------
        sub : int
            0 = CONTOUR ON/OFF (value: 0=OFF, 1=ON).
            1 = CONTOUR FREQ (value: 10–3200 Hz).
            2 = APF ON/OFF (value: 0=OFF, 1=ON).
            3 = APF FREQ (value: 0–50 → -250 to +250 Hz).
        value : int
        """
        if sub not in (0, 1, 2, 3):
            return False
        self._execute(f"CO0{sub}{int(value):04d}")
        return True

    def get_contour(self, sub):
        """
        Read contour or APF setting.

        Parameters
        ----------
        sub : int
            0–3, same as set_contour.

        Returns
        -------
        int or None
        """
        if sub not in (0, 1, 2, 3):
            return None
        resp = self._execute(f"CO0{sub}", read=True)
        if resp and resp.startswith("CO") and len(resp) >= 7:
            digits = resp[-4:]
            if digits.isdigit():
                return int(digits)
        return None

    # ------------------------------------------------------------------ #
    # CS – CW SPOT
    # ------------------------------------------------------------------ #
    def set_cw_spot(self, enabled):
        """
        Enable or disable CW Spot.

        Parameters
        ----------
        enabled : bool or int
            0/False = OFF, 1/True = ON.
        """
        val = 1 if enabled else 0
        self._execute(f"CS{val}")

    def get_cw_spot(self):
        """
        Read CW Spot state.

        Returns
        -------
        int or None
            0 = OFF, 1 = ON; None on error.
        """
        resp = self._execute("CS", read=True)
        if resp and resp.startswith("CS") and len(resp) >= 3:
            ch = resp[-1]
            if ch.isdigit():
                return int(ch)
        return None

    # ------------------------------------------------------------------ #
    # CT – CTCSS MODE
    # ------------------------------------------------------------------ #
    def set_ctcss_mode(self, mode):
        """
        Set CTCSS/DCS mode.

        Parameters
        ----------
        mode : int
            0 = OFF, 1 = CTCSS ENC/DEC, 2 = CTCSS ENC,
            3 = DCS ENC/DEC, 4 = DCS ENC.
        """
        if mode not in (0, 1, 2, 3, 4):
            return False
        self._execute(f"CT0{mode}")
        return True

    def get_ctcss_mode(self):
        """
        Read current CTCSS/DCS mode.

        Returns
        -------
        int or None
            0–4 per the mode table; None on error.
        """
        resp = self._execute("CT0", read=True)
        if resp and resp.startswith("CT") and len(resp) >= 4:
            ch = resp[-1]
            if ch.isdigit():
                return int(ch)
        return None

    # ------------------------------------------------------------------ #
    # DA – DIMMER
    # ------------------------------------------------------------------ #
    def set_dimmer(self, led_level, tft_level):
        """
        Set display brightness levels.

        Parameters
        ----------
        led_level : int
            LED indicator brightness: 1–2.
        tft_level : int
            TFT display brightness: 0–15.
        """
        led_level = max(1, min(2, int(led_level)))
        tft_level = max(0, min(15, int(tft_level)))
        self._execute(f"DA00{led_level:02d}{tft_level:02d}")
        return True

    def get_dimmer(self):
        """
        Read dimmer settings.

        Returns
        -------
        dict or None
            {"led": int, "tft": int} or None on error.
        """
        resp = self._execute("DA", read=True)
        if resp and resp.startswith("DA") and len(resp) >= 8:
            try:
                led = int(resp[4:6])
                tft = int(resp[6:8])
                return {"led": led, "tft": tft}
            except (ValueError, IndexError):
                pass
        return None

    # ------------------------------------------------------------------ #
    # DN – MIC DOWN  (set only)
    # ------------------------------------------------------------------ #
    def mic_down(self):
        """Decrement the microphone gain by one step (front-panel MIC DN)."""
        self._execute("DN")

    # ------------------------------------------------------------------ #
    # DT – DATE AND TIME
    # ------------------------------------------------------------------ #
    def set_date(self, yyyymmdd):
        """
        Set the internal date.

        Parameters
        ----------
        yyyymmdd : str
            Eight-digit date string, e.g. "20251231".
        """
        s = str(yyyymmdd).strip()
        if len(s) != 8 or not s.isdigit():
            return False
        self._execute(f"DT0{s}")
        return True

    def get_date(self):
        """
        Read the internal date.

        Returns
        -------
        str or None
            "yyyymmdd" string, or None on error.
        """
        resp = self._execute("DT0", read=True)
        if resp and resp.startswith("DT") and len(resp) >= 11:
            return resp[3:]
        return None

    def set_time(self, hhmmss):
        """
        Set the internal UTC time.

        Parameters
        ----------
        hhmmss : str
            Six-digit time string, e.g. "143000" for 14:30:00.
        """
        s = str(hhmmss).strip()
        if len(s) != 6 or not s.isdigit():
            return False
        self._execute(f"DT1{s}")
        return True

    def get_time(self):
        """
        Read the internal UTC time.

        Returns
        -------
        str or None
            "hhmmss" string, or None on error.
        """
        resp = self._execute("DT1", read=True)
        if resp and resp.startswith("DT") and len(resp) >= 9:
            return resp[3:]
        return None

    def set_timezone(self, offset_str):
        """
        Set the UTC time-zone offset.

        Parameters
        ----------
        offset_str : str
            Offset in "+hhmm" or "-hhmm" format, e.g. "+0530" or "-0500".
            Range: -12:00 to +14:00 in 30-minute increments.
        """
        s = str(offset_str).strip()
        if len(s) != 5 or s[0] not in ("+", "-") or not s[1:].isdigit():
            return False
        self._execute(f"DT2{s}")
        return True

    def get_timezone(self):
        """
        Read the UTC time-zone offset.

        Returns
        -------
        str or None
            "+hhmm" or "-hhmm" string, or None on error.
        """
        resp = self._execute("DT2", read=True)
        if resp and resp.startswith("DT") and len(resp) >= 8:
            return resp[3:]
        return None

    # ------------------------------------------------------------------ #
    # ED – ENCODER DOWN  (set only)
    # ------------------------------------------------------------------ #
    def encoder_down(self, encoder=0, steps=1):
        """
        Step an encoder downward.

        Parameters
        ----------
        encoder : int
            0 = MAIN, 1 = SUB, 8 = MULTI.
        steps : int
            1–99 frequency steps.
        """
        if encoder not in (0, 1, 8):
            return False
        steps = max(1, min(99, int(steps)))
        self._execute(f"ED{encoder}{steps:02d}")
        return True

    # ------------------------------------------------------------------ #
    # EK – ENT KEY  (set only)
    # ------------------------------------------------------------------ #
    def ent_key(self):
        """Simulate pressing the ENT key on the front panel."""
        self._execute("EK")

    # ------------------------------------------------------------------ #
    # EU – ENCODER UP  (set only)
    # ------------------------------------------------------------------ #
    def encoder_up(self, encoder=0, steps=1):
        """
        Step an encoder upward.

        Parameters
        ----------
        encoder : int
            0 = MAIN, 1 = SUB, 8 = MULTI.
        steps : int
            1–99 frequency steps.
        """
        if encoder not in (0, 1, 8):
            return False
        steps = max(1, min(99, int(steps)))
        self._execute(f"EU{encoder}{steps:02d}")
        return True

    # ------------------------------------------------------------------ #
    # EX – MENU ITEM
    # ------------------------------------------------------------------ #
    def set_menu(self, menu_number, value):
        """
        Set a menu item value.

        Parameters
        ----------
        menu_number : int
            Menu item number, 1–153.
        value : str or int
            Value string/integer appropriate for the given menu item.
        """
        if not (1 <= int(menu_number) <= 153):
            return False
        self._execute(f"EX{int(menu_number):03d}{value}")
        return True

    def get_menu(self, menu_number):
        """
        Read a menu item value.

        Parameters
        ----------
        menu_number : int
            Menu item number, 1–153.

        Returns
        -------
        str or None
            Raw value string, or None on error.
        """
        if not (1 <= int(menu_number) <= 153):
            return None
        resp = self._execute(f"EX{int(menu_number):03d}", read=True)
        if resp and resp.startswith("EX") and len(resp) > 5:
            return resp[5:]
        return None

    # ------------------------------------------------------------------ #
    # FB – FREQUENCY VFO-B
    # ------------------------------------------------------------------ #
    def set_frequency_b(self, mhz):
        """
        Set VFO-B frequency.

        Parameters
        ----------
        mhz : float
            Frequency in MHz (0.03–470 MHz).
        """
        hz = int(
            (Decimal(str(mhz)) * Decimal("1000000"))
            .quantize(Decimal("1"), rounding=ROUND_HALF_UP)
        )
        self._execute(f"FB{hz:09d}")

    def get_frequency_b(self):
        """
        Read VFO-B frequency.

        Returns
        -------
        float
            Frequency in MHz, or 0.0 on error.
        """
        resp = self._execute("FB", read=True)
        if not resp or not resp.startswith("FB") or len(resp) <= 2:
            return 0.0
        digits = resp[2:]
        if not digits.isdigit():
            return 0.0
        return float(Decimal(digits) / Decimal("1000000"))

    # ------------------------------------------------------------------ #
    # FS – FAST STEP
    # ------------------------------------------------------------------ #
    def set_fast_step(self, enabled):
        """
        Enable or disable the VFO-A FAST tuning step.

        Parameters
        ----------
        enabled : bool or int
            0/False = OFF, 1/True = ON.
        """
        val = 1 if enabled else 0
        self._execute(f"FS{val}")

    def get_fast_step(self):
        """
        Read FAST step state.

        Returns
        -------
        int or None
            0 = OFF, 1 = ON; None on error.
        """
        resp = self._execute("FS", read=True)
        if resp and resp.startswith("FS") and len(resp) >= 3:
            ch = resp[-1]
            if ch.isdigit():
                return int(ch)
        return None

    # ------------------------------------------------------------------ #
    # FT – FUNCTION TX (split TX VFO select)
    # ------------------------------------------------------------------ #
    def set_tx_vfo(self, vfo):
        """
        Select which VFO is used for transmit.

        Parameters
        ----------
        vfo : int
            2 = VFO-A TX, 3 = VFO-B TX.
        """
        if vfo not in (2, 3):
            return False
        self._execute(f"FT{vfo}")
        return True

    def get_tx_vfo(self):
        """
        Read which VFO is selected for transmit.

        Returns
        -------
        int or None
            0 = VFO-A TX, 1 = VFO-B TX; None on error.
        """
        resp = self._execute("FT", read=True)
        if resp and resp.startswith("FT") and len(resp) >= 3:
            ch = resp[-1]
            if ch.isdigit():
                return int(ch)
        return None

    # ------------------------------------------------------------------ #
    # GT – AGC FUNCTION
    # ------------------------------------------------------------------ #
    def set_agc(self, mode):
        """
        Set AGC mode.

        Parameters
        ----------
        mode : int
            0 = OFF, 1 = FAST, 2 = MID, 3 = SLOW, 4 = AUTO.
        """
        if mode not in (0, 1, 2, 3, 4):
            return False
        self._execute(f"GT0{mode}")
        return True

    def get_agc(self):
        """
        Read AGC mode.

        Returns
        -------
        int or None
            0–6 per the manual's answer code; None on error.
        """
        resp = self._execute("GT0", read=True)
        if resp and resp.startswith("GT") and len(resp) >= 4:
            ch = resp[-1]
            if ch.isdigit():
                return int(ch)
        return None

    # ------------------------------------------------------------------ #
    # ID – IDENTIFICATION  (read only)
    # ------------------------------------------------------------------ #
    def get_id(self):
        """
        Read radio model identification.

        Returns
        -------
        str or None
            "0670" for FT-991A; None on error.
        """
        resp = self._execute("ID", read=True)
        if resp and resp.startswith("ID") and len(resp) >= 6:
            return resp[2:]
        return None

    # ------------------------------------------------------------------ #
    # IF – INFORMATION  (read only)
    # ------------------------------------------------------------------ #
    def get_info(self):
        """
        Read comprehensive radio status (IF command).

        Returns
        -------
        dict or None
            Parsed fields: memory_channel, frequency_hz, clarifier_direction,
            clarifier_offset, rx_clar, tx_clar, mode, vfo_memory, ctcss,
            repeater_shift; or None on error.
        """
        resp = self._execute("IF", read=True)
        if not resp or not resp.startswith("IF") or len(resp) < 28:
            return None
        try:
            body = resp[2:]
            return {
                "memory_channel": int(body[0:3]),
                "frequency_hz": int(body[3:12]),
                "clarifier_direction": body[12],
                "clarifier_offset": int(body[13:17]),
                "rx_clar": int(body[17]),
                "tx_clar": int(body[18]),
                "mode": body[19],
                "vfo_memory": int(body[20]),
                "ctcss": int(body[21]),
                "scan": int(body[22:24]),
                "repeater_shift": int(body[24]),
            }
        except (ValueError, IndexError):
            return None

    # ------------------------------------------------------------------ #
    # IS – IF SHIFT
    # ------------------------------------------------------------------ #
    def set_if_shift(self, offset_hz):
        """
        Set IF shift offset.

        Parameters
        ----------
        offset_hz : int
            Offset in Hz, -1200 to +1200 (20 Hz steps).
        """
        try:
            hz = int(offset_hz)
        except (TypeError, ValueError):
            return False
        hz = max(-1200, min(1200, hz))
        sign = "+" if hz >= 0 else "-"
        self._execute(f"IS0{sign}{abs(hz):04d}")
        return True

    def get_if_shift(self):
        """
        Read IF shift offset.

        Returns
        -------
        int or None
            Offset in Hz; None on error.
        """
        resp = self._execute("IS0", read=True)
        if resp and resp.startswith("IS") and len(resp) >= 8:
            sign_char = resp[3]
            digits = resp[4:]
            if digits.isdigit() and sign_char in ("+", "-"):
                val = int(digits)
                return -val if sign_char == "-" else val
        return None

    # ------------------------------------------------------------------ #
    # KM – KEYER MEMORY
    # ------------------------------------------------------------------ #
    def set_keyer_memory(self, channel, message):
        """
        Store a CW keyer memory message.

        Parameters
        ----------
        channel : int
            Memory channel, 1–5.
        message : str
            Message text, up to 50 characters.
        """
        if not (1 <= int(channel) <= 5):
            return False
        text = str(message)[:50]
        self._execute(f"KM{channel}{text}")
        return True

    def get_keyer_memory(self, channel):
        """
        Read a CW keyer memory message.

        Parameters
        ----------
        channel : int
            Memory channel, 1–5.

        Returns
        -------
        str or None
        """
        if not (1 <= int(channel) <= 5):
            return None
        resp = self._execute(f"KM{channel}", read=True)
        if resp and resp.startswith("KM") and len(resp) > 3:
            return resp[3:]
        return None

    # ------------------------------------------------------------------ #
    # KP – KEY PITCH
    # ------------------------------------------------------------------ #
    def set_key_pitch(self, index):
        """
        Set CW key (sidetone) pitch.

        Parameters
        ----------
        index : int
            0–75, corresponding to 300–1050 Hz in 10 Hz steps.
        """
        index = max(0, min(75, int(index)))
        self._execute(f"KP{index:02d}")

    def get_key_pitch(self):
        """
        Read CW key pitch index.

        Returns
        -------
        int or None
            0–75; None on error.
        """
        resp = self._execute("KP", read=True)
        if resp and resp.startswith("KP") and len(resp) >= 4:
            digits = resp[-2:]
            if digits.isdigit():
                return int(digits)
        return None

    # ------------------------------------------------------------------ #
    # KR – KEYER ON/OFF
    # ------------------------------------------------------------------ #
    def set_keyer(self, enabled):
        """
        Enable or disable the CW keyer.

        Parameters
        ----------
        enabled : bool or int
        """
        val = 1 if enabled else 0
        self._execute(f"KR{val}")

    def get_keyer(self):
        """
        Read keyer state.

        Returns
        -------
        int or None
            0 = OFF, 1 = ON; None on error.
        """
        resp = self._execute("KR", read=True)
        if resp and resp.startswith("KR") and len(resp) >= 3:
            ch = resp[-1]
            if ch.isdigit():
                return int(ch)
        return None

    # ------------------------------------------------------------------ #
    # KS – KEY SPEED
    # ------------------------------------------------------------------ #
    def set_key_speed(self, wpm):
        """
        Set CW keyer speed.

        Parameters
        ----------
        wpm : int
            Speed in WPM, 4–60.
        """
        wpm = max(4, min(60, int(wpm)))
        self._execute(f"KS{wpm:03d}")

    def get_key_speed(self):
        """
        Read CW keyer speed.

        Returns
        -------
        int or None
            Speed in WPM; None on error.
        """
        resp = self._execute("KS", read=True)
        if resp and resp.startswith("KS") and len(resp) >= 5:
            digits = resp[-3:]
            if digits.isdigit():
                return int(digits)
        return None

    # ------------------------------------------------------------------ #
    # KY – CW KEYING  (set only)
    # ------------------------------------------------------------------ #
    def cw_key(self, memory):
        """
        Start CW keyer memory playback.

        Parameters
        ----------
        memory : int or str
            1–5 for keyer memories 1–5;
            6–9,"A" for message keyers 1–5.
        """
        self._execute(f"KY{memory}")

    # ------------------------------------------------------------------ #
    # LK – LOCK
    # ------------------------------------------------------------------ #
    def set_lock(self, enabled):
        """
        Enable or disable the VFO-A dial lock.

        Parameters
        ----------
        enabled : bool or int
        """
        val = 1 if enabled else 0
        self._execute(f"LK{val}")

    def get_lock(self):
        """
        Read VFO-A dial lock state.

        Returns
        -------
        int or None
            0 = OFF, 1 = ON; None on error.
        """
        resp = self._execute("LK", read=True)
        if resp and resp.startswith("LK") and len(resp) >= 3:
            ch = resp[-1]
            if ch.isdigit():
                return int(ch)
        return None

    # ------------------------------------------------------------------ #
    # LM – LOAD MESSAGE (DVS)
    # ------------------------------------------------------------------ #
    def set_dvs_record(self, channel):
        """
        Start or stop DVS recording.

        Parameters
        ----------
        channel : int
            0 = stop, 1–5 = start/stop recording on channel 1–5.
        """
        channel = max(0, min(5, int(channel)))
        self._execute(f"LM0{channel}")

    def get_dvs_record(self):
        """
        Read DVS recording state.

        Returns
        -------
        int or None
            0 = stopped, 1–5 = recording channel; None on error.
        """
        resp = self._execute("LM0", read=True)
        if resp and resp.startswith("LM") and len(resp) >= 4:
            ch = resp[-1]
            if ch.isdigit():
                return int(ch)
        return None

    # ------------------------------------------------------------------ #
    # MA – MEMORY CHANNEL TO VFO-A  (set only)
    # ------------------------------------------------------------------ #
    def memory_to_vfo_a(self):
        """Recall current memory channel into VFO-A."""
        self._execute("MA")

    # ------------------------------------------------------------------ #
    # MC – MEMORY CHANNEL
    # ------------------------------------------------------------------ #
    def set_memory_channel(self, channel):
        """
        Select a memory channel.

        Parameters
        ----------
        channel : int
            1–117 (1–99: regular, 100–117: PMS channels).
        """
        channel = max(1, min(117, int(channel)))
        self._execute(f"MC{channel:03d}")

    def get_memory_channel(self):
        """
        Read the current memory channel number.

        Returns
        -------
        int or None
            1–117; None on error.
        """
        resp = self._execute("MC", read=True)
        if resp and resp.startswith("MC") and len(resp) >= 5:
            digits = resp[-3:]
            if digits.isdigit():
                return int(digits)
        return None

    # ------------------------------------------------------------------ #
    # MG – MIC GAIN
    # ------------------------------------------------------------------ #
    def set_mic_gain(self, level):
        """
        Set microphone gain.

        Parameters
        ----------
        level : int
            0–100.
        """
        level = max(0, min(100, int(level)))
        self._execute(f"MG{level:03d}")

    def get_mic_gain(self):
        """
        Read microphone gain.

        Returns
        -------
        int or None
            0–100; None on error.
        """
        resp = self._execute("MG", read=True)
        if resp and resp.startswith("MG") and len(resp) >= 5:
            digits = resp[-3:]
            if digits.isdigit():
                return int(digits)
        return None

    # ------------------------------------------------------------------ #
    # ML – MONITOR LEVEL
    # ------------------------------------------------------------------ #
    def set_monitor(self, enabled):
        """
        Enable or disable the TX monitor.

        Parameters
        ----------
        enabled : bool or int
            False/0 = OFF, True/1 = ON.
        """
        val = 1 if enabled else 0
        self._execute(f"ML0{val:03d}")

    def set_monitor_level(self, level):
        """
        Set TX monitor level.

        Parameters
        ----------
        level : int
            0–100.
        """
        level = max(0, min(100, int(level)))
        self._execute(f"ML1{level:03d}")

    def get_monitor(self):
        """
        Read monitor ON/OFF state and level.

        Returns
        -------
        dict or None
            {"enabled": bool, "level": int} or None on error.
        """
        resp_state = self._execute("ML0", read=True)
        resp_level = self._execute("ML1", read=True)
        result = {}
        if resp_state and resp_state.startswith("ML") and len(resp_state) >= 6:
            digits = resp_state[-3:]
            if digits.isdigit():
                result["enabled"] = int(digits) == 1
        if resp_level and resp_level.startswith("ML") and len(resp_level) >= 6:
            digits = resp_level[-3:]
            if digits.isdigit():
                result["level"] = int(digits)
        return result if result else None

    # ------------------------------------------------------------------ #
    # MR – MEMORY CHANNEL READ  (read only)
    # ------------------------------------------------------------------ #
    def read_memory_channel(self, channel):
        """
        Read all data stored in a memory channel.

        Parameters
        ----------
        channel : int
            1–117.

        Returns
        -------
        dict or None
            Parsed memory channel data or None on error.
        """
        channel = max(1, min(117, int(channel)))
        resp = self._execute(f"MR{channel:03d}", read=True)
        if not resp or not resp.startswith("MR") or len(resp) < 28:
            return None
        try:
            body = resp[2:]
            return {
                "channel": int(body[0:3]),
                "frequency_hz": int(body[3:12]),
                "clarifier_direction": body[12],
                "clarifier_offset": int(body[13:17]),
                "rx_clar": int(body[17]),
                "tx_clar": int(body[18]),
                "mode": body[19],
                "vfo_memory": int(body[20]),
                "ctcss": int(body[21]),
                "scan": int(body[22:24]),
                "repeater_shift": int(body[24]),
            }
        except (ValueError, IndexError):
            return None

    # ------------------------------------------------------------------ #
    # MS – METER SW
    # ------------------------------------------------------------------ #
    def set_meter(self, meter_type):
        """
        Select the front-panel meter reading.

        Parameters
        ----------
        meter_type : int
            0 = COMP, 1 = ALC, 2 = PO, 3 = SWR, 4 = ID, 5 = VDD.
        """
        if meter_type not in (0, 1, 2, 3, 4, 5):
            return False
        self._execute(f"MS{meter_type}")
        return True

    def get_meter_type(self):
        """
        Read the currently selected meter type.

        Returns
        -------
        int or None
            0–5; None on error.
        """
        resp = self._execute("MS", read=True)
        if resp and resp.startswith("MS") and len(resp) >= 3:
            ch = resp[-1]
            if ch.isdigit():
                return int(ch)
        return None

    # ------------------------------------------------------------------ #
    # MT – MEMORY CHANNEL WRITE/TAG
    # ------------------------------------------------------------------ #
    def write_memory_channel_tag(self, channel, freq_hz, mode_code,
                                  tag="", clarifier_dir="+",
                                  clarifier_offset=0,
                                  rx_clar=0, tx_clar=0,
                                  ctcss=0, repeater_shift=0):
        """
        Write frequency, mode, and optional tag to a memory channel.

        Parameters
        ----------
        channel : int
            1–117.
        freq_hz : int
            Frequency in Hz.
        mode_code : str
            Single character mode code (1=LSB, 2=USB, 3=CW, etc.).
        tag : str
            Memory tag up to 12 ASCII characters.
        clarifier_dir : str
            "+" or "-".
        clarifier_offset : int
            0–9999 Hz.
        rx_clar : int
            0 = OFF, 1 = ON.
        tx_clar : int
            0 = OFF, 1 = ON.
        ctcss : int
            0–4 (CTCSS/DCS mode).
        repeater_shift : int
            0 = Simplex, 1 = Plus, 2 = Minus.
        """
        channel = max(1, min(117, int(channel)))
        tag_field = str(tag)[:12].ljust(12)
        cmd = (
            f"MT{channel:03d}{int(freq_hz):09d}"
            f"{clarifier_dir}{int(clarifier_offset):04d}"
            f"{int(rx_clar)}{int(tx_clar)}{mode_code}"
            f"0{ctcss}00{int(repeater_shift)}0{tag_field}"
        )
        self._execute(cmd)
        return True

    # ------------------------------------------------------------------ #
    # MW – MEMORY CHANNEL WRITE  (set only)
    # ------------------------------------------------------------------ #
    def write_memory_channel(self, channel, freq_hz, mode_code,
                              clarifier_dir="+", clarifier_offset=0,
                              rx_clar=0, tx_clar=0,
                              ctcss=0, repeater_shift=0):
        """
        Write frequency and mode to a memory channel (no tag).

        Parameters
        ----------
        channel : int
            1–117.
        freq_hz : int
            Frequency in Hz.
        mode_code : str
            Single character mode code.
        clarifier_dir : str
            "+" or "-".
        clarifier_offset : int
            0–9999 Hz.
        rx_clar : int
            0 = OFF, 1 = ON.
        tx_clar : int
            0 = OFF, 1 = ON.
        ctcss : int
            0–4.
        repeater_shift : int
            0 = Simplex, 1 = Plus, 2 = Minus.
        """
        channel = max(1, min(117, int(channel)))
        cmd = (
            f"MW{channel:03d}{int(freq_hz):09d}"
            f"{clarifier_dir}{int(clarifier_offset):04d}"
            f"{int(rx_clar)}{int(tx_clar)}{mode_code}"
            f"00{ctcss}00{int(repeater_shift)}"
        )
        self._execute(cmd)
        return True

    # ------------------------------------------------------------------ #
    # MX – MOX SET
    # ------------------------------------------------------------------ #
    def set_mox(self, enabled):
        """
        Enable or disable MOX (manual transmit).

        Parameters
        ----------
        enabled : bool or int
        """
        val = 1 if enabled else 0
        self._execute(f"MX{val}")

    def get_mox(self):
        """
        Read MOX state.

        Returns
        -------
        int or None
            0 = OFF, 1 = ON; None on error.
        """
        resp = self._execute("MX", read=True)
        if resp and resp.startswith("MX") and len(resp) >= 3:
            ch = resp[-1]
            if ch.isdigit():
                return int(ch)
        return None

    # ------------------------------------------------------------------ #
    # NA – NARROW
    # ------------------------------------------------------------------ #
    def set_narrow(self, enabled):
        """
        Enable or disable narrow filter mode.

        Parameters
        ----------
        enabled : bool or int
            0/False = OFF, 1/True = ON.
        """
        val = 1 if enabled else 0
        self._execute(f"NA0{val}")

    def get_narrow(self):
        """
        Read narrow filter state.

        Returns
        -------
        int or None
            0 = OFF, 1 = ON; None on error.
        """
        resp = self._execute("NA0", read=True)
        if resp and resp.startswith("NA") and len(resp) >= 4:
            ch = resp[-1]
            if ch.isdigit():
                return int(ch)
        return None

    # ------------------------------------------------------------------ #
    # NB – NOISE BLANKER
    # ------------------------------------------------------------------ #
    def set_noise_blanker(self, enabled):
        """
        Enable or disable the noise blanker.

        Parameters
        ----------
        enabled : bool or int
        """
        val = 1 if enabled else 0
        self._execute(f"NB0{val}")

    def get_noise_blanker(self):
        """
        Read noise blanker state.

        Returns
        -------
        int or None
            0 = OFF, 1 = ON; None on error.
        """
        resp = self._execute("NB0", read=True)
        if resp and resp.startswith("NB") and len(resp) >= 4:
            ch = resp[-1]
            if ch.isdigit():
                return int(ch)
        return None

    # ------------------------------------------------------------------ #
    # NL – NOISE BLANKER LEVEL
    # ------------------------------------------------------------------ #
    def set_noise_blanker_level(self, level):
        """
        Set noise blanker attenuation level.

        Parameters
        ----------
        level : int
            0–10.
        """
        level = max(0, min(10, int(level)))
        self._execute(f"NL0{level:03d}")

    def get_noise_blanker_level(self):
        """
        Read noise blanker level.

        Returns
        -------
        int or None
            0–10; None on error.
        """
        resp = self._execute("NL0", read=True)
        if resp and resp.startswith("NL") and len(resp) >= 6:
            digits = resp[-3:]
            if digits.isdigit():
                return int(digits)
        return None

    # ------------------------------------------------------------------ #
    # NR – NOISE REDUCTION ON/OFF
    # ------------------------------------------------------------------ #
    def set_noise_reduction(self, enabled):
        """
        Enable or disable DSP noise reduction.

        Parameters
        ----------
        enabled : bool or int
        """
        val = 1 if enabled else 0
        self._execute(f"NR0{val}")

    def get_noise_reduction(self):
        """
        Read noise reduction state.

        Returns
        -------
        int or None
            0 = OFF, 1 = ON; None on error.
        """
        resp = self._execute("NR0", read=True)
        if resp and resp.startswith("NR") and len(resp) >= 4:
            ch = resp[-1]
            if ch.isdigit():
                return int(ch)
        return None

    # ------------------------------------------------------------------ #
    # OI – OPPOSITE BAND INFORMATION  (read only)
    # ------------------------------------------------------------------ #
    def get_opposite_band_info(self):
        """
        Read VFO-B (opposite band) status information.

        Returns
        -------
        dict or None
            Same structure as get_info() but for the opposite band.
        """
        resp = self._execute("OI", read=True)
        if not resp or not resp.startswith("OI") or len(resp) < 28:
            return None
        try:
            body = resp[2:]
            return {
                "memory_channel": int(body[0:3]),
                "frequency_hz": int(body[3:12]),
                "clarifier_direction": body[12],
                "clarifier_offset": int(body[13:17]),
                "rx_clar": int(body[17]),
                "tx_clar": int(body[18]),
                "mode": body[19],
                "vfo_memory": int(body[20]),
                "ctcss": int(body[21]),
                "scan": int(body[22:24]),
                "repeater_shift": int(body[24]),
            }
        except (ValueError, IndexError):
            return None

    # ------------------------------------------------------------------ #
    # OS – OFFSET (REPEATER SHIFT)
    # ------------------------------------------------------------------ #
    def set_repeater_shift(self, shift):
        """
        Set repeater shift direction.

        Parameters
        ----------
        shift : int
            0 = Simplex, 1 = Plus shift, 2 = Minus shift.
        """
        if shift not in (0, 1, 2):
            return False
        self._execute(f"OS0{shift}")
        return True

    def get_repeater_shift(self):
        """
        Read repeater shift direction.

        Returns
        -------
        int or None
            0 = Simplex, 1 = Plus, 2 = Minus; None on error.
        """
        resp = self._execute("OS0", read=True)
        if resp and resp.startswith("OS") and len(resp) >= 4:
            ch = resp[-1]
            if ch.isdigit():
                return int(ch)
        return None

    # ------------------------------------------------------------------ #
    # PA – PRE-AMP / IPO
    # ------------------------------------------------------------------ #
    def set_preamp(self, mode):
        """
        Set pre-amplifier / IPO mode.

        Parameters
        ----------
        mode : int
            0 = IPO (no preamp), 1 = AMP 1, 2 = AMP 2.
        """
        if mode not in (0, 1, 2):
            return False
        self._execute(f"PA0{mode}")
        return True

    def get_preamp(self):
        """
        Read pre-amplifier mode.

        Returns
        -------
        int or None
            0 = IPO, 1 = AMP 1, 2 = AMP 2; None on error.
        """
        resp = self._execute("PA0", read=True)
        if resp and resp.startswith("PA") and len(resp) >= 4:
            ch = resp[-1]
            if ch.isdigit():
                return int(ch)
        return None

    # ------------------------------------------------------------------ #
    # PB – DVS PLAYBACK
    # ------------------------------------------------------------------ #
    def set_dvs_playback(self, channel):
        """
        Start or stop DVS playback.

        Parameters
        ----------
        channel : int
            0 = stop, 1–5 = play channel 1–5.
        """
        channel = max(0, min(5, int(channel)))
        self._execute(f"PB0{channel}")

    def get_dvs_playback(self):
        """
        Read DVS playback state.

        Returns
        -------
        int or None
            0 = stopped, 1–5 = playing channel; None on error.
        """
        resp = self._execute("PB0", read=True)
        if resp and resp.startswith("PB") and len(resp) >= 4:
            ch = resp[-1]
            if ch.isdigit():
                return int(ch)
        return None

    # ------------------------------------------------------------------ #
    # PL – SPEECH PROCESSOR LEVEL
    # ------------------------------------------------------------------ #
    def set_speech_processor_level(self, level):
        """
        Set speech processor output level.

        Parameters
        ----------
        level : int
            0–100.
        """
        level = max(0, min(100, int(level)))
        self._execute(f"PL{level:03d}")

    def get_speech_processor_level(self):
        """
        Read speech processor level.

        Returns
        -------
        int or None
            0–100; None on error.
        """
        resp = self._execute("PL", read=True)
        if resp and resp.startswith("PL") and len(resp) >= 5:
            digits = resp[-3:]
            if digits.isdigit():
                return int(digits)
        return None

    # ------------------------------------------------------------------ #
    # PR – SPEECH PROCESSOR ON/OFF
    # ------------------------------------------------------------------ #
    def set_speech_processor(self, processor_type, enabled):
        """
        Enable or disable the speech processor or parametric EQ.

        Parameters
        ----------
        processor_type : int
            0 = Speech Processor, 1 = Parametric Microphone EQ.
        enabled : bool or int
            False/1 = OFF (note: per manual P2=1 is OFF), True/2 = ON.
        """
        if processor_type not in (0, 1):
            return False
        val = 2 if enabled else 1
        self._execute(f"PR{processor_type}{val}")
        return True

    def get_speech_processor(self, processor_type):
        """
        Read speech processor or parametric EQ state.

        Parameters
        ----------
        processor_type : int
            0 = Speech Processor, 1 = Parametric Microphone EQ.

        Returns
        -------
        int or None
            1 = OFF, 2 = ON; None on error.
        """
        if processor_type not in (0, 1):
            return None
        resp = self._execute(f"PR{processor_type}", read=True)
        if resp and resp.startswith("PR") and len(resp) >= 4:
            ch = resp[-1]
            if ch.isdigit():
                return int(ch)
        return None

    # ------------------------------------------------------------------ #
    # PS – POWER SWITCH
    # ------------------------------------------------------------------ #
    def set_power_switch(self, on):
        """
        Power the radio on or off.

        Note: The manual requires dummy data be sent first, then this
        command within 1–2 seconds.

        Parameters
        ----------
        on : bool or int
            False/0 = OFF, True/1 = ON.
        """
        val = 1 if on else 0
        self._execute(f"PS{val}")

    def get_power_switch(self):
        """
        Read power state.

        Returns
        -------
        int or None
            0 = OFF, 1 = ON; None on error.
        """
        resp = self._execute("PS", read=True)
        if resp and resp.startswith("PS") and len(resp) >= 3:
            ch = resp[-1]
            if ch.isdigit():
                return int(ch)
        return None

    # ------------------------------------------------------------------ #
    # QI – QMB STORE  (set only)
    # ------------------------------------------------------------------ #
    def qmb_store(self):
        """Store the current VFO into the Quick Memory Bank."""
        self._execute("QI")

    # ------------------------------------------------------------------ #
    # QR – QMB RECALL  (set only)
    # ------------------------------------------------------------------ #
    def qmb_recall(self):
        """Recall a Quick Memory Bank entry into VFO-A."""
        self._execute("QR")

    # ------------------------------------------------------------------ #
    # QS – QUICK SPLIT  (set only)
    # ------------------------------------------------------------------ #
    def quick_split(self):
        """Activate the Quick Split function."""
        self._execute("QS")

    # ------------------------------------------------------------------ #
    # RA – RF ATTENUATOR
    # ------------------------------------------------------------------ #
    def set_rf_attenuator(self, enabled):
        """
        Enable or disable the RF attenuator.

        Parameters
        ----------
        enabled : bool or int
        """
        val = 1 if enabled else 0
        self._execute(f"RA0{val}")

    def get_rf_attenuator(self):
        """
        Read RF attenuator state.

        Returns
        -------
        int or None
            0 = OFF, 1 = ON; None on error.
        """
        resp = self._execute("RA0", read=True)
        if resp and resp.startswith("RA") and len(resp) >= 4:
            ch = resp[-1]
            if ch.isdigit():
                return int(ch)
        return None

    # ------------------------------------------------------------------ #
    # RC – CLAR CLEAR  (set only)
    # ------------------------------------------------------------------ #
    def clar_clear(self):
        """Clear (zero) the clarifier offset."""
        self._execute("RC")

    # ------------------------------------------------------------------ #
    # RD – CLAR DOWN  (set only)
    # ------------------------------------------------------------------ #
    def clar_down(self, hz):
        """
        Shift the clarifier downward.

        Parameters
        ----------
        hz : int
            Offset step in Hz, 0–9999.
        """
        hz = max(0, min(9999, int(hz)))
        self._execute(f"RD{hz:04d}")

    # ------------------------------------------------------------------ #
    # RG – RF GAIN
    # ------------------------------------------------------------------ #
    def set_rf_gain(self, level):
        """
        Set RF gain.

        Parameters
        ----------
        level : int
            0–255.
        """
        level = max(0, min(255, int(level)))
        self._execute(f"RG0{level:03d}")

    def get_rf_gain(self):
        """
        Read RF gain.

        Returns
        -------
        int or None
            0–255; None on error.
        """
        resp = self._execute("RG0", read=True)
        if resp and resp.startswith("RG") and len(resp) >= 6:
            digits = resp[-3:]
            if digits.isdigit():
                return int(digits)
        return None

    # ------------------------------------------------------------------ #
    # RI – RADIO INFORMATION  (read only)
    # ------------------------------------------------------------------ #
    # P1 values: 0=Hi-SWR, 3=REC, 4=PLAY, 5=VFO-A TX, 6=VFO-B TX, 7=VFO-A RX, A=TX LED
    def get_radio_info(self, info_type):
        """
        Read a specific radio status indicator.

        Parameters
        ----------
        info_type : int or str
            0 = Hi-SWR, 3 = REC, 4 = PLAY, 5 = VFO-A TX, 6 = VFO-B TX,
            7 = VFO-A RX, "A" = TX LED.

        Returns
        -------
        int or None
            0 = OFF, 1 = ON; None on error.
        """
        resp = self._execute(f"RI{info_type}", read=True)
        if resp and resp.startswith("RI") and len(resp) >= 4:
            ch = resp[-1]
            if ch.isdigit():
                return int(ch)
        return None

    # ------------------------------------------------------------------ #
    # RL – NOISE REDUCTION LEVEL
    # ------------------------------------------------------------------ #
    def set_noise_reduction_level(self, level):
        """
        Set DSP noise reduction level.

        Parameters
        ----------
        level : int
            1–15.
        """
        level = max(1, min(15, int(level)))
        self._execute(f"RL0{level:02d}")

    def get_noise_reduction_level(self):
        """
        Read DSP noise reduction level.

        Returns
        -------
        int or None
            1–15; None on error.
        """
        resp = self._execute("RL0", read=True)
        if resp and resp.startswith("RL") and len(resp) >= 5:
            digits = resp[-2:]
            if digits.isdigit():
                return int(digits)
        return None

    # ------------------------------------------------------------------ #
    # RM – READ METER  (already partial: get_swr_meter, get_s_meter)
    # ------------------------------------------------------------------ #
    def read_meter(self, meter_type):
        """
        Read a meter value by type.

        Parameters
        ----------
        meter_type : int
            0 = front-panel meter, 1 = S-meter, 3 = COMP, 4 = ALC,
            5 = PO (RF power), 6 = SWR, 7 = ID current, 8 = VDD voltage.

        Returns
        -------
        int or None
            Raw meter value 0–255; None on error.
        """
        resp = self._execute(f"RM{meter_type}", read=True)
        if resp and resp.startswith("RM") and len(resp) >= 6:
            digits = resp[-3:]
            if digits.isdigit():
                return int(digits)
        return None

    # ------------------------------------------------------------------ #
    # RS – RADIO STATUS  (read only)
    # ------------------------------------------------------------------ #
    def get_radio_status(self):
        """
        Read radio operating status.

        Returns
        -------
        int or None
            0 = Normal mode, 1 = Menu mode; None on error.
        """
        resp = self._execute("RS", read=True)
        if resp and resp.startswith("RS") and len(resp) >= 3:
            ch = resp[-1]
            if ch.isdigit():
                return int(ch)
        return None

    # ------------------------------------------------------------------ #
    # RT – RX CLARIFIER ON/OFF
    # ------------------------------------------------------------------ #
    def set_rx_clarifier(self, enabled):
        """
        Enable or disable the RX clarifier.

        Parameters
        ----------
        enabled : bool or int
        """
        val = 1 if enabled else 0
        self._execute(f"RT{val}")

    def get_rx_clarifier(self):
        """
        Read RX clarifier state.

        Returns
        -------
        int or None
            0 = OFF, 1 = ON; None on error.
        """
        resp = self._execute("RT", read=True)
        if resp and resp.startswith("RT") and len(resp) >= 3:
            ch = resp[-1]
            if ch.isdigit():
                return int(ch)
        return None

    # ------------------------------------------------------------------ #
    # RU – CLAR UP  (set only)
    # ------------------------------------------------------------------ #
    def clar_up(self, hz):
        """
        Shift the clarifier upward.

        Parameters
        ----------
        hz : int
            Offset step in Hz, 0–9999.
        """
        hz = max(0, min(9999, int(hz)))
        self._execute(f"RU{hz:04d}")

    # ------------------------------------------------------------------ #
    # SC – SCAN
    # ------------------------------------------------------------------ #
    def set_scan(self, mode):
        """
        Start or stop scanning.

        Parameters
        ----------
        mode : int
            0 = OFF, 1 = Scan UP, 2 = Scan DOWN.
        """
        if mode not in (0, 1, 2):
            return False
        self._execute(f"SC{mode}")
        return True

    def get_scan(self):
        """
        Read scan state.

        Returns
        -------
        int or None
            0 = OFF, 1 = scanning UP, 2 = scanning DOWN; None on error.
        """
        resp = self._execute("SC", read=True)
        if resp and resp.startswith("SC") and len(resp) >= 3:
            ch = resp[-1]
            if ch.isdigit():
                return int(ch)
        return None

    # ------------------------------------------------------------------ #
    # SD – CW SEMI BREAK-IN DELAY TIME
    # ------------------------------------------------------------------ #
    def set_break_in_delay(self, ms):
        """
        Set CW semi break-in delay.

        Parameters
        ----------
        ms : int
            Delay in milliseconds, 30–3000.
        """
        ms = max(30, min(3000, int(ms)))
        self._execute(f"SD{ms:04d}")

    def get_break_in_delay(self):
        """
        Read CW semi break-in delay.

        Returns
        -------
        int or None
            Delay in ms; None on error.
        """
        resp = self._execute("SD", read=True)
        if resp and resp.startswith("SD") and len(resp) >= 6:
            digits = resp[-4:]
            if digits.isdigit():
                return int(digits)
        return None

    # ------------------------------------------------------------------ #
    # SH – DSP WIDTH (PASSBAND)
    # ------------------------------------------------------------------ #
    def set_width(self, width_code):
        """
        Set DSP IF passband width.

        Parameters
        ----------
        width_code : int
            00 = default; 01–21 per the width table in the manual.
        """
        width_code = max(0, min(21, int(width_code)))
        self._execute(f"SH0{width_code:02d}")

    def get_width(self):
        """
        Read DSP IF passband width code.

        Returns
        -------
        int or None
            0–21; None on error.
        """
        resp = self._execute("SH0", read=True)
        if resp and resp.startswith("SH") and len(resp) >= 5:
            digits = resp[-2:]
            if digits.isdigit():
                return int(digits)
        return None

    # ------------------------------------------------------------------ #
    # SQ – SQUELCH LEVEL
    # ------------------------------------------------------------------ #
    def set_squelch(self, level):
        """
        Set squelch level.

        Parameters
        ----------
        level : int
            0–100.
        """
        level = max(0, min(100, int(level)))
        self._execute(f"SQ0{level:03d}")

    def get_squelch(self):
        """
        Read squelch level.

        Returns
        -------
        int or None
            0–100; None on error.
        """
        resp = self._execute("SQ0", read=True)
        if resp and resp.startswith("SQ") and len(resp) >= 6:
            digits = resp[-3:]
            if digits.isdigit():
                return int(digits)
        return None

    # ------------------------------------------------------------------ #
    # SV – SWAP VFO  (set only)
    # ------------------------------------------------------------------ #
    def swap_vfo(self):
        """Swap VFO-A and VFO-B frequencies and modes."""
        self._execute("SV")

    # ------------------------------------------------------------------ #
    # TS – TXW (TX WATCH / MONITOR)
    # ------------------------------------------------------------------ #
    def set_txw(self, enabled):
        """
        Enable or disable TX monitor (TXW).

        Parameters
        ----------
        enabled : bool or int
        """
        val = 1 if enabled else 0
        self._execute(f"TS{val}")

    def get_txw(self):
        """
        Read TXW state.

        Returns
        -------
        int or None
            0 = OFF, 1 = ON; None on error.
        """
        resp = self._execute("TS", read=True)
        if resp and resp.startswith("TS") and len(resp) >= 3:
            ch = resp[-1]
            if ch.isdigit():
                return int(ch)
        return None

    # ------------------------------------------------------------------ #
    # TX – TX SET (extends ptt_on / ptt_off)
    # ------------------------------------------------------------------ #
    def get_tx_state(self):
        """
        Read the current TX state.

        Returns
        -------
        int or None
            0 = Radio TX OFF / CAT TX OFF,
            1 = Radio TX OFF / CAT TX ON,
            2 = Radio TX ON / CAT TX OFF (answer only);
            None on error.
        """
        resp = self._execute("TX", read=True)
        if resp and resp.startswith("TX") and len(resp) >= 3:
            ch = resp[-1]
            if ch.isdigit():
                return int(ch)
        return None

    # ------------------------------------------------------------------ #
    # UL – PLL UNLOCK STATUS  (read only)
    # ------------------------------------------------------------------ #
    def get_pll_lock(self):
        """
        Read PLL lock status.

        Returns
        -------
        bool or None
            True = locked, False = unlocked; None on error.
        """
        resp = self._execute("UL", read=True)
        if resp and resp.startswith("UL") and len(resp) >= 3:
            ch = resp[-1]
            if ch.isdigit():
                return ch == "0"
        return None

    # ------------------------------------------------------------------ #
    # UP – UP  (set only)
    # ------------------------------------------------------------------ #
    def up(self):
        """Simulate pressing the UP key on the front panel."""
        self._execute("UP")

    # ------------------------------------------------------------------ #
    # VD – VOX DELAY TIME
    # ------------------------------------------------------------------ #
    def set_vox_delay(self, ms):
        """
        Set VOX (or Data VOX) delay time.

        The parameter controlled depends on Menu item 142 (VOX SELECT):
        "MIC" changes VOX delay; "DATA" changes Data VOX delay.

        Parameters
        ----------
        ms : int
            30–3000 ms in 10 ms multiples.
        """
        ms = max(30, min(3000, int(ms)))
        self._execute(f"VD{ms:04d}")

    def get_vox_delay(self):
        """
        Read VOX delay time.

        Returns
        -------
        int or None
            Delay in ms; None on error.
        """
        resp = self._execute("VD", read=True)
        if resp and resp.startswith("VD") and len(resp) >= 6:
            digits = resp[-4:]
            if digits.isdigit():
                return int(digits)
        return None

    # ------------------------------------------------------------------ #
    # VG – VOX GAIN
    # ------------------------------------------------------------------ #
    def set_vox_gain(self, level):
        """
        Set VOX gain.

        Parameters
        ----------
        level : int
            0–100.
        """
        level = max(0, min(100, int(level)))
        self._execute(f"VG{level:03d}")

    def get_vox_gain(self):
        """
        Read VOX gain.

        Returns
        -------
        int or None
            0–100; None on error.
        """
        resp = self._execute("VG", read=True)
        if resp and resp.startswith("VG") and len(resp) >= 5:
            digits = resp[-3:]
            if digits.isdigit():
                return int(digits)
        return None

    # ------------------------------------------------------------------ #
    # VM – [V/M] KEY FUNCTION  (set only)
    # ------------------------------------------------------------------ #
    def vm_key(self):
        """Simulate pressing the [V/M] (VFO/Memory) key on the front panel."""
        self._execute("VM")

    # ------------------------------------------------------------------ #
    # VX – VOX STATUS
    # ------------------------------------------------------------------ #
    def set_vox(self, enabled):
        """
        Enable or disable VOX.

        Parameters
        ----------
        enabled : bool or int
        """
        val = 1 if enabled else 0
        self._execute(f"VX{val}")

    def get_vox(self):
        """
        Read VOX state.

        Returns
        -------
        int or None
            0 = OFF, 1 = ON; None on error.
        """
        resp = self._execute("VX", read=True)
        if resp and resp.startswith("VX") and len(resp) >= 3:
            ch = resp[-1]
            if ch.isdigit():
                return int(ch)
        return None

    # ------------------------------------------------------------------ #
    # XT – TX CLARIFIER ON/OFF
    # ------------------------------------------------------------------ #
    def set_tx_clarifier(self, enabled):
        """
        Enable or disable the TX clarifier.

        Parameters
        ----------
        enabled : bool or int
        """
        val = 1 if enabled else 0
        self._execute(f"XT{val}")

    def get_tx_clarifier(self):
        """
        Read TX clarifier state.

        Returns
        -------
        int or None
            0 = OFF, 1 = ON; None on error.
        """
        resp = self._execute("XT", read=True)
        if resp and resp.startswith("XT") and len(resp) >= 3:
            ch = resp[-1]
            if ch.isdigit():
                return int(ch)
        return None

    # ------------------------------------------------------------------ #
    # ZI – ZERO IN  (set only)
    # ------------------------------------------------------------------ #
    def zero_in(self):
        """Activate CW Auto Zero-In function."""
        self._execute("ZI")