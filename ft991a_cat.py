import threading
import serial
from decimal import Decimal, ROUND_HALF_UP


class Yaesu991AControl:
    """
    Minimal CAT controller for Yaesu FT-991A using Yaesu ASCII CAT over serial.

    This module intentionally contains NO GUI code.
    """
    def __init__(self, port="/dev/ttyUSB0", baud=38400, timeout=1):
        self.port = port
        self.baud = baud
        self.timeout = timeout
        self.conn = None  # don't auto-connect on launch
        self._io_lock = threading.Lock()

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
            self.conn = serial.Serial(
                port=self.port,
                baudrate=self.baud,
                timeout=self.timeout,
                stopbits=serial.STOPBITS_ONE
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