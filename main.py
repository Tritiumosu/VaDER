#! /usr/bin/python3
import serial
import time
import csv
import threading
import tkinter as tk
from tkinter import ttk
from datetime import datetime

# --- Constants & Band Plans ---
BANDS = {
    '2m': {'start': 144.0, 'end': 148.0, 'step': 0.015, 'mode': 'FM'},
    '20m': {'start': 14.0, 'end': 14.35, 'step': 0.001, 'mode': 'USB'},
    '10m': {'start': 28.3, 'end': 29.7, 'step': 0.005, 'mode': 'USB'},
}


# --- Radio Control Class ---
class Yaesu991AControl:
    def __init__(self, port='/dev/ttyUSB0', baud=38400):
        self.port = port
        self.baud = baud
        self.conn = None  # don't auto-connect on launch

        # CTCSS Tone Mapping (Index 001-050)
        self.tone_map = {
            67.0: "001", 69.3: "002", 71.9: "003", 74.4: "004", 77.0: "005",
            79.7: "006", 82.5: "007", 85.4: "008", 88.5: "009", 91.5: "010",
            100.0: "013", 103.5: "014", 123.0: "019", 141.3: "023", 151.4: "025"
            # (Add additional tones from the manual as needed)
        }

    def is_connected(self):
        return self.conn is not None and getattr(self.conn, "is_open", False)

    def connect(self):
        """Open serial connection. Returns (True, None) on success, (False, error_message) on failure."""
        if self.is_connected():
            return True, None
        try:
            self.conn = serial.Serial(
                port=self.port,
                baudrate=self.baud,
                timeout=1,
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
        """Standard Yaesu ASCII CAT execution: [CMD];"""
        if not self.is_connected():
            return None
        try:
            self.conn.write(f"{cmd};".encode('ascii'))
            if read:
                return self.conn.read_until(b';').decode().strip().replace(';', '')
        except Exception as e:
            print(f"Serial Error: {e}")
        return None

    def set_frequency(self, mhz):
        hz = int(mhz * 1_000_000)
        self._execute(f"FA{hz:09d}")

    def get_frequency(self):
        resp = self._execute("FA", read=True)
        return float(resp[2:]) / 1_000_000 if resp and len(resp) > 2 else 0.0

    def set_mode(self, mode_str):
        modes = {'LSB': '1', 'USB': '2', 'CW': '3', 'FM': '4', 'AM': '5', 'C4FM': 'E'}
        if mode_str in modes:
            self._execute(f"MD0{modes[mode_str]}")

    def get_mode(self):
        """
        Read current mode using the Yaesu CAT 'MD' command.

        Common FT-991A response formats seen in the wild include:
          - Query:  MD0;   Reply: MD0x;   (x is the mode code)
          - Query:  MD;    Reply: MD0x;   (some firmwares accept MD without the 0)

        Returns currently active mode or None if unknown/unavailable.
        """
        code_to_mode = {'1': 'LSB', '2': 'USB', '3': 'CW-U', '4': 'FM', '5': 'AM', '6': 'RTTY-L', '7': 'CW-L', '8': 'DATA-L', '9': 'RTTY-U', 'A': 'DATA-FM', 'C': 'DATA-U','E': 'C4FM'}

        resp = self._execute("MD0", read=True)
        if not resp:
            resp = self._execute("MD", read=True)
        if not resp:
            return None

        # Expected something like "MD02" or "MD0E"
        if resp.startswith("MD") and len(resp) >= 4:
            code = resp[-1].upper()
            return code_to_mode.get(code)

        return None
    def get_swr_meter(self):
        resp = self._execute("RM6", read=True)
        return int(resp[3:6]) if resp and len(resp) >= 6 else 0

    def get_s_meter(self):
        """
        Read S-meter using the Yaesu CAT 'SM' command
        Typical FT-991A behavior:
          - Query:  SM0;
          - Reply:  SM0xxx;   where xxx is 000-255
        """
        resp = self._execute("SM0", read=True)
        if not resp:
            return 0

        # Common case: "SM0" + 3 digits
        if resp.startswith("SM") and len(resp) >= 6:
            digits = resp[-3:]
            if digits.isdigit():
                return int(digits)

        return 0

    def get_rf_power(self):
        """
        Read current RF power level using the Yaesu CAT 'PC' command.

        Typical behavior:
          - Query:  PC;
          - Reply:  PCxxx;   where xxx is 005-100 (percent / watts setting depending on rig)
        """
        resp = self._execute("PC", read=True)
        if not resp:
            return 0

        # Expected: "PC" + 3 digits (e.g., "PC050")
        if resp.startswith("PC") and len(resp) >= 5:
            digits = resp[-3:]
            if digits.isdigit():
                return int(digits)

        return 0

    def set_rf_power(self, level):
        """
        Set RF power level using the Yaesu CAT 'PC' command.

        `level` is clamped to the radio's supported range: 5..100
        and sent as a 3-digit value (005..100), e.g. "PC050;".
        """
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


# --- GUI & Features Class ---
class RadioGUI:
    def __init__(self, root, radio):
        self.root = root
        self.radio = radio
        self.scanning = False
        self.root.title("FT-991A Command Center")
        self.root.geometry("500x750")
        self.setup_ui()
        self.refresh_connection_ui()
        self.update_loop()

    def setup_ui(self):
        # Frequency Display
        self.freq_disp = tk.Label(self.root, text="DISCONNECTED", font=("Consolas", 36), fg="lime", bg="black")
        self.freq_disp.pack(pady=15, fill=tk.X)

        # Connection controls
        conn_frame = tk.LabelFrame(self.root, text="Connection")
        conn_frame.pack(padx=20, pady=10, fill=tk.X)

        self.conn_status = tk.Label(conn_frame, text="Status: DISCONNECTED", anchor="w")
        self.conn_status.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.X, expand=True)

        self.conn_btn = tk.Button(conn_frame, text="CONNECT", bg="lightgreen", command=self.toggle_connection)
        self.conn_btn.pack(side=tk.RIGHT, padx=5, pady=5)

        # S-Meter Progress Bar
        tk.Label(self.root, text="S-METER").pack()
        self.meter_var = tk.IntVar()
        self.s_meter = ttk.Progressbar(self.root, maximum=255, variable=self.meter_var)
        self.s_meter.pack(padx=30, pady=5, fill=tk.X)

        # RF Power Control
        pwr_frame = tk.LabelFrame(self.root, text="RF Power (PC)")
        pwr_frame.pack(padx=20, pady=10, fill=tk.X)

        tk.Label(pwr_frame, text="Level (5-100):").pack(side=tk.LEFT, padx=5)

        self.rf_power_var = tk.IntVar(value=0)
        self.rf_power_spin = tk.Spinbox(
            pwr_frame,
            from_=5,
            to=100,
            width=5,
            textvariable=self.rf_power_var
        )
        self.rf_power_spin.pack(side=tk.LEFT, padx=5)
        # Press either Enter key in the spinbox to apply the RF power value
        self.rf_power_spin.bind("<Return>", lambda e: self.apply_rf_power())
        self.rf_power_spin.bind("<KP_Enter>", lambda e: self.apply_rf_power())
        # Or just use the "Apply" button like a caveman
        self.rf_power_apply_btn = tk.Button(pwr_frame, text="APPLY", command=self.apply_rf_power)
        self.rf_power_apply_btn.pack(side=tk.LEFT, padx=5)

        self.rf_power_status = tk.Label(pwr_frame, text="", anchor="w")
        self.rf_power_status.pack(side=tk.LEFT, padx=8, fill=tk.X, expand=True)

        # Mode Control (MD)
        mode_frame = tk.LabelFrame(self.root, text="Mode (MD)")
        mode_frame.pack(padx=20, pady=10, fill=tk.X)

        tk.Label(mode_frame, text="Mode:").pack(side=tk.LEFT, padx=5)

        self.mode_var = tk.StringVar(value="USB")
        self.mode_combo = ttk.Combobox(
            mode_frame,
            textvariable=self.mode_var,
            values=("LSB", "USB", "CW", "FM", "AM", "C4FM"),
            state="readonly",
            width=8
        )
        self.mode_combo.pack(side=tk.LEFT, padx=5)

        self.mode_apply_btn = tk.Button(mode_frame, text="APPLY", command=self.apply_mode)
        self.mode_apply_btn.pack(side=tk.LEFT, padx=5)

        self.mode_status = tk.Label(mode_frame, text="", anchor="w")
        self.mode_status.pack(side=tk.LEFT, padx=8, fill=tk.X, expand=True)

        # Apply on Enter as well
        self.mode_combo.bind("<<ComboboxSelected>>", lambda e: self.apply_mode())
        self.mode_combo.bind("<Return>", lambda e: self.apply_mode())
        self.mode_combo.bind("<KP_Enter>", lambda e: self.apply_mode())

        # Band Selection
        band_frame = tk.LabelFrame(self.root, text="Quick Band Select")
        band_frame.pack(padx=20, pady=10, fill=tk.X)
        for b in BANDS:
            tk.Button(
                band_frame,
                text=b,
                command=lambda name=b: self.goto_band(name)
            ).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)

        # Scanner
        scan_frame = tk.LabelFrame(self.root, text="Scanner Controls")
        scan_frame.pack(padx=20, pady=10, fill=tk.X)
        self.scan_btn = tk.Button(scan_frame, text="START SCAN", bg="lightgray", command=self.toggle_scan)
        self.scan_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5, pady=5)

        tk.Label(scan_frame, text="Squelch:").pack(side=tk.LEFT)
        self.thresh_entry = tk.Entry(scan_frame, width=5)
        self.thresh_entry.insert(0, "40")
        self.thresh_entry.pack(side=tk.LEFT, padx=5)

        # PTT
        self.ptt_btn = tk.Button(self.root, text="PUSH TO TALK", bg="darkred", fg="white", font=("Arial", 12, "bold"))
        self.ptt_btn.pack(pady=10, ipadx=30)
        self.ptt_btn.bind("<ButtonPress-1>", lambda e: self.radio.ptt_on())
        self.ptt_btn.bind("<ButtonRelease-1>", lambda e: self.radio.ptt_off())

        # Logger
        log_frame = tk.LabelFrame(self.root, text="Signal Log")
        log_frame.pack(padx=20, pady=10, fill=tk.BOTH, expand=True)

        self.note_entry = tk.Entry(log_frame)
        self.note_entry.insert(0, "Enter callsign/note...")
        self.note_entry.pack(fill=tk.X, padx=5, pady=5)

        tk.Button(log_frame, text="MANUAL LOG", command=self.manual_log, bg="lightblue").pack(fill=tk.X, padx=5)

        self.log_box = tk.Text(log_frame, height=10, font=("Consolas", 9))
        self.log_box.pack(padx=5, pady=5, fill=tk.BOTH, expand=True)

    def refresh_connection_ui(self):
        connected = self.radio.is_connected()
        if connected:
            self.conn_status.config(text=f"Status: CONNECTED ({self.radio.port} @ {self.radio.baud})")
            self.conn_btn.config(text="DISCONNECT", bg="orange")
            self.rf_power_apply_btn.config(state=tk.NORMAL)
            self.rf_power_spin.config(state="normal")
            self.mode_apply_btn.config(state=tk.NORMAL)
            self.mode_combo.config(state="readonly")
        else:
            self.conn_status.config(text=f"Status: DISCONNECTED ({self.radio.port} @ {self.radio.baud})")
            self.conn_btn.config(text="CONNECT", bg="lightgreen")
            self.meter_var.set(0)
            self.rf_power_var.set(0)
            self.rf_power_status.config(text="")
            self.rf_power_apply_btn.config(state=tk.DISABLED)
            self.rf_power_spin.config(state="disabled")
            self.mode_status.config(text="")
            self.mode_apply_btn.config(state=tk.DISABLED)
            self.mode_combo.config(state="disabled")

    def toggle_connection(self):
        if self.radio.is_connected():
            # Stop scanning before disconnecting to avoid thread using a closed port
            self.scanning = False
            self.scan_btn.config(text="START SCAN", bg="lightgray")

            self.radio.disconnect()
            self.freq_disp.config(text="DISCONNECTED")
            self.meter_var.set(0)
            self.rf_power_var.set(0)
            self.rf_power_status.config(text="")
            self.refresh_connection_ui()
            return

        ok, err = self.radio.connect()
        if not ok:
            # Show failure in the UI (and keep "disconnected" state)
            self.conn_status.config(text=f"Status: ERROR ({err})")
            self.freq_disp.config(text="DISCONNECTED")
            self.meter_var.set(0)
            self.rf_power_var.set(0)
            self.rf_power_status.config(text="")
            self.refresh_connection_ui()
            return

        # Connected successfully
        self.refresh_connection_ui()
        p = self.radio.get_rf_power()
        self.rf_power_var.set(p)
        self.rf_power_status.config(text=f"READ {p:03d}")
        m = self.radio.get_mode()
        if m:
            self.mode_var.set(m)
            self.mode_status.config(text=f"READ {m}")

    def apply_rf_power(self):
        """Apply RF power from the GUI control via PC command."""
        if not self.radio.is_connected():
            self.rf_power_status.config(text="DISCONNECTED")
            return

        try:
            desired = int(self.rf_power_var.get())
        except (TypeError, ValueError):
            self.rf_power_status.config(text="INVALID")
            return

        ok = self.radio.set_rf_power(desired)
        if not ok:
            self.rf_power_status.config(text="INVALID")
            return

        # Read back to show the clamped/actual value
        actual = self.radio.get_rf_power()
        self.rf_power_var.set(actual)
        self.rf_power_status.config(text=f"SET {actual:03d}")

    def apply_mode(self):
        """Apply selected mode via CAT using set_mode()."""
        if not self.radio.is_connected():
            self.mode_status.config(text="DISCONNECTED")
            return

        mode = (self.mode_var.get() or "").strip().upper()
        if mode not in {"LSB", "USB", "CW", "FM", "AM", "C4FM"}:
            self.mode_status.config(text="INVALID")
            return

        self.radio.set_mode(mode)
        self.mode_status.config(text=f"SET {mode}")

    def update_loop(self):
        """Updates frequency and meter if connected and not busy scanning."""
        if self.radio.is_connected() and not self.scanning:
            f = self.radio.get_frequency()
            self.freq_disp.config(text=f"{f:09.4f}")
            self.meter_var.set(self.radio.get_s_meter())

            # Keep RF power display in sync, but don't fight the user while editing/applying
            try:
                if str(self.root.focus_get()) != str(self.rf_power_spin):
                    self.rf_power_var.set(self.radio.get_rf_power())
            except Exception:
                pass

            # Keep Mode display in sync, but don't fight the user while the combobox has focus
            try:
                if str(self.root.focus_get()) != str(self.mode_combo):
                    m = self.radio.get_mode()
                    if m:
                        self.mode_var.set(m)
            except Exception:
                pass

        elif not self.radio.is_connected():
            # Keep a clear disconnected display
            self.freq_disp.config(text="DISCONNECTED")
            self.meter_var.set(0)
            self.rf_power_var.set(0)
            self.rf_power_status.config(text="")
            self.mode_status.config(text="")

        self.root.after(500, self.update_loop)

    def log_to_file(self, freq, strength, notes):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open('radio_log.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, f"{freq:.4f}", strength, notes])
        self.log_box.insert(tk.END, f"[{timestamp[-8:]}] {freq:.4f} S:{strength} | {notes}\n")
        self.log_box.see(tk.END)

    def manual_log(self):
        self.log_to_file(self.radio.get_frequency(), self.radio.get_s_meter(), self.note_entry.get())

    def toggle_scan(self):
        if not self.scanning:
            if not self.radio.is_connected():
                # Refuse to scan when disconnected
                self.conn_status.config(text="Status: DISCONNECTED (click CONNECT to scan)")
                self.scan_btn.config(text="START SCAN", bg="lightgray")
                self.scanning = False
                return

            self.scanning = True
            self.scan_btn.config(text="STOP SCAN", bg="orange")
            threading.Thread(target=self.scan_thread, daemon=True).start()
        else:
            self.scanning = False
            self.scan_btn.config(text="START SCAN", bg="lightgray")

    def scan_thread(self):
        curr_f = self.radio.get_frequency()
        thresh = int(self.thresh_entry.get())
        while self.scanning:
            self.radio.set_frequency(curr_f)
            time.sleep(0.15)
            s = self.radio.get_s_meter()
            if s >= thresh:
                self.log_to_file(curr_f, s, "AUTO-FOUND")
                time.sleep(3)  # Dwell on signal
            curr_f += 0.005
            if curr_f > 148.0: curr_f = 144.0  # Wrap-around example


if __name__ == "__main__":
    # Change '/dev/ttyUSB0' to 'COM3' or similar if on Windows
    radio = Yaesu991AControl(port='COM7', baud=38400)
    root = tk.Tk()
    gui = RadioGUI(root, radio)
    root.mainloop()