#! /usr/bin/python3
import time
import csv
import threading
import tkinter as tk
from tkinter import ttk
from datetime import datetime
import queue

from ft991a_cat import Yaesu991AControl

# --- Constants & Band Plans ---
BANDS = {
    '2m': {'start': 144.0, 'end': 148.0, 'step': 0.015, 'mode': 'FM'},
    '20m': {'start': 14.0, 'end': 14.35, 'step': 0.001, 'mode': 'USB'},
    '10m': {'start': 28.3, 'end': 29.7, 'step': 0.005, 'mode': 'USB'},
}

# --- Radio Control Class ---
# (moved to ft991a_cat.py)

# --- GUI & Features Class ---
class RadioGUI:
    def __init__(self, root, radio):
        self.root = root
        self.radio = radio
        self.scanning = False
        self.active_band = None

        self._ui_queue = queue.Queue()
        self._shutdown = threading.Event()
        self._poll_thread = None

        self.root.title("FT-991A Command Center")
        self.root.geometry("500x750")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.setup_ui()
        self.refresh_connection_ui()

        self.process_ui_queue()
        self.start_polling()

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
        self.ptt_btn = tk.Button(
            self.root,
            text="PUSH TO TALK",
            bg="darkred",
            fg="white",
            font=("Arial", 12, "bold"),
        )
        self.ptt_btn.pack(pady=10, ipadx=30)
        self.ptt_btn.bind("<ButtonPress-1>", self._on_ptt_press)
        self.ptt_btn.bind("<ButtonRelease-1>", self._on_ptt_release)

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

    def _ptt_allowed(self) -> bool:
        """True when the GUI should allow PTT interaction."""
        try:
            if str(self.ptt_btn.cget("state")) == str(tk.DISABLED):
                return False
        except Exception:
            pass
        return self.radio.is_connected() and (not self.scanning)

    def _on_ptt_press(self, event):
        if not self._ptt_allowed():
            return "break"
        self.radio.ptt_on()
        return "break"

    def _on_ptt_release(self, event):
        if not self._ptt_allowed():
            return "break"
        self.radio.ptt_off()
        return "break"

    def on_close(self):
        """Stop worker threads and close serial cleanly before exiting."""
        self.scanning = False
        self._shutdown.set()
        try:
            if self.radio.is_connected():
                self.radio.disconnect()
        except Exception:
            pass
        self.root.destroy()

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

    def process_ui_queue(self):
        """Drain UI events produced by worker threads (Tkinter-safe)."""
        try:
            while True:
                item = self._ui_queue.get_nowait()
                kind = item[0]

                if kind == "freq":
                    _, f = item
                    self.freq_disp.config(text=f"{f:09.4f}")

                elif kind == "s_meter":
                    _, s = item
                    self.meter_var.set(s)

                elif kind == "rf_power":
                    _, p = item
                    # Don't fight the user while editing the spinbox
                    try:
                        if str(self.root.focus_get()) != str(self.rf_power_spin):
                            self.rf_power_var.set(p)
                    except Exception:
                        self.rf_power_var.set(p)

                elif kind == "mode":
                    _, m = item
                    # Don't fight the user while interacting with the combobox
                    try:
                        if str(self.root.focus_get()) != str(self.mode_combo):
                            if m:
                                self.mode_var.set(m)
                    except Exception:
                        if m:
                            self.mode_var.set(m)

                elif kind == "status":
                    _, text = item
                    self.conn_status.config(text=text)

                elif kind == "ptt_state":
                    _, enabled = item
                    self.ptt_btn.config(state=(tk.NORMAL if enabled else tk.DISABLED))

                elif kind == "log":
                    _, freq, strength, notes, timestamp = item
                    self.log_box.insert(
                        tk.END,
                        f"[{timestamp[-8:]}] {freq:.4f} S:{strength} | {notes}\n"
                    )
                    self.log_box.see(tk.END)

        except queue.Empty:
            pass

        self.root.after(50, self.process_ui_queue)

    def start_polling(self):
        """Start radio polling worker thread once (idempotent)."""
        if self._poll_thread and self._poll_thread.is_alive():
            return
        self._poll_thread = threading.Thread(target=self.radio_poll_thread, daemon=True)
        self._poll_thread.start()

    def _append_log_line(self, freq, strength, notes, timestamp):
        """UI-thread-only: update the log textbox."""
        self.log_box.insert(tk.END, f"[{timestamp[-8:]}] {freq:.4f} S:{strength} | {notes}\n")
        self.log_box.see(tk.END)

    def radio_poll_thread(self):
        """
        Background polling loop:
        - Reads rig state periodically
        - Pushes UI updates via queue
        - Avoids polling while scanning to reduce CAT traffic contention
        """
        last_mode = None
        last_pwr = None
        last_status_ts = 0.0
        last_s_ts = 0.0
        last_f_ts = 0.0
        last_slow_ts = 0.0

        while not self._shutdown.is_set():
            if not self.radio.is_connected():
                now = time.monotonic()
                if now - last_status_ts > 1.0:
                    self._ui_queue.put(("status", f"Status: DISCONNECTED ({self.radio.port} @ {self.radio.baud})"))
                    self._ui_queue.put(("freq", 0.0))
                    self._ui_queue.put(("s_meter", 0))
                    self._ui_queue.put(("rf_power", 0))
                    self._ui_queue.put(("mode", ""))
                    last_status_ts = now
                time.sleep(0.2)
                continue

            # Connected
            if self.scanning:
                # Let scanning own the CAT link; UI remains responsive anyway.
                time.sleep(0.1)
                continue

            now = time.monotonic()

            # Frequency: moderately fast
            if now - last_f_ts > 0.25:
                f = self.radio.get_frequency()
                self._ui_queue.put(("freq", f))
                last_f_ts = now

            # S-meter: fast-ish
            if now - last_s_ts > 0.20:
                s = self.radio.get_s_meter()
                self._ui_queue.put(("s_meter", s))
                last_s_ts = now

            # Mode + Power: slower (reduces CAT traffic)
            if now - last_slow_ts > 1.5:
                m = self.radio.get_mode()
                p = self.radio.get_rf_power()

                if m != last_mode and m:
                    self._ui_queue.put(("mode", m))
                    last_mode = m

                if p != last_pwr:
                    self._ui_queue.put(("rf_power", p))
                    last_pwr = p

                last_slow_ts = now

            time.sleep(0.02)

    def log_to_file(self, freq, strength, notes):
        """Thread-safe logging: file write + enqueue UI update."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open('radio_log.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, f"{freq:.4f}", strength, notes])

        self._ui_queue.put(("log", freq, strength, notes, timestamp))

    def manual_log(self):
        self.log_to_file(self.radio.get_frequency(), self.radio.get_s_meter(), self.note_entry.get())

    def infer_band_from_freq(self, mhz: float):
        """Return band name if mhz is inside a defined band plan, else None."""
        for name, plan in BANDS.items():
            if plan["start"] <= mhz <= plan["end"]:
                return name
        return None

    def goto_band(self, band_name: str):
        """Set radio to band start + mode, and make it the active scan band."""
        if band_name not in BANDS:
            return

        self.active_band = band_name
        plan = BANDS[band_name]

        if not self.radio.is_connected():
            self.conn_status.config(text=f"Status: DISCONNECTED (selected {band_name})")
            return

        # Do CAT sequencing off the UI thread, with a small settle + verify/retry.
        def worker():
            try:
                target = float(plan["start"])
                mode = plan["mode"]

                # 1) Mode first (some rigs behave better this way)
                self.radio.set_mode(mode)
                time.sleep(0.05)

                # 2) Set frequency
                self.radio.set_frequency(target)
                time.sleep(0.05)

                # 3) Verify; if we're off by > ~50 Hz, retry once
                actual = self.radio.get_frequency()
                if abs(actual - target) > 0.00005:  # 0.00005 MHz == 50 Hz
                    self.radio.set_frequency(target)
                    time.sleep(0.05)

                self._ui_queue.put(("mode", mode))
                self._ui_queue.put(
                    ("status", f"Status: CONNECTED ({self.radio.port} @ {self.radio.baud}) [{band_name}]"))
            except Exception as e:
                self._ui_queue.put(("status", f"Status: ERROR (band set failed: {e})"))

        threading.Thread(target=worker, daemon=True).start()

        # Update UI intent immediately
        self.mode_var.set(plan["mode"])
        self.mode_status.config(text=f"SET {plan['mode']} ({band_name})")

    def toggle_scan(self):
        if not self.scanning:
            if not self.radio.is_connected():
                self._ui_queue.put(("status", "Status: DISCONNECTED (click CONNECT to scan)"))
                self.scan_btn.config(text="START SCAN", bg="lightgray")
                self.scanning = False
                return

            if not self.active_band:
                current_f = self.radio.get_frequency()
                self.active_band = self.infer_band_from_freq(current_f)

            if not self.active_band:
                self._ui_queue.put(("status", "Status: ERROR (Select a band before scanning)"))
                self.scan_btn.config(text="START SCAN", bg="lightgray")
                self.scanning = False
                return

            # Parse squelch threshold on the UI thread (Tkinter-safe)
            try:
                thresh = int(self.thresh_entry.get())
            except (TypeError, ValueError):
                thresh = 40
                self._ui_queue.put(("status", "Status: SCAN (invalid squelch; default 40)"))

            self._ui_queue.put(("ptt_state", False))

            self.scanning = True
            self.scan_btn.config(text="STOP SCAN", bg="orange")

            band = self.active_band  # snapshot to avoid changes mid-scan
            threading.Thread(target=self.scan_thread, args=(band, thresh), daemon=True).start()
        else:
            self.scanning = False
            self.scan_btn.config(text="START SCAN", bg="lightgray")
            self._ui_queue.put(("ptt_state", True))

    def scan_thread(self, band: str, thresh: int):
        plan = BANDS.get(band)
        if not plan:
            self.scanning = False
            self._ui_queue.put(("ptt_state", True))
            self._ui_queue.put(("status", "Status: ERROR (No band selected / unknown band plan)"))
            return

        start = float(plan["start"])
        end = float(plan["end"])
        step = float(plan["step"])

        if step <= 0:
            self.scanning = False
            self._ui_queue.put(("ptt_state", True))
            self._ui_queue.put(("status", f"Status: ERROR (Invalid step for {band})"))
            return

        curr_f = self.radio.get_frequency()
        if curr_f < start or curr_f > end:
            curr_f = start

        while self.scanning and not self._shutdown.is_set():
            if not self.radio.is_connected():
                self._ui_queue.put(("status", "Status: DISCONNECTED (scan stopped)"))
                break

            if curr_f < start or curr_f > end:
                curr_f = start

            self.radio.set_frequency(curr_f)
            time.sleep(0.15)

            s = self.radio.get_s_meter()
            if s >= thresh:
                self.log_to_file(curr_f, s, f"AUTO-FOUND ({band})")

                # Dwell, but remain responsive to STOP SCAN
                dwell_until = time.monotonic() + 3.0
                while self.scanning and not self._shutdown.is_set() and time.monotonic() < dwell_until:
                    time.sleep(0.05)

            curr_f += step

        self.scanning = False
        self._ui_queue.put(("ptt_state", True))

    def toggle_connection(self):
        if self.radio.is_connected():
            # Stop scanning before disconnecting
            self.scanning = False
            self.scan_btn.config(text="START SCAN", bg="lightgray")
            self.ptt_btn.config(state=tk.NORMAL)

            self.radio.disconnect()
            self.freq_disp.config(text="DISCONNECTED")
            self.meter_var.set(0)
            self.rf_power_var.set(0)
            self.rf_power_status.config(text="")
            self.refresh_connection_ui()
            return

        ok, err = self.radio.connect()
        if not ok:
            self.conn_status.config(text=f"Status: ERROR ({err})")
            self.freq_disp.config(text="DISCONNECTED")
            self.meter_var.set(0)
            self.rf_power_var.set(0)
            self.rf_power_status.config(text="")
            self.refresh_connection_ui()
            return

        self.refresh_connection_ui()

        # Ensure polling restarts after a successful connect
        self.start_polling()

        # Infer active band on connect if possible
        f = self.radio.get_frequency()
        inferred = self.infer_band_from_freq(f)
        if inferred:
            self.active_band = inferred
            self.conn_status.config(text=f"Status: CONNECTED ({self.radio.port} @ {self.radio.baud}) [{inferred}]")

        p = self.radio.get_rf_power()
        self.rf_power_var.set(p)
        self.rf_power_status.config(text=f"READ {p:03d}")
        m = self.radio.get_mode()
        if m:
            self.mode_var.set(m)
            self.mode_status.config(text=f"READ {m}")

if __name__ == "__main__":
    # Change '/dev/ttyUSB0' to 'COM3' or similar if on Windows
    radio = Yaesu991AControl(port='COM7', baud=38400)
    root = tk.Tk()
    gui = RadioGUI(root, radio)
    root.mainloop()