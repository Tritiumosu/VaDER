#! /usr/bin/python3
import time
import csv
import configparser
import os
import threading
import tkinter as tk
from tkinter import ttk, messagebox
from datetime import datetime
import queue

try:
    import serial.tools.list_ports as _list_ports
    def _enum_serial_ports() -> list[str]:
        """Return sorted list of available serial port names."""
        return sorted(p.device for p in _list_ports.comports())
except Exception:
    def _enum_serial_ports() -> list[str]:  # type: ignore[misc]
        return []

from ft991a_cat import Yaesu991AControl
from digi_input import SoundCardAudioSource
from ft8_decode import FT8ConsoleDecoder, format_ft8_message

# --- Constants & Band Plans ---
BANDS = {
    '2m': {'start': 144.0, 'end': 148.0, 'step': 0.015, 'mode': 'FM'},
    '20m': {'start': 14.0, 'end': 14.35, 'step': 0.001, 'mode': 'USB'},
    '10m': {'start': 28.3, 'end': 29.7, 'step': 0.005, 'mode': 'USB'},
}

# --- Persistent Configuration ---
# Settings are stored in vader.cfg next to main.py.
_CFG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vader.cfg")

class AppConfig:
    """
    Thin wrapper around configparser for reading and writing vader.cfg.

    All values have safe defaults so the app runs even if the file is missing
    or was created by an older version that lacked a particular key.
    """

    _DEFAULTS = {
        "serial": {
            "port":     "",       # empty → user must choose via Settings on first run
            "baud":     "38400",
            "stopbits": "1",
        },
        "audio": {
            "device_index": "-1",  # -1 = not yet chosen
            "device_label": "",
        },
    }

    def __init__(self, path: str = _CFG_PATH) -> None:
        self._path = path
        self._cfg = configparser.ConfigParser()
        # Seed with defaults so missing sections/keys always resolve
        for section, pairs in self._DEFAULTS.items():
            self._cfg[section] = dict(pairs)
        # Overlay with whatever is on disk (silently ignored if absent)
        self._cfg.read(self._path, encoding="utf-8")

    # ── Serial helpers ────────────────────────────────────────────────────

    @property
    def port(self) -> str:
        return self._cfg.get("serial", "port", fallback="").strip()

    @property
    def baud(self) -> int:
        try:
            return int(self._cfg.get("serial", "baud", fallback="38400"))
        except ValueError:
            return 38400

    @property
    def stopbits(self) -> float:
        try:
            return float(self._cfg.get("serial", "stopbits", fallback="1"))
        except ValueError:
            return 1.0

    def save_serial(self, port: str, baud: int, stopbits: float) -> None:
        if not self._cfg.has_section("serial"):
            self._cfg.add_section("serial")
        self._cfg.set("serial", "port",     port.strip())
        self._cfg.set("serial", "baud",     str(int(baud)))
        self._cfg.set("serial", "stopbits", str(float(stopbits)))
        self._write()

    # ── Audio helpers ─────────────────────────────────────────────────────

    @property
    def audio_device_index(self) -> int:
        try:
            return int(self._cfg.get("audio", "device_index", fallback="-1"))
        except ValueError:
            return -1

    @property
    def audio_device_label(self) -> str:
        return self._cfg.get("audio", "device_label", fallback="").strip()

    def save_audio(self, device_index: int, device_label: str) -> None:
        if not self._cfg.has_section("audio"):
            self._cfg.add_section("audio")
        self._cfg.set("audio", "device_index", str(int(device_index)))
        self._cfg.set("audio", "device_label", device_label.strip())
        self._write()

    # ── Internal ──────────────────────────────────────────────────────────

    def _write(self) -> None:
        try:
            with open(self._path, "w", encoding="utf-8") as fh:
                self._cfg.write(fh)
        except OSError as exc:
            # Non-fatal — settings simply won't persist this run
            print(f"[Config] Could not write {self._path}: {exc}", flush=True)


# --- Radio Control Class ---
# (moved to ft991a_cat.py)

# --- Settings Dialog ---
class SettingsDialog:
    """
    Modal dialog for configuring the radio serial connection.

    Opens as a child of the given parent window and blocks until
    the user clicks Apply or Cancel.  On Apply the supplied
    `on_apply` callback is called with (port, baud, stopbits).
    """

    BAUD_RATES = [1200, 2400, 4800, 9600, 19200, 38400, 57600, 115200]
    STOP_BITS  = [1, 1.5, 2]

    def __init__(
        self,
        parent: tk.Misc,
        *,
        current_port: str = "",
        current_baud: int = 38400,
        current_stopbits: float = 1,
        on_apply,          # callable(port: str, baud: int, stopbits: float)
    ) -> None:
        self._on_apply = on_apply

        self._win = tk.Toplevel(parent)
        self._win.title("Radio Connection Settings")
        self._win.resizable(False, False)
        self._win.grab_set()          # make modal
        self._win.focus_set()

        # ── Port row ──────────────────────────────────────────────────────
        port_frame = tk.LabelFrame(self._win, text="Serial Port")
        port_frame.pack(padx=16, pady=(12, 6), fill=tk.X)

        tk.Label(port_frame, text="Port:").pack(side=tk.LEFT, padx=6)

        self._port_var = tk.StringVar(value=current_port)
        self._port_combo = ttk.Combobox(
            port_frame,
            textvariable=self._port_var,
            values=(),
            width=14,
        )
        self._port_combo.pack(side=tk.LEFT, padx=4)

        tk.Button(
            port_frame,
            text="↻ Refresh",
            command=self._refresh_ports,
        ).pack(side=tk.LEFT, padx=6)

        self._port_status = tk.Label(port_frame, text="", anchor="w", fg="gray")
        self._port_status.pack(side=tk.LEFT, padx=4)

        # ── Baud rate row ─────────────────────────────────────────────────
        baud_frame = tk.LabelFrame(self._win, text="Baud Rate")
        baud_frame.pack(padx=16, pady=6, fill=tk.X)

        tk.Label(baud_frame, text="Baud:").pack(side=tk.LEFT, padx=6)

        self._baud_var = tk.StringVar(value=str(current_baud))
        self._baud_combo = ttk.Combobox(
            baud_frame,
            textvariable=self._baud_var,
            values=[str(b) for b in self.BAUD_RATES],
            state="readonly",
            width=10,
        )
        self._baud_combo.pack(side=tk.LEFT, padx=4)

        # ── Stop bits row ─────────────────────────────────────────────────
        stop_frame = tk.LabelFrame(self._win, text="Stop Bits")
        stop_frame.pack(padx=16, pady=6, fill=tk.X)

        tk.Label(stop_frame, text="Stop bits:").pack(side=tk.LEFT, padx=6)

        self._stop_var = tk.StringVar(value=str(current_stopbits))
        self._stop_combo = ttk.Combobox(
            stop_frame,
            textvariable=self._stop_var,
            values=[str(s) for s in self.STOP_BITS],
            state="readonly",
            width=6,
        )
        self._stop_combo.pack(side=tk.LEFT, padx=4)

        # ── Buttons ───────────────────────────────────────────────────────
        btn_frame = tk.Frame(self._win)
        btn_frame.pack(padx=16, pady=(10, 12), fill=tk.X)

        tk.Button(
            btn_frame,
            text="Apply",
            bg="lightgreen",
            width=10,
            command=self._apply,
        ).pack(side=tk.RIGHT, padx=4)

        tk.Button(
            btn_frame,
            text="Cancel",
            width=10,
            command=self._win.destroy,
        ).pack(side=tk.RIGHT, padx=4)

        # Bind Enter to Apply, Escape to Cancel
        self._win.bind("<Return>",  lambda _e: self._apply())
        self._win.bind("<KP_Enter>", lambda _e: self._apply())
        self._win.bind("<Escape>",  lambda _e: self._win.destroy())

        # Populate ports immediately
        self._refresh_ports(initial=current_port)

        # Centre over the parent
        self._win.update_idletasks()
        pw = parent.winfo_rootx() + parent.winfo_width()  // 2
        ph = parent.winfo_rooty() + parent.winfo_height() // 2
        w  = self._win.winfo_width()
        h  = self._win.winfo_height()
        self._win.geometry(f"+{pw - w // 2}+{ph - h // 2}")

    # ── Internal helpers ──────────────────────────────────────────────────

    def _refresh_ports(self, initial: str = "") -> None:
        ports = _enum_serial_ports()
        self._port_combo.config(values=ports)

        if ports:
            # Keep current value if still present; otherwise pick first in list
            current = self._port_var.get()
            if initial and initial in ports:
                self._port_var.set(initial)
            elif current in ports:
                pass          # already set correctly
            else:
                self._port_var.set(ports[0])
            self._port_status.config(text=f"{len(ports)} port(s) found", fg="gray")
        else:
            self._port_var.set("")
            self._port_status.config(text="No ports found", fg="red")

    def _apply(self) -> None:
        port = self._port_var.get().strip()
        if not port:
            messagebox.showerror(
                "Settings",
                "Please select a serial port.",
                parent=self._win,
            )
            return

        try:
            baud = int(self._baud_var.get())
        except ValueError:
            messagebox.showerror(
                "Settings",
                "Invalid baud rate.",
                parent=self._win,
            )
            return

        try:
            stopbits = float(self._stop_var.get())
        except ValueError:
            messagebox.showerror(
                "Settings",
                "Invalid stop-bits value.",
                parent=self._win,
            )
            return

        self._win.destroy()
        self._on_apply(port, baud, stopbits)


# --- GUI & Features Class ---
class RadioGUI:
    def __init__(self, root, radio, config: "AppConfig | None" = None):
        self.root = root
        self.radio = radio
        self._config = config or AppConfig()
        self.scanning = False
        self.active_band = None

        self._ui_queue = queue.Queue()
        self._shutdown = threading.Event()
        self._poll_thread = None

        # Audio device selection state (index to pass to sounddevice / SoundCardAudioSource)
        # Restore from config; -1 means not yet chosen.
        _saved_idx = self._config.audio_device_index
        self.audio_device_index = _saved_idx if _saved_idx >= 0 else None
        self._audio_devices = []  # list[tuple[int, str]] -> (index, display_name)

        # Live audio stream state
        self._audio_src = None
        self._audio_thread = None
        self._audio_stop = threading.Event()

        # FT8 decode (console) scaffold
        self._ft8 = FT8ConsoleDecoder(on_decode=self._on_ft8_decode)
        self._ft8.start()

        self.root.title("FT-991A Command Center")
        self.root.geometry("500x750")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.setup_ui()
        self.refresh_connection_ui()

        # Restore saved audio device label into the combo (best-effort)
        _saved_label = self._config.audio_device_label
        if _saved_label:
            self.audio_device_var.set(_saved_label)
            self.audio_status.config(text=f"Audio: restored — click APPLY to activate")

        self.process_ui_queue()
        self.start_polling()

    def _query_audio_input_devices(self):
        """
        Return list of (device_index, display_name) for INPUT-capable devices.

        Uses sounddevice if available; otherwise returns ([], "reason").
        """
        try:
            import sounddevice as sd  # type: ignore
        except Exception as e:
            return [], f"sounddevice not available ({e})"

        try:
            devices = sd.query_devices()
            hostapis = sd.query_hostapis()
        except Exception as e:
            return [], f"Failed to query devices ({e})"

        def _hostapi_name(dev: dict) -> str:
            try:
                hidx = int(dev.get("hostapi", -1))
                if 0 <= hidx < len(hostapis):
                    return str(hostapis[hidx].get("name", f"hostapi {hidx}"))
            except Exception:
                pass
            return "unknown hostapi"

        def _norm(s: str) -> str:
            return " ".join((s or "").strip().lower().split())

        out: list[tuple[int, str]] = []
        seen_keys: set[tuple[str, int]] = set()

        for idx, d in enumerate(devices):
            hostapi_label = _hostapi_name(d)
            if hostapi_label != "Windows WASAPI":
                continue

            # 1) Must claim to support input channels
            try:
                max_in = int(d.get("max_input_channels", 0))
            except Exception:
                max_in = 0
            if max_in <= 0:
                continue

            # 2) Must be actually usable ("active") by attempting to validate input settings
            name = str(d.get("name", f"Device {idx}")).strip()
            default_fs = d.get("default_samplerate", None)
            try:
                # Prefer the device's default samplerate; fall back to 48k (your app default)
                fs = int(default_fs) if default_fs else 48_000
                sd.check_input_settings(device=idx, channels=1, samplerate=fs)
            except Exception:
                continue  # not usable right now

            # 3) Deduplicate: collapse duplicates across host APIs / repeated endpoints
            # (hostapi is constant now, since we only keep WASAPI)
            key = (_norm(name), max_in)
            if key in seen_keys:
                continue
            seen_keys.add(key)

            out.append((idx, f"{idx}: {name} ({hostapi_label})"))

        if not out:
            return [], "No active WASAPI input devices found"
        return out, ""

    def refresh_audio_devices(self):
        """Refresh the dropdown with currently available audio input devices."""
        devices, err = self._query_audio_input_devices()
        self._audio_devices = devices

        if err:
            self.audio_status.config(text=f"Audio: {err}")
            self.audio_device_combo.config(values=())
            self.audio_device_var.set("")
            self.audio_apply_btn.config(state=tk.DISABLED)
            return

        values = [label for (_, label) in devices]
        self.audio_device_combo.config(values=values)
        self.audio_apply_btn.config(state=tk.NORMAL)

        # Keep current selection if possible, else select the first device.
        if self.audio_device_index is not None:
            for idx, label in devices:
                if idx == self.audio_device_index:
                    self.audio_device_var.set(label)
                    self.audio_status.config(text=f"Audio: selected {label}")
                    return

        self.audio_device_var.set(values[0])
        self.audio_status.config(text="Audio: select a device and click APPLY")

    def _parse_selected_audio_device_index(self):
        label = (self.audio_device_var.get() or "").strip()
        if not label:
            return None
        try:
            idx_str = label.split(":", 1)[0].strip()
            return int(idx_str)
        except Exception:
            return None

    def _audio_worker(self):
        """
        Consume audio chunks so the input stream stays active.
        Updates GUI with a simple RMS indicator (proves we're getting live samples).
        """
        try:
            src = self._audio_src
            if src is None:
                return

            last_ui = 0.0
            for chunk in src.chunks(timeout_s=0.5):
                if self._shutdown.is_set() or self._audio_stop.is_set():
                    break

                # Feed decoder scaffold (prints to console)
                try:
                    self._ft8.feed(fs=chunk.fs, samples=chunk.samples, t0_monotonic=chunk.t0)
                except Exception:
                    pass

                # Basic RMS level; fast and good enough as a "signal present" indicator
                x = chunk.samples
                if x.size:
                    rms = float((x * x).mean() ** 0.5)
                else:
                    rms = 0.0

                now = time.monotonic()
                if now - last_ui > 0.2:
                    self._ui_queue.put(("audio_rms", rms))
                    last_ui = now

        except Exception as e:
            self._ui_queue.put(("audio_status", f"Audio: ERROR ({e})"))

    def _start_audio_stream(self, device_index: int) -> None:
        self._stop_audio_stream()

        self._audio_stop.clear()
        self._audio_src = SoundCardAudioSource(device=device_index)
        self._audio_src.start()

        self._audio_thread = threading.Thread(target=self._audio_worker, daemon=True)
        self._audio_thread.start()

    def _stop_audio_stream(self) -> None:
        self._audio_stop.set()

        src = self._audio_src
        self._audio_src = None
        if src is not None:
            try:
                src.stop()
            except Exception:
                pass

        self._audio_thread = None

    def _restart_audio_stream(self, device_index: int) -> None:
        self._start_audio_stream(device_index)
        label = (self.audio_device_var.get() or "").strip()
        self.audio_status.config(text=f"Audio: LIVE on {label}")

    def _on_ft8_decode(self, utc: str, freq_hz: float, snr_db: float, message: str) -> None:
        """
        Called from the FT8 decoder thread on every successful LDPC+CRC decode.
        Must be thread-safe — puts a UI update onto the queue; never touches
        Tkinter widgets directly.
        """
        line = format_ft8_message(utc, snr_db, freq_hz, message)
        # Print to terminal (includes all debug info)
        print(line, flush=True)
        # Send formatted line to GUI display queue
        self._ui_queue.put(("ft8_decoded", line + "\n"))

    def _clear_ft8_log(self) -> None:
        """Clear the FT8 decoded messages panel (called from GUI thread)."""
        self.ft8_log.config(state=tk.NORMAL)
        self.ft8_log.delete("1.0", tk.END)
        self.ft8_log.config(state=tk.DISABLED)

    def apply_audio_device(self):
        """
        Live switch:
        - store selected device index
        - restart the running audio stream immediately on that device
        """
        idx = self._parse_selected_audio_device_index()
        if idx is None:
            self.audio_status.config(text="Audio: no device selected (or parse failed)")
            return

        self.audio_device_index = idx
        label = self.audio_device_var.get().strip()

        try:
            self._restart_audio_stream(idx)
            # Persist the chosen device so it's restored on next launch
            self._config.save_audio(idx, label)
        except Exception as e:
            self.audio_status.config(text=f"Audio: ERROR (failed to start: {e})")

    def setup_ui(self):
        # Frequency Display
        self.freq_disp = tk.Label(self.root, text="DISCONNECTED", font=("Consolas", 36), fg="lime", bg="black")
        self.freq_disp.pack(pady=15, fill=tk.X)

        # Connection controls
        conn_frame = tk.LabelFrame(self.root, text="Connection")
        conn_frame.pack(padx=20, pady=10, fill=tk.X)

        self.conn_status = tk.Label(conn_frame, text="Status: DISCONNECTED", anchor="w")
        self.conn_status.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.X, expand=True)

        self.settings_btn = tk.Button(
            conn_frame, text="⚙ SETTINGS", bg="lightblue", command=self.open_settings
        )
        self.settings_btn.pack(side=tk.RIGHT, padx=5, pady=5)

        self.conn_btn = tk.Button(conn_frame, text="CONNECT", bg="lightgreen", command=self.toggle_connection)
        self.conn_btn.pack(side=tk.RIGHT, padx=5, pady=5)

        # Audio input device selection
        audio_frame = tk.LabelFrame(self.root, text="Audio Input")
        audio_frame.pack(padx=20, pady=10, fill=tk.X)

        tk.Label(audio_frame, text="Device:").pack(side=tk.LEFT, padx=5)

        self.audio_device_var = tk.StringVar(value="")
        self.audio_device_combo = ttk.Combobox(
            audio_frame,
            textvariable=self.audio_device_var,
            values=(),
            state="readonly",
            width=30
        )
        self.audio_device_combo.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        self.audio_refresh_btn = tk.Button(audio_frame, text="REFRESH", command=self.refresh_audio_devices)
        self.audio_refresh_btn.pack(side=tk.LEFT, padx=5)

        self.audio_apply_btn = tk.Button(audio_frame, text="APPLY", command=self.apply_audio_device, state=tk.DISABLED)
        self.audio_apply_btn.pack(side=tk.LEFT, padx=5)

        self.audio_status = tk.Label(self.root, text="Audio: (not selected)", anchor="w")
        self.audio_status.pack(padx=20, pady=(0, 10), fill=tk.X)

        # Live-switch immediately when user picks a different item (no need to click APPLY twice)
        self.audio_device_combo.bind("<<ComboboxSelected>>", lambda e: self.apply_audio_device())

        # Populate devices on startup (non-fatal if sounddevice isn't installed)
        self.refresh_audio_devices()

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

        self.log_box = tk.Text(log_frame, height=6, font=("Consolas", 9))
        self.log_box.pack(padx=5, pady=5, fill=tk.BOTH, expand=True)

        # FT8 Decoded Messages panel
        ft8_frame = tk.LabelFrame(self.root, text="FT8 Decoded Messages")
        ft8_frame.pack(padx=20, pady=(0, 10), fill=tk.BOTH, expand=True)

        ft8_btn_row = tk.Frame(ft8_frame)
        ft8_btn_row.pack(fill=tk.X, padx=5, pady=(4, 0))
        tk.Label(ft8_btn_row, text="Live FT8 decodes appear here →", fg="gray",
                 font=("Consolas", 8)).pack(side=tk.LEFT)
        tk.Button(ft8_btn_row, text="Clear", font=("Arial", 8),
                  command=self._clear_ft8_log).pack(side=tk.RIGHT)

        self.ft8_log = tk.Text(
            ft8_frame,
            height=8,
            font=("Consolas", 9),
            bg="#0a0a1a",
            fg="#00ff88",
            insertbackground="white",
            state=tk.DISABLED,
        )
        ft8_scroll = ttk.Scrollbar(ft8_frame, orient=tk.VERTICAL, command=self.ft8_log.yview)
        self.ft8_log.configure(yscrollcommand=ft8_scroll.set)
        ft8_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.ft8_log.pack(padx=5, pady=(2, 5), fill=tk.BOTH, expand=True)

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

        # Stop decoder scaffold
        try:
            self._ft8.stop()
        except Exception:
            pass

        # Stop live audio cleanly
        try:
            self._stop_audio_stream()
        except Exception:
            pass

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

                elif kind == "audio_rms":
                    _, rms = item
                    # Keep it simple: show level as a hint that the stream is alive
                    self.audio_status.config(text=f"Audio: LIVE (RMS {rms:.4f})")

                elif kind == "audio_status":
                    _, text = item
                    self.audio_status.config(text=text)

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

                elif kind == "ft8_decoded":
                    _, line = item
                    self.ft8_log.config(state=tk.NORMAL)
                    self.ft8_log.insert(tk.END, line)
                    self.ft8_log.see(tk.END)
                    self.ft8_log.config(state=tk.DISABLED)

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

    def open_settings(self) -> None:
        """Open the radio connection settings modal dialog."""
        # Read current stopbits from the serial object if connected, else default to 1
        try:
            import serial
            sb_map = {
                serial.STOPBITS_ONE: 1,
                serial.STOPBITS_ONE_POINT_FIVE: 1.5,
                serial.STOPBITS_TWO: 2,
            }
            if self.radio.conn is not None and getattr(self.radio.conn, "is_open", False):
                current_stopbits = sb_map.get(self.radio.conn.stopbits, 1)
            else:
                current_stopbits = getattr(self.radio, "_stopbits", 1)
        except Exception:
            current_stopbits = getattr(self.radio, "_stopbits", 1)

        SettingsDialog(
            self.root,
            current_port=self.radio.port,
            current_baud=self.radio.baud,
            current_stopbits=current_stopbits,
            on_apply=self._apply_settings,
        )

    def _apply_settings(self, port: str, baud: int, stopbits: float) -> None:
        """
        Apply new serial settings from the SettingsDialog.

        If the radio is currently connected, disconnects first, updates the
        parameters, then reconnects automatically.
        """
        was_connected = self.radio.is_connected()

        if was_connected:
            self.scanning = False
            self.radio.disconnect()

        # Update parameters on the radio object
        self.radio.port = port
        self.radio.baud = baud
        self.radio._stopbits = stopbits

        # Patch the stopbits into the serial module constant so connect() picks it up
        try:
            import serial
            _sb_map = {1: serial.STOPBITS_ONE, 1.5: serial.STOPBITS_ONE_POINT_FIVE, 2: serial.STOPBITS_TWO}
            self.radio._stopbits_serial = _sb_map.get(stopbits, serial.STOPBITS_ONE)
        except Exception:
            self.radio._stopbits_serial = None

        # Persist to vader.cfg
        self._config.save_serial(port, baud, stopbits)

        self.refresh_connection_ui()

        if was_connected:
            ok, err = self.radio.connect()
            if ok:
                self.refresh_connection_ui()
                self.conn_status.config(
                    text=f"Status: CONNECTED ({self.radio.port} @ {self.radio.baud})"
                )
            else:
                self.conn_status.config(text=f"Status: ERROR reconnecting ({err})")
        else:
            self.conn_status.config(
                text=f"Status: DISCONNECTED ({self.radio.port} @ {self.radio.baud})"
            )

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
    # Load persisted settings from vader.cfg (created automatically on first save).
    _cfg = AppConfig()

    # Determine the startup port: prefer the saved value; fall back to the first
    # detected port so the app is functional out-of-the-box for new testers.
    _port = _cfg.port
    if not _port:
        _detected = _enum_serial_ports()
        _port = _detected[0] if _detected else "COM1"

    radio = Yaesu991AControl(port=_port, baud=_cfg.baud, stopbits=_cfg.stopbits)
    root = tk.Tk()
    gui = RadioGUI(root, radio, config=_cfg)
    root.mainloop()