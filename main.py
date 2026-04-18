#! /usr/bin/python3
import logging
import time
import csv
import configparser
import os
import threading
import tkinter as tk
from tkinter import ttk, messagebox
from datetime import datetime, timezone
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
from audio_passthrough import AudioPassthrough, AudioTxCapture
from ft8_tx import Ft8TxCoordinator, TxJob, TxState, validate_operator
from ft8_ntp import Ft8SlotTimer, default_slot_timer
from ft8_qso import Ft8QsoManager, OperatorConfig, QsoState, ReceivedMessage

_log = logging.getLogger(__name__)

# --- Constants & Band Plans ---
BANDS = {
    '160m': {'start': 1.800,  'end': 2.000,   'step': 0.001, 'mode': 'LSB'},
    '80m':  {'start': 3.500,  'end': 4.000,   'step': 0.001, 'mode': 'LSB'},
    '40m':  {'start': 7.000,  'end': 7.300,   'step': 0.001, 'mode': 'LSB'},
    '30m':  {'start': 10.100, 'end': 10.150,  'step': 0.001, 'mode': 'USB'},
    '20m':  {'start': 14.000, 'end': 14.350,  'step': 0.001, 'mode': 'USB'},
    '17m':  {'start': 18.068, 'end': 18.168,  'step': 0.001, 'mode': 'USB'},
    '15m':  {'start': 21.000, 'end': 21.450,  'step': 0.001, 'mode': 'USB'},
    '12m':  {'start': 24.890, 'end': 24.990,  'step': 0.001, 'mode': 'USB'},
    '10m':  {'start': 28.000, 'end': 29.700,  'step': 0.005, 'mode': 'USB'},
    '6m':   {'start': 50.000, 'end': 54.000,  'step': 0.025, 'mode': 'USB'},
    '2m':   {'start': 144.000,'end': 148.000, 'step': 0.025, 'mode': 'FM'},
    '70cm': {'start': 430.000,'end': 440.000, 'step': 0.025, 'mode': 'FM'},
}

# Standard FT8 calling frequencies per band (MHz)
FT8_FREQS = {
    '160m': 1.840,  '80m': 3.573,  '40m': 7.074,  '30m': 10.136,
    '20m':  14.074, '17m': 18.100, '15m': 21.074, '12m': 24.915,
    '10m':  28.074, '6m':  50.313, '2m':  144.174,
}

# Log file for FT8 decoded messages (append mode)
FT8_LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ft8_messages.log")

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
            "port":     "",       # empty -> user must choose via Settings on first run
            "baud":     "38400",
            "stopbits": "1",
        },
        "audio": {
            "device_index":        "-1",  # -1 = not yet chosen
            "device_label":        "",
            "output_device_index": "-1",
            "output_device_label": "",
        },
        "tx_audio": {
            "mic_device_index":        "-1",  # computer microphone for TX
            "mic_device_label":        "",
            "radio_out_device_index":  "-1",  # soundcard output → radio audio input
            "radio_out_device_label":  "",
            "ft8_base_tone_hz":        "1500",  # FT8 base tone frequency in Hz (50–3000)
        },
        "operator": {
            "callsign": "",   # operator callsign, e.g. W4ABC — used in FT8 messages
            "grid":     "",   # 4-char Maidenhead grid locator, e.g. EN52
        },
        "ntp": {
            # Comma-separated list of NTP servers tried in order on startup.
            # Defaults to NIST, Cloudflare, Google, and the NTP Pool.
            "servers":          "time.nist.gov,time.cloudflare.com,time.google.com,pool.ntp.org",
            # Automatically query NTP when the application starts.
            "sync_on_startup":  "true",
            # Per-server socket timeout in seconds.
            "timeout_s":        "3.0",
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

    # -- Serial helpers ---------------------------------------------------

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

    # -- Audio input helpers -----------------------------------------------

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

    # -- Audio output helpers ----------------------------------------------

    @property
    def audio_output_device_index(self) -> int:
        try:
            return int(self._cfg.get("audio", "output_device_index", fallback="-1"))
        except ValueError:
            return -1

    @property
    def audio_output_device_label(self) -> str:
        return self._cfg.get("audio", "output_device_label", fallback="").strip()

    def save_audio_output(self, device_index: int, device_label: str) -> None:
        if not self._cfg.has_section("audio"):
            self._cfg.add_section("audio")
        self._cfg.set("audio", "output_device_index", str(int(device_index)))
        self._cfg.set("audio", "output_device_label", device_label.strip())
        self._write()

    # -- TX Audio helpers --------------------------------------------------

    @property
    def tx_mic_device_index(self) -> int:
        try:
            return int(self._cfg.get("tx_audio", "mic_device_index", fallback="-1"))
        except ValueError:
            return -1

    @property
    def tx_mic_device_label(self) -> str:
        return self._cfg.get("tx_audio", "mic_device_label", fallback="").strip()

    @property
    def tx_radio_out_device_index(self) -> int:
        try:
            return int(self._cfg.get("tx_audio", "radio_out_device_index", fallback="-1"))
        except ValueError:
            return -1

    @property
    def tx_radio_out_device_label(self) -> str:
        return self._cfg.get("tx_audio", "radio_out_device_label", fallback="").strip()

    @property
    def ft8_base_tone_hz(self) -> float:
        """FT8 base tone frequency in Hz (50–3000). Defaults to 1500."""
        try:
            return float(self._cfg.get("tx_audio", "ft8_base_tone_hz", fallback="1500"))
        except ValueError:
            return 1500.0

    def save_ft8_base_tone_hz(self, hz: float) -> None:
        """Persist the FT8 base tone frequency to vader.cfg."""
        if not self._cfg.has_section("tx_audio"):
            self._cfg.add_section("tx_audio")
        self._cfg.set("tx_audio", "ft8_base_tone_hz", str(float(hz)))
        self._write()

    def save_tx_audio(
        self,
        mic_idx: int,
        mic_label: str,
        radio_out_idx: int,
        radio_out_label: str,
    ) -> None:
        if not self._cfg.has_section("tx_audio"):
            self._cfg.add_section("tx_audio")
        self._cfg.set("tx_audio", "mic_device_index",       str(int(mic_idx)))
        self._cfg.set("tx_audio", "mic_device_label",       mic_label.strip())
        self._cfg.set("tx_audio", "radio_out_device_index", str(int(radio_out_idx)))
        self._cfg.set("tx_audio", "radio_out_device_label", radio_out_label.strip())
        self._write()

    # -- Operator identity helpers ----------------------------------------

    @property
    def operator_callsign(self) -> str:
        """Operator callsign stored in vader.cfg (empty string if not set)."""
        return self._cfg.get("operator", "callsign", fallback="").strip().upper()

    @property
    def operator_grid(self) -> str:
        """4-char Maidenhead grid locator stored in vader.cfg (empty if not set)."""
        return self._cfg.get("operator", "grid", fallback="").strip().upper()

    def save_operator(self, callsign: str, grid: str) -> None:
        """Persist the operator callsign and grid locator to vader.cfg."""
        if not self._cfg.has_section("operator"):
            self._cfg.add_section("operator")
        self._cfg.set("operator", "callsign", callsign.strip().upper())
        self._cfg.set("operator", "grid",     grid.strip().upper())
        self._write()

    # -- NTP helpers -------------------------------------------------------

    @property
    def ntp_servers(self) -> list[str]:
        """
        Ordered list of NTP servers from vader.cfg.

        Returns the DEFAULT_NTP_SERVERS list from ft8_ntp if the config key
        is empty or missing.
        """
        raw = self._cfg.get("ntp", "servers", fallback="").strip()
        if not raw:
            from ft8_ntp import DEFAULT_NTP_SERVERS
            return list(DEFAULT_NTP_SERVERS)
        return [s.strip() for s in raw.split(",") if s.strip()]

    @property
    def ntp_sync_on_startup(self) -> bool:
        """True when VaDER should run an NTP sync on application startup."""
        raw = self._cfg.get("ntp", "sync_on_startup", fallback="true").strip().lower()
        return raw not in ("false", "0", "no", "off")

    @property
    def ntp_timeout_s(self) -> float:
        """Per-server NTP socket timeout in seconds."""
        try:
            return float(self._cfg.get("ntp", "timeout_s", fallback="3.0"))
        except ValueError:
            return 3.0

    def save_ntp(
        self,
        servers: list[str],
        sync_on_startup: bool = True,
        timeout_s: float = 3.0,
    ) -> None:
        """Persist NTP settings to vader.cfg."""
        if not self._cfg.has_section("ntp"):
            self._cfg.add_section("ntp")
        self._cfg.set("ntp", "servers",         ",".join(s.strip() for s in servers))
        self._cfg.set("ntp", "sync_on_startup", "true" if sync_on_startup else "false")
        self._cfg.set("ntp", "timeout_s",       str(float(timeout_s)))
        self._write()

    # -- Internal ----------------------------------------------------------

    def _write(self) -> None:
        try:
            with open(self._path, "w", encoding="utf-8") as fh:
                self._cfg.write(fh)
        except OSError as exc:
            # Non-fatal -- settings simply won't persist this run
            print(f"[Config] Could not write {self._path}: {exc}", flush=True)


# --- Audio Device Enumeration ---

def _enum_audio_devices() -> tuple[list[tuple[int, str]], list[tuple[int, str]], str]:
    """
    Enumerate preferred Windows audio input/output devices.

    Returns (input_devices, output_devices, error_message).
    Each device list contains (index, display_label) tuples.
    error_message is an empty string on success.
    """
    try:
        import sounddevice as sd  # type: ignore
    except Exception as e:
        return [], [], f"sounddevice not available ({e})"

    try:
        devices  = sd.query_devices()
        hostapis = sd.query_hostapis()
    except Exception as e:
        return [], [], f"Failed to query devices ({e})"

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

    preferred_apis = ("Windows WASAPI", "MME")

    inputs:   list[tuple[int, str]] = []
    outputs:  list[tuple[int, str]] = []
    seen_in:  set[tuple[str, int, str]]  = set()
    seen_out: set[tuple[str, int, str]]  = set()

    for api_name in preferred_apis:
        for idx, d in enumerate(devices):
            if _hostapi_name(d) != api_name:
                continue

            name       = str(d.get("name", f"Device {idx}")).strip()
            norm_name  = _norm(name)
            default_fs = d.get("default_samplerate", None)
            fs         = int(default_fs) if default_fs else 48_000
            api_short  = "WASAPI" if api_name == "Windows WASAPI" else "MME"
            label      = f"{idx}: {name} ({api_short})"

            max_in = int(d.get("max_input_channels", 0) or 0)
            if max_in > 0:
                key = (norm_name, max_in, api_short)
                if key not in seen_in:
                    try:
                        sd.check_input_settings(device=idx, channels=1, samplerate=fs)
                        seen_in.add(key)
                        inputs.append((idx, label))
                    except Exception:
                        pass

            max_out = int(d.get("max_output_channels", 0) or 0)
            if max_out > 0:
                key = (norm_name, max_out, api_short)
                if key not in seen_out:
                    try:
                        sd.check_output_settings(device=idx, channels=1, samplerate=fs)
                        seen_out.add(key)
                        outputs.append((idx, label))
                    except Exception:
                        pass

    err = ""
    if not inputs and not outputs:
        err = "No usable WASAPI/MME audio devices found"
    return inputs, outputs, err


# --- Radio Control Class ---
# (moved to ft991a_cat.py)

# --- Settings Dialog ---
class SettingsDialog:
    """
    Modal dialog for configuring the radio serial connection and audio devices.

    Opens as a child of the given parent window and blocks until
    the user clicks Apply or Cancel.  On Apply the supplied
    `on_apply` callback is called with:
      (port, baud, stopbits,
       audio_in_idx, audio_in_label, audio_out_idx, audio_out_label,
       tx_mic_idx, tx_mic_label, tx_radio_out_idx, tx_radio_out_label)
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
        current_audio_in_label:      str = "",
        current_audio_out_label:     str = "",
        current_tx_mic_label:        str = "",
        current_tx_radio_out_label:  str = "",
        on_apply,   # callable(port, baud, stopbits, in_idx, in_label, out_idx, out_label,
                    #          tx_mic_idx, tx_mic_label, tx_radio_out_idx, tx_radio_out_label)
    ) -> None:
        self._on_apply = on_apply

        self._win = tk.Toplevel(parent)
        self._win.title("Settings")
        self._win.resizable(False, False)
        self._win.grab_set()          # make modal
        self._win.focus_set()

        # -- Serial Port row -----------------------------------------------
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
            text="Refresh",
            command=self._refresh_ports,
        ).pack(side=tk.LEFT, padx=6)

        self._port_status = tk.Label(port_frame, text="", anchor="w", fg="gray")
        self._port_status.pack(side=tk.LEFT, padx=4)

        # -- Baud rate row -------------------------------------------------
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

        # -- Stop bits row -------------------------------------------------
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

        # -- Audio Input row -----------------------------------------------
        ain_frame = tk.LabelFrame(self._win, text="Audio Input Device (RX: Radio → Computer)")
        ain_frame.pack(padx=16, pady=6, fill=tk.X)

        tk.Label(ain_frame, text="Device:").pack(side=tk.LEFT, padx=6)

        self._ain_var = tk.StringVar(value=current_audio_in_label)
        self._ain_combo = ttk.Combobox(
            ain_frame,
            textvariable=self._ain_var,
            values=(),
            state="readonly",
            width=32,
        )
        self._ain_combo.pack(side=tk.LEFT, padx=4, fill=tk.X, expand=True)

        tk.Button(
            ain_frame,
            text="Refresh",
            command=self._refresh_audio,
        ).pack(side=tk.LEFT, padx=4)

        # -- Audio Output row ----------------------------------------------
        aout_frame = tk.LabelFrame(self._win, text="Audio Output Device (RX Monitor: Computer → Speakers)")
        aout_frame.pack(padx=16, pady=6, fill=tk.X)

        tk.Label(aout_frame, text="Device:").pack(side=tk.LEFT, padx=6)

        self._aout_var = tk.StringVar(value=current_audio_out_label)
        self._aout_combo = ttk.Combobox(
            aout_frame,
            textvariable=self._aout_var,
            values=(),
            state="readonly",
            width=32,
        )
        self._aout_combo.pack(side=tk.LEFT, padx=4, fill=tk.X, expand=True)

        tk.Button(
            aout_frame,
            text="Refresh",
            command=self._refresh_audio,
        ).pack(side=tk.LEFT, padx=4)

        # -- TX Microphone row ---------------------------------------------
        txmic_frame = tk.LabelFrame(self._win, text="TX Microphone (Computer Mic → Radio)")
        txmic_frame.pack(padx=16, pady=6, fill=tk.X)

        tk.Label(txmic_frame, text="Device:").pack(side=tk.LEFT, padx=6)

        self._tx_mic_var = tk.StringVar(value=current_tx_mic_label)
        self._tx_mic_combo = ttk.Combobox(
            txmic_frame,
            textvariable=self._tx_mic_var,
            values=(),
            state="readonly",
            width=32,
        )
        self._tx_mic_combo.pack(side=tk.LEFT, padx=4, fill=tk.X, expand=True)

        tk.Button(
            txmic_frame,
            text="Refresh",
            command=self._refresh_audio,
        ).pack(side=tk.LEFT, padx=4)

        # -- TX Radio Output row -------------------------------------------
        txout_frame = tk.LabelFrame(self._win, text="TX Radio Output (Computer → Radio Audio Input)")
        txout_frame.pack(padx=16, pady=6, fill=tk.X)

        tk.Label(txout_frame, text="Device:").pack(side=tk.LEFT, padx=6)

        self._tx_radio_out_var = tk.StringVar(value=current_tx_radio_out_label)
        self._tx_radio_out_combo = ttk.Combobox(
            txout_frame,
            textvariable=self._tx_radio_out_var,
            values=(),
            state="readonly",
            width=32,
        )
        self._tx_radio_out_combo.pack(side=tk.LEFT, padx=4, fill=tk.X, expand=True)

        tk.Button(
            txout_frame,
            text="Refresh",
            command=self._refresh_audio,
        ).pack(side=tk.LEFT, padx=4)

        self._audio_status_lbl = tk.Label(self._win, text="", anchor="w", fg="gray")
        self._audio_status_lbl.pack(padx=16, fill=tk.X)

        # -- Buttons -------------------------------------------------------
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
        self._win.bind("<Return>",   lambda _e: self._apply())
        self._win.bind("<KP_Enter>", lambda _e: self._apply())
        self._win.bind("<Escape>",   lambda _e: self._win.destroy())

        # Populate serial ports immediately
        self._refresh_ports(initial=current_port)

        # Populate audio devices
        self._refresh_audio(
            initial_in=current_audio_in_label,
            initial_out=current_audio_out_label,
            initial_tx_mic=current_tx_mic_label,
            initial_tx_radio_out=current_tx_radio_out_label,
        )

        # Centre over the parent
        self._win.update_idletasks()
        pw = parent.winfo_rootx() + parent.winfo_width()  // 2
        ph = parent.winfo_rooty() + parent.winfo_height() // 2
        w  = self._win.winfo_width()
        h  = self._win.winfo_height()
        self._win.geometry(f"+{pw - w // 2}+{ph - h // 2}")

    # -- Internal helpers --------------------------------------------------

    def _refresh_ports(self, initial: str = "") -> None:
        ports = _enum_serial_ports()
        self._port_combo.config(values=ports)

        if ports:
            current = self._port_var.get()
            if initial and initial in ports:
                self._port_var.set(initial)
            elif current in ports:
                pass
            else:
                self._port_var.set(ports[0])
            self._port_status.config(text=f"{len(ports)} port(s) found", fg="gray")
        else:
            self._port_var.set("")
            self._port_status.config(text="No ports found", fg="red")

    def _refresh_audio(
        self,
        initial_in: str = "",
        initial_out: str = "",
        initial_tx_mic: str = "",
        initial_tx_radio_out: str = "",
    ) -> None:
        """Query preferred audio devices (WASAPI/MME) and populate all four dropdowns."""
        self._audio_status_lbl.config(text="Scanning audio devices...", fg="gray")
        self._win.update_idletasks()

        ins, outs, err = _enum_audio_devices()

        in_labels  = [lbl for (_, lbl) in ins]
        out_labels = [lbl for (_, lbl) in outs]

        self._ain_combo.config(values=in_labels)
        self._aout_combo.config(values=out_labels)
        self._tx_mic_combo.config(values=in_labels)
        self._tx_radio_out_combo.config(values=out_labels)

        # Restore or auto-select RX input
        cur_in = initial_in or self._ain_var.get()
        if cur_in in in_labels:
            self._ain_var.set(cur_in)
        elif in_labels:
            self._ain_var.set(in_labels[0])
        else:
            self._ain_var.set("")

        # Restore or auto-select RX output (monitor)
        cur_out = initial_out or self._aout_var.get()
        if cur_out in out_labels:
            self._aout_var.set(cur_out)
        elif out_labels:
            self._aout_var.set(out_labels[0])
        else:
            self._aout_var.set("")

        # Restore or auto-select TX microphone
        cur_tx_mic = initial_tx_mic or self._tx_mic_var.get()
        if cur_tx_mic in in_labels:
            self._tx_mic_var.set(cur_tx_mic)
        elif in_labels:
            self._tx_mic_var.set(in_labels[0])
        else:
            self._tx_mic_var.set("")

        # Restore or auto-select TX radio output
        cur_tx_out = initial_tx_radio_out or self._tx_radio_out_var.get()
        if cur_tx_out in out_labels:
            self._tx_radio_out_var.set(cur_tx_out)
        elif out_labels:
            self._tx_radio_out_var.set(out_labels[0])
        else:
            self._tx_radio_out_var.set("")

        if err:
            self._audio_status_lbl.config(text=err, fg="orange")
        else:
            self._audio_status_lbl.config(
                text=f"{len(ins)} input(s), {len(outs)} output(s) found", fg="gray"
            )

    @staticmethod
    def _parse_device_index(label: str) -> int:
        """Extract numeric device index from a '42: Device Name (API)' label."""
        try:
            return int((label or "").split(":", 1)[0].strip())
        except Exception:
            return -1

    def _apply(self) -> None:
        port = self._port_var.get().strip()
        if not port:
            messagebox.showerror("Settings", "Please select a serial port.", parent=self._win)
            return

        try:
            baud = int(self._baud_var.get())
        except ValueError:
            messagebox.showerror("Settings", "Invalid baud rate.", parent=self._win)
            return

        try:
            stopbits = float(self._stop_var.get())
        except ValueError:
            messagebox.showerror("Settings", "Invalid stop-bits value.", parent=self._win)
            return

        ain_label          = self._ain_var.get().strip()
        aout_label         = self._aout_var.get().strip()
        tx_mic_label       = self._tx_mic_var.get().strip()
        tx_radio_out_label = self._tx_radio_out_var.get().strip()

        ain_idx          = self._parse_device_index(ain_label)
        aout_idx         = self._parse_device_index(aout_label)
        tx_mic_idx       = self._parse_device_index(tx_mic_label)
        tx_radio_out_idx = self._parse_device_index(tx_radio_out_label)

        self._win.destroy()
        self._on_apply(
            port, baud, stopbits,
            ain_idx, ain_label, aout_idx, aout_label,
            tx_mic_idx, tx_mic_label, tx_radio_out_idx, tx_radio_out_label,
        )


# --- GUI & Features Class ---
class RadioGUI:
    def __init__(self, root, radio, config: "AppConfig | None" = None):
        self.root   = root
        self.radio  = radio
        self._config = config or AppConfig()
        self.scanning    = False
        self.active_band = None

        self._ui_queue   = queue.Queue()
        self._shutdown   = threading.Event()
        self._poll_thread = None

        # Operating mode: "voice" or "data"
        self._op_mode = "voice"

        # Frequency step for tuning buttons (MHz)
        self._freq_step = 0.001

        # Audio input device (index to pass to SoundCardAudioSource / AudioPassthrough)
        _saved_idx = self._config.audio_device_index
        self.audio_device_index = _saved_idx if _saved_idx >= 0 else None

        # Audio output device (headphones / speakers for RX monitoring)
        _saved_out = self._config.audio_output_device_index
        self.audio_output_device_index = _saved_out if _saved_out >= 0 else None

        # TX audio devices
        _tx_mic = self._config.tx_mic_device_index
        self.tx_mic_device_index = _tx_mic if _tx_mic >= 0 else None

        _tx_out = self._config.tx_radio_out_device_index
        self.tx_radio_out_device_index = _tx_out if _tx_out >= 0 else None

        # Live audio stream state (data mode / FT8)
        self._audio_src    = None
        self._audio_thread = None
        self._audio_stop   = threading.Event()

        # RX audio passthrough (voice mode: radio audio → computer speakers)
        self._audio_passthrough: "AudioPassthrough | None" = None

        # TX audio capture (PTT active: computer mic → radio audio input)
        self._tx_capture: "AudioTxCapture | None" = None

        # FT8 decoder -- created once, started/stopped with operating mode
        self._ft8 = FT8ConsoleDecoder(on_decode=self._on_ft8_decode)

        # FT8 TX coordinator (manual-assisted TX, Milestone 4)
        self._slot_timer = default_slot_timer
        self._tx_coord = Ft8TxCoordinator(
            radio=self.radio,
            slot_timer=self._slot_timer,
        )
        self._tx_coord.on_state_change = self._on_tx_state_change
        # Countdown refresh timer handle (stored so it can be cancelled)
        self._tx_countdown_after: str | None = None

        # CQ QSO assist — operator-confirmed reply pre-fill (Milestone 4)
        # Never auto-transmits; only pre-fills the TX message field and shows
        # a hint so the operator can review and press Arm TX manually.
        self._qso_mgr: "Ft8QsoManager | None" = None
        self._qso_assist_active: bool = False
        # Dedup guard: last message we pre-filled via the assist path.
        # Prevents the same suggestion being written to the TX field more
        # than once for the same QSO step.  Cleared when TX reaches COMPLETE
        # (meaning the message was sent); preserved on ERROR/CANCELED so the
        # operator's last suggestion remains visible.
        self._qso_assist_prefilled: str = ""

        self.root.title("FT-991A Command Center")
        self.root.geometry("500x900")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.setup_ui()
        self.refresh_connection_ui()

        # Apply initial operating mode (voice) to show/hide appropriate sections
        self._apply_op_mode("voice")

        self.process_ui_queue()
        self.start_polling()

    # -- Audio stream helpers ----------------------------------------------

    def _audio_worker(self):
        """
        Consume audio chunks so the input stream stays active.
        Only feeds the FT8 decoder when in data mode.
        Updates the GUI with a simple RMS indicator.
        """
        try:
            src = self._audio_src
            if src is None:
                return

            last_ui = 0.0
            for chunk in src.chunks(timeout_s=0.5):
                if self._shutdown.is_set() or self._audio_stop.is_set():
                    break

                # Feed decoder only in data mode
                if self._op_mode == "data":
                    try:
                        self._ft8.feed(
                            fs=chunk.fs,
                            samples=chunk.samples,
                            t0_monotonic=chunk.t0,
                        )
                    except Exception:  # noqa: BLE001
                        _log.exception("FT8 decoder feed error")

                # Basic RMS level indicator
                x   = chunk.samples
                rms = float((x * x).mean() ** 0.5) if x.size else 0.0

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

    def _on_start_audio(self) -> None:
        """Start audio stream using the configured input device."""
        if self.audio_device_index is None:
            self.audio_status.config(
                text="Audio: no device configured -- open Settings first"
            )
            return
        try:
            self._start_audio_stream(self.audio_device_index)
            self.audio_status.config(
                text=f"Audio: LIVE on device {self.audio_device_index}"
            )
            self._audio_start_btn.config(state=tk.DISABLED)
            self._audio_stop_btn.config(state=tk.NORMAL)
        except Exception as e:
            self.audio_status.config(text=f"Audio: ERROR (failed to start: {e})")

    def _on_stop_audio(self) -> None:
        """Stop the running audio stream."""
        self._stop_audio_stream()
        self.audio_status.config(text="Audio: stopped")
        self._audio_start_btn.config(state=tk.NORMAL)
        self._audio_stop_btn.config(state=tk.DISABLED)

    # -- Voice RX monitoring helpers ---------------------------------------

    def _start_rx_monitor(self) -> None:
        """
        Start RX audio passthrough: radio audio input → computer speakers/headphones.
        Requires both audio_device_index (radio input) and audio_output_device_index
        (headphones) to be configured in Settings.
        """
        self._stop_rx_monitor()
        if self.audio_device_index is None or self.audio_output_device_index is None:
            self._ui_queue.put(("voice_audio_status",
                                "Voice Audio: configure input & output devices in Settings"))
            return
        try:
            self._audio_passthrough = AudioPassthrough(
                input_device=self.audio_device_index,
                output_device=self.audio_output_device_index,
                rms_callback=lambda rms: self._ui_queue.put(("voice_rx_rms", rms)),
            )
            self._audio_passthrough.start()
            self._ui_queue.put((
                "voice_audio_status",
                f"RX Monitor: LIVE (in={self.audio_device_index} → out={self.audio_output_device_index})",
            ))
        except Exception as e:
            self._audio_passthrough = None
            self._ui_queue.put(("voice_audio_status", f"RX Monitor: ERROR ({e})"))

    def _stop_rx_monitor(self) -> None:
        """Stop RX audio passthrough if running."""
        pt = self._audio_passthrough
        self._audio_passthrough = None
        if pt is not None:
            try:
                pt.stop()
            except Exception:
                pass

    def _on_start_rx_monitor(self) -> None:
        """GUI callback: start RX audio monitoring."""
        self._start_rx_monitor()
        if self._audio_passthrough is not None:
            self._voice_rx_start_btn.config(state=tk.DISABLED)
            self._voice_rx_stop_btn.config(state=tk.NORMAL)

    def _on_stop_rx_monitor(self) -> None:
        """GUI callback: stop RX audio monitoring."""
        self._stop_rx_monitor()
        self._voice_rx_start_btn.config(state=tk.NORMAL)
        self._voice_rx_stop_btn.config(state=tk.DISABLED)
        self.voice_audio_status.config(text="RX Monitor: stopped")

    # -- TX audio helpers --------------------------------------------------

    def _start_tx_audio(self) -> None:
        """
        Start TX audio capture: computer microphone → radio audio input.
        Called automatically on PTT press.
        """
        if self.tx_mic_device_index is None or self.tx_radio_out_device_index is None:
            # TX audio not configured; PTT still works via CAT PTT command
            return
        try:
            self._tx_capture = AudioTxCapture(
                mic_device=self.tx_mic_device_index,
                radio_out_device=self.tx_radio_out_device_index,
                rms_callback=lambda rms: self._ui_queue.put(("voice_tx_rms", rms)),
            )
            self._tx_capture.start()
        except Exception as e:
            self._tx_capture = None
            self._ui_queue.put(("voice_audio_status", f"TX Audio: ERROR ({e})"))

    def _stop_tx_audio(self) -> None:
        """Stop TX audio capture. Called automatically on PTT release."""
        tx = self._tx_capture
        self._tx_capture = None
        if tx is not None:
            try:
                tx.stop()
            except Exception:
                pass

    # -- FT8 helpers -------------------------------------------------------

    def _on_ft8_decode(self, utc: str, freq_hz: float, snr_db: float, message: str) -> None:
        """
        Called from the FT8 decoder thread on every successful LDPC+CRC decode.
        Thread-safe -- puts a UI update onto the queue; never touches
        Tkinter widgets directly.
        """
        line = format_ft8_message(utc, snr_db, freq_hz, message)
        print(line, flush=True)
        # Include the raw message and SNR so the UI thread can route it to
        # the CQ QSO assist watcher without re-parsing the formatted line.
        self._ui_queue.put(("ft8_decoded", line + "\n", message, snr_db))

    def _clear_ft8_log(self) -> None:
        """Clear the FT8 decoded messages panel (called from GUI thread)."""
        self.ft8_log.config(state=tk.NORMAL)
        self.ft8_log.delete("1.0", tk.END)
        self.ft8_log.config(state=tk.DISABLED)

    def _save_ft8_log_to_file(self) -> None:
        """
        Append all messages currently in the FT8 panel to FT8_LOG_PATH.
        Uses append mode so previous sessions are preserved.
        Called automatically when switching to voice mode or on app close.
        """
        try:
            content = self.ft8_log.get("1.0", tk.END).strip()
            if not content:
                return
            with open(FT8_LOG_PATH, "a", encoding="utf-8") as fh:
                fh.write(
                    f"\n--- Session {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')} ---\n"
                )
                fh.write(content + "\n")
            print(f"[FT8 Log] Saved to {FT8_LOG_PATH}", flush=True)
        except Exception as e:
            print(f"[FT8 Log] Could not save: {e}", flush=True)

    # -- FT8 TX helpers ----------------------------------------------------

    def _on_save_operator(self) -> None:
        """Validate and persist the operator callsign/grid from the TX panel."""
        call = self._tx_callsign_var.get().strip().upper()
        grid = self._tx_grid_var.get().strip().upper()
        ok, reason = validate_operator(call, grid)
        if not ok:
            messagebox.showerror("Operator Settings", reason, parent=self.root)
            return
        self._config.save_operator(call, grid)
        self._tx_status_var.set(f"Operator saved: {call} / {grid}")

    def _on_compose_cq(self) -> None:
        """Pre-fill TX message with a CQ using the current operator settings."""
        call = self._tx_callsign_var.get().strip().upper()
        grid = self._tx_grid_var.get().strip().upper()
        ok, reason = validate_operator(call, grid)
        if not ok:
            messagebox.showerror(
                "Operator Settings",
                f"Cannot compose CQ: {reason}",
                parent=self.root,
            )
            return
        self._tx_msg_var.set(f"CQ {call} {grid}")

    def _on_ft8_log_double_click(self, event) -> None:
        """
        Double-click on a decoded FT8 log line to pre-fill the TX message
        with an appropriate reply.

        The raw line format from format_ft8_message is:
            HH:MM:SS  SNR dB  freq Hz  message
        The FT8 message text is everything after the third whitespace-delimited
        field.  We extract it and pass it to the reply helper.
        """
        try:
            # Get the line that was double-clicked
            index = self.ft8_log.index(f"@{event.x},{event.y}")
            line_start = self.ft8_log.index(f"{index} linestart")
            line_end   = self.ft8_log.index(f"{index} lineend")
            raw_line   = self.ft8_log.get(line_start, line_end).strip()
            if not raw_line:
                return
            # Extract the FT8 message: everything from the 4th token onward
            parts = raw_line.split()
            if len(parts) < 4:
                return
            ft8_msg = " ".join(parts[3:])
            self._prefill_reply(ft8_msg)
        except Exception:
            pass  # Ignore parse errors silently

    def _prefill_reply(self, received_msg: str) -> None:
        """
        Given a received FT8 message, pre-fill the TX message box with the
        recommended reply.

        Only pre-fills; the operator must still press "Arm TX" to transmit.
        Validates operator settings before composing.
        """
        call = self._tx_callsign_var.get().strip().upper()
        grid = self._tx_grid_var.get().strip().upper()
        ok, reason = validate_operator(call, grid)
        if not ok:
            self._tx_status_var.set(f"Cannot reply: {reason}")
            return

        parts = received_msg.strip().upper().split()
        if not parts:
            return

        if parts[0] in ("CQ", "QRZ") and len(parts) >= 3:
            # CQ DX_CALL GRID — reply: DX_CALL OUR_CALL +00
            dx_call = parts[1]
            self._tx_msg_var.set(f"{dx_call} {call} +00")
        elif len(parts) >= 3 and parts[0] == call:
            # DX is calling us: OUR_CALL DX_CALL REPORT or RR73/RRR/73
            dx_call = parts[1]
            extra   = parts[2] if len(parts) > 2 else ""
            if extra in ("RRR", "RR73"):
                self._tx_msg_var.set(f"{dx_call} {call} 73")
            elif extra == "73":
                self._tx_msg_var.set("")  # QSO complete — clear field
            elif extra.startswith("R"):
                self._tx_msg_var.set(f"{dx_call} {call} RR73")
            else:
                self._tx_msg_var.set(f"{dx_call} {call} R+00")
        else:
            # Generic: just copy the message so operator can edit
            self._tx_msg_var.set(received_msg.strip())

        self._tx_status_var.set(f"Reply pre-filled from: {received_msg.strip()}")

    # -- CQ QSO assist methods (Milestone 4) ---------------------------------

    def _on_start_cq_session(self) -> None:
        """
        Start a CQ QSO assist session.

        Creates a new :class:`~ft8_qso.Ft8QsoManager`, composes the CQ
        message, pre-fills the TX message field, and activates the
        decode-driven reply-assist watcher.

        The operator must still press **Arm TX** to transmit.  The assist
        only pre-fills the TX field and updates the status label — it never
        calls :meth:`_on_arm_tx` automatically.
        """
        call = self._tx_callsign_var.get().strip().upper()
        grid = self._tx_grid_var.get().strip().upper()
        ok, reason = validate_operator(call, grid)
        if not ok:
            messagebox.showerror(
                "Operator Settings",
                f"Cannot start CQ session: {reason}",
                parent=self.root,
            )
            return

        op = OperatorConfig(callsign=call, grid=grid)
        self._qso_mgr = Ft8QsoManager(operator=op, slot_timer=self._slot_timer)
        cq_msg = self._qso_mgr.start_cq()

        # Pre-fill TX message and activate the assist watcher
        self._tx_msg_var.set(cq_msg)
        self._qso_assist_active    = True
        self._qso_assist_prefilled = ""

        # Update button states
        self._cq_session_btn.config(state=tk.DISABLED)
        self._stop_session_btn.config(state=tk.NORMAL)
        self._tx_status_var.set(
            f"CQ Session: {cq_msg} — Arm TX, then watch for replies"
        )

    def _on_stop_cq_session(self) -> None:
        """
        Stop the active CQ session and reset all QSO assist state.

        Safe to call at any time; clears assist flags, aborts the QSO
        manager (if present), and restores the session button states.
        """
        if self._qso_mgr is not None:
            self._qso_mgr.abort()
        self._qso_mgr              = None
        self._qso_assist_active    = False
        self._qso_assist_prefilled = ""
        # Milestone 6: clear DX callsign AP passes — no more QSO context.
        self._ft8.set_dx_callsign(None)
        self._cq_session_btn.config(state=tk.NORMAL)
        self._stop_session_btn.config(state=tk.DISABLED)
        self._tx_status_var.set("QSO Session stopped")

    def _maybe_assist_prefill(self, message: str, snr_db: float) -> None:
        """
        Inspect *message* against the active CQ QSO state machine and
        pre-fill the TX message field with the appropriate next response.

        Called on the Tkinter thread (via the UI queue), so it may safely
        update :class:`tkinter.StringVar` instances and button states.
        **It never calls** :meth:`_on_arm_tx` — the operator must press
        Arm TX to transmit.

        Guards
        ------
        - Only runs while ``_qso_assist_active`` is True.
        - Skips if the TX coordinator is ARMED, TX_PREP, or TX_ACTIVE
          (a transmission is scheduled or in flight; don't overwrite it).
        - Dedup: skips if this exact message text was already pre-filled
          for the current QSO cycle (prevents redundant UI flicker when
          the same FT8 message is decoded in multiple consecutive slots).
        """
        if not self._qso_assist_active or self._qso_mgr is None:
            return

        # Gate: don't interfere with an active or armed TX job
        tx_state = self._tx_coord.state
        if tx_state in (TxState.ARMED, TxState.TX_PREP, TxState.TX_ACTIVE):
            return

        try:
            next_msg = self._qso_mgr.advance(message, snr_db=round(snr_db))
        except Exception:
            return  # Defensive: never crash the UI queue drain

        if next_msg is None:
            return  # Message did not advance QSO state (not addressed to us, etc.)

        # Milestone 6: Callsign-aware AP passes — notify the decoder of the
        # active DX partner so it can inject their callsign bits into LDPC AP
        # passes.  The partner is first known once the QSO manager accepts the
        # reply (i.e. dx_callsign is populated after advance() returns non-None).
        if self._qso_mgr.dx_callsign:
            self._ft8.set_dx_callsign(self._qso_mgr.dx_callsign)

        # Dedup: avoid writing the same suggestion repeatedly
        if next_msg == self._qso_assist_prefilled:
            return

        # Pre-fill the TX message field
        self._tx_msg_var.set(next_msg)
        self._qso_assist_prefilled = next_msg

        # Compose a human-readable status hint
        rx  = ReceivedMessage(message)
        dx  = rx.call2 if rx.call2 else "?"
        qso_state = self._qso_mgr.state
        if qso_state == QsoState.EXCHANGE_SENT:
            hint = f"Reply from {dx} — review exchange, then Arm TX"
        elif qso_state == QsoState.COMPLETE:
            if rx.is_rrr:
                completion_text = "RRR"
            elif rx.is_rr73:
                completion_text = "RR73"
            else:
                completion_text = "Completion reply"
            hint = f"{completion_text} from {dx} — 73 pre-filled, Arm TX to close QSO"
        else:
            hint = f"Next reply from {dx} — review then Arm TX"

        self._tx_status_var.set(f"CQ Assist: {hint}")

    def _on_arm_tx(self) -> None:
        """Arm the TX coordinator for the next FT8 slot."""
        msg = self._tx_msg_var.get().strip()
        if not msg:
            messagebox.showerror("TX Error", "TX message is empty.", parent=self.root)
            return

        # Validate operator settings
        call = self._tx_callsign_var.get().strip().upper()
        grid = self._tx_grid_var.get().strip().upper()
        ok, reason = validate_operator(call, grid)
        if not ok:
            messagebox.showerror(
                "Operator Settings",
                f"Cannot arm TX: {reason}",
                parent=self.root,
            )
            return

        # Persist operator settings
        self._config.save_operator(call, grid)

        # Validate and read base tone frequency
        try:
            f0_hz = float(self._tx_base_tone_var.get().strip())
        except ValueError:
            messagebox.showerror(
                "TX Error", "Base tone frequency must be a number (50–3000 Hz).",
                parent=self.root,
            )
            return
        if not (50.0 <= f0_hz <= 3000.0):
            messagebox.showerror(
                "TX Error",
                f"Base tone frequency {f0_hz:.0f} Hz is out of range (50–3000 Hz).",
                parent=self.root,
            )
            return
        self._config.save_ft8_base_tone_hz(f0_hz)

        # Build TxJob with the configured radio output device
        dev = self.tx_radio_out_device_index
        job = TxJob(
            msg,
            f0_hz=f0_hz,
            audio_device=dev,
        )

        try:
            self._tx_coord.arm(job)
        except RuntimeError as exc:
            messagebox.showerror("TX Error", str(exc), parent=self.root)
            return

        # Update button states
        self._arm_btn.config(state=tk.DISABLED)
        self._cancel_btn.config(state=tk.NORMAL)
        # Start countdown display
        self._update_tx_countdown()

    def _on_cancel_tx(self) -> None:
        """Cancel the armed TX job before slot start."""
        accepted = self._tx_coord.cancel()
        if not accepted:
            self._tx_status_var.set("TX: Cannot cancel — already active or idle")

    def _on_tx_state_change(self, state: TxState, message: str) -> None:
        """
        Callback from Ft8TxCoordinator (worker thread) — marshals update to GUI.
        """
        self._ui_queue.put(("tx_state", state, message))

    def _apply_tx_state_update(self, state: TxState, message: str) -> None:
        """
        Update TX panel widgets based on a new TxState (GUI thread only).
        """
        state_colors = {
            TxState.IDLE:      "gray",
            TxState.ARMED:     "#4a90d9",
            TxState.TX_PREP:   "orange",
            TxState.TX_ACTIVE: "red",
            TxState.COMPLETE:  "green",
            TxState.ERROR:     "red",
            TxState.CANCELED:  "gray",
        }
        color = state_colors.get(state, "gray")
        self._tx_status_lbl.config(fg=color)
        self._tx_status_var.set(f"TX: {message}")

        if state in (TxState.IDLE, TxState.COMPLETE, TxState.ERROR, TxState.CANCELED):
            self._arm_btn.config(state=tk.NORMAL)
            self._cancel_btn.config(state=tk.DISABLED)
            # Clear the dedup guard so the next decode can trigger a fresh
            # assist prefill for the subsequent QSO step.
            if state == TxState.COMPLETE:
                self._qso_assist_prefilled = ""
            # Auto-reset coordinator to IDLE so next arm() works immediately
            self._tx_coord.reset()
        elif state == TxState.ARMED:
            self._arm_btn.config(state=tk.DISABLED)
            self._cancel_btn.config(state=tk.NORMAL)
        else:  # TX_PREP or TX_ACTIVE
            self._arm_btn.config(state=tk.DISABLED)
            self._cancel_btn.config(state=tk.DISABLED)

    def _update_tx_countdown(self) -> None:
        """
        Periodic GUI-thread callback that updates the TX status label with a
        live countdown when the coordinator is ARMED.
        """
        state = self._tx_coord.state
        if state == TxState.ARMED:
            try:
                secs = self._tx_coord.seconds_to_next_slot()
                self._tx_status_var.set(f"TX: Armed — slot in {secs:.1f} s")
            except Exception:
                pass
            # Reschedule every 250 ms
            self._tx_countdown_after = self.root.after(250, self._update_tx_countdown)
        else:
            self._tx_countdown_after = None

    # -- Operating mode helpers --------------------------------------------

    def _switch_to_voice(self) -> None:
        """Switch to voice operating mode."""
        # Stop any active CQ QSO session before switching modes
        if self._qso_assist_active:
            self._on_stop_cq_session()
        # Save decoded FT8 messages before hiding the panel
        self._save_ft8_log_to_file()
        # Stop data-mode audio stream and FT8 decoder
        self._stop_audio_stream()
        try:
            self._ft8.stop()
        except Exception:  # noqa: BLE001
            _log.exception("FT8 decoder stop error during mode switch to voice")
        self._op_mode = "voice"
        self._apply_op_mode("voice")

    def _switch_to_data(self) -> None:
        """Switch to data/FT8 operating mode."""
        # Stop voice-mode audio passthrough before switching
        self._stop_rx_monitor()
        self._stop_tx_audio()
        self._op_mode = "data"
        # Start FT8 decoder (it will receive audio once the stream is started)
        try:
            self._ft8.start()
        except Exception:  # noqa: BLE001
            _log.exception("FT8 decoder start error during mode switch to data")
        self._apply_op_mode("data")

    def _apply_op_mode(self, mode: str) -> None:
        """
        Show/hide GUI sections based on operating mode.

        Voice: PTT + Voice Audio Monitor + Signal Log visible; FT8 + Audio Controls hidden.
        Data:  FT8 panel + Audio Controls visible; PTT + Voice Audio + Signal Log hidden.
        """
        # Default button background (platform-safe: read from the widget itself)
        _btn_bg = self._voice_btn.cget("bg")

        # Update toggle button styles
        if mode == "voice":
            self._voice_btn.config(relief=tk.SUNKEN, bg="#4a90d9", fg="white")
            self._data_btn.config( relief=tk.RAISED,  bg=_btn_bg,   fg="black")
        else:
            self._voice_btn.config(relief=tk.RAISED,  bg=_btn_bg,   fg="black")
            self._data_btn.config( relief=tk.SUNKEN, bg="#e8a020", fg="white")

        # Remove all mode-conditional sections first (preserves pack order on re-add)
        self._ptt_frame.pack_forget()
        self._voice_audio_frame.pack_forget()
        self.log_frame.pack_forget()
        self._audio_ctrl_frame.pack_forget()
        self.ft8_frame.pack_forget()
        self._tx_frame.pack_forget()

        if mode == "voice":
            self._ptt_frame.pack(pady=5, fill=tk.X, padx=20)
            self._voice_audio_frame.pack(padx=20, pady=(0, 6), fill=tk.X)
            self.log_frame.pack(padx=20, pady=(0, 10), fill=tk.BOTH, expand=True)
        else:
            # Reset audio button states
            self._audio_start_btn.config(state=tk.NORMAL)
            self._audio_stop_btn.config(state=tk.DISABLED)
            self._audio_ctrl_frame.pack(padx=20, pady=10, fill=tk.X)
            self.ft8_frame.pack(padx=20, pady=(0, 4), fill=tk.BOTH, expand=True)
            self._tx_frame.pack(padx=20, pady=(0, 10), fill=tk.X)

    # -- Frequency step helpers --------------------------------------------

    def _freq_step_up(self) -> None:
        """Tune frequency up by one step."""
        if not self.radio.is_connected():
            return
        def _worker():
            f = self.radio.get_frequency()
            if f > 0:
                self.radio.set_frequency(round(f + self._freq_step, 9))
        threading.Thread(target=_worker, daemon=True).start()

    def _freq_step_down(self) -> None:
        """Tune frequency down by one step."""
        if not self.radio.is_connected():
            return
        def _worker():
            f = self.radio.get_frequency()
            if f > 0:
                self.radio.set_frequency(round(f - self._freq_step, 9))
        threading.Thread(target=_worker, daemon=True).start()

    # -- UI Setup ----------------------------------------------------------

    def setup_ui(self):
        # -- Operating mode toggle -----------------------------------------
        mode_row = tk.Frame(self.root, bd=1, relief=tk.GROOVE)
        mode_row.pack(padx=20, pady=(10, 4), fill=tk.X)

        tk.Label(mode_row, text="Mode:", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=8)

        self._voice_btn = tk.Button(
            mode_row, text="VOICE", width=12,
            command=self._switch_to_voice,
        )
        self._voice_btn.pack(side=tk.LEFT, padx=4, pady=4)

        self._data_btn = tk.Button(
            mode_row, text="DATA / FT8", width=14,
            command=self._switch_to_data,
        )
        self._data_btn.pack(side=tk.LEFT, padx=4, pady=4)

        # -- Frequency display + step buttons ------------------------------
        freq_row = tk.Frame(self.root)
        freq_row.pack(pady=(4, 0), fill=tk.X)

        tk.Button(
            freq_row, text="\u25bc", font=("Consolas", 14),
            width=3, command=self._freq_step_down,
        ).pack(side=tk.LEFT, padx=(20, 2))

        self.freq_disp = tk.Label(
            freq_row, text="DISCONNECTED",
            font=("Consolas", 36), fg="lime", bg="black",
        )
        self.freq_disp.pack(side=tk.LEFT, fill=tk.X, expand=True)

        tk.Button(
            freq_row, text="\u25b2", font=("Consolas", 14),
            width=3, command=self._freq_step_up,
        ).pack(side=tk.LEFT, padx=(2, 20))

        # Step size selector
        step_row = tk.Frame(self.root)
        step_row.pack(fill=tk.X, padx=20)
        tk.Label(step_row, text="Step:", font=("Arial", 8)).pack(side=tk.LEFT)
        self._step_var = tk.StringVar(value="1 kHz")
        _step_options = {
            "10 Hz":   0.00001,
            "100 Hz":  0.0001,
            "1 kHz":   0.001,
            "5 kHz":   0.005,
            "10 kHz":  0.010,
            "25 kHz":  0.025,
            "100 kHz": 0.100,
        }
        self._step_map = _step_options
        step_cb = ttk.Combobox(
            step_row,
            textvariable=self._step_var,
            values=list(_step_options.keys()),
            state="readonly",
            width=8,
        )
        step_cb.pack(side=tk.LEFT, padx=4)
        step_cb.bind("<<ComboboxSelected>>",
                     lambda e: setattr(
                         self, "_freq_step",
                         self._step_map.get(self._step_var.get(), 0.001)
                     ))

        # -- Connection controls -------------------------------------------
        conn_frame = tk.LabelFrame(self.root, text="Connection")
        conn_frame.pack(padx=20, pady=8, fill=tk.X)

        self.conn_status = tk.Label(conn_frame, text="Status: DISCONNECTED", anchor="w")
        self.conn_status.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.X, expand=True)

        self.settings_btn = tk.Button(
            conn_frame, text="SETTINGS", bg="lightblue", command=self.open_settings
        )
        self.settings_btn.pack(side=tk.RIGHT, padx=5, pady=5)

        self.conn_btn = tk.Button(
            conn_frame, text="CONNECT", bg="lightgreen", command=self.toggle_connection
        )
        self.conn_btn.pack(side=tk.RIGHT, padx=5, pady=5)

        # -- S-Meter -------------------------------------------------------
        tk.Label(self.root, text="S-METER").pack()
        self.meter_var = tk.IntVar()
        self.s_meter = ttk.Progressbar(self.root, maximum=255, variable=self.meter_var)
        self.s_meter.pack(padx=30, pady=5, fill=tk.X)

        # -- RF Power ------------------------------------------------------
        pwr_frame = tk.LabelFrame(self.root, text="RF Power (PC)")
        pwr_frame.pack(padx=20, pady=6, fill=tk.X)

        tk.Label(pwr_frame, text="Level (5-100):").pack(side=tk.LEFT, padx=5)

        self.rf_power_var = tk.IntVar(value=0)
        self.rf_power_spin = tk.Spinbox(
            pwr_frame, from_=5, to=100, width=5, textvariable=self.rf_power_var
        )
        self.rf_power_spin.pack(side=tk.LEFT, padx=5)
        self.rf_power_spin.bind("<Return>",   lambda e: self.apply_rf_power())
        self.rf_power_spin.bind("<KP_Enter>", lambda e: self.apply_rf_power())

        self.rf_power_apply_btn = tk.Button(pwr_frame, text="APPLY", command=self.apply_rf_power)
        self.rf_power_apply_btn.pack(side=tk.LEFT, padx=5)

        self.rf_power_status = tk.Label(pwr_frame, text="", anchor="w")
        self.rf_power_status.pack(side=tk.LEFT, padx=8, fill=tk.X, expand=True)

        # -- Radio Mode (MD) -----------------------------------------------
        mode_frame = tk.LabelFrame(self.root, text="Radio Mode (MD)")
        mode_frame.pack(padx=20, pady=6, fill=tk.X)

        tk.Label(mode_frame, text="Mode:").pack(side=tk.LEFT, padx=5)

        self.mode_var = tk.StringVar(value="USB")
        self.mode_combo = ttk.Combobox(
            mode_frame,
            textvariable=self.mode_var,
            values=("LSB", "USB", "CW", "FM", "AM", "C4FM", "DATA-L", "DATA-U"),
            state="readonly",
            width=9,
        )
        self.mode_combo.pack(side=tk.LEFT, padx=5)

        self.mode_apply_btn = tk.Button(mode_frame, text="APPLY", command=self.apply_mode)
        self.mode_apply_btn.pack(side=tk.LEFT, padx=5)

        self.mode_status = tk.Label(mode_frame, text="", anchor="w")
        self.mode_status.pack(side=tk.LEFT, padx=8, fill=tk.X, expand=True)

        self.mode_combo.bind("<<ComboboxSelected>>", lambda e: self.apply_mode())
        self.mode_combo.bind("<Return>",   lambda e: self.apply_mode())
        self.mode_combo.bind("<KP_Enter>", lambda e: self.apply_mode())

        # -- Band Selection ------------------------------------------------
        band_frame = tk.LabelFrame(self.root, text="Band Select")
        band_frame.pack(padx=20, pady=6, fill=tk.X)

        band_row1 = tk.Frame(band_frame)
        band_row1.pack(fill=tk.X, padx=4, pady=(2, 1))
        band_row2 = tk.Frame(band_frame)
        band_row2.pack(fill=tk.X, padx=4, pady=(1, 2))

        _band_names = list(BANDS.keys())
        _mid = len(_band_names) // 2
        for b in _band_names[:_mid]:
            tk.Button(
                band_row1, text=b, font=("Arial", 8),
                command=lambda name=b: self.goto_band(name)
            ).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)
        for b in _band_names[_mid:]:
            tk.Button(
                band_row2, text=b, font=("Arial", 8),
                command=lambda name=b: self.goto_band(name)
            ).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)

        # FT8 shortcut
        ft8_jump_row = tk.Frame(band_frame)
        ft8_jump_row.pack(fill=tk.X, padx=4, pady=(0, 4))
        tk.Button(
            ft8_jump_row, text="Jump to FT8 Freq", font=("Arial", 8),
            bg="#e8c060", command=self._goto_ft8_freq,
        ).pack(side=tk.LEFT, padx=2)

        # -- Scanner -------------------------------------------------------
        scan_frame = tk.LabelFrame(self.root, text="Scanner Controls")
        scan_frame.pack(padx=20, pady=6, fill=tk.X)
        self.scan_btn = tk.Button(
            scan_frame, text="START SCAN", bg="lightgray", command=self.toggle_scan
        )
        self.scan_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5, pady=5)

        tk.Label(scan_frame, text="Squelch:").pack(side=tk.LEFT)
        self.thresh_entry = tk.Entry(scan_frame, width=5)
        self.thresh_entry.insert(0, "40")
        self.thresh_entry.pack(side=tk.LEFT, padx=5)

        # -- PTT (voice-only section) --------------------------------------
        self._ptt_frame = tk.Frame(self.root)
        self.ptt_btn = tk.Button(
            self._ptt_frame,
            text="PUSH TO TALK",
            bg="darkred", fg="white",
            font=("Arial", 12, "bold"),
        )
        self.ptt_btn.pack(pady=6, ipadx=30, fill=tk.X)
        self.ptt_btn.bind("<ButtonPress-1>",   self._on_ptt_press)
        self.ptt_btn.bind("<ButtonRelease-1>", self._on_ptt_release)

        # -- Voice Audio Monitor (voice-only section) ----------------------
        self._voice_audio_frame = tk.LabelFrame(self.root, text="Voice Audio")

        voice_audio_btn_row = tk.Frame(self._voice_audio_frame)
        voice_audio_btn_row.pack(fill=tk.X, padx=5, pady=4)

        self._voice_rx_start_btn = tk.Button(
            voice_audio_btn_row, text="Start RX Monitor", bg="lightgreen",
            command=self._on_start_rx_monitor,
        )
        self._voice_rx_start_btn.pack(side=tk.LEFT, padx=4)

        self._voice_rx_stop_btn = tk.Button(
            voice_audio_btn_row, text="Stop RX Monitor", bg="orange", state=tk.DISABLED,
            command=self._on_stop_rx_monitor,
        )
        self._voice_rx_stop_btn.pack(side=tk.LEFT, padx=4)

        self.voice_audio_status = tk.Label(
            self._voice_audio_frame,
            text="RX Monitor: configure devices in Settings, then Start",
            anchor="w", fg="gray",
        )
        self.voice_audio_status.pack(padx=5, pady=(0, 4), fill=tk.X)

        # -- Signal Log (voice-only section) -------------------------------
        self.log_frame = tk.LabelFrame(self.root, text="Signal Log")

        self.note_entry = tk.Entry(self.log_frame)
        self.note_entry.insert(0, "Enter callsign/note...")
        self.note_entry.pack(fill=tk.X, padx=5, pady=5)

        tk.Button(
            self.log_frame, text="MANUAL LOG", command=self.manual_log, bg="lightblue"
        ).pack(fill=tk.X, padx=5)

        self.log_box = tk.Text(self.log_frame, height=6, font=("Consolas", 9))
        self.log_box.pack(padx=5, pady=5, fill=tk.BOTH, expand=True)

        # -- Audio Controls (data-only section) ----------------------------
        self._audio_ctrl_frame = tk.LabelFrame(self.root, text="Audio Connection")

        audio_btn_row = tk.Frame(self._audio_ctrl_frame)
        audio_btn_row.pack(fill=tk.X, padx=5, pady=4)

        self._audio_start_btn = tk.Button(
            audio_btn_row, text="Start Audio", bg="lightgreen",
            command=self._on_start_audio,
        )
        self._audio_start_btn.pack(side=tk.LEFT, padx=4)

        self._audio_stop_btn = tk.Button(
            audio_btn_row, text="Stop Audio", bg="orange", state=tk.DISABLED,
            command=self._on_stop_audio,
        )
        self._audio_stop_btn.pack(side=tk.LEFT, padx=4)

        self.audio_status = tk.Label(
            self._audio_ctrl_frame,
            text="Audio: configure device in Settings, then Start",
            anchor="w", fg="gray",
        )
        self.audio_status.pack(padx=5, pady=(0, 4), fill=tk.X)

        # -- FT8 Decoded Messages (data-only section) ----------------------
        self.ft8_frame = tk.LabelFrame(self.root, text="FT8 Decoded Messages")

        ft8_btn_row = tk.Frame(self.ft8_frame)
        ft8_btn_row.pack(fill=tk.X, padx=5, pady=(4, 0))
        tk.Label(ft8_btn_row, text="Live FT8 decodes appear here \u2192",
                 fg="gray", font=("Consolas", 8)).pack(side=tk.LEFT)
        tk.Button(
            ft8_btn_row, text="Clear", font=("Arial", 8),
            command=self._clear_ft8_log,
        ).pack(side=tk.RIGHT)

        self.ft8_log = tk.Text(
            self.ft8_frame,
            height=8,
            font=("Consolas", 9),
            bg="#0a0a1a", fg="#00ff88",
            insertbackground="white",
            state=tk.DISABLED,
        )
        ft8_scroll = ttk.Scrollbar(self.ft8_frame, orient=tk.VERTICAL, command=self.ft8_log.yview)
        self.ft8_log.configure(yscrollcommand=ft8_scroll.set)
        ft8_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.ft8_log.pack(padx=5, pady=(2, 5), fill=tk.BOTH, expand=True)

        # Bind double-click on a decoded message to pre-fill the TX message
        self.ft8_log.bind("<Double-Button-1>", self._on_ft8_log_double_click)

        # -- FT8 TX Panel (data-only section) ------------------------------
        self._tx_frame = tk.LabelFrame(self.root, text="FT8 Transmit")

        # Row 1: Operator callsign + grid
        op_row = tk.Frame(self._tx_frame)
        op_row.pack(fill=tk.X, padx=5, pady=(4, 0))
        tk.Label(op_row, text="My Call:", font=("Arial", 9)).pack(side=tk.LEFT)
        self._tx_callsign_var = tk.StringVar(
            value=self._config.operator_callsign
        )
        self._tx_callsign_entry = tk.Entry(
            op_row, textvariable=self._tx_callsign_var, width=10,
            font=("Consolas", 9),
        )
        self._tx_callsign_entry.pack(side=tk.LEFT, padx=(2, 8))
        tk.Label(op_row, text="Grid:", font=("Arial", 9)).pack(side=tk.LEFT)
        self._tx_grid_var = tk.StringVar(
            value=self._config.operator_grid
        )
        self._tx_grid_entry = tk.Entry(
            op_row, textvariable=self._tx_grid_var, width=6,
            font=("Consolas", 9),
        )
        self._tx_grid_entry.pack(side=tk.LEFT, padx=(2, 0))
        tk.Button(
            op_row, text="Save", font=("Arial", 8),
            command=self._on_save_operator,
        ).pack(side=tk.LEFT, padx=(6, 0))

        # Row 2: TX message entry
        msg_row = tk.Frame(self._tx_frame)
        msg_row.pack(fill=tk.X, padx=5, pady=(4, 0))
        tk.Label(msg_row, text="TX Msg:", font=("Arial", 9)).pack(side=tk.LEFT)
        self._tx_msg_var = tk.StringVar()
        self._tx_msg_entry = tk.Entry(
            msg_row, textvariable=self._tx_msg_var, width=30,
            font=("Consolas", 9),
        )
        self._tx_msg_entry.pack(side=tk.LEFT, padx=(2, 4), fill=tk.X, expand=True)
        tk.Button(
            msg_row, text="CQ", font=("Arial", 8),
            command=self._on_compose_cq,
        ).pack(side=tk.LEFT, padx=(0, 2))

        # Row 2b: Base tone frequency
        tone_row = tk.Frame(self._tx_frame)
        tone_row.pack(fill=tk.X, padx=5, pady=(4, 0))
        tk.Label(tone_row, text="Base Tone (Hz):", font=("Arial", 9)).pack(side=tk.LEFT)
        self._tx_base_tone_var = tk.StringVar(
            value=f"{self._config.ft8_base_tone_hz:g}"
        )
        self._tx_base_tone_entry = tk.Entry(
            tone_row, textvariable=self._tx_base_tone_var, width=6,
            font=("Consolas", 9),
        )
        self._tx_base_tone_entry.pack(side=tk.LEFT, padx=(2, 4))
        tk.Label(
            tone_row, text="(50–3000 Hz, default 1500)",
            font=("Arial", 8), fg="gray",
        ).pack(side=tk.LEFT)

        # Row 3: CQ Session assist buttons
        session_row = tk.Frame(self._tx_frame)
        session_row.pack(fill=tk.X, padx=5, pady=(4, 0))
        self._cq_session_btn = tk.Button(
            session_row,
            text="\u25b6 Start CQ Session",
            bg="#4a90d9", fg="white",
            font=("Arial", 8, "bold"),
            command=self._on_start_cq_session,
        )
        self._cq_session_btn.pack(side=tk.LEFT, padx=(0, 4))
        self._stop_session_btn = tk.Button(
            session_row,
            text="\u25a0 Stop QSO",
            bg="orange", state=tk.DISABLED,
            font=("Arial", 8),
            command=self._on_stop_cq_session,
        )
        self._stop_session_btn.pack(side=tk.LEFT, padx=(0, 0))

        # Row 4: Arm / Cancel buttons + status
        ctrl_row = tk.Frame(self._tx_frame)
        ctrl_row.pack(fill=tk.X, padx=5, pady=(4, 2))
        self._arm_btn = tk.Button(
            ctrl_row, text="Arm TX (next slot)", bg="lightgreen",
            font=("Arial", 9, "bold"),
            command=self._on_arm_tx,
        )
        self._arm_btn.pack(side=tk.LEFT, padx=(0, 4))
        self._cancel_btn = tk.Button(
            ctrl_row, text="Cancel TX", bg="orange", state=tk.DISABLED,
            font=("Arial", 9),
            command=self._on_cancel_tx,
        )
        self._cancel_btn.pack(side=tk.LEFT, padx=(0, 8))

        # Status label (shows countdown + TX state)
        self._tx_status_var = tk.StringVar(value="TX: Idle")
        self._tx_status_lbl = tk.Label(
            self._tx_frame, textvariable=self._tx_status_var,
            font=("Consolas", 9), fg="gray", anchor="w",
        )
        self._tx_status_lbl.pack(padx=5, pady=(0, 4), fill=tk.X)

    # -- Connection helpers ------------------------------------------------

    def refresh_connection_ui(self):
        connected = self.radio.is_connected()
        if connected:
            self.conn_status.config(
                text=f"Status: CONNECTED ({self.radio.port} @ {self.radio.baud})"
            )
            self.conn_btn.config(text="DISCONNECT", bg="orange")
            self.rf_power_apply_btn.config(state=tk.NORMAL)
            self.rf_power_spin.config(state="normal")
            self.mode_apply_btn.config(state=tk.NORMAL)
            self.mode_combo.config(state="readonly")
        else:
            self.conn_status.config(
                text=f"Status: DISCONNECTED ({self.radio.port} @ {self.radio.baud})"
            )
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
        self._start_tx_audio()
        self.ptt_btn.config(bg="red")
        return "break"

    def _on_ptt_release(self, event):
        if not self._ptt_allowed():
            return "break"
        self._stop_tx_audio()
        self.radio.ptt_off()
        self.ptt_btn.config(bg="darkred")
        return "break"

    def on_close(self):
        """Save FT8 log, stop worker threads, and close serial cleanly before exiting."""
        self.scanning = False
        self._shutdown.set()

        # Cancel any armed TX and cancel the countdown timer
        try:
            self._tx_coord.cancel()
        except Exception:
            pass
        if self._tx_countdown_after is not None:
            try:
                self.root.after_cancel(self._tx_countdown_after)
            except Exception:
                pass

        # Save any pending FT8 messages before exit
        try:
            self._save_ft8_log_to_file()
        except Exception:
            pass

        try:
            self._ft8.stop()
        except Exception:
            pass

        try:
            self._stop_audio_stream()
        except Exception:
            pass

        try:
            self._stop_rx_monitor()
        except Exception:
            pass

        try:
            self._stop_tx_audio()
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

        actual = self.radio.get_rf_power()
        self.rf_power_var.set(actual)
        self.rf_power_status.config(text=f"SET {actual:03d}")

    def apply_mode(self):
        """Apply selected mode via CAT using set_mode()."""
        if not self.radio.is_connected():
            self.mode_status.config(text="DISCONNECTED")
            return

        mode  = (self.mode_var.get() or "").strip().upper()
        valid = {"LSB", "USB", "CW", "FM", "AM", "C4FM", "DATA-L", "DATA-U"}
        if mode not in valid:
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
                    self.audio_status.config(text=f"Audio: LIVE (RMS {rms:.4f})")

                elif kind == "audio_status":
                    _, text = item
                    self.audio_status.config(text=text)

                elif kind == "voice_audio_status":
                    _, text = item
                    self.voice_audio_status.config(text=text)

                elif kind == "voice_rx_rms":
                    _, rms = item
                    self.voice_audio_status.config(
                        text=f"RX Monitor: LIVE  RMS {rms:.4f}"
                    )

                elif kind == "voice_tx_rms":
                    _, rms = item
                    self.voice_audio_status.config(
                        text=f"TX ACTIVE  Mic RMS {rms:.4f}"
                    )

                elif kind == "rf_power":
                    _, p = item
                    try:
                        if str(self.root.focus_get()) != str(self.rf_power_spin):
                            self.rf_power_var.set(p)
                    except Exception:
                        self.rf_power_var.set(p)

                elif kind == "mode":
                    _, m = item
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
                    line = item[1]
                    self.ft8_log.config(state=tk.NORMAL)
                    self.ft8_log.insert(tk.END, line)
                    self.ft8_log.see(tk.END)
                    self.ft8_log.config(state=tk.DISABLED)
                    # CQ QSO assist: route raw message + SNR to the prefill
                    # watcher when a session is active.  Fields are present
                    # when emitted by _on_ft8_decode (len == 4).
                    if len(item) >= 4 and self._qso_assist_active:
                        self._maybe_assist_prefill(item[2], item[3])

                elif kind == "tx_state":
                    _, state, message = item
                    self._apply_tx_state_update(state, message)

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
        last_mode      = None
        last_pwr       = None
        last_status_ts = 0.0
        last_s_ts      = 0.0
        last_f_ts      = 0.0
        last_slow_ts   = 0.0

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
        with open("radio_log.csv", "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, f"{freq:.4f}", strength, notes])
        self._ui_queue.put(("log", freq, strength, notes, timestamp))

    def manual_log(self):
        self.log_to_file(
            self.radio.get_frequency(),
            self.radio.get_s_meter(),
            self.note_entry.get(),
        )

    def infer_band_from_freq(self, mhz: float):
        """Return band name if mhz is inside a defined band plan, else None."""
        for name, plan in BANDS.items():
            if plan["start"] <= mhz <= plan["end"]:
                return name
        return None

    def _goto_ft8_freq(self) -> None:
        """Jump to the FT8 calling frequency for the currently active band."""
        band = self.active_band
        if not band:
            if self.radio.is_connected():
                f    = self.radio.get_frequency()
                band = self.infer_band_from_freq(f)
        if not band or band not in FT8_FREQS:
            self.conn_status.config(
                text="Status: select a band before jumping to FT8 frequency"
            )
            return
        freq = FT8_FREQS[band]
        def _worker():
            self.radio.set_mode("USB")
            time.sleep(0.05)
            self.radio.set_frequency(freq)
            self._ui_queue.put(("mode", "USB"))
            self._ui_queue.put(("status", f"Status: FT8 {band} {freq:.3f} MHz"))
        if self.radio.is_connected():
            threading.Thread(target=_worker, daemon=True).start()

    def goto_band(self, band_name: str):
        """Set radio to band start + mode, and make it the active scan band."""
        if band_name not in BANDS:
            return

        self.active_band = band_name
        plan = BANDS[band_name]

        if not self.radio.is_connected():
            self.conn_status.config(
                text=f"Status: DISCONNECTED (selected {band_name})"
            )
            return

        def worker():
            try:
                target = float(plan["start"])
                mode   = plan["mode"]

                # 1) Mode first (some rigs behave better this way)
                self.radio.set_mode(mode)
                time.sleep(0.05)

                # 2) Set frequency
                self.radio.set_frequency(target)
                time.sleep(0.05)

                # 3) Verify; if we are off by > ~50 Hz, retry once
                actual = self.radio.get_frequency()
                if abs(actual - target) > 0.00005:
                    self.radio.set_frequency(target)
                    time.sleep(0.05)

                # Update step size to match band default
                self._freq_step = plan["step"]
                self._ui_queue.put(("mode", mode))
                self._ui_queue.put(
                    ("status", f"Status: CONNECTED ({self.radio.port} @ {self.radio.baud}) [{band_name}]")
                )
            except Exception as e:
                self._ui_queue.put(("status", f"Status: ERROR (band set failed: {e})"))

        threading.Thread(target=worker, daemon=True).start()

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

            try:
                thresh = int(self.thresh_entry.get())
            except (TypeError, ValueError):
                thresh = 40
                self._ui_queue.put(("status", "Status: SCAN (invalid squelch; default 40)"))

            self._ui_queue.put(("ptt_state", False))

            self.scanning = True
            self.scan_btn.config(text="STOP SCAN", bg="orange")

            band = self.active_band
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
        end   = float(plan["end"])
        step  = float(plan["step"])

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

                dwell_until = time.monotonic() + 3.0
                while self.scanning and not self._shutdown.is_set() and time.monotonic() < dwell_until:
                    time.sleep(0.05)

            curr_f += step

        self.scanning = False
        self._ui_queue.put(("ptt_state", True))

    def open_settings(self) -> None:
        """Open the settings modal dialog (serial port + audio devices)."""
        try:
            import serial
            sb_map = {
                serial.STOPBITS_ONE:            1,
                serial.STOPBITS_ONE_POINT_FIVE: 1.5,
                serial.STOPBITS_TWO:            2,
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
            current_audio_in_label=self._config.audio_device_label,
            current_audio_out_label=self._config.audio_output_device_label,
            current_tx_mic_label=self._config.tx_mic_device_label,
            current_tx_radio_out_label=self._config.tx_radio_out_device_label,
            on_apply=self._apply_settings,
        )

    def _apply_settings(
        self,
        port: str,
        baud: int,
        stopbits: float,
        audio_in_idx: int,
        audio_in_label: str,
        audio_out_idx: int,
        audio_out_label: str,
        tx_mic_idx: int,
        tx_mic_label: str,
        tx_radio_out_idx: int,
        tx_radio_out_label: str,
    ) -> None:
        """
        Apply new settings from the SettingsDialog.

        Handles serial port reconnection and persists audio device selections.
        """
        was_connected = self.radio.is_connected()

        if was_connected:
            self.scanning = False
            self.radio.disconnect()

        # Update serial parameters
        self.radio.port     = port
        self.radio.baud     = baud
        self.radio._stopbits = stopbits

        try:
            import serial
            _sb_map = {
                1:   serial.STOPBITS_ONE,
                1.5: serial.STOPBITS_ONE_POINT_FIVE,
                2:   serial.STOPBITS_TWO,
            }
            self.radio._stopbits_serial = _sb_map.get(stopbits, serial.STOPBITS_ONE)
        except Exception:
            self.radio._stopbits_serial = None

        self._config.save_serial(port, baud, stopbits)

        # Update audio input device (RX: radio → computer)
        if audio_in_idx >= 0:
            self.audio_device_index = audio_in_idx
            self._config.save_audio(audio_in_idx, audio_in_label)
            self.audio_status.config(
                text=f"Audio: device {audio_in_idx} configured -- click Start Audio"
            )

        # Update audio output device (RX monitor: computer → headphones)
        if audio_out_idx >= 0:
            self.audio_output_device_index = audio_out_idx
            self._config.save_audio_output(audio_out_idx, audio_out_label)

        # Update TX audio devices
        tx_changed = False
        if tx_mic_idx >= 0:
            self.tx_mic_device_index = tx_mic_idx
            tx_changed = True
        if tx_radio_out_idx >= 0:
            self.tx_radio_out_device_index = tx_radio_out_idx
            tx_changed = True
        if tx_changed:
            self._config.save_tx_audio(
                tx_mic_idx, tx_mic_label,
                tx_radio_out_idx, tx_radio_out_label,
            )
            self.voice_audio_status.config(
                text="Voice Audio: TX devices updated — re-start RX Monitor if active"
            )

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
            self.conn_status.config(
                text=f"Status: CONNECTED ({self.radio.port} @ {self.radio.baud}) [{inferred}]"
            )

        p = self.radio.get_rf_power()
        self.rf_power_var.set(p)
        self.rf_power_status.config(text=f"READ {p:03d}")
        m = self.radio.get_mode()
        if m:
            self.mode_var.set(m)
            self.mode_status.config(text=f"READ {m}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s [%(name)s] %(message)s",
    )

    # Load persisted settings from vader.cfg (created automatically on first save).
    _cfg = AppConfig()

    # Determine the startup port: prefer the saved value; fall back to the first
    # detected port so the app is functional out-of-the-box for new testers.
    _port = _cfg.port
    if not _port:
        _detected = _enum_serial_ports()
        _port = _detected[0] if _detected else "COM1"

    radio = Yaesu991AControl(port=_port, baud=_cfg.baud, stopbits=_cfg.stopbits)
    root  = tk.Tk()
    gui   = RadioGUI(root, radio, config=_cfg)
    root.mainloop()
