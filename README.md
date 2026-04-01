# VaDER — Voice and Data Enhanced Radio

> **73 de W8TSB** — a learning project bringing Python-driven radio control and FT8 digital-mode decoding to the Yaesu FT-991A.

---

## Table of Contents

- [What is VaDER?](#what-is-vader)
- [Current State at a Glance](#current-state-at-a-glance)
- [Feature Overview](#feature-overview)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Module Descriptions](#module-descriptions)
- [Running Tests](#running-tests)
- [Known Limitations](#known-limitations)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License / Notes](#license--notes)

---

## What is VaDER?

VaDER is a Python desktop application for controlling a **Yaesu FT-991A** transceiver over USB/Serial (CAT), monitoring radio telemetry in real time, and decoding live **FT8 digital-mode signals** — all from a single GUI window.

The project started as a hands-on way to learn Python and explore the capabilities (and quirks) of Yaesu's ASCII CAT protocol. It has since grown into a working radio-control dashboard with a genuine FT8 decoder pipeline built from scratch in pure Python/NumPy, compatible with the WSJT-X FT8 specification.

The project is currently **hardware-specific** (Yaesu FT-991A), but the CAT layer is designed to be extensible as other radios are added in the future.

---

## Current State at a Glance

| Area | Status |
|------|--------|
| FT-991A CAT control (70+ commands) | ✅ Working |
| Real-time frequency / S-meter / RF power display | ✅ Working |
| Radio mode selection (LSB/USB/CW/FM/AM/C4FM) | ✅ Working |
| Band quick-select (160 m – 70 cm, 12 bands) | ✅ Working |
| PTT via CAT | ✅ Working |
| Voice RX monitor (radio → speakers) | ✅ Working |
| Voice TX capture (mic → radio) | ✅ Working |
| Live FT8 decoding from soundcard | ✅ Working |
| FT8 message log (on-screen + file) | ✅ Working |
| Settings persistence (vader.cfg) | ✅ Working |
| Automated test suite (pytest) | ✅ Working |
| FT8 TX tone generation | 🔲 Not yet implemented |
| Full FT8 QSO automation | 🔲 Not yet implemented |
| Contact / QSO logging to file | 🔲 Partial (UI exists, no file save) |
| Support for radios other than FT-991A | 🔲 Planned |

---

## Feature Overview

### CAT Radio Control
VaDER communicates with the FT-991A using Yaesu's ASCII CAT protocol over USB/serial. The CAT library (`ft991a_cat.py`) implements **70+ commands**, including:

- **Frequency**: set/get VFO A & B, step tuning (10 Hz – 100 kHz), band selection
- **Mode**: LSB, USB, CW, FM, AM, RTTY, C4FM
- **Metering**: S-meter (0–255), RF power output (%), SWR
- **PTT**: TX on/off via CAT
- **VFO operations**: A↔B copy, memory access, channel up/down
- **Audio**: AF gain, key pitch
- **Filters**: Manual notch, auto notch, IF shift, contour, AGC
- **CW**: Keyer on/off, key speed, CW spot, keyer memories, direct CW send
- **CTCSS/DCS**: Tone mode and number selection
- **Antenna tuner**: Enable/disable
- **Date/time/timezone**: Radio clock sync
- **Menu access**: Direct menu item read/write
- **Misc**: Lock, dimmer, break-in, TX VFO select, auto-information mode

### Real-Time GUI Dashboard
The main window (`main.py`) provides:
- Large frequency display (36 pt) with fine-tune step buttons
- S-meter progress bar updated every ~250 ms
- RF power spinbox with live apply
- Dropdown for operating mode (MD command)
- Quick-select buttons for 12 amateur bands + FT8 sub-band jump
- Band scanner (up/down + squelch)
- Operating mode toggle: **VOICE** vs **DATA/FT8**

### Voice Audio (VOICE mode)
- **RX monitoring**: Routes radio audio line → computer speakers in real time
- **TX capture**: Routes computer microphone → radio audio input when PTT is pressed
- Both paths use non-blocking queued audio (sounddevice) with live RMS level metering
- Audio device selection is saved to `vader.cfg`

### FT8 Decoder (DATA mode)
VaDER includes a clean-room Python implementation of the WSJT-X FT8 decoder:

1. Resample incoming audio to 12 kHz mono
2. Align to UTC 15-second transmission slots
3. Detect FT8 candidates via incoherent 8-tone matched filter
4. Grid-search Costas array sync (time and frequency offset)
5. Extract 79 symbols × 8-tone energy matrix
6. Gray decode 58 data symbols → 174 channel LLRs
7. De-interleave LLRs via bit-reversal permutation
8. LDPC belief-propagation decode (174,91 code, up to 200 iterations)
9. CRC-14 verification (poly 0x2757)
10. Unpack 77-bit payload → callsign / grid / SNR / free-text

Decoded messages are displayed live in the FT8 log panel (with UTC, SNR, and frequency offset) and appended to `ft8_messages.log`.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      main.py  (GUI)                         │
│   tkinter window: frequency · S-meter · RF power · FT8 log │
└──────┬─────────────────────────┬───────────────────────┬───┘
       │                         │                       │
       ▼                         ▼                       ▼
ft991a_cat.py            ft8_decode.py          audio_passthrough.py
Yaesu ASCII CAT          FT8 decoder            Voice audio routing
(serial/USB)             (NumPy + SciPy)        (sounddevice)
                               ▲
                               │
                         digi_input.py
                         Soundcard capture
                         (SoundCardAudioSource)
```

**Data flow**

1. User clicks **Connect** → `Yaesu991AControl.connect()` opens serial port.
2. A polling thread queries frequency, S-meter, RF power, and mode every ~250 ms and updates the GUI.
3. In **VOICE** mode, `AudioPassthrough` streams radio audio to speakers; when PTT is pressed, `AudioTxCapture` routes mic audio to the radio.
4. In **DATA** mode, `SoundCardAudioSource` feeds raw audio to `FT8ConsoleDecoder`, which processes each UTC-aligned 15-second slot and fires a callback for each decoded message.
5. Decoded messages are added to the on-screen FT8 log and appended to `ft8_messages.log`.

---

## Prerequisites

| Requirement | Version |
|-------------|---------|
| Python | 3.9 or newer |
| numpy | ≥ 1.20 |
| scipy | ≥ 1.5 |
| pyserial | ≥ 3.5 |
| sounddevice | ≥ 0.4.5 |
| Yaesu FT-991A | (or future supported radio) |
| USB CAT cable | Standard USB-A to USB-B (or built-in USB on FT-991A) |

On Windows, VaDER uses WASAPI for audio device enumeration. Linux and macOS should work but are less tested.

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/Tritiumosu/VaDER.git
cd VaDER

# 2. (Optional but recommended) create a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 3. Install dependencies
pip install numpy scipy pyserial sounddevice

# 4. Run VaDER
python main.py
```

---

## Quick Start

1. **Connect your FT-991A** via USB, then launch `python main.py`.
2. Click **Settings** and select your serial port (e.g. `COM3` on Windows or `/dev/ttyUSB0` on Linux), baud rate (default **38400**), and audio devices.
3. Click **Connect** — the frequency display and S-meter should start updating within a second.
4. Toggle **VOICE** mode to hear the radio through your speakers and use PTT.
5. Toggle **DATA** mode and click **Start Audio** to begin live FT8 decoding. Decoded messages appear in the FT8 log panel and are saved to `ft8_messages.log`.
6. Use the **Band** buttons or frequency step arrows to navigate between amateur bands.
7. Click **Disconnect** when done.

### Console-only FT8 decoder (no hardware required)

```bash
# List available audio devices
python live_test.py --list

# Decode from a live soundcard input (replace 3 with your device index)
python live_test.py --device 3 --fs 48000

# Decode an offline WAV file
python -c "from ft8_decode import decode_wav; decode_wav('live_ft8_audio_traffic.wav')"
```

---

## Module Descriptions

| File | Purpose |
|------|---------|
| `main.py` | Main tkinter GUI — radio dashboard, FT8 log, settings |
| `ft991a_cat.py` | Yaesu FT-991A ASCII CAT library (70+ serial commands) |
| `ft8_decode.py` | Full FT8 decoder: resample → sync → LDPC → CRC → unpack |
| `audio_passthrough.py` | Real-time voice audio routing (RX monitor, TX capture) |
| `digi_input.py` | Soundcard capture front-end for the FT8 decoder |
| `live_test.py` | Console FT8 decoder — useful without a radio attached |
| `diag.py` | Internal diagnostics: CRC, Gray, interleave, LDPC self-tests |
| `gen_ldpc_matrix.py` | Fetches authoritative (174,91) LDPC matrix from ft8_lib |
| `debug_73.py` | Debug helper for FT8 message-unpacking edge cases |
| `debug_offline.py` | Offline debug for gray decode / de-interleave / LDPC |

---

## Running Tests

VaDER has a growing pytest suite covering the FT8 pipeline, CAT command encoding, audio routing, and GUI modes:

```bash
# Run all tests
pytest test_*.py -v

# Run individual test modules
pytest test_ft991a_cat.py -v        # CAT command encoding (no hardware needed)
pytest test_ldpc_pipeline.py -v     # LDPC encode → decode round-trip
pytest test_ft8_decode_output.py -v # Message formatting against live WAV files
pytest test_msg_unpack.py -v        # Callsign / grid / free-text unpacking
pytest test_audio_passthrough.py -v # Audio routing with mocked sounddevice
pytest test_main_gui_mode.py -v     # GUI operating modes and config persistence
```

All tests run without physical hardware; serial and audio I/O are mocked.

---

## Known Limitations

- **FT8 TX not implemented** — VaDER can decode FT8 but cannot yet generate or transmit FT8 tones. A full QSO would require tone generation and sequenced timing.
- **QSO/contact logging is incomplete** — The log UI panel exists but does not yet save contacts to an ADIF or other file format.
- **PTT is voice-only useful** — CAT PTT works, but without FT8 TX, it is only useful for voice SSB/FM operation.
- **Yaesu FT-991A only** — The CAT library is specific to this radio model. Other radios are a future goal.
- **Windows-first audio** — WASAPI device enumeration is used in the GUI; Linux/macOS audio device discovery is less polished.
- **No rig database / memory management** — Memories and other menu-level settings are accessible via CAT but are not presented in the GUI yet.
- **I am not a professional developer** — This project makes heavy use of AI assistance, and the code reflects an active learning process. Bugs and rough edges are expected.

---

## Roadmap

Progress is tracked here as features move from planned to implemented. Items are loosely grouped by milestone.

### ✅ Milestone 1 — Core CAT Control & Basic GUI (Complete)
- [x] Yaesu FT-991A serial CAT library (frequency, mode, PTT, S-meter, RF power)
- [x] Persistent settings (vader.cfg) for port, baud rate, and audio devices
- [x] Real-time frequency and S-meter display (~250 ms polling)
- [x] RF power control and radio mode (MD) selection
- [x] 12-band quick-select and band scanner
- [x] PTT button via CAT

### ✅ Milestone 2 — FT8 Decoder (Complete)
- [x] Clean-room Python FT8 decoder (Costas sync, Gray decode, LDPC BP, CRC-14)
- [x] UTC-aligned 15-second frame buffering
- [x] Live FT8 message log in GUI (SNR, frequency offset, timestamp)
- [x] Offline WAV-file decode support
- [x] FT8 message log persisted to `ft8_messages.log`
- [x] Console-mode decoder (`live_test.py`) for testing without a radio

### ✅ Milestone 3 — Voice Audio Passthrough (Complete)
- [x] RX monitor: radio audio line → computer speakers
- [x] TX capture: computer microphone → radio audio input
- [x] RMS level metering in GUI
- [x] Audio device selection with persistence

### 🔄 Milestone 4 — FT8 Transmit & Basic QSO (In Progress / Next Up)
- [ ] FT8 tone generation (8-FSK, 6.25 Hz spacing, 160 ms symbols)
- [ ] TX sequencing: auto-time transmission to 15-second UTC slots
- [ ] PTT integration for DATA mode (CAT PTT triggered around FT8 TX window)
- [ ] Minimal FT8 QSO exchange: CQ / reply / RRR / 73
- [ ] SNR / grid reporting in outbound FT8 messages

### 🗺️ Milestone 5 — QSO Logging
- [ ] Save completed QSOs to ADIF file format
- [ ] In-session log view (callsign, band, mode, time, RST)
- [ ] Optional integration with Hamlog / LOTW / QRZ for lookups

### 🗺️ Milestone 6 — Multi-Radio Support
- [ ] Abstract CAT interface (base class) for multiple radio models
- [ ] Icom IC-7300 support (CI-V protocol)
- [ ] Kenwood TS-590S / TS-890S support (ASCII CAT, similar to Yaesu)
- [ ] Hamlib back-end option as universal fallback

### 🗺️ Milestone 7 — UI / UX Modernization
- [ ] Migrate from raw tkinter to a modern UI framework (e.g. CustomTkinter or PyQt6)
- [ ] Dark/light theme support
- [ ] Resizable layout and proper high-DPI handling
- [ ] Waterfall display (spectrum / panadapter view from audio)
- [ ] Accessible keyboard shortcuts and screen-reader support

### 🗺️ Milestone 8 — Advanced Digital Modes
- [ ] JS8Call / JS8 decode
- [ ] WSPR decode (2-minute slots, 4-FSK)
- [ ] PSK31 / PSK63 decode
- [ ] RTTY decode / encode
- [ ] Plugin architecture for adding new modes without modifying core

### 💡 Aspirational Goals (Long Term)
- **Cross-platform packaging**: one-click installer for Windows, macOS, and Linux (PyInstaller / cx_Freeze)
- **Remote operation**: web interface for headless/remote radio control (WebSocket or HTTP API)
- **Cluster / DX spotting integration**: pull DX spots and click-to-QSY directly in the GUI
- **AI-assisted CW decode**: real-time Morse code recognition using an ML model
- **Contest mode**: rapid exchange logging, dupe checking, and score tracking
- **SDR back-end support**: integrate RTL-SDR / HackRF for panadapter and decode without a transceiver

---

## Contributing

This is a personal learning project and contributions are welcome in the spirit of Elmer-style knowledge sharing.

- **Bug reports**: Please open a GitHub Issue with the serial command output (if CAT-related) or a description of the audio problem.
- **Pull requests**: Keep changes focused; follow the existing code style (no mandatory formatter yet). Add or update `test_*.py` tests when possible.
- **Feature ideas**: Open an Issue tagged `enhancement` — especially for supporting new radio models or digital modes.
- **Elmers / experienced devs**: If something in the code is embarrassingly wrong, please say so kindly. This is a learning project. 😄

---

## License / Notes

No formal license has been applied yet. The code is shared openly on GitHub for educational and amateur-radio purposes. Please do not use it for commercial purposes without asking first.

The FT8 decoder implementation is a clean-room Python re-implementation of the algorithm described in the WSJT-X source code and associated publications. WSJT-X is copyright © Joe Taylor K1JT and the WSJT Development Group.

**73 de W8TSB** 📻
