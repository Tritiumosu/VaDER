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
- [TX Safety & Compliance](#tx-safety--compliance)
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
| FT8 TX tone generation | ✅ Working (`ft8_encode.py`) |
| FT8 message packing & LDPC encode | ✅ Working |
| FT8 QSO state machine & message composition | ✅ Working (`ft8_qso.py`) |
| Operator callsign/grid in settings | ✅ Working (`vader.cfg [operator]`) |
| NTP-disciplined TX slot timing | ✅ Working (`ft8_ntp.py`) |
| Manual-assisted FT8 TX orchestration | ✅ Working (`ft8_tx.py`) |
| GUI TX panel (arm/cancel/status/countdown) | ✅ Working (`main.py`) |
| Select-and-reply from decoded messages | ✅ Working (`main.py`) |
| CAT PTT key/unkey around FT8 TX window | ✅ Working (`ft8_tx.py`) |
| CQ-initiated QSO assist (auto-prefill, no auto-TX) | ✅ Working (`main.py`, `ft8_qso.py`) |
| Unattended full QSO automation | 🚫 Not permitted by project safety policy |
| Contact / QSO logging to file | 🔲 Partial (`QsoRecord` struct ready, no file write yet) |
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
| Python | 3.10 or newer |
| numpy | ≥ 1.20 |
| scipy | ≥ 1.5 |
| pyserial | ≥ 3.5 |
| sounddevice | ≥ 0.4.5 |
| ntplib | ≥ 0.4.0 |
| Yaesu FT-991A | (or future supported radio) |
| USB CAT cable | Standard USB-A to USB-B (or built-in USB on FT-991A) |

On Windows, VaDER now enumerates both WASAPI and MME devices (WASAPI first, MME fallback). Linux and macOS should work but are untested.

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
pip install numpy scipy pyserial sounddevice ntplib

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

### Live TX helper scripts (user-run, supervised)

```bash
# Preflight checks only (CAT + TX audio stream open), no RF transmission
python live_tx_preflight.py

# One-shot manual-gated FT8 TX helper (requires typed confirmation)
python live_ft8_single_tx.py
```

On some Windows USB audio codecs, WASAPI/WDM-KS stream startup can fail with
`PaErrorCode -9999` / `WdmSyncIoctl ... GLE=0x00000490`. If that occurs, select
the matching MME TX output endpoint in Settings or set `[tx_audio]`
`radio_out_device_index` to the MME device in `vader.cfg`.

---

## TX Safety & Compliance

VaDER supports **manual-assisted TX only**. Contributors and users must follow these non-optional safety and legal boundaries:

This section intentionally mirrors the TX policy in `AGENTS.md`; update both documents together.

- **No unattended or autonomous transmissions**: TX must remain operator-initiated with a human control operator in the loop.
- **Human pre-transmit approval is required**: before any live/bench TX check, confirm the station is configured for safe testing.
- **Safe RF setup is required for live TX validation**: minimum practical power and/or a dummy load (or equivalent non-radiating setup) to avoid unintended on-air transmission during development tests.
- **Real-time abort path is required**: maintain an immediate stop path (for example **Cancel TX**, PTT release, or CAT unkey).
- **If a feature would increase automation of TX behavior, stop and ask first**: discuss and approve control-operator and safety implications before implementation.

---

## Module Descriptions

| File | Purpose |
|------|---------|
| `main.py` | Main tkinter GUI — radio dashboard, FT8 log, settings |
| `ft991a_cat.py` | Yaesu FT-991A ASCII CAT library (70+ serial commands) |
| `ft8_decode.py` | Full FT8 decoder: resample → sync → LDPC → CRC → unpack |
| `ft8_encode.py` | FT8 encoder: pack → CRC → LDPC → tones → audio synthesis |
| `ft8_qso.py` | QSO state machine, message composition, received-message parser |
| `ft8_ntp.py` | NTP-disciplined clock: multi-server sync, offset caching, `Ft8SlotTimer` |
| `audio_passthrough.py` | Real-time voice audio routing (RX monitor, TX capture) |
| `digi_input.py` | Soundcard capture front-end for the FT8 decoder |
| `ft8_tx.py` | TX orchestration: state machine, slot timing, PTT, audio dispatch |
| `audio_bridge.py` | Full-duplex audio bridge (RX monitor → headphones, mic → radio TX) |
| `live_test.py` | Console FT8 decoder — useful without a radio attached |
| `offline_test.py` | Offline WAV-file decoder — processes a recording through the full pipeline |
| `diag.py` | Internal diagnostics: CRC, Gray, interleave, LDPC self-tests |
| `gen_ldpc_matrix.py` | Utility: fetch authoritative (174,91) LDPC matrix from ft8_lib on GitHub |

> **`archive/`** — Historical development artefacts kept for reference but not part of the active codebase. `archive/debug_sessions/` contains the 27 diagnostic scripts written during the Milestone 2 LDPC decoder debugging sessions. `archive/ldpc_matrix.py` is the matrix snapshot generated by `gen_ldpc_matrix.py` (data is embedded in `ft8_decode.py`). `archive/LDPC_FIX_SUMMARY.md` documents the decoder investigation history.

---

## Running Tests

VaDER uses a two-tier validation model:

- **Tier 1 (automated, default)**: `pytest` with stubs/mocks only (no physical radio/audio hardware required).
- **Tier 2 (user-run live validation)**: supervised hardware checks run manually by the user only.

VaDER's automated suite covers FT8 pipeline behavior, CAT command encoding, audio routing, NTP timing, and GUI modes:

```bash
# Run all tests (cross-shell friendly; uses pytest discovery)
pytest -v

# Run individual test modules
pytest test_ft991a_cat.py -v        # CAT command encoding (no hardware needed)
pytest test_ldpc_pipeline.py -v     # LDPC encode → decode round-trip
pytest test_ft8_decode_output.py -v # Message formatting against live WAV files
pytest test_msg_unpack.py -v        # Callsign / grid / free-text unpacking
pytest test_audio_passthrough.py -v # Audio routing with mocked sounddevice
pytest test_audio_bridge.py -v      # Full-duplex audio bridge with mocked sounddevice
pytest test_main_gui_mode.py -v     # GUI operating modes, config persistence, CQ assist
pytest test_ft8_encode.py -v        # FT8 encoder, QSO state machine, round-trip
pytest test_ft8_qso_assist.py -v    # CQ-initiated QSO assist: parser, state machine, QsoRecord
pytest test_ft8_ntp.py -v           # NTP sync, slot math, AppConfig NTP settings
pytest test_ft8_tx.py -v            # TX orchestration: state machine, PTT, slot timing
pytest test_ft8_reference.py -v     # Reference verification against ft8_lib constants
pytest test_pipeline_e2e.py -v      # Full encode → Gray → LDPC → unpack round-trip
pytest test_symbol_pipeline.py -v   # Symbol extraction and Gray decode pipeline
pytest test_utc_framer.py -v        # UTC 15-second slot boundary alignment
```

All automated tests run without physical hardware; serial, audio I/O, and NTP network calls are mocked.

If you touch TX behavior (`ft8_tx.py`, CAT PTT control, TX audio routing, TX UI flow), add or update automated regression tests in the same change.

Any script that may exercise live hardware should be clearly named (for example `live_*.py` or `hardware_*.py`) and kept out of normal `pytest` discovery/CI paths.

Live hardware scripts/checks are post-test validation, not a CI gate.

---

## Known Limitations

- **Unattended full QSO automation is intentionally not implemented** — By project policy, TX remains manual-assisted and operator-controlled.
- **QSO/contact logging is incomplete** — `QsoRecord` (in `ft8_qso.py`) captures completed QSO data, but writing to ADIF or another log format is deferred to Milestone 5.
- **Yaesu FT-991A only** — The CAT library is specific to this radio model. Other radios are a future goal.
- **Windows-first audio** — GUI enumeration and TX fallback paths are tuned for Windows host APIs (WASAPI first, MME fallback); Linux/macOS audio discovery is less polished.
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

### 🔄 Milestone 4 — FT8 Transmit & Basic QSO (In Progress)
- [x] FT8 message packing (callsign, grid, SNR, special tokens → 77 bits)
- [x] CRC-14 generation (polynomial 0x2757, matching WSJT-X / ft8_lib)
- [x] LDPC (174, 91) encoding via GF(2) matrix factorisation
- [x] Gray-coded 8-FSK tone symbol generation (58 payload + 21 Costas sync)
- [x] Phase-continuous audio synthesis with configurable base frequency and sample rate
- [x] Operator callsign and grid persistence in `vader.cfg` (`[operator]` section)
- [x] Standard QSO message composition helpers (CQ, reply, exchange, RRR, RR73, 73)
- [x] FT8 QSO state machine (`Ft8QsoManager`) — CQ and reply-to-CQ workflows
- [x] Decoded-message parser (`ReceivedMessage`) for classifying received FT8 lines
- [x] Full encode → audio → decode round-trip validation (WSJT-X compatible)
- [x] NTP-disciplined slot timer (`ft8_ntp.py`) — queries NIST and other public NTP servers, caches clock offset, `Ft8SlotTimer` exposes corrected `seconds_to_next_slot()` / `current_slot_parity()` / `next_slot_utc()`
- [x] NTP server list configurable in `vader.cfg` (`[ntp]` section), with graceful fallback to system clock
- [x] `Ft8QsoManager` wired to `Ft8SlotTimer` for accurate TX scheduling
- [x] **Manual-assisted TX orchestration** (`ft8_tx.py`) — `Ft8TxCoordinator` with IDLE → ARMED → TX_PREP → TX_ACTIVE → COMPLETE / ERROR / CANCELED state machine; PTT always unkeyed in `finally`; missed-slot detection; cancel before slot start
- [x] **GUI TX panel** (DATA/FT8 mode) — operator callsign/grid entry + Save, TX message field, CQ quick-fill, Arm TX / Cancel TX buttons, live countdown to next slot, colour-coded TX state indicator
- [x] **Select-and-reply assist** — double-click any decoded FT8 line to pre-populate the TX message with the recommended reply; manual confirmation (Arm TX) still required before any transmission
- [x] **CAT PTT integration** — `Ft8TxCoordinator` keys and unkeys CAT PTT around the FT8 audio window with configurable pre-key and post-key guard intervals
- [x] TX guardrails: blocks arm if CAT disconnected, invalid callsign/grid, empty message, negative audio device, or another TX already active/armed
- [x] **CQ-initiated QSO assist** — operator clicks **▶ Start CQ Session** to begin a tracked exchange; after sending CQ the system watches decoded traffic, recognises replies addressed to the operator, and pre-fills the next standard exchange message; the operator reviews the suggestion then presses **Arm TX** to transmit; the assist watcher continues through the full exchange (CQ → exchange → 73); **never transmits autonomously**
- [x] `QsoRecord` dataclass added to `ft8_qso.py` with ADIF helpers (`adif_date()`, `adif_time()`) — foundation for Milestone 5 contact logging

#### Milestone 4 Usage Notes

**Quick start (DATA mode)**

1. Connect to radio and switch to **DATA / FT8** mode.
2. In the **FT8 Transmit** panel: enter your callsign and grid, then click **Save**.
3. Type a message in the **TX Msg** field, or click **CQ** to auto-fill `CQ MYCALL MYGRID`, or double-click a decoded message to pre-fill a reply.
4. Click **Arm TX (next slot)** — the status label will count down to the next 15-second boundary.
5. The coordinator automatically keys PTT, plays the FT8 tones, then unkeys PTT.
6. Click **Cancel TX** at any time before the slot fires to abort without transmitting.

**CQ Session assist (operator-confirmed)**

1. Enter your callsign and grid, click **Save**.
2. Click **▶ Start CQ Session** — the TX message is filled with `CQ MYCALL MYGRID` and the assist watcher is activated.
3. Click **Arm TX (next slot)** to send the CQ.
4. After the CQ transmits, decoded replies addressed to your callsign are automatically detected and the appropriate exchange response is pre-filled in the TX field.
5. Review the suggested message (edit if needed), then click **Arm TX** again to respond.
6. The watcher continues through the exchange:  CQ → exchange reply → 73 confirmation.
7. Click **■ Stop QSO** at any time to end the session and clear the assist state.

**Guard behaviour (CQ assist)**
- The assist watcher **never calls Arm TX** — operator confirmation is always required.
- Prefill is suppressed while a TX job is ARMED, in TX_PREP, or TX_ACTIVE.
- A duplicate decode of the same message for the current QSO step is silently ignored (dedup guard).
- Switching to VOICE mode automatically stops any active CQ session.
- Messages from stations other than the first valid responder are ignored after lock-in.

**Safety behaviour**
- PTT is always unkeyed in a `finally` block — a Python exception, audio failure, or serial error will not leave the rig stuck in TX.
- If the OS scheduler wakes the TX thread more than 0.5 s late, the job is aborted and reported as `ERROR` rather than starting a late transmission.
- Only one TX job can be armed at a time; arming a second job while one is active raises an error visible in the status label.

**Configuration (`vader.cfg`)**
```ini
[operator]
callsign = W4ABC
grid     = EN52
```
Saved automatically when you click **Save** in the TX panel.

**Module structure**
| Module | Responsibility |
|--------|----------------|
| `ft8_tx.py` | TX orchestration: state machine, slot timing, PTT, audio dispatch |
| `ft8_encode.py` | FT8 tone generation and audio synthesis |
| `ft8_ntp.py` | NTP-corrected slot timer |
| `ft8_qso.py` | Message composition helpers, QSO state machine, `QsoRecord` |
| `main.py` | GUI TX panel, select-and-reply assist, CQ session assist, UI queue dispatch |

### 🗺️ Milestone 5 — QSO Logging
- [x] `QsoRecord` dataclass with ADIF helpers (`adif_date()`, `adif_time()`) — data structure ready
- [ ] Write completed QSOs to ADIF file (`.adi`) on `build_record()` completion
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
- **Pull requests**: Keep changes focused; follow the existing code style (no mandatory formatter yet).
- **TX-related pull requests**: Must include automated regression coverage updates (especially around `ft8_tx.py`, CAT PTT sequencing, TX audio routing, and error/unkey safety).
- **Live hardware validation**: Can only be performed by a human operator. Do not claim live TX validation unless the user confirms they ran it.
- **Safety policy**: Do not implement unattended/autonomous TX workflows.
- **Feature ideas**: Open an Issue tagged `enhancement` — especially for supporting new radio models or digital modes.
- **Elmers / experienced devs**: If something in the code is embarrassingly wrong, please say so kindly. This is a learning project. 😄

### TX Work Handoff (AI-assisted changes)

- Report exactly which automated tests were added/updated for TX behavior and which focused suites were run.
- Provide a short, explicit live-validation checklist for the user (pre-transmit approval, safe RF setup, abort path, expected result).
- Do not claim live TX validation was performed unless the user confirms they ran it.

---

## License / Notes

No formal license has been applied yet. The code is shared openly on GitHub for educational and amateur-radio purposes. Please do not use it for commercial purposes without asking first.

The FT8 decoder implementation is a clean-room Python re-implementation of the algorithm described in the WSJT-X source code and associated publications. WSJT-X is copyright © Joe Taylor K1JT and the WSJT Development Group.

**73 de W8TSB** 📻
