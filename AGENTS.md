# AGENTS.md

## Big picture
- `main.py` is the orchestration layer: Tkinter UI, config persistence, worker-thread startup, and mode switching. Most feature wiring lands here, not in the lower-level modules.
- The project has three main boundaries:
  - CAT control in `ft991a_cat.py` (`Yaesu991AControl`) for FT-991A serial commands.
  - FT8 DSP in `digi_input.py` → `ft8_decode.py` → `ft8_qso.py` / `ft8_tx.py`.
  - Voice audio routing in `audio_passthrough.py` (`AudioPassthrough`, `AudioTxCapture`).
- `audio_bridge.py` is an alternative full-duplex abstraction, but `main.py` currently uses `audio_passthrough.py` directly for live voice mode. Treat `audio_passthrough.py` as the active integration point unless you are intentionally refactoring.
- `archive/` is historical reference only; active code and tests exclude it (`pytest.ini`).

## Runtime/data flow
- Voice mode: `RadioGUI._start_rx_monitor()` routes radio RX audio to speakers; PTT press/release starts/stops `AudioTxCapture` and calls `radio.ptt_on()` / `radio.ptt_off()`.
- Data mode: `SoundCardAudioSource.chunks()` feeds `RadioGUI._audio_worker()`, which pushes chunks into a long-lived `FT8ConsoleDecoder` via `feed(fs=..., samples=..., t0_monotonic=...)`.
- Successful FT8 decodes call `RadioGUI._on_ft8_decode()`, which formats lines with `format_ft8_message()` and appends them to the GUI log; the on-screen FT8 log is flushed to `ft8_messages.log` when switching back to voice mode or closing the app.
- FT8 TX is manual-assisted, not fully automatic: the GUI builds a `TxJob`, `Ft8TxCoordinator.arm()` schedules the next 15 s slot, keys CAT PTT, plays generated tones, and always unkeys in `finally`.
- Slot timing is intentionally separated into `ft8_ntp.py`; `default_slot_timer` is shared, but it is **not** auto-synced at import time.

## Project conventions that matter
- Keep GUI work on the Tk thread. Background threads communicate through `RadioGUI._ui_queue`, drained by `process_ui_queue()` every 50 ms. Follow this pattern instead of touching widgets from worker threads.
- The radio polling cadence is deliberate: frequency ~250 ms, S-meter ~200 ms, mode/RF power ~1.5 s (`RadioGUI.radio_poll_thread()`) to reduce CAT contention.
- `AppConfig` in `main.py` is the source of truth for persisted settings in `vader.cfg`; new persisted settings should follow its “safe defaults + property helpers + save_* methods” pattern.
- Device selections are stored as both index and human label in `vader.cfg`; GUI code expects `-1` to mean “not configured”.
- Windows audio behavior matters: several paths prefer WASAPI devices, and `ft8_tx.py` contains Windows-specific WASAPI/MME fallback logic for PortAudio host errors.
- CAT methods in `ft991a_cat.py` are thin command wrappers. Preserve the existing pattern: validate/clamp inputs, call `_execute("CMD...")`, and parse exact Yaesu response shapes.
- FT8 message composition/validation belongs in `ft8_qso.py` / `ft8_tx.py`, not in the GUI. `main.py` should mostly validate, prefill, and dispatch.

## TX legal and safety policy (non-optional)
- Keep `README.md` section `TX Safety & Compliance` aligned with this policy text to prevent drift.
- **No unattended or autonomous transmissions.** AI agents must not implement, enable, or test unattended TX loops. TX must remain operator-initiated with a human control operator in the loop.
- **Human pre-transmit approval is required** before any live/bench TX check. Agent output must include a user confirmation step that the station is configured for safe testing.
- **Safe RF setup is required for live TX validation:** minimum practical power and/or a dummy load (or equivalent non-radiating setup) to avoid unintended on-air transmission during development tests.
- **Real-time abort path is required** for any TX workflow. The user must be able to stop transmission immediately (for example: UI cancel/PTT release/CAT unkey).
- **If a feature would increase automation of TX behavior, stop and ask first.** Do not proceed without explicit user approval of the control-operator and safety implications.

## Developer workflows
- Main app: `python main.py`
- Live decoder without rig: `python live_test.py --list` or `python live_test.py --device <index> --fs 48000`
- Offline decoder: `python -c "from ft8_decode import decode_wav; decode_wav('live_ft8_audio_traffic.wav')"`
- Test suite: `pytest -v`
- High-value focused tests while editing:
  - `pytest test_main_gui_mode.py -v`
  - `pytest test_ft8_tx.py -v`
  - `pytest test_ft991a_cat.py -v`
  - `pytest test_audio_passthrough.py -v`
  - `pytest test_ft8_ntp.py -v`

## Testing/debugging patterns
- Treat tests as two explicit tiers:
  - **Tier 1 (automated, default):** `pytest` tests with stubs/mocks only; no physical radio/audio hardware required.
  - **Tier 2 (user-run, live hardware):** manual validation scripts/checks run only by the user in a supervised session.
- Tests are designed to run without hardware: serial, audio, Tkinter, and NTP are commonly stubbed or mocked. Match that style when adding coverage.
- **Any change touching TX paths must include regression coverage in automated tests** (new tests or updates), especially around `ft8_tx.py`, CAT PTT control, TX audio routing, and error/unkey safety behavior.
- Local testing on physical hardware can only be done by the user, so make sure to run the full test suite with stubs before proposing live validation for hardware integration changes.
- Any script that may exercise live hardware should be clearly named (for example `live_*.py` or `hardware_*.py`) and kept out of normal `pytest` discovery/CI paths.
- Hardware/live checks must never be a required gate for CI pass; they are a post-test user validation step.
- `test_main_gui_mode.py` shows how to import `main.py` safely with Tk/audio/serial stubs.
- `test_ft991a_cat.py` asserts exact CAT command strings; if you change command formatting, update tests carefully.
- `FT8ConsoleDecoder` is chatty by default (`_debug = True`) and prints sync/decode progress to stdout; use that before adding new logging.
- `radio_log.csv` is only for the manual signal log; FT8 decodes persist separately in `ft8_messages.log`.

## Agent handoff requirements for TX-related work
- Report exactly which automated tests were added/updated for TX behavior and which focused suites were run.
- Provide a short, explicit live-validation checklist for the user (pre-transmit approval, safe RF setup, abort path, expected result).
- Do not claim live TX validation was performed unless the user confirms they ran it.
