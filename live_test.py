"""
live_test.py — VADER FT8 live decoder console test.

Captures audio from a sound card and runs the full FT8 decode pipeline,
printing every candidate and every successful decode to the console.

Usage:
    python live_test.py                   # auto-select first WASAPI input device
    python live_test.py --device 3        # use device index 3
    python live_test.py --list            # list available input devices and exit
    python live_test.py --device 3 --fs 48000   # override sample rate

The decoder is UTC-aligned: decodes fire at the end of each 15-second FT8 slot
(i.e. at :00, :15, :30, :45 past the minute).

Ctrl-C to stop.
"""
import argparse
import configparser
import os
import sys
import time

from digi_input import SoundCardAudioSource
from ft8_decode import FT8ConsoleDecoder


# ── Load vader.cfg defaults ───────────────────────────────────────────────────

_CFG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vader.cfg")

def _load_cfg() -> tuple[int | None, int]:
    """Return (device_index, fs) from vader.cfg, or (None, 48000) if absent."""
    cfg = configparser.ConfigParser()
    cfg.read(_CFG_PATH, encoding="utf-8")
    try:
        idx = int(cfg.get("audio", "device_index", fallback="-1"))
        device = idx if idx >= 0 else None
    except ValueError:
        device = None
    try:
        # vader.cfg doesn't store fs; default to 48000 (USB Audio CODEC standard rate)
        fs = int(cfg.get("audio", "fs", fallback="48000"))
    except ValueError:
        fs = 48_000
    return device, fs

_CFG_DEVICE, _CFG_FS = _load_cfg()

# ── Argument parsing ──────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="VADER FT8 live console decoder")
    p.add_argument("--device", type=int, default=_CFG_DEVICE,
                   help=f"sounddevice input device index (default from vader.cfg: {_CFG_DEVICE})")
    p.add_argument("--fs", type=int, default=_CFG_FS,
                   help=f"capture sample rate in Hz (default from vader.cfg: {_CFG_FS})")
    p.add_argument("--list", action="store_true",
                   help="list available audio input devices and exit")
    p.add_argument("--fmin", type=float, default=200.0,
                   help="lower FT8 audio frequency bound in Hz (default: 200)")
    p.add_argument("--fmax", type=float, default=3200.0,
                   help="upper FT8 audio frequency bound in Hz (default: 3200)")
    p.add_argument("--quiet", action="store_true",
                   help="suppress per-candidate debug output; show decoded messages only")
    return p.parse_args()


def _list_devices() -> None:
    try:
        import sounddevice as sd
        devs = sd.query_devices()
        hostapis = sd.query_hostapis()
        print(f"{'Idx':>4}  {'Name':<40}  {'In':>4}  {'Out':>4}  HostAPI")
        print("-" * 75)
        for i, d in enumerate(devs):
            if int(d.get("max_input_channels", 0)) > 0:
                ha = d.get("hostapi", 0)
                ha_name = hostapis[ha].get("name", "") if ha < len(hostapis) else ""
                marker = " *" if d.get("name", "") == sd.query_devices(kind="input").get("name","") else "  "
                print(f"{i:>4}{marker} {d['name']:<40}  {int(d['max_input_channels']):>4}  "
                      f"{int(d.get('max_output_channels',0)):>4}  {ha_name}")
    except ImportError:
        print("sounddevice not installed. Run: pip install sounddevice")
    except Exception as e:
        print(f"Error listing devices: {e}")


# ── Decode callback ───────────────────────────────────────────────────────────

_decode_count = 0

def _on_decode(utc: str, freq_hz: float, snr_db: float, message: str) -> None:
    global _decode_count
    _decode_count += 1
    # Bold/bright green for actual decodes — stands out from the debug noise
    print(
        f"\n{'='*60}\n"
        f"  *** DECODE #{_decode_count} ***\n"
        f"  UTC:  {utc}\n"
        f"  Freq: {freq_hz:.1f} Hz\n"
        f"  SNR:  {snr_db:+.1f} dB\n"
        f"  MSG:  {message}\n"
        f"{'='*60}\n",
        flush=True,
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()

    if args.list:
        _list_devices()
        return

    print(f"VADER FT8 Live Decoder")
    print(f"  Device index : {args.device if args.device is not None else 'default'}")
    print(f"  Capture fs   : {args.fs} Hz")
    print(f"  FT8 band     : {args.fmin:.0f} – {args.fmax:.0f} Hz")
    print(f"  Debug output : {'OFF (--quiet)' if args.quiet else 'ON'}")
    print(f"  Ctrl-C to stop\n")

    # Build decoder
    decoder = FT8ConsoleDecoder(
        fmin_hz=args.fmin,
        fmax_hz=args.fmax,
        on_decode=_on_decode,
    )
    decoder._debug = not args.quiet
    decoder.start()

    # Build audio source
    src = SoundCardAudioSource(
        fs=args.fs,
        block_size=int(args.fs * 0.1),   # 100 ms chunks
        device=args.device,
    )

    print(f"Opening audio stream... ", end="", flush=True)
    try:
        src.start()
    except Exception as e:
        print(f"FAILED\n  {e}")
        print("\nAvailable input devices:")
        _list_devices()
        decoder.stop()
        sys.exit(1)
    print("OK\n")

    # Print a UTC clock so the user can see we're aligned to FT8 slots
    import datetime as _dt
    def _next_slot_in() -> float:
        t = _dt.datetime.now(_dt.timezone.utc).timestamp()
        return 15.0 - (t % 15.0)

    last_slot_announce = 0.0

    try:
        for chunk in src.chunks(timeout_s=0.5):
            # Feed the decoder
            decoder.feed(fs=chunk.fs, samples=chunk.samples, t0_monotonic=chunk.t0)

            # Print a "next slot in Xs" heartbeat once per second
            now = time.monotonic()
            if now - last_slot_announce >= 1.0:
                ns = _next_slot_in()
                print(f"\r  [audio live]  next FT8 slot in {ns:4.1f}s   decodes so far: {_decode_count}   ",
                      end="", flush=True)
                last_slot_announce = now

    except KeyboardInterrupt:
        print(f"\n\nStopped by user.  Total decodes: {_decode_count}")
    finally:
        src.stop()
        decoder.stop()


if __name__ == "__main__":
    main()

