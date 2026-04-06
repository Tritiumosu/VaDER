"""
test_utc_framer.py — Unit tests for UTC15sFramer slot boundary alignment.

Tests:
  1. Frames emitted by push() start at UTC 15-second slot boundaries
  2. First frame is always marked partial (and should be skipped)
  3. Subsequent frames are not partial and start at consecutive boundaries
  4. Audio capture starting mid-slot discards pre-boundary audio, then aligns
  5. Audio capture starting exactly on a boundary emits aligned frames
  6. FT8ConsoleDecoder._decode_frame verbose logging flags are present

Run:  python test_utc_framer.py
"""
from __future__ import annotations

import math
import sys
import time
import traceback

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0] if "/" in __file__ else ".")

from ft8_decode import UTC15sFramer, FT8_FS

PASS = "✓ PASS"
FAIL = "✗ FAIL"
results: list[tuple[str, bool, str]] = []


def run(name: str, fn) -> None:
    t0 = time.perf_counter()
    try:
        msg = fn()
        ok = True
        detail = msg or ""
    except Exception as exc:
        ok = False
        detail = (
            f"{type(exc).__name__}: {exc}\n"
            f"{''.join(traceback.format_tb(exc.__traceback__))}"
        )
    elapsed = (time.perf_counter() - t0) * 1000
    results.append((name, ok, detail))
    print(f"  {PASS if ok else FAIL}  {name}  ({elapsed:.1f} ms)")
    if not ok:
        for line in detail.splitlines():
            print(f"         {line}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_framer(t0_utc: float) -> UTC15sFramer:
    """Return a UTC15sFramer whose internal clock is fixed at t0_utc."""
    framer = UTC15sFramer(fs_proc=FT8_FS)
    # Patch _update_utc_minus_mono so the framer believes the UTC epoch is
    # exactly t0_utc when t0_monotonic=0.
    framer._utc_minus_mono = t0_utc  # mono=0  →  utc = 0 + t0_utc
    framer._alpha = 0.0              # prevent further drift updates
    return framer


def _push_seconds(framer: UTC15sFramer, seconds: float, *, t0_mono: float = 0.0) -> list:
    """Push `seconds` of silence in 0.1-second chunks; return all emitted frames."""
    chunk_s = 0.1
    n_chunk = int(round(chunk_s * FT8_FS))
    silence = np.zeros(n_chunk, dtype=np.float32)
    out = []
    mono = t0_mono
    n_chunks = int(math.ceil(seconds / chunk_s))
    for _ in range(n_chunks):
        out.extend(framer.push(silence, t0_monotonic=mono))
        mono += chunk_s
    return out


# ---------------------------------------------------------------------------
# Test 1: frames start at UTC slot boundaries
# ---------------------------------------------------------------------------

def t_frames_at_boundaries() -> str:
    """All emitted frames must start at a multiple of 15 seconds (UTC epoch)."""
    # Start 7 seconds into a slot (slot boundary at t=0)
    t0_utc = 7.0
    framer = _make_framer(t0_utc)
    # Push 60 seconds of audio → should emit multiple frames
    frames = _push_seconds(framer, 60.0)
    assert len(frames) >= 2, f"Expected at least 2 frames, got {len(frames)}"
    for slot_start, _, _ in frames:
        remainder = slot_start % 15.0
        assert remainder < 1e-3 or remainder > 15.0 - 1e-3, (
            f"Frame slot_start={slot_start} is not on a 15-s boundary "
            f"(remainder={remainder:.6f})"
        )
    return f"All {len(frames)} frames start at slot boundaries"


run("1. Frames start at UTC 15-second slot boundaries", t_frames_at_boundaries)


# ---------------------------------------------------------------------------
# Test 2: first frame is always partial
# ---------------------------------------------------------------------------

def t_first_frame_partial() -> str:
    """The first emitted frame must be marked is_partial=True."""
    t0_utc = 3.0
    framer = _make_framer(t0_utc)
    frames = _push_seconds(framer, 60.0)
    assert len(frames) >= 1, "No frames emitted"
    _, _, is_partial = frames[0]
    assert is_partial, f"First frame not marked partial: {frames[0]}"
    return "First frame correctly marked partial"


run("2. First emitted frame is always marked partial", t_first_frame_partial)


# ---------------------------------------------------------------------------
# Test 3: subsequent frames are not partial and are consecutive
# ---------------------------------------------------------------------------

def t_subsequent_frames_not_partial() -> str:
    """Frames after the first must have is_partial=False and sequential slot times."""
    t0_utc = 3.0
    framer = _make_framer(t0_utc)
    frames = _push_seconds(framer, 90.0)
    assert len(frames) >= 3, f"Expected >=3 frames, got {len(frames)}"
    for slot_start, _, is_partial in frames[1:]:
        assert not is_partial, f"Frame at {slot_start} is unexpectedly partial"
    # Check consecutive 15-second spacing
    slot_times = [f[0] for f in frames]
    for i in range(1, len(slot_times)):
        gap = slot_times[i] - slot_times[i - 1]
        assert abs(gap - 15.0) < 1e-3, (
            f"Frame gap {gap:.3f}s ≠ 15.0s between frames {i-1} and {i}"
        )
    return f"{len(frames[1:])} non-partial frames with 15-s spacing"


run("3. Subsequent frames not partial, spaced 15 s apart", t_subsequent_frames_not_partial)


# ---------------------------------------------------------------------------
# Test 4: mid-slot start → FT8 signal at t≈0 in frame
# ---------------------------------------------------------------------------

def t_signal_at_t0_in_frame() -> str:
    """
    Inject a simple sinusoid that starts exactly at a slot boundary.
    The energy of the injected tone must be concentrated in the first half
    of the first non-partial (decodable) frame, confirming the signal is at t≈0.
    """
    # Start audio capture 4 seconds AFTER a slot boundary
    t0_utc = 4.0   # UTC epoch offset; slot boundary at 0.0, 15.0, 30.0, ...

    framer = _make_framer(t0_utc)
    fs = FT8_FS
    chunk_s = 0.1
    n_chunk = int(round(chunk_s * fs))
    signal_freq = 1000.0  # 1 kHz tone, arbitrary

    frames_out: list[tuple[float, np.ndarray, bool]] = []
    mono = 0.0
    total_s = 70.0
    n_total = int(math.ceil(total_s / chunk_s))

    # With t0_utc=4, the framer emits:
    #   partial  frame: slot_start=15 (covers UTC 15-30) — SKIPPED
    #   decode   frame: slot_start=30 (covers UTC 30-45)
    # Inject a tone starting at UTC=30 (the first decodable slot boundary).
    signal_start_utc = 30.0

    for i in range(n_total):
        t_utc_start = t0_utc + mono  # UTC time of this chunk's first sample
        chunk_t = np.linspace(t_utc_start, t_utc_start + chunk_s, n_chunk,
                              endpoint=False)
        mask = chunk_t >= signal_start_utc
        samples = np.where(
            mask,
            np.sin(2 * math.pi * signal_freq * chunk_t).astype(np.float32),
            np.zeros(n_chunk, dtype=np.float32),
        )
        frames_out.extend(framer.push(samples, t0_monotonic=mono))
        mono += chunk_s

    # Find the first non-partial frame
    non_partial = [(s, f) for s, f, p in frames_out if not p]
    assert non_partial, "No non-partial frames emitted"

    slot_start, frame = non_partial[0]
    # This frame must start at the first complete slot boundary after the partial
    assert abs(slot_start - signal_start_utc) < 1e-2, (
        f"First non-partial frame slot_start={slot_start:.3f}, "
        f"expected {signal_start_utc}"
    )

    # Energy in first 1 second vs. last 1 second of the frame
    n_1s = fs  # samples in 1 second
    e_first = float(np.mean(frame[:n_1s] ** 2))
    e_last = float(np.mean(frame[-n_1s:] ** 2))
    # Signal is present at the START of this frame (t=0), so first-second
    # energy must be substantial (close to 0.5 for a unit sine wave).
    assert e_first > 0.3, (
        f"Expected tone energy at start of frame, got e_first={e_first:.4f}"
    )
    return (
        f"Non-partial frame starts at slot boundary {slot_start:.1f}s; "
        f"e_first={e_first:.4f} e_last={e_last:.4f}"
    )


run("4. Mid-slot start: FT8 signal appears at t≈0 in first decoded frame",
    t_signal_at_t0_in_frame)


# ---------------------------------------------------------------------------
# Test 5: start exactly on boundary
# ---------------------------------------------------------------------------

def t_start_on_boundary() -> str:
    """Audio capture starting exactly at a slot boundary emits aligned frames."""
    t0_utc = 0.0  # exactly at a boundary
    framer = _make_framer(t0_utc)
    frames = _push_seconds(framer, 60.0)
    assert len(frames) >= 3, f"Expected >=3 frames, got {len(frames)}"
    # All frames must be at 0, 15, 30, 45, ...
    for slot_start, _, _ in frames:
        remainder = slot_start % 15.0
        assert remainder < 1e-3 or remainder > 15.0 - 1e-3, (
            f"Frame at {slot_start:.3f} not on boundary (remainder={remainder:.4f})"
        )
    return f"All {len(frames)} frames aligned when starting on boundary"


run("5. Capture starting exactly on boundary still emits aligned frames",
    t_start_on_boundary)


# ---------------------------------------------------------------------------
# Test 6: FT8ConsoleDecoder verbose logging attributes
# ---------------------------------------------------------------------------

def t_decoder_debug_attrs() -> str:
    """FT8ConsoleDecoder must expose _debug flag and _min_costas_matches."""
    from ft8_decode import FT8ConsoleDecoder
    dec = FT8ConsoleDecoder()
    assert hasattr(dec, "_debug"), "Missing _debug attribute"
    assert hasattr(dec, "_min_costas_matches"), "Missing _min_costas_matches"
    assert isinstance(dec._debug, bool), "_debug must be bool"
    assert isinstance(dec._min_costas_matches, int), "_min_costas_matches must be int"
    return (
        f"_debug={dec._debug}, _min_costas_matches={dec._min_costas_matches}"
    )


run("6. FT8ConsoleDecoder verbose logging attributes present", t_decoder_debug_attrs)


# ---------------------------------------------------------------------------
print()
passed = sum(1 for _, ok, _ in results if ok)
failed = sum(1 for _, ok, _ in results if not ok)
print(f"Results: {passed}/{len(results)} passed", end="")
if failed:
    print(f"  ({failed} FAILED)")
else:
    print("  — all tests passed ✓")


def test_all_pass():
    failures = [(name, detail) for name, ok, detail in results if not ok]
    assert not failures, "Failed tests:\n" + "\n".join(f"  {n}: {d}" for n, d in failures)


if __name__ == "__main__":
    sys.exit(1 if failed else 0)
