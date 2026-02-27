"""
test_ft8_decode_output.py — Unit tests for FT8 decoded message formatting and
terminal output.

Tests:
  1. format_ft8_message — standard message (CALL1 CALL2 REPORT)
  2. format_ft8_message — CQ message (CQ CALL GRID)
  3. format_ft8_message — grid exchange (CALL1 CALL2 GRID)
  4. format_ft8_message — SNR rounding (positive, negative, fractional)
  5. format_ft8_message — frequency formatting (8 chars, 3 decimal places)
  6. format_ft8_message — SNR width always 3 chars with sign
  7. decode_wav with live_ft8_traffic_2.wav — produces decodable messages
  8. decode_wav results — all messages have non-empty utc_time, message fields
  9. decode_wav results — known CQ message appears in live_ft8_traffic_2.wav
 10. decode_wav — standard message format (CALL1 CALL2 report) appears in results
 11. decode_wav with live_ft8_audio_sample_3.wav — produces decodable messages
 12. decode_wav sample_3 — all result fields are valid
 13. decode_wav sample_3 — CQ message appears
 14. decode_wav sample_3 — format_ft8_message produces valid lines for all results

Run:  python test_ft8_decode_output.py
"""
from __future__ import annotations

import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(__file__))

from ft8_decode import format_ft8_message, decode_wav

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
        detail = f"{type(exc).__name__}: {exc}\n{''.join(traceback.format_tb(exc.__traceback__))}"
    elapsed = (time.perf_counter() - t0) * 1000
    results.append((name, ok, detail))
    print(f"  {PASS if ok else FAIL}  {name}  ({elapsed:.1f} ms)")
    if not ok:
        for line in detail.splitlines():
            print(f"         {line}")


# ---------------------------------------------------------------------------
# format_ft8_message tests
# ---------------------------------------------------------------------------

def t_standard_message() -> str:
    """Standard message: CALL1 CALL2 REPORT."""
    line = format_ft8_message("12:34:56", -10.0, 1234.567, "W4ABC K9XYZ -10")
    assert line == "12:34:56 -10 1234.567 W4ABC K9XYZ -10", f"Got: {line!r}"
    return f"format OK: {line!r}"

run("1. format_ft8_message — standard message (CALL1 CALL2 REPORT)", t_standard_message)


def t_cq_message() -> str:
    """CQ message: CQ CALL GRID."""
    line = format_ft8_message("23:00:15", -7.0, 975.0, "CQ W4ABC EM73")
    assert line == "23:00:15  -7  975.000 CQ W4ABC EM73", f"Got: {line!r}"
    return f"format OK: {line!r}"

run("2. format_ft8_message — CQ message (CQ CALL GRID)", t_cq_message)


def t_grid_exchange() -> str:
    """Grid exchange: CALL1 CALL2 GRID."""
    line = format_ft8_message("00:00:00", 3.5, 2000.0, "KG5QIX W2EAD FN31")
    assert line == "00:00:00  +4 2000.000 KG5QIX W2EAD FN31", f"Got: {line!r}"
    return f"format OK: {line!r}"

run("3. format_ft8_message — grid exchange (CALL1 CALL2 GRID)", t_grid_exchange)


def t_snr_rounding() -> str:
    """SNR values should be rounded to nearest integer."""
    # Positive fractional rounds up
    line = format_ft8_message("00:00:00", 2.6, 1000.0, "MSG")
    assert " +3 " in line, f"Expected '+3', got: {line!r}"
    # Negative fractional rounds toward zero for -2.4
    line2 = format_ft8_message("00:00:00", -2.4, 1000.0, "MSG")
    assert " -2 " in line2, f"Expected '-2', got: {line2!r}"
    # Zero
    line3 = format_ft8_message("00:00:00", 0.0, 1000.0, "MSG")
    assert " +0 " in line3, f"Expected '+0', got: {line3!r}"
    return "SNR rounding correct for +2.6→+3, -2.4→-2, 0→+0"

run("4. format_ft8_message — SNR rounding", t_snr_rounding)


def t_frequency_format() -> str:
    """Frequency field uses 8-char formatted field (3 decimal places).
    When parsed via split(), leading padding spaces are stripped."""
    line = format_ft8_message("12:00:00", 0.0, 450.0, "MSG")
    parts = line.split()
    # parts[2] is the frequency token (stripped of leading spaces by split)
    freq_field = parts[2]
    assert freq_field == "450.000", f"Frequency field wrong: {freq_field!r}"
    assert "." in freq_field and freq_field.endswith("000"), \
        f"Frequency not 3 decimal places: {freq_field!r}"
    # Verify the raw line contains the 8-char formatted field
    assert " 450.000 " in line, f"8-char freq field not in raw line: {line!r}"
    # Larger frequency (fills all 8 chars, no padding)
    line2 = format_ft8_message("12:00:00", 0.0, 2843.75, "MSG")
    parts2 = line2.split()
    assert parts2[2] == "2843.750", f"Frequency field wrong: {parts2[2]!r}"
    return f"Frequency format OK: {freq_field!r}"

run("5. format_ft8_message — frequency format (8-char field, 3 decimal places)", t_frequency_format)


def t_snr_field_width() -> str:
    """SNR field uses +3d format (sign + up to 2 digits, right-aligned in 3 chars).
    When parsed via split(), the padding space is stripped, leaving +N or -NN etc."""
    # Single-digit positive: raw " +5", split token "+5"
    line = format_ft8_message("12:00:00", 5.0, 1000.0, "MSG")
    parts = line.split()
    snr_field = parts[1]
    assert snr_field == "+5", f"SNR field wrong: {snr_field!r}"
    # Raw line contains " +5 " (the field with its padding space)
    assert " +5 " in line, f"Expected ' +5 ' in raw line: {line!r}"
    # Two-digit negative: raw "-22", split token "-22"
    line2 = format_ft8_message("12:00:00", -22.0, 1000.0, "MSG")
    parts2 = line2.split()
    assert parts2[1] == "-22", f"SNR field wrong: {parts2[1]!r}"
    return f"SNR field format OK: {snr_field!r}"

run("6. format_ft8_message — SNR with sign prefix", t_snr_field_width)


# ---------------------------------------------------------------------------
# decode_wav tests using live_ft8_traffic_2.wav
# ---------------------------------------------------------------------------

_WAV2 = os.path.join(os.path.dirname(__file__), "live_ft8_traffic_2.wav")

# Decode once at module load time (shared by tests 7-10).
# If the file is missing, _WAV2_RESULTS will be None and each test raises
# FileNotFoundError with a clear message rather than failing silently.
_WAV2_RESULTS: list | None = None
if os.path.exists(_WAV2):
    _WAV2_RESULTS = decode_wav(_WAV2)


def t_decode_wav_produces_messages() -> str:
    """decode_wav on live_ft8_traffic_2.wav must return at least one message."""
    if _WAV2_RESULTS is None:
        raise FileNotFoundError(f"Test file not found: {_WAV2}")
    assert len(_WAV2_RESULTS) > 0, "No messages decoded from live_ft8_traffic_2.wav"
    return f"Decoded {len(_WAV2_RESULTS)} messages from live_ft8_traffic_2.wav"

run("7. decode_wav with live_ft8_traffic_2.wav — produces messages", t_decode_wav_produces_messages)


def t_decode_wav_result_fields() -> str:
    """All FT8DecodeResult fields must be non-empty / valid types."""
    if _WAV2_RESULTS is None:
        raise FileNotFoundError(f"Test file not found: {_WAV2}")
    assert len(_WAV2_RESULTS) > 0
    for r in _WAV2_RESULTS:
        assert isinstance(r.utc_time, str) and r.utc_time, f"Bad utc_time: {r.utc_time!r}"
        assert isinstance(r.strength_db, float), f"Bad strength_db type: {type(r.strength_db)}"
        assert isinstance(r.frequency_hz, float) and r.frequency_hz > 0, f"Bad freq: {r.frequency_hz}"
        assert isinstance(r.message, str) and r.message, f"Bad message: {r.message!r}"
    return f"All {len(_WAV2_RESULTS)} results have valid fields"

run("8. decode_wav results — all fields are valid", t_decode_wav_result_fields)


def t_decode_wav_has_cq() -> str:
    """At least one CQ message must be present in live_ft8_traffic_2.wav."""
    if _WAV2_RESULTS is None:
        raise FileNotFoundError(f"Test file not found: {_WAV2}")
    cq_msgs = [r for r in _WAV2_RESULTS if r.message.startswith("CQ ")]
    assert cq_msgs, f"No CQ messages found among: {[r.message for r in _WAV2_RESULTS[:5]]}"
    return f"Found {len(cq_msgs)} CQ message(s): {cq_msgs[0].message!r}"

run("9. decode_wav — CQ message appears in live_ft8_traffic_2.wav", t_decode_wav_has_cq)


def t_decode_wav_format_output() -> str:
    """format_ft8_message applied to decode_wav results produces standard FT8 lines."""
    if _WAV2_RESULTS is None:
        raise FileNotFoundError(f"Test file not found: {_WAV2}")
    assert len(_WAV2_RESULTS) > 0
    for r in _WAV2_RESULTS:
        line = format_ft8_message(r.utc_time, r.strength_db, r.frequency_hz, r.message)
        parts = line.split()
        # Must have at least 4 parts: utc, snr, freq, message_start
        assert len(parts) >= 4, f"Malformed output line: {line!r}"
        # SNR field (parts[1]) must be integer with sign
        snr_str = parts[1]
        assert snr_str[0] in ("+", "-"), f"SNR has no sign: {snr_str!r} in {line!r}"
        assert snr_str[1:].isdigit(), f"SNR non-digit after sign: {snr_str!r}"
        # Freq field (parts[2]) must contain '.'
        assert "." in parts[2], f"Freq missing decimal: {parts[2]!r}"
    return f"format_ft8_message produces valid lines for all {len(_WAV2_RESULTS)} results"

run("10. decode_wav — format_ft8_message produces valid lines for all results",
    t_decode_wav_format_output)


# ---------------------------------------------------------------------------
# decode_wav tests using live_ft8_audio_sample_3.wav
# ---------------------------------------------------------------------------

_WAV3 = os.path.join(os.path.dirname(__file__), "live_ft8_audio_sample_3.wav")

# Decode sample_3 once at module load time (shared by tests 11-14).
# If the file is missing, _WAV3_RESULTS will be None and each test raises
# FileNotFoundError with a clear message rather than failing silently.
_WAV3_RESULTS: list | None = None
if os.path.exists(_WAV3):
    _WAV3_RESULTS = decode_wav(_WAV3)


def t_decode_wav3_produces_messages() -> str:
    """decode_wav on live_ft8_audio_sample_3.wav must return at least one message."""
    if _WAV3_RESULTS is None:
        raise FileNotFoundError(f"Test file not found: {_WAV3}")
    assert len(_WAV3_RESULTS) > 0, "No messages decoded from live_ft8_audio_sample_3.wav"
    return f"Decoded {len(_WAV3_RESULTS)} messages from live_ft8_audio_sample_3.wav"

run("11. decode_wav with live_ft8_audio_sample_3.wav — produces messages",
    t_decode_wav3_produces_messages)


def t_decode_wav3_result_fields() -> str:
    """All FT8DecodeResult fields from sample_3 must be non-empty / valid types."""
    if _WAV3_RESULTS is None:
        raise FileNotFoundError(f"Test file not found: {_WAV3}")
    assert len(_WAV3_RESULTS) > 0
    for r in _WAV3_RESULTS:
        assert isinstance(r.utc_time, str) and r.utc_time, f"Bad utc_time: {r.utc_time!r}"
        assert isinstance(r.strength_db, float), f"Bad strength_db type: {type(r.strength_db)}"
        assert isinstance(r.frequency_hz, float) and r.frequency_hz > 0, f"Bad freq: {r.frequency_hz}"
        assert isinstance(r.message, str) and r.message, f"Bad message: {r.message!r}"
    return f"All {len(_WAV3_RESULTS)} results have valid fields"

run("12. decode_wav sample_3 — all result fields are valid", t_decode_wav3_result_fields)


def t_decode_wav3_has_cq() -> str:
    """At least one CQ message must be present in live_ft8_audio_sample_3.wav."""
    if _WAV3_RESULTS is None:
        raise FileNotFoundError(f"Test file not found: {_WAV3}")
    cq_msgs = [r for r in _WAV3_RESULTS if r.message.startswith("CQ ")]
    assert cq_msgs, f"No CQ messages found among: {[r.message for r in _WAV3_RESULTS[:5]]}"
    return f"Found {len(cq_msgs)} CQ message(s): {cq_msgs[0].message!r}"

run("13. decode_wav sample_3 — CQ message appears", t_decode_wav3_has_cq)


def t_decode_wav3_format_output() -> str:
    """format_ft8_message on sample_3 results produces valid standard FT8 lines."""
    if _WAV3_RESULTS is None:
        raise FileNotFoundError(f"Test file not found: {_WAV3}")
    assert len(_WAV3_RESULTS) > 0
    for r in _WAV3_RESULTS:
        line = format_ft8_message(r.utc_time, r.strength_db, r.frequency_hz, r.message)
        parts = line.split()
        assert len(parts) >= 4, f"Malformed output line: {line!r}"
        snr_str = parts[1]
        assert snr_str[0] in ("+", "-"), f"SNR has no sign: {snr_str!r} in {line!r}"
        assert snr_str[1:].isdigit(), f"SNR non-digit after sign: {snr_str!r}"
        assert "." in parts[2], f"Freq missing decimal: {parts[2]!r}"
    return f"format_ft8_message produces valid lines for all {len(_WAV3_RESULTS)} sample_3 results"

run("14. decode_wav sample_3 — format_ft8_message produces valid lines for all results",
    t_decode_wav3_format_output)

print()
passed = sum(1 for _, ok, _ in results if ok)
failed = sum(1 for _, ok, _ in results if not ok)
print(f"Results: {passed}/{len(results)} passed", end="")
if failed:
    print(f"  ({failed} FAILED)")
    sys.exit(1)
else:
    print("  — all tests passed ✓")
    sys.exit(0)
