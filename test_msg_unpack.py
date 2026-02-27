"""
Stage 5 message unpacker tests.

We construct known 77-bit message payloads by hand (following the FT8
bit-packing spec from pack77.f90) and verify ft8_unpack_message() produces
the expected human-readable string.

Tests:
  1. Standard type-1: callsign + callsign + grid
  2. Standard type-1: callsign + callsign + signal report
  3. Standard type-1: CQ + callsign + grid
  4. Free text (i3=1): 'CQ DX' and similar
  5. Telemetry (i3=4): hex payload
  6. Round-trip: pack a known callsign integer, unpack, verify
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from ft8_decode import (
    ft8_unpack_message,
    _unpack_callsign_28,
    _unpack_grid_15,
)


def _int_to_bits(val: int, length: int) -> list[int]:
    """Convert integer to MSB-first bit list of given length."""
    bits = []
    for i in range(length - 1, -1, -1):
        bits.append((val >> i) & 1)
    return bits


def _pack_callsign_28(call: str) -> int:
    """Pack a standard callsign into 28 bits (inverse of _unpack_callsign_28)."""
    NBASE = 37 * 36 * 10 * 27 * 27 * 27   # 262177560

    C36 = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"   # c0: no space
    C37 = " 0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"   # c1: space allowed
    C27 = "ABCDEFGHIJKLMNOPQRSTUVWXYZ "
    C10 = "0123456789"

    call = call.upper().strip()

    # Special tokens
    if call == 'DE':    return NBASE
    if call == 'QRZ':   return NBASE + 1
    if call == 'CQ':    return NBASE + 2
    if call.startswith('CQ '):
        suffix = call[3:].strip()
        if suffix.isdigit() and len(suffix) <= 3:
            return NBASE + 3 + int(suffix)
        # 4-char alpha suffix (e.g. 'DX')
        suffix = (suffix + '    ')[:4]
        m = 0
        for ch in suffix:
            idx = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ '.index(ch) if ch in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ ' else 26
            m = m * 27 + idx
        return NBASE + 3 + 1000 + m

    # Standard callsign: find digit, split into prefix(2) + digit + suffix(3)
    digit_pos = next((i for i, ch in enumerate(call) if ch.isdigit()), -1)
    if digit_pos < 0:
        prefix, digit_c, suffix = '', '0', call[:3]
    else:
        prefix, digit_c, suffix = call[:digit_pos], call[digit_pos], call[digit_pos+1:]

    # c0 must be in C36 (no space) — pad prefix to 2 using space for c1 only
    prefix2 = (prefix + '  ')[:2]
    suffix3  = (suffix  + '   ')[:3]

    def idx36(ch): return C36.index(ch) if ch in C36 else 0
    def idx37(ch): return C37.index(ch) if ch in C37 else 0
    def idx27(ch): return C27.index(ch) if ch in C27 else 26
    def idx10(ch): return C10.index(ch) if ch in C10 else 0

    c = list(prefix2 + digit_c + suffix3)
    n = idx36(c[0])
    n = n * 37 + idx37(c[1])
    n = n * 10 + idx10(c[2])
    n = n * 27 + idx27(c[3])
    n = n * 27 + idx27(c[4])
    n = n * 27 + idx27(c[5])
    return n



def _pack_grid_15(grid: str) -> int:
    """Pack a 4-char Maidenhead grid square or special token into 15 bits."""
    grid = grid.upper().strip()

    # Special tokens — checked BEFORE integer parse to avoid treating '73' as a report
    if grid == 'RRR':  return 32767
    if grid == 'RR73': return 32766
    if grid == '73':   return 32765

    # Signal report e.g. '-15', '+05', '-24'..'+09'
    # Only treat as report if the string looks like a signed/unsigned small integer
    import re as _re
    if _re.fullmatch(r'[+-]?\d{1,2}', grid):
        try:
            db = int(grid)
            # Valid FT8 signal report range is -24..+9 dB (encoded as n+35, 0..62)
            return max(0, min(61, db + 35))
        except ValueError:
            pass

    if len(grid) < 4:
        return 0

    G = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    _GRID_LETTERS = 'ABCDEFGHIJKLMNOPQR'
    g1 = _GRID_LETTERS.index(grid[0]) if grid[0] in _GRID_LETTERS else 0
    g2 = _GRID_LETTERS.index(grid[1]) if grid[1] in _GRID_LETTERS else 0
    g3 = int(grid[2]) if grid[2].isdigit() else 0
    g4 = int(grid[3]) if grid[3].isdigit() else 0

    igrid = ((g1 * 18 + g2) * 10 + g3) * 10 + g4
    return igrid + 63


def _build_type1_bits(call1: str, r1: int, call2: str, r2: int,
                       grid: str, n3: int = 0, i3: int = 0) -> np.ndarray:
    """
    Build a 77-bit type-1 message bit array.

    FT8 77-bit layout (MSB-first, bits 0-76):
      [0:28]   = call1 (28 bits)
      [28]     = r1 flag (1 bit)
      [29:57]  = call2 (28 bits)
      [57]     = r2 flag (1 bit)
      [58:73]  = grid/report (15 bits)
      [73:74]  = spare (1 bit, set 0)  -- not n3; n3 is encoded in grid field
      [74:77]  = i3 (3 bits)
    Total = 28+1+28+1+15+1+3 = 77 bits.

    For i3=0 the sub-type n3 is packed into the top 3 bits of the 15-bit
    grid/report field, making the grid itself only 12 bits. For simplicity
    we pack n3 into the grid integer's upper bits here.
    """
    c1 = _pack_callsign_28(call1)
    c2 = _pack_callsign_28(call2)
    g  = _pack_grid_15(grid)

    # For i3=0: pack n3 into the upper 3 bits of the 15-bit grid field
    # grid_field = (n3 << 12) | (g & 0x0FFF)  -- but g from _pack_grid_15
    # is already the full 15-bit value; to keep it simple we leave n3=0
    # for standard messages (grid uses all 15 bits when n3=0).
    grid_field = g  # n3=0 means grid occupies all 15 bits

    bits = (
        _int_to_bits(c1, 28) +
        [r1 & 1] +
        _int_to_bits(c2, 28) +
        [r2 & 1] +
        _int_to_bits(grid_field, 15) +
        [0] +                            # 1 spare bit
        _int_to_bits(i3, 3)
    )
    assert len(bits) == 77, f"Expected 77 bits, got {len(bits)}"
    return np.array(bits, dtype=np.uint8)


# ══ Test 1: Standard QSO exchange ═════════════════════════════════════════════
print("Test 1: standard type-1 — QSO exchange with grid")

bits1 = _build_type1_bits('W4ABC', 0, 'K9XYZ', 0, 'EN52')
msg1  = ft8_unpack_message(bits1)
print(f"  packed bits → '{msg1}'")
assert 'W4ABC' in msg1, f"call1 missing: '{msg1}'"
assert 'K9XYZ' in msg1, f"call2 missing: '{msg1}'"
assert 'EN52'  in msg1, f"grid missing: '{msg1}'"
print("  Test 1 PASSED")


# ══ Test 2: Signal report ═════════════════════════════════════════════════════
print("\nTest 2: standard type-1 — signal report -12 dB")

bits2 = _build_type1_bits('W4ABC', 0, 'K9XYZ', 0, '-12')
msg2  = ft8_unpack_message(bits2)
print(f"  packed bits → '{msg2}'")
assert 'W4ABC' in msg2, f"call1 missing: '{msg2}'"
assert 'K9XYZ' in msg2, f"call2 missing: '{msg2}'"
assert '-12'   in msg2 or '-12' in msg2, f"report missing: '{msg2}'"
print("  Test 2 PASSED")


# ══ Test 3: CQ call ═══════════════════════════════════════════════════════════
print("\nTest 3: standard type-1 — CQ with grid")

bits3 = _build_type1_bits('CQ', 0, 'W4ABC', 0, 'EM73')
msg3  = ft8_unpack_message(bits3)
print(f"  packed bits → '{msg3}'")
assert 'W4ABC' in msg3, f"callsign missing: '{msg3}'"
assert 'EM73'  in msg3, f"grid missing: '{msg3}'"
print("  Test 3 PASSED")


# ══ Test 4: CQ DX ═════════════════════════════════════════════════════════════
print("\nTest 4: CQ DX directed call")

bits4 = _build_type1_bits('CQ DX', 0, 'VK2ABC', 0, 'QF56')
msg4  = ft8_unpack_message(bits4)
print(f"  packed bits → '{msg4}'")
assert 'VK2ABC' in msg4 or 'VK2AB' in msg4, f"callsign missing: '{msg4}'"
print("  Test 4 PASSED")


# ══ Test 5: RRR ═══════════════════════════════════════════════════════════════
print("\nTest 5: standard type-1 — RRR acknowledgement")

bits5 = _build_type1_bits('W4ABC', 0, 'K9XYZ', 0, 'RRR')
msg5  = ft8_unpack_message(bits5)
print(f"  packed bits → '{msg5}'")
assert 'RRR' in msg5, f"RRR missing: '{msg5}'"
print("  Test 5 PASSED")


# ══ Test 6: 73 ═══════════════════════════════════════════════════════════════
print("\nTest 6: standard type-1 — 73")

bits6 = _build_type1_bits('W4ABC', 0, 'K9XYZ', 0, '73')
msg6  = ft8_unpack_message(bits6)
print(f"  packed bits → '{msg6}'")
assert '73' in msg6, f"73 missing: '{msg6}'"
print("  Test 6 PASSED")


# ══ Test 7: Free text (i3=1) ══════════════════════════════════════════════════
print("\nTest 7: free text (i3=1)")

def _pack_free_text(text: str) -> np.ndarray:
    """Pack up to 13 chars of free text into 71 bits using base-42 (i3=1 message)."""
    _FT = " 0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ+-./?"   # 42 chars
    text = (text.upper() + ' ' * 13)[:13]
    n = 0
    for ch in text:
        idx = _FT.index(ch) if ch in _FT else 0
        n = n * 42 + idx
    bits71 = _int_to_bits(n, 71)
    bits77 = bits71 + [0, 0, 0] + _int_to_bits(1, 3)   # spare=0, i3=1
    return np.array(bits77[:77], dtype=np.uint8)

text_in = 'CQ W4ABC    '
bits7 = _pack_free_text(text_in.strip())
msg7  = ft8_unpack_message(bits7)
print(f"  packed '{text_in.strip()}' → '{msg7}'")
# Free text round-trip: the decoded text should contain key parts
assert 'CQ' in msg7 or 'W4ABC' in msg7, f"free text missing key content: '{msg7}'"
print("  Test 7 PASSED")


# ══ Test 8: Telemetry (i3=4) ══════════════════════════════════════════════════
print("\nTest 8: telemetry (i3=4)")

telem_val = 0x12345678ABCDEF
bits8 = np.array(
    _int_to_bits(telem_val, 71) + _int_to_bits(0, 3) + _int_to_bits(4, 3),
    dtype=np.uint8
)
msg8 = ft8_unpack_message(bits8)
print(f"  telemetry → '{msg8}'")
assert msg8.startswith('TELEMETRY:'), f"Telemetry prefix missing: '{msg8}'"
print("  Test 8 PASSED")


# ══ Test 9: Callsign pack/unpack round-trip ════════════════════════════════════
print("\nTest 9: callsign pack/unpack round-trip")

for call in ['W4ABC', 'K9XYZ', 'VK2TIM', 'G3ABC', 'DE', 'QRZ', 'CQ']:
    n    = _pack_callsign_28(call)
    back = _unpack_callsign_28(n)
    print(f"  '{call}' → {n} → '{back}'")
    assert back == call, f"Round-trip mismatch: '{call}' → '{back}'"
print("  Test 9 PASSED")


# ══ Test 10: Grid pack/unpack round-trip ═══════════════════════════════════════
print("\nTest 10: grid pack/unpack round-trip")

for grid in ['EN52', 'EM73', 'QF56', 'IO91', 'RRR', 'RR73', '73']:
    n    = _pack_grid_15(grid)
    back = _unpack_grid_15(n)
    print(f"  '{grid}' → {n} → '{back}'")
    assert back == grid, f"Round-trip mismatch: '{grid}' → '{back}'"
print("  Test 10 PASSED")


print("\n=== All message unpack tests passed ===")

