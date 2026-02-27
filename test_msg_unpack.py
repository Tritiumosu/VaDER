"""
Stage 5 message unpacker tests.

We construct known 77-bit message payloads by hand (following the FT8
bit-packing spec from ft8_lib message.c) and verify ft8_unpack_message()
produces the expected human-readable string.

Tests:
  1. Standard type-1: callsign + callsign + grid
  2. Standard type-1: callsign + callsign + signal report
  3. Standard type-1: CQ + callsign + grid
  4. CQ DX directed call
  5. RRR acknowledgement
  6. 73 final exchange
  7. Free text (i3=0, n3=0)
  8. Telemetry (i3=0, n3=4)
  9. Callsign pack/unpack round-trip
  10. Grid pack/unpack round-trip
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from ft8_decode import (
    ft8_unpack_message,
    _unpack_callsign_28,
    _unpack_grid,
)


def _int_to_bits(val: int, length: int) -> list[int]:
    """Convert integer to MSB-first bit list of given length."""
    bits = []
    for i in range(length - 1, -1, -1):
        bits.append((val >> i) & 1)
    return bits


def _pack_callsign_28(call: str) -> int:
    """Pack a standard callsign into a 28-bit integer (ft8_lib convention)."""
    # ft8_lib constants
    NTOKENS = 2063592
    MAX22   = 4194304
    C37 = " 0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"  # ALPHANUM_SPACE (space=0)
    C36 = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"   # ALPHANUM
    C27 = " ABCDEFGHIJKLMNOPQRSTUVWXYZ"            # LETTERS_SPACE (space=0)
    C10 = "0123456789"

    call = call.upper().strip()

    # Special tokens
    if call == 'DE':    return 0
    if call == 'QRZ':   return 1
    if call == 'CQ':    return 2
    if call.startswith('CQ '):
        suffix = call[3:].strip()
        if suffix.isdigit() and len(suffix) <= 3:
            return 3 + int(suffix)
        # alpha suffix (up to 4 chars A-Z)
        suffix = (suffix + '    ')[:4]
        m = 0
        for ch in suffix:
            idx = C27.index(ch) if ch in C27 else 0
            m = m * 27 + idx
        return 1003 + m

    # Standard callsign: place digit at c6[2] (ft8_lib convention)
    digit_pos = next((i for i, ch in enumerate(call) if ch.isdigit()), -1)
    if digit_pos == 2 and len(call) <= 6:
        c6 = (call + '      ')[:6]
    elif digit_pos == 1 and len(call) <= 5:
        c6 = ' ' + (call + '     ')[:5]
    else:
        c6 = ' ' + (call + '     ')[:5]  # best-effort

    def i37(ch): return C37.index(ch) if ch in C37 else 0
    def i36(ch): return C36.index(ch) if ch in C36 else 0
    def i27(ch): return C27.index(ch) if ch in C27 else 0
    def i10(ch): return C10.index(ch) if ch in C10 else 0

    n = (i37(c6[0]) * 36 * 10 * 27 * 27 * 27 +
         i36(c6[1]) * 10 * 27 * 27 * 27 +
         i10(c6[2]) * 27 * 27 * 27 +
         i27(c6[3]) * 27 * 27 +
         i27(c6[4]) * 27 +
         i27(c6[5]))
    return NTOKENS + MAX22 + n



def _pack_grid_15(grid: str) -> int:
    """Pack a 4-char Maidenhead grid, report, or special token into 15 bits (ft8_lib convention).

    Returns (igrid4, ir) where igrid4 is the 15-bit value and ir is the 1-bit R-flag.
    For convenience, callers can unpack the tuple.
    """
    MAXGRID4 = 32400  # = 18*18*10*10
    grid = grid.upper().strip()

    if grid == 'RRR':  return MAXGRID4 + 2, 0
    if grid == 'RR73': return MAXGRID4 + 3, 0
    if grid == '73':   return MAXGRID4 + 4, 0

    # Signal report: e.g. '+04', '-12', 'R+04'
    import re as _re
    ir = 0
    s = grid
    if s.startswith('R') and len(s) > 1 and s[1] in '+-0123456789':
        ir = 1; s = s[1:]
    if _re.fullmatch(r'[+-]?\d{1,2}', s):
        try:
            dd = int(s)
            irpt = dd + 35
            return MAXGRID4 + irpt, ir
        except ValueError:
            pass

    # Standard 4-letter grid
    _GL = 'ABCDEFGHIJKLMNOPQR'
    if len(grid) >= 4 and grid[0] in _GL and grid[1] in _GL and grid[2].isdigit() and grid[3].isdigit():
        igrid4 = (_GL.index(grid[0]) * 18 + _GL.index(grid[1])) * 100 + int(grid[2]) * 10 + int(grid[3])
        return igrid4, 0

    return MAXGRID4 + 1, 0  # blank


def _build_type1_bits(call1: str, r1: int, call2: str, r2: int,
                       grid: str, i3: int = 1) -> np.ndarray:
    """
    Build a 77-bit standard FT8 message bit array (ft8_lib convention).

    77-bit layout (MSB-first, bits 0-76):
      [0:28]   = n28a (28-bit callsign 1)
      [28]     = ipa  (1 bit: /R or /P flag)
      [29:57]  = n28b (28-bit callsign 2)
      [57]     = ipb  (1 bit: /R or /P flag)
      [58]     = ir   (1 bit: R-prefix on grid/report)
      [59:74]  = igrid4 (15-bit grid/report value)
      [74:77]  = i3   (3 bits; 1=standard, 2=standard+/P)
    Total = 28+1+28+1+1+15+3 = 77 bits.
    """
    c1n = _pack_callsign_28(call1)
    c2n = _pack_callsign_28(call2)
    igrid4, ir = _pack_grid_15(grid)

    bits = (
        _int_to_bits(c1n, 28) +
        [r1 & 1] +
        _int_to_bits(c2n, 28) +
        [r2 & 1] +
        [ir & 1] +
        _int_to_bits(igrid4, 15) +
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
    """Pack up to 13 chars of free text into 71 bits using base-42 (i3=0,n3=0 message)."""
    _FT = " 0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ+-./?"   # 42 chars
    text = (text.upper() + ' ' * 13)[:13]
    n = 0
    for ch in text:
        idx = _FT.index(ch) if ch in _FT else 0
        n = n * 42 + idx
    bits71 = _int_to_bits(n, 71)
    # i3=0, n3=0: bits[71:74]=000 (n3), bits[74:77]=000 (i3)
    bits77 = bits71 + [0, 0, 0] + _int_to_bits(0, 3)   # n3=000, i3=000
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
    _int_to_bits(telem_val, 71) + _int_to_bits(4, 3) + _int_to_bits(0, 3),
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
    igrid4, ir = _pack_grid_15(grid)
    back = _unpack_grid(igrid4, ir)
    print(f"  '{grid}' → igrid4={igrid4} ir={ir} → '{back}'")
    assert back == grid, f"Round-trip mismatch: '{grid}' → '{back}'"
print("  Test 10 PASSED")


print("\n=== All message unpack tests passed ===")

