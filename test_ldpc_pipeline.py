"""
test_ldpc_pipeline.py — Pipeline tests matching ft8_lib reference exactly.

Reference pipeline (ft8_lib / WSJT-X):
  1. Encode: LDPC(91 bits) → 174-bit codeword, bits 0..90 = msg+CRC, 91..173 = parity
  2. Tones: groups of 3 codeword bits → Gray-coded 8-FSK tone (no interleaver)
  3. Gray decode: tone energies → 174 LLRs, sign: positive = bit 0 more likely
  4. Normalize: scale LLRs so variance = 24
  5. BP decode: tanh sum-product → 174 bits, check parity, CRC on bits 0..76 (zero-pad to 82)
  6. Message: plain[0..76]
"""
from __future__ import annotations
import sys, time, traceback, math
import numpy as np

try:
    from ft8_decode import (
        _LDPC_CHECKS, _FT8_LDPC_FREE_COLS, _FT8_LDPC_PIVOT_COLS,
        _FT8_GRAY_DECODE, _ft8_crc14,
        ft8_ldpc_decode, ft8_gray_decode, ft8_unpack_message, ft8_deinterleave,
    )
except ImportError as e:
    print(f"IMPORT ERROR: {e}"); sys.exit(1)

PASS = "✓ PASS"; FAIL = "✗ FAIL"
results: list[tuple[str, bool, str]] = []

def run(name, fn):
    t0 = time.perf_counter()
    try:
        msg = fn(); ok = True; detail = msg or ""
    except Exception as exc:
        ok = False
        detail = f"{type(exc).__name__}: {exc}\n{''.join(traceback.format_tb(exc.__traceback__))}"
    elapsed = (time.perf_counter() - t0) * 1000
    results.append((name, ok, detail))
    print(f"  {PASS if ok else FAIL}  {name}  ({elapsed:.1f} ms)")
    if not ok:
        for line in detail.splitlines(): print(f"         {line}")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_H():
    H = np.zeros((83, 174), dtype=np.uint8)
    for r, row in enumerate(_LDPC_CHECKS):
        for c in row: H[r, c] = 1
    return H

def gf2_rank(M):
    A = M.copy().astype(np.int32); rank = 0
    for col in range(A.shape[1]):
        piv = next((r for r in range(rank, A.shape[0]) if A[r, col]), None)
        if piv is None: continue
        A[[rank, piv]] = A[[piv, rank]]
        for r in range(A.shape[0]):
            if r != rank and A[r, col]: A[r] = (A[r] + A[rank]) % 2
        rank += 1
    return rank

# Encode using ft8_lib generator matrix approach:
# bits 0..90 = systematic (msg+CRC), bits 91..173 computed from parity check H
def encode_ft8(msg77: np.ndarray) -> np.ndarray:
    """Produce valid 174-bit codeword with bits[0..90] = msg+CRC, bits[91..173] = parity."""
    assert len(msg77) == 77
    crc = _ft8_crc14(msg77)
    crc_bits = np.array([(crc >> (13 - i)) & 1 for i in range(14)], dtype=np.uint8)
    sys_bits = np.concatenate([msg77, crc_bits])   # 91 bits
    # Solve H[:,91:173] * parity = H[:,0:91] * sys  (mod 2)
    H = build_H()
    Hs = H[:, :91]; Hp = H[:, 91:]
    rhs = (Hs.astype(np.int32) @ sys_bits.astype(np.int32)) % 2
    # Gauss-Jordan solve Hp * p = rhs
    aug = np.concatenate([Hp.copy().astype(np.uint8),
                           rhs.reshape(83, 1).astype(np.uint8)], axis=1)
    for col in range(83):
        piv = next((r for r in range(col, 83) if aug[r, col]), None)
        if piv is None: raise RuntimeError(f"Singular at col {col}")
        aug[[col, piv]] = aug[[piv, col]]
        for r in range(83):
            if r != col and aug[r, col]: aug[r] = (aug[r] + aug[col]) % 2
    parity = aug[:, 83]
    cw = np.concatenate([sys_bits, parity])  # 174 bits, systematic layout
    assert np.all((H.astype(np.int32) @ cw.astype(np.int32)) % 2 == 0), "Codeword invalid"
    return cw

def cw_to_tones(cw: np.ndarray) -> np.ndarray:
    """Convert 174 codeword bits → 58 Gray-coded tones (no interleaver)."""
    gray_map = list(_FT8_GRAY_DECODE)
    inv_gray = [0] * 8
    for tone, gv in enumerate(gray_map): inv_gray[gv] = tone
    tones = np.array([
        inv_gray[(int(cw[3*s]) << 2) | (int(cw[3*s+1]) << 1) | int(cw[3*s+2])]
        for s in range(58)
    ], dtype=np.int32)
    return tones

def tones_to_E(tones: np.ndarray, snr: float = 100.0) -> np.ndarray:
    """Build (58,8) energy matrix from hard tone decisions at given SNR."""
    E = np.ones((58, 8), dtype=np.float64)
    for s, t in enumerate(tones): E[s, t] = snr
    return E

def normalize_llrs(llrs: np.ndarray) -> np.ndarray:
    """Match ft8_lib ftx_normalize_logl: scale so variance = 24."""
    var = float(np.var(llrs))
    if var < 1e-10: return llrs
    return llrs * math.sqrt(24.0 / var)

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def t_matrix_sanity():
    assert len(_LDPC_CHECKS) == 83
    for row in _LDPC_CHECKS:
        assert 1 <= len(row) <= 7
        assert len(set(row)) == len(row)
        for c in row: assert 0 <= c < 174
    return "83 checks, all indices in [0,173]"
run("1. Matrix sanity", t_matrix_sanity)

def t_parity_rank():
    H = build_H()
    assert gf2_rank(H) == 83, "H not full rank"
    # With systematic layout bits 0..90 = free, 91..173 = parity
    Hp = H[:, 91:]
    assert gf2_rank(Hp) == 83, "Parity submatrix not full rank"
    return "H rank=83, H[:,91:] rank=83"
run("2. Parity submatrix rank", t_parity_rank)

def t_encode_valid():
    rng = np.random.default_rng(42)
    msg = rng.integers(0, 2, size=77, dtype=np.uint8)
    cw = encode_ft8(msg)
    H = build_H()
    assert np.all((H.astype(np.int32) @ cw.astype(np.int32)) % 2 == 0)
    return "Random message encodes to valid codeword"
run("3. Encode produces valid codeword", t_encode_valid)

def t_crc_roundtrip():
    msg = np.array([1,0,1,0,1,0,1] + [0]*70, dtype=np.uint8)
    crc = _ft8_crc14(msg)
    # CRC should be 14-bit
    assert 0 <= crc < 16384
    # Re-encode and check CRC in codeword
    cw = encode_ft8(msg)
    crc_bits = cw[77:91]
    rx_crc = sum(int(b) << (13-i) for i,b in enumerate(crc_bits))
    assert crc == rx_crc, f"CRC mismatch: calc=0x{crc:04X} rx=0x{rx_crc:04X}"
    return f"CRC=0x{crc:04X} round-trips correctly"
run("4. CRC-14 round-trip", t_crc_roundtrip)

def t_gray_decode_allzero():
    """All tone-0 symbols → all-zero bits with negative LLRs (positive = bit 1 convention)."""
    gray_map = list(_FT8_GRAY_DECODE)
    # tone 0 → gray value 0 → bits 0,0,0
    assert gray_map[0] == 0
    E = np.ones((58, 8)); E[:, 0] = 1e6
    syms = np.zeros(58, dtype=np.int32)
    hard, llrs = ft8_gray_decode(syms, E)
    assert np.all(hard == 0), f"Expected all-zero bits"
    # positive = bit 1 more likely, so bit=0 should give negative LLRs
    assert np.all(llrs <= 0), f"Expected all non-positive LLRs (positive=bit1 convention)"
    return "Tone-0 → 174 zero bits, all LLRs ≤ 0 ✓"
run("5. Gray decode sanity (all-zero tones)", t_gray_decode_allzero)

def t_clean_decode():
    """Perfect signal: encode → tones → Gray decode → LDPC → recover message."""
    rng = np.random.default_rng(1234)
    msg = rng.integers(0, 2, size=77, dtype=np.uint8)
    cw = encode_ft8(msg)
    tones = cw_to_tones(cw)
    E = tones_to_E(tones)
    hard, llrs = ft8_gray_decode(tones, E)
    llrs_norm = normalize_llrs(llrs)
    # Hard bits should exactly match codeword
    assert np.all(hard == cw), f"Hard bits mismatch at {np.where(hard != cw)[0]}"
    ok, payload, iters = ft8_ldpc_decode(llrs_norm)
    assert ok, f"LDPC failed (iters={iters})"
    assert np.all(payload[:77] == msg), "Message mismatch"
    return f"Clean decode ok (iters={iters})"
run("6. Clean encode→decode round-trip", t_clean_decode)

def t_noisy_decode():
    """Gaussian noise at ~3dB: still decodes."""
    rng = np.random.default_rng(999)
    msg = rng.integers(0, 2, size=77, dtype=np.uint8)
    cw = encode_ft8(msg)
    tones = cw_to_tones(cw)
    E = tones_to_E(tones, snr=20.0)
    _, llrs = ft8_gray_decode(tones, E)
    noise = rng.standard_normal(174) * 1.5
    llrs_noisy = normalize_llrs(llrs + noise)
    ok, payload, iters = ft8_ldpc_decode(llrs_noisy, max_iterations=100)
    assert ok, f"Noisy decode failed (iters={iters})"
    assert np.all(payload[:77] == msg)
    return f"Noisy decode ok (iters={iters})"
run("7. Noisy decode (Gaussian noise)", t_noisy_decode)

def t_full_pipeline():
    """Full Stage 2→3→4→5 pipeline including ft8_deinterleave (pass-through)."""
    rng = np.random.default_rng(2025)
    msg = rng.integers(0, 2, size=77, dtype=np.uint8)
    cw = encode_ft8(msg)
    tones = cw_to_tones(cw)
    E = tones_to_E(tones)
    # ft8_deinterleave is a no-op pass-through at symbol level
    syms_di, E_di = ft8_deinterleave(tones, E)
    assert np.array_equal(syms_di, tones)
    hard, llrs = ft8_gray_decode(syms_di, E_di)
    llrs_norm = normalize_llrs(llrs)
    ok, payload, iters = ft8_ldpc_decode(llrs_norm)
    assert ok, f"LDPC failed (iters={iters})"
    assert np.all(payload[:77] == msg)
    decoded = ft8_unpack_message(payload[:77])
    return f"Full pipeline ok, msg='{decoded}' (iters={iters})"
run("8. Full pipeline (deinterleave+gray+LDPC+unpack)", t_full_pipeline)

def t_multiple_messages():
    """Decode 10 random messages reliably."""
    rng = np.random.default_rng(777)
    failed = []
    for i in range(10):
        msg = rng.integers(0, 2, size=77, dtype=np.uint8)
        cw = encode_ft8(msg)
        E = tones_to_E(cw_to_tones(cw))
        _, llrs = ft8_gray_decode(cw_to_tones(cw), E)
        ok, payload, _ = ft8_ldpc_decode(normalize_llrs(llrs))
        if not ok or not np.all(payload[:77] == msg):
            failed.append(i)
    assert not failed, f"Failed on messages: {failed}"
    return "10/10 random messages decode correctly"
run("9. 10 random messages", t_multiple_messages)

def t_known_crc():
    """Verify CRC value for all-zeros message matches expected."""
    msg = np.zeros(77, dtype=np.uint8)
    crc = _ft8_crc14(msg)
    # All-zeros 82-bit input → CRC should be 0
    assert crc == 0, f"CRC of all-zeros should be 0, got 0x{crc:04X}"
    return f"CRC(all-zeros 77 bits) = 0x{crc:04X} ✓"
run("10. CRC of all-zeros = 0", t_known_crc)

def t_real_interleaver_tables():
    """Verify that _FT8_DEINTERLEAVE correctly inverts _FT8_INTERLEAVE."""
    from ft8_decode import _FT8_INTERLEAVE, _FT8_DEINTERLEAVE

    # Validation 1: Check basic properties
    assert len(_FT8_INTERLEAVE) == 174
    assert len(_FT8_DEINTERLEAVE) == 174

    # Validation 2: Simulate transmission
    # cw[i] is codeword bit i
    # tx[j] = cw[_FT8_INTERLEAVE[j]] is transmitted bit j

    # Check that DEINTERLEAVE inverts INTERLEAVE
    # recovered_cw[i] = tx[_FT8_DEINTERLEAVE[i]] (if >= 0)
    #                 = cw[_FT8_INTERLEAVE[_FT8_DEINTERLEAVE[i]]]

    # So we require: _FT8_INTERLEAVE[_FT8_DEINTERLEAVE[i]] == i  FOR ALL i where _FT8_DEINTERLEAVE[i] != -1

    erased_count = 0
    for i in range(174):
        dst = _FT8_DEINTERLEAVE[i]
        if dst == -1:
            erased_count += 1
            continue

        src_recheck = _FT8_INTERLEAVE[dst]
        assert src_recheck == i, f"Mismatch at cw bit {i}: maps to tx {dst}, but tx {dst} comes from cw {src_recheck}"

    return f"De-interleaver table validates against interleaver. Erased bits: {erased_count}"
run("11. Real Interleaver tables check", t_real_interleaver_tables)

# ---------------------------------------------------------------------------
print()
passed = sum(1 for _, ok, _ in results if ok)
failed = sum(1 for _, ok, _ in results if not ok)
print(f"Results: {passed}/{len(results)} passed", end="")
if failed:
    print(f"  ({failed} FAILED)"); sys.exit(1)
else:
    print("  — all tests passed ✓"); sys.exit(0)
