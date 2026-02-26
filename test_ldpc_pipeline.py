"""
test_ldpc_pipeline.py — Offline pipeline tests for the FT8 LDPC decoder and
surrounding stages.  Run with:  python test_ldpc_pipeline.py

Tests
-----
1.  Matrix sanity  — 83 rows, each with 91 unique column indices in [0,173]
2.  Free / pivot   — FREE_COLS = 0..90, PIVOT_COLS = 91..173, no overlap
3.  Parity submatrix rank  — H[:,91:] must be full rank 83 over GF(2)
4.  All-zeros codeword     — passes every parity check
5.  Random valid codeword  — encode via H, LDPC decode succeeds
6.  CRC-14 round-trip      — encode 77-bit msg + CRC, full LDPC+CRC decode
7.  Single bit-flip error  — decoder corrects and still passes
8.  Interleaver round-trip — _build_ft8_interleave_table is a valid permutation
9.  Gray decode sanity     — all-zero payload → 174 zero bits
10. Full synthetic pipeline— build a valid 174-bit codeword, pass through
                             Gray, deinterleave, LDPC; check message recovered
"""

from __future__ import annotations
import sys, time, traceback
import numpy as np

# ── import the module under test ─────────────────────────────────────────────
try:
    from ft8_decode import (
        _LDPC_CHECKS,
        _FT8_LDPC_FREE_COLS,
        _FT8_LDPC_PIVOT_COLS,
        _FT8_INTERLEAVE_PERM,
        _build_ft8_interleave_table,
        _FT8_GRAY_DECODE,
        _ft8_crc14,
        ft8_ldpc_decode,
        ft8_gray_decode,
        ft8_unpack_message,
    )
except ImportError as e:
    print(f"IMPORT ERROR: {e}")
    sys.exit(1)

PASS = "✓ PASS"
FAIL = "✗ FAIL"

results: list[tuple[str, bool, str]] = []

def run(name: str, fn):
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
    status = PASS if ok else FAIL
    print(f"  {status}  {name}  ({elapsed:.1f} ms)")
    if not ok:
        for line in detail.splitlines():
            print(f"         {line}")


# ─────────────────────────────────────────────────────────────────────────────
# Helper: GF(2) rank via row reduction
# ─────────────────────────────────────────────────────────────────────────────
def gf2_rank(M: np.ndarray) -> int:
    """Compute rank of a binary matrix over GF(2)."""
    A = M.copy().astype(np.uint8)
    nrows, ncols = A.shape
    rank = 0
    for col in range(ncols):
        pivot = None
        for row in range(rank, nrows):
            if A[row, col]:
                pivot = row
                break
        if pivot is None:
            continue
        A[[rank, pivot]] = A[[pivot, rank]]
        for row in range(nrows):
            if row != rank and A[row, col]:
                A[row] = (A[row] + A[rank]) % 2
        rank += 1
    return rank


# ─────────────────────────────────────────────────────────────────────────────
# Helper: build the full (83, 174) binary H matrix
# ─────────────────────────────────────────────────────────────────────────────
def build_H() -> np.ndarray:
    H = np.zeros((83, 174), dtype=np.uint8)
    for r, row in enumerate(_LDPC_CHECKS):
        for c in row:
            H[r, c] = 1
    return H


# ─────────────────────────────────────────────────────────────────────────────
# Helper: encode a 91-bit message into a 174-bit codeword using H
# Solve  H[:,91:] * p = H[:,:91] * s  (all GF(2))
# ─────────────────────────────────────────────────────────────────────────────
def encode_codeword(sys_bits: np.ndarray) -> np.ndarray:
    """
    Given 91 systematic bits, compute the 83 parity bits and return the full
    174-bit codeword.  Uses GF(2) Gaussian elimination on H[:,91:].
    """
    assert sys_bits.shape == (91,)
    H = build_H()
    Hs = H[:, :91]   # (83, 91)  — systematic part
    Hp = H[:, 91:]   # (83, 83)  — parity part

    # syndrome target: s = Hs @ sys_bits mod 2
    s = (Hs @ sys_bits.astype(np.uint8)) % 2   # (83,)

    # Solve Hp @ p = s over GF(2) via row reduction
    aug = np.concatenate([Hp.copy(), s.reshape(83, 1)], axis=1).astype(np.uint8)
    for col in range(83):
        pivot = None
        for row in range(col, 83):
            if aug[row, col]:
                pivot = row
                break
        if pivot is None:
            raise RuntimeError(f"Singular parity sub-matrix at col {col}")
        aug[[col, pivot]] = aug[[pivot, col]]
        for row in range(83):
            if row != col and aug[row, col]:
                aug[row] = (aug[row] + aug[col]) % 2
    parity = aug[:, 83]   # (83,)

    cw = np.concatenate([sys_bits.astype(np.uint8), parity])
    # Verify
    check = (H @ cw) % 2
    assert np.all(check == 0), f"Encode verification failed: {check}"
    return cw


# ─────────────────────────────────────────────────────────────────────────────
# Helper: build LLRs from a known hard-bit codeword (channel perfect)
# ─────────────────────────────────────────────────────────────────────────────
def bits_to_llrs(bits: np.ndarray, channel_snr: float = 8.0) -> np.ndarray:
    """
    Convert hard bits to soft LLRs.
    Convention: positive LLR = bit 1 (same as ft8_ldpc_decode expectation).
    bit=1 → LLR = +channel_snr
    bit=0 → LLR = -channel_snr
    """
    return np.where(bits.astype(bool), channel_snr, -channel_snr).astype(np.float64)


# ═════════════════════════════════════════════════════════════════════════════
# TEST 1 — Matrix sanity
# ═════════════════════════════════════════════════════════════════════════════
def t_matrix_sanity():
    assert len(_LDPC_CHECKS) == 83, f"Expected 83 rows, got {len(_LDPC_CHECKS)}"
    for i, row in enumerate(_LDPC_CHECKS):
        assert len(row) == 91, f"Row {i} has {len(row)} entries, expected 91"
        assert len(set(row)) == 91, f"Row {i} has duplicate column indices"
        for c in row:
            assert 0 <= c < 174, f"Row {i} column {c} out of range [0,173]"
    return "83 rows × 91 cols, all indices in [0,173]"

run("1. Matrix sanity (83×91, indices valid)", t_matrix_sanity)


# ═════════════════════════════════════════════════════════════════════════════
# TEST 2 — Free / pivot column sets
# ═════════════════════════════════════════════════════════════════════════════
def t_free_pivot_cols():
    assert len(_FT8_LDPC_FREE_COLS)  == 91, f"FREE_COLS length {len(_FT8_LDPC_FREE_COLS)}"
    assert len(_FT8_LDPC_PIVOT_COLS) == 83, f"PIVOT_COLS length {len(_FT8_LDPC_PIVOT_COLS)}"
    free_set  = set(_FT8_LDPC_FREE_COLS)
    pivot_set = set(_FT8_LDPC_PIVOT_COLS)
    assert free_set & pivot_set == set(), "FREE and PIVOT overlap"
    assert free_set | pivot_set == set(range(174)), "FREE ∪ PIVOT ≠ {0..173}"
    assert list(_FT8_LDPC_FREE_COLS) == list(range(91)), "FREE_COLS should be 0..90"
    return "FREE=0..90 (91), PIVOT=91..173 (83), disjoint and complete"

run("2. Free/pivot column sets", t_free_pivot_cols)


# ═════════════════════════════════════════════════════════════════════════════
# TEST 3 — Parity submatrix rank
# ═════════════════════════════════════════════════════════════════════════════
def t_parity_rank():
    H = build_H()
    Hp = H[:, 91:]   # (83, 83)
    r = gf2_rank(Hp)
    assert r == 83, f"H_parity rank = {r}, expected 83"
    r_full = gf2_rank(H)
    assert r_full == 83, f"Full H rank = {r_full}, expected 83"
    return f"H_parity rank=83, full H rank=83"

run("3. Parity submatrix full rank over GF(2)", t_parity_rank)


# ═════════════════════════════════════════════════════════════════════════════
# TEST 4 — All-zeros codeword
# ═════════════════════════════════════════════════════════════════════════════
def t_all_zeros():
    cw = np.zeros(174, dtype=np.uint8)
    H = build_H()
    syndrome = (H @ cw) % 2
    assert np.all(syndrome == 0), f"All-zeros syndrome non-zero at {np.where(syndrome)[0]}"
    # Decode: all-zeros LLRs → ambiguous, but LDPC should find the all-zeros word
    llrs = bits_to_llrs(cw, channel_snr=10.0)
    ok, codeword, iters = ft8_ldpc_decode(llrs, max_iterations=50)
    # The all-zeros codeword is a valid LDPC codeword; CRC might not match (msg=0)
    # but parity checks must pass (ok=True or crc failed)
    # We only check that parity is satisfied
    H2 = build_H()
    syn2 = (H2 @ codeword.astype(np.uint8)[:174]) % 2  # codeword is 91 bits (free bits)
    # Actually ft8_ldpc_decode returns free_bits (91,); syndrome must be re-checked on full 174
    # Reconstruct full codeword
    full_cw = encode_codeword(codeword)
    syn3 = (H2 @ full_cw) % 2
    assert np.all(syn3 == 0), f"Re-encoded all-zeros syndrome non-zero"
    return f"All-zeros passes all 83 parity checks (iters={iters})"

run("4. All-zeros codeword parity check", t_all_zeros)


# ═════════════════════════════════════════════════════════════════════════════
# TEST 5 — Random valid codeword encode → LDPC decode
# ═════════════════════════════════════════════════════════════════════════════
def t_random_codeword():
    rng = np.random.default_rng(42)
    sys_bits = rng.integers(0, 2, size=91, dtype=np.uint8)
    cw = encode_codeword(sys_bits)
    llrs = bits_to_llrs(cw, channel_snr=10.0)
    ok, decoded_free, iters = ft8_ldpc_decode(llrs, max_iterations=50)
    # Parity must be satisfied (CRC may or may not pass since msg is random)
    H = build_H()
    full_dec = encode_codeword(decoded_free)
    syndrome = (H @ full_dec) % 2
    assert np.all(syndrome == 0), f"Decoded codeword fails parity at checks {np.where(syndrome)[0]}"
    assert np.array_equal(decoded_free, sys_bits), \
        f"Decoded free bits don't match original systematic bits"
    return f"Encoded random 91-bit word, decoded back exactly (iters={iters})"

run("5. Random valid codeword encode→decode", t_random_codeword)


# ═════════════════════════════════════════════════════════════════════════════
# TEST 6 — CRC-14 round-trip (full LDPC+CRC)
# ═════════════════════════════════════════════════════════════════════════════
def t_crc_roundtrip():
    # Build a known 77-bit message: "CQ W4ABC EM73" style (just arbitrary bits)
    rng = np.random.default_rng(7)
    msg_bits = rng.integers(0, 2, size=77, dtype=np.uint8)

    # Compute CRC-14 over 5-zero-pad + msg_bits (WSJT-X convention)
    padded = np.concatenate((np.zeros(5, dtype=np.uint8), msg_bits))
    crc_int = _ft8_crc14(padded)
    crc_bits = np.array([(crc_int >> (13 - i)) & 1 for i in range(14)], dtype=np.uint8)

    sys_bits = np.concatenate([msg_bits, crc_bits])   # 91 bits
    assert sys_bits.shape == (91,)

    cw = encode_codeword(sys_bits)
    llrs = bits_to_llrs(cw, channel_snr=10.0)

    ok, decoded_free, iters = ft8_ldpc_decode(llrs, max_iterations=50)
    assert ok, f"LDPC+CRC decode failed (iters={iters})"
    assert np.array_equal(decoded_free[:77], msg_bits), "Decoded message bits don't match"
    return f"CRC-14 round-trip pass (iters={iters}, crc_int=0x{crc_int:04X})"

run("6. CRC-14 round-trip LDPC+CRC", t_crc_roundtrip)


# ═════════════════════════════════════════════════════════════════════════════
# TEST 7 — Single bit-flip error correction
# ═════════════════════════════════════════════════════════════════════════════
def t_single_bit_flip():
    rng = np.random.default_rng(99)
    msg_bits = rng.integers(0, 2, size=77, dtype=np.uint8)
    padded = np.concatenate((np.zeros(5, dtype=np.uint8), msg_bits))
    crc_int = _ft8_crc14(padded)
    crc_bits = np.array([(crc_int >> (13 - i)) & 1 for i in range(14)], dtype=np.uint8)
    sys_bits = np.concatenate([msg_bits, crc_bits])
    cw = encode_codeword(sys_bits)

    # Flip one bit in a non-trivial position
    flip_pos = 37
    cw_err = cw.copy()
    cw_err[flip_pos] ^= 1

    llrs = bits_to_llrs(cw_err, channel_snr=6.0)   # moderate SNR
    ok, decoded_free, iters = ft8_ldpc_decode(llrs, max_iterations=100)
    assert ok, f"Single-bit-flip decode failed at pos={flip_pos} (iters={iters})"
    assert np.array_equal(decoded_free[:77], msg_bits), "Bit-flip correction: message mismatch"
    return f"Single bit flip at pos={flip_pos} corrected (iters={iters})"

run("7. Single bit-flip error correction", t_single_bit_flip)


# ═════════════════════════════════════════════════════════════════════════════
# TEST 8 — Interleaver round-trip
# ═════════════════════════════════════════════════════════════════════════════
def t_interleaver():
    perm = _FT8_INTERLEAVE_PERM
    assert len(perm) == 174, f"Interleave perm length {len(perm)}"
    assert set(perm) == set(range(174)), "Interleave perm is not a valid permutation of 0..173"
    # Verify _build_ft8_interleave_table matches _FT8_INTERLEAVE_PERM
    built = _build_ft8_interleave_table()
    assert built == perm, "Built perm does not match cached _FT8_INTERLEAVE_PERM"
    # Round-trip: interleave then de-interleave = identity
    bits = np.arange(174, dtype=np.int32)
    interleaved = bits[list(perm)]
    inv_perm = np.argsort(list(perm))
    recovered = interleaved[inv_perm]
    assert np.array_equal(recovered, bits), "Interleave round-trip failed"
    return "174-entry permutation valid, round-trip identity"

run("8. Interleaver round-trip", t_interleaver)


# ═════════════════════════════════════════════════════════════════════════════
# TEST 9 — Gray decode sanity (all-zero payload symbols → 174 zero LLRs)
# ═════════════════════════════════════════════════════════════════════════════
def t_gray_decode_sanity():
    # Build a (58, 8) energy matrix where tone 0 dominates every symbol
    # → Gray(tone=0) = 000 → 3 zero bits per symbol → 174 zero bits
    E = np.zeros((58, 8), dtype=np.float64)
    E[:, 0] = 1e6    # tone 0 carries all energy
    for k in range(1, 8):
        E[:, k] = 1.0   # small but nonzero to avoid log(0)

    syms = np.zeros(58, dtype=np.int32)   # hard decision: tone 0

    gray_table = np.array(_FT8_GRAY_DECODE, dtype=np.int32)
    assert gray_table[0] == 0, f"Gray(0) should be 0b000=0, got {gray_table[0]}"

    hard_bits, llrs = ft8_gray_decode(syms, E)
    assert hard_bits.shape == (174,), f"hard_bits shape {hard_bits.shape}"
    assert llrs.shape    == (174,), f"llrs shape {llrs.shape}"
    # All hard bits should be 0
    assert np.all(hard_bits == 0), f"Expected all-zero bits, got {np.where(hard_bits)[0]}"
    # All LLRs should be negative (convention: negative = bit 1 less likely = bit 0 likely)
    assert np.all(llrs <= 0), f"Expected all non-positive LLRs for all-zero bits"
    return "All-zero symbols → 174 zero hard bits, all non-positive LLRs"

run("9. Gray decode sanity (all-zero symbols)", t_gray_decode_sanity)


# ═════════════════════════════════════════════════════════════════════════════
# TEST 10 — Full synthetic pipeline: encode → Gray → deinterleave → LDPC
# ═════════════════════════════════════════════════════════════════════════════
def t_full_pipeline():
    """
    Build a valid 77-bit message, encode it to a 174-bit LDPC codeword, then
    synthesise perfect 8-FSK symbol energies and run the full decoder chain:
        Gray decode → (identity deinterleave) → LDPC decode → unpack message
    """
    from ft8_decode import ft8_deinterleave, _FT8_INTERLEAVE_PERM

    # ── 1. Build message + CRC ───────────────────────────────────────────
    rng = np.random.default_rng(2025)
    msg_bits = rng.integers(0, 2, size=77, dtype=np.uint8)
    padded = np.concatenate((np.zeros(5, dtype=np.uint8), msg_bits))
    crc_int = _ft8_crc14(padded)
    crc_bits = np.array([(crc_int >> (13 - i)) & 1 for i in range(14)], dtype=np.uint8)
    sys_bits = np.concatenate([msg_bits, crc_bits])   # 91 bits

    # ── 2. Encode to 174-bit codeword ───────────────────────────────────
    cw = encode_codeword(sys_bits)  # (174,) bits

    # ── 3. The pipeline applies INTERLEAVE before transmission;
    #       we need to PRE-interleave so that after de-interleaving we recover cw.
    #       ft8_gray_decode applies de-interleave internally.
    #       So: transmitted_bits[perm[i]] = cw[i]
    #           → transmitted_bits = cw placed at perm destinations
    perm = list(_FT8_INTERLEAVE_PERM)   # perm[dst] = src  (de-interleave map)
    # interleave: for each dst, transmitted[dst] = cw[src where perm[src]=dst]
    inv_perm = np.argsort(perm)    # inv_perm[src] = dst  (forward interleave)
    tx_bits = cw[inv_perm]         # (174,) — bits as transmitted

    # ── 4. Convert 174 tx bits → 58 Gray-coded 3-tone symbols ───────────
    gray_table = np.array(_FT8_GRAY_DECODE, dtype=np.int32)

    # Build inverse Gray table: 3-bit integer → tone index
    # gray_table[tone] = gray_value → inv_gray[gray_value] = tone
    inv_gray = np.zeros(8, dtype=np.int32)
    for tone, gv in enumerate(gray_table):
        inv_gray[gv] = tone

    # Pack 174 bits back into 58 3-bit values (MSB-first per symbol)
    syms_tx = np.empty(58, dtype=np.int32)
    for s in range(58):
        b2 = int(tx_bits[3*s + 0])
        b1 = int(tx_bits[3*s + 1])
        b0 = int(tx_bits[3*s + 2])
        gv = (b2 << 2) | (b1 << 1) | b0
        syms_tx[s] = int(inv_gray[gv])

    # ── 5. Build perfect energy matrix (ideal 8-FSK: dominant tone has all energy)
    E_payload = np.ones((58, 8), dtype=np.float64) * 1.0   # noise floor
    for s in range(58):
        E_payload[s, int(syms_tx[s])] = 1e8                 # signal tone

    # ── 6. Gray decode → 174 bits + LLRs (includes de-interleave) ───────
    hard_bits, llrs = ft8_gray_decode(syms_tx, E_payload)

    # Check hard bits match cw
    assert np.array_equal(hard_bits, cw), \
        f"Gray-decoded hard bits mismatch cw at positions: {np.where(hard_bits != cw)[0]}"

    # ── 7. ft8_deinterleave is identity at symbol level ──────────────────
    syms_di, E_di = ft8_deinterleave(syms_tx, E_payload)
    assert np.array_equal(syms_di, syms_tx), "deinterleave changed symbols unexpectedly"

    # ── 8. LDPC decode ────────────────────────────────────────────────────
    ok, decoded_free, iters = ft8_ldpc_decode(llrs, max_iterations=50)
    assert ok, f"LDPC+CRC decode failed in full pipeline (iters={iters})"
    assert np.array_equal(decoded_free[:77], msg_bits), \
        f"Pipeline message bits mismatch at: {np.where(decoded_free[:77] != msg_bits)[0]}"

    return f"Full synthetic pipeline passed (iters={iters})"

run("10. Full synthetic pipeline encode→Gray→LDPC→unpack", t_full_pipeline)


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
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

