"""
ft8_decode.py  —  Clean-slate FT8 decoder based on the WSJT-X / ft8_lib reference.

Pipeline (matches ft8_lib / WSJT-X exactly):

  Audio  →  Resample → 12 kHz mono
         →  UTC-aligned 15 s frames
         →  Candidate detection (8-tone matched filter)
         →  Sync search (Costas array score over dt/df grid)
         →  Symbol energy extraction  (79 symbols × 8 tones)
         →  Payload extraction        (58 payload symbols)
         →  Gray decode               (58 symbols → 174 channel LLRs)
         →  LLR normalisation         (variance = 24  [ftx_normalize_logl])
         →  LDPC BP decode            (174,91) sum-product
         →  CRC-14 check
         →  Message unpack            (77 bits → human string)

Key reference parameters (WSJT-X / ft8_lib constants):
  Sample rate       12 000 Hz
  Symbol duration   160 ms  (1920 samples at 12 kHz)
  Tone spacing      6.25 Hz  (= 1 / 0.160)
  Symbols per frame 79  (58 data + 21 Costas sync)
  Costas positions  0-6, 36-42, 72-78
  Costas pattern    {3,1,4,0,6,5,2}  (WSJT-X source ft8b.f90)
  FEC               LDPC (174, 91)  — 83 check nodes, col-weight 3
  CRC               14-bit, polynomial 0x2757
  Transmission time 12.64 s

Correctness notes (vs the previous broken implementation):
  1. Gray map  _FT8_GRAY_DECODE = (0,1,3,2,6,4,5,7)
              This is the exact inverse of ft8_lib kFT8_Gray_map = {0,1,3,2,5,6,4,7}.
  2. LDPC BP   C→V message = −2·atanh(product) with ft8_lib positive-means-1 LLR convention.
  3. No bit interleaver in encode/decode path (ft8_lib uses none).
"""
from __future__ import annotations

import math
import queue
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import numpy as np
from scipy import signal as _scipy_signal


# ═══════════════════════════════════════════════════════════════════════════════
# § 1  FT8 Protocol Constants
# ═══════════════════════════════════════════════════════════════════════════════

FT8_TONE_SPACING_HZ: float = 6.25
FT8_SYMBOL_DURATION_S: float = 0.160        # 1 / 6.25
FT8_NSYMS: int = 79
FT8_NDATASYM: int = 58
FT8_FS: int = 12_000
FT8_SYM_SAMPLES: int = int(round(FT8_SYMBOL_DURATION_S * FT8_FS))  # 1920
FT8_TX_DURATION_S: float = FT8_NSYMS * FT8_SYMBOL_DURATION_S       # 12.64 s

# Costas sync array positions (3 blocks of 7, at symbols 0-6, 36-42, 72-78)
FT8_COSTAS_POSITIONS: tuple[int, ...] = (
    0, 1, 2, 3, 4, 5, 6,
    36, 37, 38, 39, 40, 41, 42,
    72, 73, 74, 75, 76, 77, 78,
)

# Costas sync tone pattern (from WSJT-X ft8b.f90 / ft8_lib)
FT8_COSTAS_TONES: tuple[int, ...] = (3, 1, 4, 0, 6, 5, 2)

# Payload symbol positions (all 79 minus the 21 Costas positions)
FT8_PAYLOAD_POSITIONS: tuple[int, ...] = tuple(
    s for s in range(FT8_NSYMS) if s not in set(FT8_COSTAS_POSITIONS)
)

# ---------------------------------------------------------------------------
# Gray code  (ft8_lib constants.c  kFT8_Gray_map)
# ---------------------------------------------------------------------------
# Forward map: gray_value (3-bit integer) → tone index
#   kFT8_Gray_map[8] = {0, 1, 3, 2, 5, 6, 4, 7}
_FT8_GRAY_MAP: tuple[int, ...] = (0, 1, 3, 2, 5, 6, 4, 7)

# Inverse map: tone index → gray_value
#   Derived: for gv, tone in enumerate(_FT8_GRAY_MAP): inv[tone] = gv
#   Result:  (0, 1, 3, 2, 6, 4, 5, 7)
_tmp_inv = [0] * 8
for _gv, _tone in enumerate(_FT8_GRAY_MAP):
    _tmp_inv[_tone] = _gv
_FT8_GRAY_DECODE: tuple[int, ...] = tuple(_tmp_inv)
del _tmp_inv, _gv, _tone


# ═══════════════════════════════════════════════════════════════════════════════
# § 2  LDPC (174, 91) Parity-Check Matrix
# ═══════════════════════════════════════════════════════════════════════════════
# Source: ft8_lib constants.c  kFTX_LDPC_Nm  (0-based column indices).
# 83 check rows, 174 variable columns, column weight 3 (regular LDPC).
# Systematic bits occupy columns 0–90; parity bits columns 91–173.

_LDPC_CHECKS: tuple[tuple[int, ...], ...] = (
    (3, 30, 58, 90, 91, 95, 152),
    (4, 31, 59, 92, 114, 145),
    (5, 23, 60, 93, 121, 150),
    (6, 32, 61, 94, 95, 142),
    (7, 24, 62, 82, 92, 95, 147),
    (5, 31, 63, 96, 125, 137),
    (4, 33, 64, 77, 97, 106, 153),
    (8, 34, 65, 98, 138, 145),
    (9, 35, 66, 99, 106, 125),
    (10, 36, 66, 86, 100, 138, 157),
    (11, 37, 67, 101, 104, 154),
    (12, 38, 68, 102, 148, 161),
    (7, 39, 69, 81, 103, 113, 144),
    (13, 40, 70, 87, 101, 122, 155),
    (14, 41, 58, 105, 122, 158),
    (0, 32, 71, 105, 106, 156),
    (15, 42, 72, 107, 140, 159),
    (16, 36, 73, 80, 108, 130, 153),
    (10, 43, 74, 109, 120, 165),
    (44, 54, 63, 110, 129, 160, 172),
    (7, 45, 70, 111, 118, 165),
    (17, 35, 75, 88, 112, 113, 142),
    (18, 37, 76, 103, 115, 162),
    (19, 46, 69, 91, 137, 164),
    (1, 47, 73, 112, 127, 159),
    (20, 44, 77, 82, 116, 120, 150),
    (21, 46, 57, 117, 126, 163),
    (15, 38, 61, 111, 133, 157),
    (22, 42, 78, 119, 130, 144),
    (18, 34, 58, 72, 109, 124, 160),
    (19, 35, 62, 93, 135, 160),
    (13, 30, 78, 97, 131, 163),
    (2, 43, 79, 123, 126, 168),
    (18, 45, 80, 116, 134, 166),
    (6, 48, 57, 89, 99, 104, 167),
    (11, 49, 60, 117, 118, 143),
    (12, 50, 63, 113, 117, 156),
    (23, 51, 75, 128, 147, 148),
    (24, 52, 68, 89, 100, 129, 155),
    (19, 45, 64, 79, 119, 139, 169),
    (20, 53, 76, 99, 139, 170),
    (34, 81, 132, 141, 170, 173),
    (13, 29, 82, 112, 124, 169),
    (3, 28, 67, 119, 133, 172),
    (0, 3, 51, 56, 85, 135, 151),
    (25, 50, 55, 90, 121, 136, 167),
    (51, 83, 109, 114, 144, 167),
    (6, 49, 80, 98, 131, 172),
    (22, 54, 66, 94, 171, 173),
    (25, 40, 76, 108, 140, 147),
    (1, 26, 40, 60, 61, 114, 132),
    (26, 39, 55, 123, 124, 125),
    (17, 48, 54, 123, 140, 166),
    (5, 32, 84, 107, 115, 155),
    (27, 47, 69, 84, 104, 128, 157),
    (8, 53, 62, 130, 146, 154),
    (21, 52, 67, 108, 120, 173),
    (2, 12, 47, 77, 94, 122),
    (30, 68, 132, 149, 154, 168),
    (11, 42, 65, 88, 96, 134, 158),
    (4, 38, 74, 101, 135, 166),
    (1, 53, 85, 100, 134, 163),
    (14, 55, 86, 107, 118, 170),
    (9, 43, 81, 90, 110, 143, 148),
    (22, 33, 70, 93, 126, 152),
    (10, 48, 87, 91, 141, 156),
    (28, 33, 86, 96, 146, 161),
    (29, 49, 59, 85, 136, 141, 161),
    (9, 52, 65, 83, 111, 127, 164),
    (21, 56, 84, 92, 139, 158),
    (27, 31, 71, 102, 131, 165),
    (27, 28, 83, 87, 116, 142, 149),
    (0, 25, 44, 79, 127, 146),
    (16, 26, 88, 102, 115, 152),
    (50, 56, 97, 162, 164, 171),
    (20, 36, 72, 137, 151, 168),
    (15, 46, 75, 129, 136, 153),
    (2, 23, 29, 71, 103, 138),
    (8, 39, 89, 105, 133, 150),
    (14, 57, 59, 73, 110, 149, 162),
    (17, 41, 78, 143, 145, 151),
    (24, 37, 64, 98, 121, 159),
    (16, 41, 74, 128, 169, 171),
)

# Pre-built adjacency lists for the BP decoder.
# _BP_Nm[m] = variable indices for check m  (check → variables)
# _BP_Mn[n] = check indices for variable n  (variable → checks, length always 3)
_BP_Nm: list[list[int]] = [list(row) for row in _LDPC_CHECKS]
_BP_Mn: list[list[int]] = [[] for _ in range(174)]
for _m, _row in enumerate(_BP_Nm):
    for _n in _row:
        _BP_Mn[_n].append(_m)
del _m, _row, _n


def _compute_ft8_ldpc_free_cols() -> tuple[tuple[int, ...], tuple[int, ...]]:
    """GF(2) Gaussian elimination → free (systematic) and pivot (parity) columns."""
    N, M = 174, 83
    H = np.zeros((M, N), dtype=np.uint8)
    for r, cols in enumerate(_LDPC_CHECKS):
        for c in cols:
            H[r, c] = 1
    A = H.copy().astype(np.int32)
    pivot_cols: list[int] = []
    row = 0
    for col in range(N):
        found = next((r for r in range(row, M) if A[r, col]), None)
        if found is None:
            continue
        A[[row, found]] = A[[found, row]]
        for r in range(M):
            if r != row and A[r, col]:
                A[r] = (A[r] + A[row]) % 2
        pivot_cols.append(col)
        row += 1
        if row == M:
            break
    pivot_set = set(pivot_cols)
    free_cols = [c for c in range(N) if c not in pivot_set]
    return tuple(free_cols), tuple(pivot_cols)


_FT8_LDPC_FREE_COLS: tuple[int, ...]
_FT8_LDPC_PIVOT_COLS: tuple[int, ...]
_FT8_LDPC_FREE_COLS, _FT8_LDPC_PIVOT_COLS = _compute_ft8_ldpc_free_cols()


# ---------------------------------------------------------------------------
# Interleave tables (kept for API compatibility with existing test code).
# ft8_lib does NOT use a bit interleaver; these are provided as no-op stubs.
# ---------------------------------------------------------------------------

def _build_ft8_interleave_tables() -> tuple[tuple[int, ...], tuple[int, ...]]:
    def bit_rev7(x: int) -> int:
        r = 0
        for _ in range(7):
            r = (r << 1) | (x & 1)
            x >>= 1
        return r
    N = 174
    interleave = tuple(bit_rev7(i) for i in range(N))
    deint: list[int] = [-1] * N
    for i in range(N):
        deint[bit_rev7(i)] = i   # last write wins
    return interleave, tuple(deint)


_FT8_INTERLEAVE: tuple[int, ...]
_FT8_DEINTERLEAVE: tuple[int, ...]
_FT8_INTERLEAVE, _FT8_DEINTERLEAVE = _build_ft8_interleave_tables()

# Legacy name aliases for test compatibility
_FT8_BIT_REV7_TABLE: tuple[int, ...] = _FT8_INTERLEAVE
_FT8_INTERLEAVE_DST_TO_SRC: tuple[int, ...] = _FT8_INTERLEAVE
_FT8_DEINTERLEAVE_SRC_TO_DST: tuple[int, ...] = tuple(
    d if d >= 0 else 0 for d in _FT8_DEINTERLEAVE
)
_FT8_INTERLEAVE_PERM: tuple[int, ...] = _FT8_INTERLEAVE


# ═══════════════════════════════════════════════════════════════════════════════
# § 3  CRC-14
# ═══════════════════════════════════════════════════════════════════════════════

_FT8_CRC14_POLY: int = 0x2757   # from ft8_lib constants.h


def _ft8_crc14(bits: np.ndarray) -> int:
    """
    CRC-14 over 77 message bits zero-extended to 82 bits.

    Matches ft8_lib ftx_compute_crc(a91, 96-14):
      polynomial 0x2757, width 14, init 0, no final XOR.

    Algorithm: each byte is XOR'd into the top 8 bits of the 14-bit remainder
    register before 8 shift/reduce steps — this matches ft8_lib exactly.
    """
    # Pack 77 bits (MSB-first) into 11 bytes, with bits 77-81 as zeros (5-bit pad)
    padded = np.zeros(82, dtype=np.uint8)
    padded[:77] = np.asarray(bits[:77], dtype=np.uint8)
    n_bytes = (82 + 7) // 8  # 11 bytes to cover 82 bits
    msg_bytes = np.zeros(n_bytes, dtype=np.uint8)
    for i in range(82):
        msg_bytes[i // 8] |= int(padded[i]) << (7 - i % 8)

    # ft8_lib ftx_compute_crc: XOR next byte into top 8 bits, then shift each bit
    crc = 0
    idx_byte = 0
    for idx_bit in range(82):
        if idx_bit % 8 == 0:
            crc ^= int(msg_bytes[idx_byte]) << (14 - 8)   # shift byte to top 8 bits
            idx_byte += 1
        if crc & 0x2000:   # TOPBIT = 1 << 13
            crc = ((crc << 1) ^ _FT8_CRC14_POLY) & 0xFFFF
        else:
            crc = (crc << 1) & 0xFFFF
    return crc & 0x3FFF


# ═══════════════════════════════════════════════════════════════════════════════
# § 4  Symbol Energy Extractor
# ═══════════════════════════════════════════════════════════════════════════════

class FT8SymbolEnergyExtractor:
    """
    Compute per-tone energy for each of the 79 FT8 symbols.

    For each symbol position, a sym_n-point coherent DFT is evaluated at the
    8 tone frequencies f0 + k·6.25 Hz (k=0..7).  This is equivalent to the
    matched-filter approach used in WSJT-X / ft8_lib.
    """

    def __init__(self, fs: int = FT8_FS) -> None:
        self.fs = int(fs)
        self.sym_n = int(round(FT8_SYMBOL_DURATION_S * self.fs))  # 1920 @ 12 kHz

    def extract_all_79(
        self,
        frame: np.ndarray,
        *,
        t0_s: float,
        f0_hz: float,
    ) -> np.ndarray:
        """
        Return (79, 8) energy matrix for one FT8 slot.

        Parameters
        ----------
        frame  : 1-D float audio at self.fs (≥15 s)
        t0_s   : signal start time offset from frame start (seconds)
        f0_hz  : lowest-tone frequency of the candidate signal

        Returns
        -------
        E79 : (79, 8) float64 — |DFT|² at each (symbol, tone)
        """
        x = np.asarray(frame, dtype=np.float64)
        sym_n = self.sym_n
        t0_n = int(round(t0_s * self.fs))

        # Pre-compute basis vectors: exp(−j·2π·f_k/fs · t) for t=0..sym_n-1
        phase_inc = 2.0 * math.pi * np.array(
            [f0_hz + k * FT8_TONE_SPACING_HZ for k in range(8)]
        ) / float(self.fs)               # (8,) rad/sample
        t_sym = np.arange(sym_n, dtype=np.float64)
        basis = np.exp(-1j * np.outer(phase_inc, t_sym))   # (8, sym_n)

        E79 = np.zeros((FT8_NSYMS, 8), dtype=np.float64)
        for s in range(FT8_NSYMS):
            start = t0_n + s * sym_n
            end = start + sym_n
            if start < 0 or end > len(x):
                continue
            dft = basis @ x[start:end]             # (8,) complex
            E79[s] = (dft * np.conj(dft)).real     # |DFT|²
        return E79


# ═══════════════════════════════════════════════════════════════════════════════
# § 5  Payload Symbol Extraction  (pass-through with API compatibility)
# ═══════════════════════════════════════════════════════════════════════════════

def ft8_extract_payload_symbols(
    E79: np.ndarray,
    *,
    shift: int = 0,
    inverted: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract (58, 8) payload energies from a (79, 8) energy matrix.

    Parameters 'shift' and 'inverted' are accepted for API compatibility but
    not applied; ft8_lib never permutes the energy matrix for decoding.

    Returns
    -------
    E_payload : (58, 8) float64
    hard_syms : (58,)  int32 — argmax tone per payload symbol
    """
    E79 = np.asarray(E79, dtype=np.float64)
    if E79.shape != (FT8_NSYMS, 8):
        raise ValueError(f"E79 must be shape ({FT8_NSYMS}, 8), got {E79.shape}")
    E_payload = E79[list(FT8_PAYLOAD_POSITIONS), :].copy()
    hard_syms = np.argmax(E_payload, axis=1).astype(np.int32)
    return E_payload, hard_syms


def ft8_deinterleave(
    hard_syms: np.ndarray,
    E_payload: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Symbol-level deinterleave — identity pass-through (ft8_lib uses none)."""
    return (
        np.asarray(hard_syms, dtype=np.int32),
        np.asarray(E_payload, dtype=np.float64),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# § 6  Gray Decode → Channel LLRs
# ═══════════════════════════════════════════════════════════════════════════════

def ft8_gray_decode(
    syms: np.ndarray,
    E: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Gray-decode 58 payload symbols → 174 channel LLRs.

    Matches ft8_lib ft8_extract_symbol() exactly:

      s2[gv] = E[tone that has gray_value gv]   (forward Gray map)

      LLR bit 2 (MSB):  L[0] = max(s2[4..7]) − max(s2[0..3])
      LLR bit 1:        L[1] = max(s2[2,3,6,7]) − max(s2[0,1,4,5])
      LLR bit 0 (LSB):  L[2] = max(s2[1,3,5,7]) − max(s2[0,2,4,6])

    LLR sign convention (ft8_lib): positive → bit = 1 more likely.

    Parameters
    ----------
    syms : (58,) int   — hard-decision tone indices 0..7
    E    : (58, 8) float — symbol energies indexed by tone

    Returns
    -------
    hard_bits : (174,) uint8
    llrs      : (174,) float64
    """
    syms = np.asarray(syms, dtype=np.int32)
    E = np.asarray(E, dtype=np.float64)

    # gray_fwd[gv] = tone  (kFT8_Gray_map)
    gray_fwd = np.array(_FT8_GRAY_MAP, dtype=np.int32)
    # gray_inv[tone] = gv  (_FT8_GRAY_DECODE)
    gray_inv = np.array(_FT8_GRAY_DECODE, dtype=np.int32)

    N = FT8_NDATASYM
    hard_bits = np.empty(N * 3, dtype=np.uint8)
    llrs = np.empty(N * 3, dtype=np.float64)
    E_safe = np.maximum(E, 1e-30)

    for s in range(N):
        row = E_safe[s]
        # s2[gv] = energy at the tone that carries gray_value gv
        s2 = np.empty(8, dtype=np.float64)
        for gv in range(8):
            s2[gv] = row[gray_fwd[gv]]

        llrs[3 * s + 0] = (
            max(s2[4], s2[5], s2[6], s2[7]) - max(s2[0], s2[1], s2[2], s2[3])
        )
        llrs[3 * s + 1] = (
            max(s2[2], s2[3], s2[6], s2[7]) - max(s2[0], s2[1], s2[4], s2[5])
        )
        llrs[3 * s + 2] = (
            max(s2[1], s2[3], s2[5], s2[7]) - max(s2[0], s2[2], s2[4], s2[6])
        )

        gv = int(gray_inv[int(syms[s])])
        hard_bits[3 * s + 0] = (gv >> 2) & 1
        hard_bits[3 * s + 1] = (gv >> 1) & 1
        hard_bits[3 * s + 2] = gv & 1

    return hard_bits, llrs


# ═══════════════════════════════════════════════════════════════════════════════
# § 7  LDPC (174, 91) Sum-Product (Belief-Propagation) Decoder
# ═══════════════════════════════════════════════════════════════════════════════

def _ldpc_check(plain: np.ndarray) -> int:
    """Count parity-check failures (0 = valid codeword)."""
    errors = 0
    for row in _LDPC_CHECKS:
        x = 0
        for n in row:
            x ^= int(plain[n])
        if x:
            errors += 1
    return errors


def ft8_ldpc_decode(
    llrs: np.ndarray,
    *,
    max_iterations: int = 50,
) -> tuple[bool, np.ndarray, int]:
    """
    Sum-product LDPC decoder for the FT8 (174, 91) code.

    Algorithm (matches ft8_lib bp_decode() from lib/ldpc.cpp with the
    ft8_lib LLR convention where positive means bit = 1 more likely):

      Initialise  tov[174][3] = 0,  toc[83][≤7] = 0

      Each iteration:
        V→C:  Lmn = ch_llr[n] + Σ_{k≠m} tov[n][k]
              toc[m][n_idx] = tanh(−Lmn / 2)    ← ft8_lib uses negative sign

        C→V:  prod_all = Π toc[m][k]
              prod_excl = prod_all / toc[m][n_idx]   (or loop if |toc| < eps)
              prod_excl = clamp(prod_excl, −0.9999, +0.9999)
              tov[n][m_idx] = −2 · atanh(prod_excl)

        Hard: plain[n] = 1 if (ch_llr[n] + Σ tov[n]) > 0 else 0
        Check parity; exit early if 0 errors.

    The −2·atanh sign is correct for the ft8_lib positive-means-1 LLR
    convention (verified empirically against the reference LDPC matrix).

    Returns (success, payload[0:91], iterations_used).
    success=True only when all 83 parity checks pass AND CRC-14 matches.
    """
    codeword = np.asarray(llrs, dtype=np.float64)
    if codeword.shape != (174,):
        raise ValueError(f"Expected (174,) LLRs, got {codeword.shape}")

    N = 174
    CLAMP = 0.9999
    EPS = 1e-10

    tov: list[list[float]] = [[0.0, 0.0, 0.0] for _ in range(N)]
    toc: list[list[float]] = [[0.0] * len(row) for row in _BP_Nm]

    plain = np.zeros(N, dtype=np.uint8)
    best_errors = 83
    best_plain = plain.copy()
    iterations_used = max_iterations

    for iteration in range(max_iterations):
        # ── V→C ──────────────────────────────────────────────────────────
        for m, row in enumerate(_BP_Nm):
            for n_idx, n in enumerate(row):
                Lmn = float(codeword[n])
                for j in range(3):
                    if _BP_Mn[n][j] != m:
                        Lmn += tov[n][j]
                toc[m][n_idx] = math.tanh(-Lmn / 2.0)  # ft8_lib uses tanh(-Tnm/2)

        # ── C→V ──────────────────────────────────────────────────────────
        for m, row in enumerate(_BP_Nm):
            prod_all = 1.0
            for k in range(len(row)):
                prod_all *= toc[m][k]

            for n_idx, n in enumerate(row):
                m_idx = _BP_Mn[n].index(m)
                t = toc[m][n_idx]
                if abs(t) > EPS:
                    prod_excl = prod_all / t
                else:
                    prod_excl = 1.0
                    for k, nv in enumerate(row):
                        if nv != n:
                            prod_excl *= toc[m][k]

                prod_excl = max(-CLAMP, min(CLAMP, prod_excl))
                tov[n][m_idx] = -2.0 * math.atanh(prod_excl)

        # ── Hard decision ─────────────────────────────────────────────────
        for n in range(N):
            total = float(codeword[n]) + tov[n][0] + tov[n][1] + tov[n][2]
            plain[n] = 1 if total > 0.0 else 0

        errors = _ldpc_check(plain)
        if errors < best_errors:
            best_errors = errors
            best_plain = plain.copy()
            if errors == 0:
                iterations_used = iteration + 1
                break

    plain = best_plain
    payload = plain[:91].copy()

    if best_errors > 0:
        return False, payload, iterations_used

    # CRC-14 verification
    msg_bits = payload[:77]
    rx_crc_bits = payload[77:91]
    rx_crc = int(sum(int(b) << (13 - i) for i, b in enumerate(rx_crc_bits)))
    calc_crc = _ft8_crc14(msg_bits)
    return (calc_crc == rx_crc), payload, iterations_used


# ═══════════════════════════════════════════════════════════════════════════════
# § 8  Message Unpacker (77 bits → human string)
# ═══════════════════════════════════════════════════════════════════════════════
# Reference: ft8_lib message.c (ftx_message_decode / pack28 / unpack28).
#
# 77-bit layout — i3 field at bits 74-76:
#   i3=1 or 2  Standard QSO/CQ: [28 n28a][1 ipa][28 n28b][1 ipb][1 ir][15 igrid4][3 i3]
#   i3=0       sub-type via n3 (bits 71-73):
#                n3=0: Free text [71 text][3 n3=000][3 i3=000]
#                n3=1: DXpedition  n3=2: EU-VHF  n3=3: ARRL FD  n3=4: Telemetry
#   i3=3       Non-standard call (58-bit hash pair)
#   i3=4       WWROF contest
#
# ft8_lib constants (message.c):
_NTOKENS: int = 2063592   # special tokens (DE/QRZ/CQ/CQ nnn/CQ a[bcd])
_MAX22: int = 4194304     # 22-bit hash space (2^22)
# Standard callsigns start at _NTOKENS + _MAX22 = 6257896
_MAXGRID4: int = 32400    # = 18*18*10*10; values >= this are special/reports

# Character tables matching ft8_lib charn() / nchar():
_C37 = " 0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"  # ALPHANUM_SPACE (37 chars, space=0)
_C36 = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"   # ALPHANUM (36 chars)
_C27 = " ABCDEFGHIJKLMNOPQRSTUVWXYZ"            # LETTERS_SPACE (27 chars, space=0)
_C10 = "0123456789"
_FREETEXT_CHARS = " 0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ+-./?"  # FULL (42 chars)
_GRID_LETTERS = "ABCDEFGHIJKLMNOPQR"


def _bits_to_int(bits: np.ndarray, start: int, length: int) -> int:
    val = 0
    for i in range(length):
        val = (val << 1) | int(bits[start + i])
    return val


def _unpack_callsign_28(n28: int, ip: int = 0, i3: int = 1) -> str:
    """
    Decode a 28-bit packed callsign number to a string.

    Matches ft8_lib unpack28() exactly:
      n28 < _NTOKENS              → special token (DE/QRZ/CQ/CQ nnn/CQ a[bcd])
      _NTOKENS ≤ n28 < _NTOKENS+_MAX22  → 22-bit hash (shown as <hash>)
      n28 ≥ _NTOKENS+_MAX22      → standard callsign (decoded with character tables)

    ip=1 appends '/R' (i3=1) or '/P' (i3=2) to indicate portable/rover operation.
    """
    if n28 < _NTOKENS:
        if n28 == 0: return 'DE'
        if n28 == 1: return 'QRZ'
        if n28 == 2: return 'CQ'
        if n28 <= 1002:
            return f'CQ {n28 - 3:03d}'
        if n28 <= 532443:
            m = n28 - 1003
            suffix = ''
            for _ in range(4):
                suffix = _C27[m % 27] + suffix
                m //= 27
            return f'CQ {suffix.strip()}'
        return f'<token{n28}>'
    if n28 < _NTOKENS + _MAX22:
        return f'<{n28 - _NTOKENS:06X}>'
    # Standard callsign
    n = n28 - _NTOKENS - _MAX22
    c5 = _C27[n % 27]; n //= 27
    c4 = _C27[n % 27]; n //= 27
    c3 = _C27[n % 27]; n //= 27
    c2 = _C10[n % 10]; n //= 10
    c1 = _C36[n % 36]; n //= 36
    c0 = _C37[n] if n < 37 else '?'
    call = (c0 + c1 + c2 + c3 + c4 + c5).strip()
    if ip:
        call += '/R' if i3 == 1 else '/P'
    return call or '?'


def _unpack_grid(igrid4: int, ir: int = 0) -> str:
    """
    Decode a 15-bit igrid4 value (plus 1-bit ir flag) to a grid/report string.

    Matches ft8_lib unpackgrid():
      igrid4 < 32400 : 4-letter Maidenhead locator, prefixed 'R' if ir=1
      32402 / 32403 / 32404 : RRR / RR73 / 73
      32405+ : signal report dB (dd = igrid4 − 32400 − 35), prefixed 'R' if ir=1
      32400 / 32401 : blank (two-callsign-only message)
    """
    if igrid4 >= _MAXGRID4:
        special = {_MAXGRID4 + 2: 'RRR', _MAXGRID4 + 3: 'RR73', _MAXGRID4 + 4: '73'}
        if igrid4 in special:
            return special[igrid4]
        if igrid4 > _MAXGRID4 + 4:
            dd = igrid4 - _MAXGRID4 - 35
            report = f'{dd:+03d}'
            return ('R' + report) if ir else report
        return ''  # blank (32400 or 32401)
    ig = igrid4
    g4 = ig % 10; ig //= 10
    g3 = ig % 10; ig //= 10
    g2 = ig % 18; g1 = ig // 18
    grid = _GRID_LETTERS[g1] + _GRID_LETTERS[g2] + str(g3) + str(g4)
    return ('R' + grid) if ir else grid


def _unpack_type1(bits: np.ndarray, i3: int = 1) -> str:
    """Decode a standard FT8 message (i3=1 or i3=2) from 77 bits."""
    n28a = _bits_to_int(bits, 0, 28)
    ipa  = int(bits[28])
    n28b = _bits_to_int(bits, 29, 28)
    ipb  = int(bits[57])
    ir   = int(bits[58])
    igrid4 = _bits_to_int(bits, 59, 15)   # bits 59-73
    c1 = _unpack_callsign_28(n28a, ipa, i3)
    c2 = _unpack_callsign_28(n28b, ipb, i3)
    extra = _unpack_grid(igrid4, ir)
    return f'{c1} {c2} {extra}'.strip()


def _unpack_free_text(bits: np.ndarray) -> str:
    """Decode 71-bit free text using 42-character alphabet (ft8_lib FT8_CHAR_TABLE_FULL)."""
    n = _bits_to_int(bits, 0, 71)
    chars = []
    for _ in range(13):
        n, r = divmod(n, 42)
        chars.append(_FREETEXT_CHARS[r] if r < len(_FREETEXT_CHARS) else '?')
    return ''.join(reversed(chars)).strip()


def _unpack_telemetry(bits: np.ndarray) -> str:
    """Decode 71-bit telemetry payload as hex string."""
    return f'TELEMETRY:{_bits_to_int(bits, 0, 71):018X}'


def ft8_unpack_message(msg_bits: np.ndarray) -> str:
    """
    Decode 77 message bits to a human-readable FT8 message string.

    Parameters
    ----------
    msg_bits : (77,) uint8

    Returns
    -------
    str  e.g. 'W4ABC K9XYZ EN52', 'CQ W4ABC EM73', 'CQ DX5ABC +04'
    """
    msg_bits = np.asarray(msg_bits, dtype=np.uint8)
    if msg_bits.shape != (77,):
        return f'?bad_len={len(msg_bits)}'
    i3 = _bits_to_int(msg_bits, 74, 3)
    if i3 == 1 or i3 == 2:
        # Standard message: two callsigns + grid/report
        return _unpack_type1(msg_bits, i3)
    if i3 == 0:
        # Sub-type determined by n3 field (bits 71-73)
        n3 = _bits_to_int(msg_bits, 71, 3)
        if n3 == 0:
            return _unpack_free_text(msg_bits)
        if n3 == 1:
            return 'DXPEDITION ' + _unpack_type1(msg_bits, i3=1)
        if n3 == 2:
            return 'EU-VHF ' + _unpack_type1(msg_bits, i3=2)
        if n3 == 3:
            # ARRL Field Day: [28 c1][1 R][28 c2][1 R][13 exch][3 n3][3 i3]
            c1 = _unpack_callsign_28(_bits_to_int(msg_bits, 0, 28))
            c2 = _unpack_callsign_28(_bits_to_int(msg_bits, 29, 28))
            exch = _bits_to_int(msg_bits, 58, 13)
            return f'FD {c1} {c2} {exch}'
        if n3 == 4:
            return _unpack_telemetry(msg_bits)
        return f'?i3=0,n3={n3}'
    if i3 == 3:
        # Non-standard call (58-bit hash pair)
        raw = _bits_to_int(msg_bits, 0, 77)
        return f'NONSTD:{raw:021X}'
    # i3=4 (WWROF) and others
    raw = _bits_to_int(msg_bits, 0, 77)
    return f'?i3={i3}:{raw:021X}'


# ═══════════════════════════════════════════════════════════════════════════════
# § 9  Costas Sync Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _costas_score(E79: np.ndarray) -> tuple[int, int, bool]:
    """
    Count how many of the 21 Costas symbols match the expected FT8 pattern.

    Returns (matches, total=21, inverted).
    'inverted' indicates the signal is reflected (7-n tone).
    """
    E79 = np.asarray(E79, dtype=np.float64)
    best = -1
    best_inv = False
    for inv in (False, True):
        m = 0
        for i, pos in enumerate(FT8_COSTAS_POSITIONS):
            expected = FT8_COSTAS_TONES[i % 7]
            if inv:
                expected = 7 - expected
            if int(np.argmax(E79[pos])) == expected:
                m += 1
        if m > best:
            best, best_inv = m, inv
    return best, 21, best_inv


# Legacy API shims used by some existing test files
def ft8_costas_ok(E: np.ndarray) -> tuple[int, int, int]:
    m, t, _ = _costas_score(E)
    return m, t, 0


def ft8_costas_ok_costasE(Ec: np.ndarray) -> tuple[int, int, int, bool]:
    m, t, inv = _costas_score(Ec)
    return m, t, 0, inv


def ft8_costas_margin_score(
    Ec: np.ndarray,
) -> tuple[float, tuple[float, float, float], int, int, int, bool]:
    m, t, inv = _costas_score(Ec)
    return 0.0, (0.0, 0.0, 0.0), m, t, 0, inv


def ft8_costas_rank_stats(Ec: np.ndarray) -> dict[str, float]:
    m, t, inv = _costas_score(Ec)
    return {
        "matches": float(m), "total": float(t), "shift": 0.0,
        "inverted": float(inv), "rank_mean": 0.0, "rank_median": 0.0,
        "margin_db_mean": 0.0, "margin_db_median": 0.0,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# § 10  Offline Batch Decoder  (decode_wav)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class FT8DecodeResult:
    utc_time: str
    strength_db: float
    frequency_hz: float
    message: str


def format_ft8_message(utc: str, snr_db: float, freq_hz: float, message: str) -> str:
    """
    Format an FT8 decoded message matching the reference decoder output format.

    Output: ``HHMMSS +NN NNNN.NNN MESSAGE``

    Parameters
    ----------
    utc      : UTC timestamp string (e.g. ``HH:MM:SS`` or ``HHMMSS``)
    snr_db   : signal-to-noise ratio in dB (rounded to nearest integer)
    freq_hz  : audio frequency in Hz
    message  : decoded FT8 message text (e.g. ``CQ W4ABC EM73``)

    Returns
    -------
    str  formatted line ready for terminal or log output
    """
    snr_int = int(round(snr_db))
    return f"{utc} {snr_int:+3d} {freq_hz:8.3f} {message}"


def decode_wav(
    wav_path: str,
    *,
    fmin_hz: float = 200.0,
    fmax_hz: float = 3000.0,
    max_iterations: int = 50,
    dt_step_s: float = 0.04,
    debug: bool = False,
) -> list[FT8DecodeResult]:
    """
    Decode FT8 messages from a WAV file.

    Resamples to 12 kHz, segments into 15 s FT8 slots, performs a full
    time/frequency grid search for Costas candidates, then runs LDPC+CRC
    on each.  Deduplicates by (slot, message) text.

    The time search covers the entire valid FT8 window within a slot:
      t0 ∈ [−0.5 s, +2.5 s]  (signal must be fully within the 15 s slot)

    Parameters
    ----------
    wav_path     : path to the WAV file (any sample rate, mono or stereo)
    fmin_hz      : lower bound of search band (Hz)
    fmax_hz      : upper bound of search band (Hz)
    max_iterations : LDPC max iterations
    dt_step_s    : time search step size (s); default 0.04 s ≈ quarter symbol
    debug        : print per-candidate diagnostics

    Returns
    -------
    list of FT8DecodeResult, sorted by (utc_time, frequency_hz)
    """
    import wave
    from scipy.signal import resample_poly

    # ── Load WAV ──────────────────────────────────────────────────────────────
    with wave.open(wav_path, 'rb') as w:
        nch = w.getnchannels()
        fs_in = w.getframerate()
        raw = w.readframes(w.getnframes())
    samples = np.frombuffer(raw, dtype=np.int16).reshape(-1, nch)
    audio = samples[:, 0].astype(np.float32) / 32768.0   # left channel, float

    # ── Resample to 12 kHz ────────────────────────────────────────────────────
    if fs_in != FT8_FS:
        g = math.gcd(fs_in, FT8_FS)
        audio = resample_poly(audio, FT8_FS // g, fs_in // g).astype(np.float32)
    fs = FT8_FS
    sym_n = FT8_SYM_SAMPLES   # 1920

    total_samples = len(audio)
    slot_samples = int(round(15.0 * fs))   # 180 000 samples
    # FT8 signals can start at most (15 - 12.64) = 2.36 s after a slot boundary.
    # Allow 0.5 s margin on each side: search t0 ∈ [−0.5, +2.86] s.
    t0_min_s = -0.5
    t0_max_s = 15.0 - FT8_TX_DURATION_S + 0.5   # ≈ 2.86 s
    n_slots = total_samples // slot_samples

    if debug:
        print(f"[decode_wav] {wav_path}: {total_samples/fs:.1f}s @ {fs} Hz → {n_slots} slots")
        print(f"[decode_wav] time search: [{t0_min_s:.2f}, {t0_max_s:.2f}] s  step={dt_step_s:.3f} s")

    extractor = FT8SymbolEnergyExtractor(fs=fs)
    results: list[FT8DecodeResult] = []
    seen: set[tuple[str, str]] = set()   # (slot_label, message) dedup

    for slot_idx in range(n_slots):
        slot_start_n = slot_idx * slot_samples
        # Frame: take the full slot plus a small guard on each side to cover
        # timing offsets.  The guard is |t0_min_s| samples before slot start
        # and t0_max_s samples after slot start (the signal itself is 12.64 s).
        pre_guard = max(0, int(math.ceil(abs(t0_min_s) * fs)))
        post_guard = int(math.ceil((t0_max_s + FT8_TX_DURATION_S) * fs))
        frame_start_n = max(0, slot_start_n - pre_guard)
        frame_end_n = min(total_samples, slot_start_n + post_guard)
        frame = audio[frame_start_n:frame_end_n]
        # t0_s is measured from the beginning of `frame`
        slot_offset_in_frame = (slot_start_n - frame_start_n) / float(fs)

        utc_label = f"slot{slot_idx:02d}"

        # ── Candidate frequency search ─────────────────────────────────────────
        # Build time-averaged power spectrum at 6.25 Hz resolution
        n_wins = len(frame) // sym_n
        if n_wins == 0:
            continue

        pwr = np.zeros(sym_n // 2 + 1, dtype=np.float64)
        for w in range(n_wins):
            chunk = frame[w * sym_n: (w + 1) * sym_n].astype(np.float64)
            spec = np.fft.rfft(chunk)
            pwr += np.abs(spec) ** 2
        pwr /= n_wins

        df = float(fs) / sym_n   # 6.25 Hz per bin
        lo_bin = max(0, int(math.floor(fmin_hz / df)))
        hi_bin = min(len(pwr) - 9, int(math.ceil(fmax_hz / df)))

        # 8-tone matched filter: sum of 8 adjacent bins vs local noise
        noise_w = 12
        cand_freqs: list[tuple[float, float]] = []
        for b in range(lo_bin, hi_bin):
            sig_pow = float(np.sum(pwr[b: b + 8]))
            lo_n = max(0, b - noise_w)
            hi_n = min(len(pwr), b + 8 + noise_w)
            nbins = np.concatenate([pwr[lo_n:b], pwr[b + 8:hi_n]])
            if len(nbins) < 4:
                continue
            noise_pow = float(np.mean(nbins)) * 8
            if noise_pow < 1e-30:
                continue
            snr_db = 10.0 * math.log10(max(sig_pow / noise_pow, 1e-9))
            if snr_db > 1.0:
                cand_freqs.append((b * df, snr_db))

        if not cand_freqs:
            continue

        # Sort by SNR, keep top 20, de-duplicate within 4 bins (25 Hz)
        cand_freqs.sort(key=lambda x: -x[1])
        filtered: list[tuple[float, float]] = []
        for f0, snr in cand_freqs:
            if not any(abs(f0 - ff) < 4 * df for ff, _ in filtered):
                filtered.append((f0, snr))
            if len(filtered) >= 20:
                break

        if debug:
            print(f"[decode_wav] slot {utc_label}: {len(filtered)} freq candidates")

        # ── Time/frequency search across the full slot window ─────────────────
        # t_offsets are relative to the slot boundary (slot_start_n).
        t_offsets = np.arange(t0_min_s, t0_max_s + dt_step_s / 2, dt_step_s)

        # Set of (t_key, f_key) already attempted — avoid re-scoring
        tried: set[tuple[int, int]] = set()

        for f0_hz, _ in filtered:
            for dt in t_offsets:
                # t0_s is measured from the frame start
                t0_s = slot_offset_in_frame + float(dt)
                tk = int(round(float(dt) / (dt_step_s / 2)))
                fk = int(round(f0_hz / (df / 2)))
                if (tk, fk) in tried:
                    continue
                tried.add((tk, fk))

                E79 = extractor.extract_all_79(frame, t0_s=t0_s, f0_hz=f0_hz)
                costas_m, _, _ = _costas_score(E79)
                if costas_m < 7:
                    continue

                E_payload, hard_syms = ft8_extract_payload_symbols(E79)
                _, ch_llrs = ft8_gray_decode(hard_syms, E_payload)

                # Normalise to variance=24 (ftx_normalize_logl)
                var = float(np.var(ch_llrs))
                if var > 1e-10:
                    ch_llrs = ch_llrs * math.sqrt(24.0 / var)

                ok, payload, iters = ft8_ldpc_decode(ch_llrs, max_iterations=max_iterations)
                if not ok:
                    continue

                message = ft8_unpack_message(payload[:77])
                snr_db = 10.0 * math.log10(max(costas_m / 21.0, 1e-6))

                key = (utc_label, message)
                if key in seen:
                    continue
                seen.add(key)

                if debug:
                    print(
                        f"  [DECODE] slot={utc_label} f0={f0_hz:.1f}Hz "
                        f"dt={dt:+.2f}s costas={costas_m}/21 "
                        f"iters={iters} msg='{message}'"
                    )

                results.append(FT8DecodeResult(
                    utc_time=utc_label,
                    strength_db=snr_db,
                    frequency_hz=f0_hz,
                    message=message,
                ))

    results.sort(key=lambda r: (r.utc_time, r.frequency_hz))
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# § 11  Streaming Infrastructure
# ═══════════════════════════════════════════════════════════════════════════════

class PolyphaseResampler:
    """Streaming resampler wrapping scipy.signal.resample_poly."""

    def __init__(self, fs_in: int, fs_out: int) -> None:
        self.fs_in = int(fs_in)
        self.fs_out = int(fs_out)
        g = math.gcd(self.fs_in, self.fs_out)
        self.up = self.fs_out // g
        self.down = self.fs_in // g

    def process(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        if x.size == 0:
            return x
        return _scipy_signal.resample_poly(x, self.up, self.down).astype(
            np.float32, copy=False
        )


class UTC15sFramer:
    """
    Buffer a continuous fs_proc audio stream and emit exact 15 s UTC-aligned frames.

    push() → list of (slot_start_utc_epoch, frame_samples, is_partial)
    The first frame emitted may be partial (audio started mid-slot).
    """

    def __init__(self, fs_proc: int, frame_s: float = 15.0) -> None:
        self.fs = int(fs_proc)
        self.frame_s = float(frame_s)
        self.frame_n = int(round(self.fs * self.frame_s))
        self._buf = np.zeros(0, dtype=np.float32)
        self._t0_utc: Optional[float] = None
        self._first_frame_emitted: bool = False
        self._utc_minus_mono: Optional[float] = None
        self._alpha = 0.01

    @staticmethod
    def _utc_now_epoch() -> float:
        return datetime.now(timezone.utc).timestamp()

    @staticmethod
    def _slot_start_epoch(t_utc_epoch: float, slot_s: float = 15.0) -> float:
        return math.floor(t_utc_epoch / slot_s) * slot_s

    def _update_utc_minus_mono(self) -> float:
        utc_now = self._utc_now_epoch()
        mono_now = time.monotonic()
        sample = utc_now - mono_now
        if self._utc_minus_mono is None:
            self._utc_minus_mono = sample
        else:
            a = float(self._alpha)
            self._utc_minus_mono = (1.0 - a) * float(self._utc_minus_mono) + a * float(sample)
        return float(self._utc_minus_mono)

    def push(
        self, x: np.ndarray, *, t0_monotonic: float
    ) -> list[tuple[float, np.ndarray, bool]]:
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        utc_offset = self._update_utc_minus_mono()
        t0_utc = t0_monotonic + utc_offset
        if self._t0_utc is None:
            self._t0_utc = t0_utc
        self._buf = np.concatenate([self._buf, x])
        out: list[tuple[float, np.ndarray, bool]] = []
        while len(self._buf) >= self.frame_n:
            frame = self._buf[: self.frame_n].copy()
            self._buf = self._buf[self.frame_n:]
            slot_start = self._slot_start_epoch(float(self._t0_utc))
            is_partial = not self._first_frame_emitted
            out.append((slot_start, frame, is_partial))
            self._first_frame_emitted = True
            self._t0_utc = float(self._t0_utc) + self.frame_s
        return out


# ═══════════════════════════════════════════════════════════════════════════════
# § 12  FT8 Signal Detector  (candidate seed frequencies)
# ═══════════════════════════════════════════════════════════════════════════════

class FT8SignalDetector:
    """
    Detect candidate FT8 base frequencies via an 8-tone matched-filter
    response on the time-averaged spectrogram.
    """

    def __init__(
        self,
        fs: int = FT8_FS,
        fmin_hz: float = 200.0,
        fmax_hz: float = 3200.0,
    ) -> None:
        self.fs = int(fs)
        self.fmin_hz = float(fmin_hz)
        self.fmax_hz = float(fmax_hz)
        self._sym_n = int(round(FT8_SYMBOL_DURATION_S * self.fs))
        self._df = float(self.fs) / self._sym_n   # 6.25 Hz

    def detect(self, frame: np.ndarray, *, top_n: int = 20) -> list[tuple[float, float]]:
        """Return list of (f0_hz, score_db) sorted descending."""
        x = np.asarray(frame, dtype=np.float64)
        sym_n = self._sym_n
        df = self._df
        n_wins = len(x) // sym_n
        if n_wins == 0:
            return []

        pwr = np.zeros(sym_n // 2 + 1, dtype=np.float64)
        for w in range(n_wins):
            spec = np.fft.rfft(x[w * sym_n: (w + 1) * sym_n])
            pwr += np.abs(spec) ** 2
        pwr /= n_wins

        lo_bin = max(0, int(math.floor(self.fmin_hz / df)))
        hi_bin = min(len(pwr) - 9, int(math.ceil(self.fmax_hz / df)))
        noise_w = 16
        candidates: list[tuple[float, float]] = []

        for b in range(lo_bin, hi_bin):
            sig = float(np.sum(pwr[b: b + 8]))
            lo_n, hi_n = max(0, b - noise_w), min(len(pwr), b + 8 + noise_w)
            nbins = np.concatenate([pwr[lo_n:b], pwr[b + 8:hi_n]])
            if len(nbins) < 4:
                continue
            noise = float(np.mean(nbins)) * 8
            if noise < 1e-20:
                continue
            score = 10.0 * math.log10(max(sig / noise, 1e-9))
            if score > 0.0:
                candidates.append((b * df, score))

        candidates.sort(key=lambda t: -t[1])
        return candidates[:top_n]


# ═══════════════════════════════════════════════════════════════════════════════
# § 13  FT8 Sync Search
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class FT8SyncCandidate:
    slot_utc: str
    time_offset_s: float
    freq_hz: float
    score_db: float


class FT8SyncSearch:
    """
    Search for FT8 signals by sliding over time/frequency offsets around
    each seed frequency and scoring the Costas array match.
    """

    def __init__(
        self,
        *,
        fs: int = FT8_FS,
        fmin_hz: float = 200.0,
        fmax_hz: float = 3200.0,
        sym_s: float = FT8_SYMBOL_DURATION_S,
        tone_spacing_hz: float = FT8_TONE_SPACING_HZ,
        extractor: Optional["FT8SymbolEnergyExtractor"] = None,
    ) -> None:
        self.fs = int(fs)
        self.fmin_hz = float(fmin_hz)
        self.fmax_hz = float(fmax_hz)
        self.sym_s = float(sym_s)
        self.tone_hz = float(tone_spacing_hz)
        self._extractor = extractor or FT8SymbolEnergyExtractor(fs=self.fs)

    def search(
        self,
        frame: np.ndarray,
        *,
        seed_freqs_hz: list[float],
        time_search_s: float = 0.5,
        freq_search_bins: int = 3,
        max_candidates: int = 10,
    ) -> list[tuple[float, float, float]]:
        """
        Return list of (t0_s, f0_hz, costas_score_db) for the best candidates.
        """
        df_hz = self.tone_hz
        dt_step = self.sym_s / 4.0
        t_offsets = np.arange(-time_search_s, time_search_s + dt_step / 2, dt_step)

        results: list[tuple[float, float, float]] = []
        tried: set[tuple[int, int]] = set()

        for f0 in seed_freqs_hz:
            for b in range(-freq_search_bins, freq_search_bins + 1):
                f_try = float(f0) + b * df_hz
                if f_try < self.fmin_hz or f_try > self.fmax_hz - 7 * df_hz:
                    continue
                for t0 in t_offsets:
                    tk = int(round(float(t0) / (dt_step / 2)))
                    fk = int(round(f_try / (df_hz / 2)))
                    if (tk, fk) in tried:
                        continue
                    tried.add((tk, fk))

                    E79 = self._extractor.extract_all_79(
                        frame, t0_s=float(t0), f0_hz=f_try
                    )
                    m, _, _ = _costas_score(E79)
                    score = 10.0 * math.log10(max(m / 21.0, 1e-6))
                    results.append((float(t0), f_try, score))

        results.sort(key=lambda t: -t[2])
        return results[:max_candidates]


# ═══════════════════════════════════════════════════════════════════════════════
# § 14  FT8ConsoleDecoder  (end-to-end streaming decoder)
# ═══════════════════════════════════════════════════════════════════════════════

class FT8ConsoleDecoder:
    """
    End-to-end FT8 decoder that consumes a live audio stream.

    Audio is fed via feed() calls from any thread; a background worker
    thread performs resampling, framing, detection, and decoding.

    Every successful decode triggers the on_decode callback:
        on_decode(utc: str, freq_hz: float, snr_db: float, message: str)

    Usage
    -----
        decoder = FT8ConsoleDecoder(on_decode=my_callback)
        decoder.start()
        while running:
            decoder.feed(fs=chunk.fs, samples=chunk.samples, t0_monotonic=chunk.t0)
        decoder.stop()
    """

    def __init__(
        self,
        *,
        fs_proc: int = FT8_FS,
        fmin_hz: float = 200.0,
        fmax_hz: float = 3200.0,
        on_decode=None,
    ) -> None:
        self.fs_proc = int(fs_proc)
        self.fmin_hz = float(fmin_hz)
        self.fmax_hz = float(fmax_hz)
        self._on_decode = on_decode

        self._q: queue.Queue = queue.Queue(maxsize=32)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self._resampler: Optional[PolyphaseResampler] = None
        self._framer = UTC15sFramer(self.fs_proc)
        self._extractor = FT8SymbolEnergyExtractor(fs=self.fs_proc)
        self._detector = FT8SignalDetector(
            fs=self.fs_proc, fmin_hz=self.fmin_hz, fmax_hz=self.fmax_hz
        )
        self._sync = FT8SyncSearch(
            fs=self.fs_proc,
            fmin_hz=self.fmin_hz,
            fmax_hz=self.fmax_hz,
            extractor=self._extractor,
        )

        self._debug: bool = True
        self._min_costas_matches: int = 7

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._worker, daemon=True, name="ft8-decode"
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def feed(self, *, fs: int, samples: np.ndarray, t0_monotonic: float) -> None:
        try:
            self._q.put_nowait(
                (int(fs), np.asarray(samples, dtype=np.float32), float(t0_monotonic))
            )
        except queue.Full:
            if self._debug:
                print("[FT8] audio queue full — dropping chunk", flush=True)

    @staticmethod
    def _fmt_utc(epoch_s: float) -> str:
        return datetime.fromtimestamp(epoch_s, tz=timezone.utc).strftime("%H:%M:%S")

    def _decode_frame(self, frame: np.ndarray, utc: str) -> None:
        """Run full decode pipeline on one 15 s frame."""
        _T0_MIN = -0.50
        _T0_MAX = 15.0 - FT8_TX_DURATION_S   # ≈ +2.36 s

        # Detect candidate seed frequencies
        candidates = self._detector.detect(frame, top_n=20)
        seed_freqs = [f for f, _ in candidates[:10]]

        if self._debug:
            print(
                f"[FT8] slot {utc}Z  seeds={len(seed_freqs)}", flush=True
            )

        sync_cands = self._sync.search(
            frame,
            seed_freqs_hz=seed_freqs,
            time_search_s=0.5,
            freq_search_bins=3,
            max_candidates=5,
        )

        seen_msg: set[str] = set()
        for t0_s, f0_hz, _score in sync_cands:
            if float(t0_s) < _T0_MIN or float(t0_s) > _T0_MAX:
                continue

            E79 = self._extractor.extract_all_79(frame, t0_s=t0_s, f0_hz=f0_hz)
            costas_m, _, _ = _costas_score(E79)
            if costas_m < self._min_costas_matches:
                continue

            E_payload, hard_syms = ft8_extract_payload_symbols(E79)
            _, ch_llrs = ft8_gray_decode(hard_syms, E_payload)

            var = float(np.var(ch_llrs))
            if var > 1e-10:
                ch_llrs = ch_llrs * math.sqrt(24.0 / var)

            ok, payload, iters = ft8_ldpc_decode(ch_llrs)
            if not ok:
                continue

            message = ft8_unpack_message(payload[:77])
            if message in seen_msg:
                continue
            seen_msg.add(message)

            snr_db = 10.0 * math.log10(max(costas_m / 21.0, 1e-6))

            if self._debug:
                print(
                    f"[FT8] {utc} DECODE f0={f0_hz:.1f}Hz t0={t0_s:+.2f}s "
                    f"snr≈{snr_db:+.1f}dB iters={iters} '{message}'",
                    flush=True,
                )

            if self._on_decode is not None:
                self._on_decode(utc, f0_hz, snr_db, message)

    def _worker(self) -> None:
        while not self._stop.is_set():
            try:
                fs_in, x_in, t0 = self._q.get(timeout=0.25)
            except queue.Empty:
                continue

            if self._resampler is None or self._resampler.fs_in != fs_in:
                self._resampler = PolyphaseResampler(fs_in=fs_in, fs_out=self.fs_proc)
                if self._debug:
                    print(
                        f"[FT8] resampler: {fs_in} → {self.fs_proc} Hz", flush=True
                    )

            x = self._resampler.process(x_in)
            frames = self._framer.push(x, t0_monotonic=t0)

            for slot_start_utc, frame, is_partial in frames:
                utc = self._fmt_utc(slot_start_utc)
                if is_partial:
                    if self._debug:
                        print(
                            f"[FT8] slot {utc}Z skipped (partial)", flush=True
                        )
                    continue
                if self._debug:
                    print(f"[FT8] slot {utc}Z — decoding", flush=True)
                self._decode_frame(frame, utc)
