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
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
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

# Pre-computed numpy array of payload positions for efficient SNR calculation.
_PAYLOAD_POS_ARRAY: np.ndarray = np.array(FT8_PAYLOAD_POSITIONS, dtype=np.intp)

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

# ─── Vectorized BP decode pre-computed tables ─────────────────────────────────
# Built once at module import to eliminate per-call Python-loop overhead inside
# ft8_ldpc_decode.  The FT8 (174,91) code has column weight exactly 3 (each
# variable node connects to exactly 3 check nodes) and check-node degree 6 or 7.


def _build_bp_numpy_tables() -> tuple:
    """
    Pre-compute dense/flat index arrays for the fully-vectorised numpy LDPC BP
    decoder.

    Returns
    -------
    nm_dense   : (83, max_row) int32  — variable-node indices, -1 for padding
    nm_valid   : (83, max_row) bool   — True only for real (non-padded) edges
    max_row    : int                  — maximum check-node degree (7 for FT8)
    edges_vn   : (E,) int32           — variable-node index for each edge
    edges_midx : (E,) int32           — tov column (0/1/2) for that variable node
    edges_ravel: (E,) int32           — flat index into (83, max_row) matrix
    """
    M = len(_BP_Nm)
    max_row = max(len(r) for r in _BP_Nm)   # 7 for the FT8 (174,91) code

    nm_dense = np.full((M, max_row), -1, dtype=np.int32)
    nm_valid = np.zeros((M, max_row), dtype=bool)

    ev: list[int] = []    # variable-node index per edge
    em: list[int] = []    # tov slot (m_idx) for that variable node
    er: list[int] = []    # flat index into (83, max_row) padded matrix

    for m, row in enumerate(_BP_Nm):
        for n_idx, n in enumerate(row):
            nm_dense[m, n_idx] = n
            nm_valid[m, n_idx] = True
            m_idx = _BP_Mn[n].index(m)
            ev.append(n)
            em.append(m_idx)
            er.append(m * max_row + n_idx)

    return (
        nm_dense,
        nm_valid,
        max_row,
        np.array(ev, dtype=np.int32),
        np.array(em, dtype=np.int32),
        np.array(er, dtype=np.int32),
    )


(
    _BP_NM_DENSE,
    _BP_NM_VALID,
    _BP_MAX_ROW_LEN,
    _BP_EDGES_VN,
    _BP_EDGES_MIDX,
    _BP_EDGES_RAVEL,
) = _build_bp_numpy_tables()

# Parity-check matrix H (83 × 174, uint8) for vectorised syndrome evaluation.
# H[m, n] = 1 iff variable n participates in parity check m.
_H_LDPC = np.zeros((83, 174), dtype=np.uint8)
for _r, _cols in enumerate(_LDPC_CHECKS):
    for _c in _cols:
        _H_LDPC[_r, _c] = 1
del _r, _cols, _c

# Pre-built numpy copies of the Gray code tables — avoids per-call np.array().
_GRAY_FWD_NP = np.array(_FT8_GRAY_MAP,    dtype=np.int32)  # gray_value → tone index
_GRAY_INV_NP = np.array(_FT8_GRAY_DECODE, dtype=np.int32)  # tone index → gray_value

# ---------------------------------------------------------------------------
# A priori (AP) decode constants.
# WSJT-X ft8b.f90 performs multiple BP passes with known structural bits clamped
# to high-confidence LLRs, substantially improving convergence on marginal signals.
# LLR convention: positive → bit = 1 more likely.
# Bits 74-76 (MSB first) encode the i3 message type field in the 77-bit payload.
# Bits 0-27 encode the first callsign n28a (MSB first).
#   n28a=0 → DE   (bits 0-27 all 0)
#   n28a=2 → CQ   (bits 0-25=0, bit 26=1, bit 27=0)
# Bit 28 = ipa (portable/rover flag for n28a)
# ---------------------------------------------------------------------------
# AP LLR magnitude: applied after normalization to variance=24 (ftx_normalize_logl),
# where the typical channel LLR magnitude for a correct decision is ~sqrt(24) ≈ 4.9.
# A value of 50 (~10 sigma) acts as a near-certain prior for the constrained bits.
_AP_LLR_MAGNITUDE: float = 50.0

# Threshold: only run AP passes when baseline LDPC had this many or fewer parity
# errors.  Signals with many parity errors are likely noise, not near-miss signals;
# adding AP passes would waste time without improving decodes.
_AP_PARITY_ERROR_THRESHOLD: int = 15

# Fine sub-bin frequency offsets (Hz) used after the coarse 6.25 Hz grid sync.
# FT8 transmitters may have calibration offsets of a few Hz, causing a signal
# centred between two 6.25 Hz grid points.  Trying these offsets finds the
# optimal base frequency and significantly improves Costas scores and LLR quality
# for off-grid signals.
# Milestone 6: Wider search (±4 Hz) with ±0.5 Hz sub-steps catches transmitters
# with ±3.5–4 Hz calibration offset (common with older rigs and cheap SDRs).
_FINE_FREQ_OFFSETS_HZ: tuple[float, ...] = (
    -4.0, -3.5, -3.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 3.0, 3.5, 4.0
)

# ---------------------------------------------------------------------------
# Milestone 6 tunable decoder parameters
# ---------------------------------------------------------------------------
# Minimum Costas-array match count to accept a sync candidate.
# Lowered from 7 to 5 so LDPC+CRC acts as the final arbiter (matching WSJT-X
# behaviour) and weak signals are not discarded at the sync stage.
_MIN_COSTAS_MATCHES: int = 5

# Maximum number of sync candidates to process per slot after deduplication.
# The waterfall search returns up to 40 coarse hits; capping at 10 left valid
# weak signals on the table.  25 gives headroom for dense band openings.
_MAX_SYNC_CANDIDATES: int = 25

# Fine time-search step size as a fraction of one symbol duration.
# 1/8  = 20 ms (former default) → 1/16 = 10 ms halves worst-case ISI error
# and doubles the number of tested time offsets per candidate (from 9 to 17).
_FINE_DT_FRACTION: float = 1.0 / 16.0

# Adaptive LDPC: if the baseline 50-iteration decode leaves this many or fewer
# unsatisfied parity checks, retry with a higher iteration budget before giving
# up.  Signals in this "near-miss" zone often converge with more iterations.
_ADAPTIVE_LDPC_ERROR_THRESHOLD: int = 5
_ADAPTIVE_LDPC_MAX_ITERATIONS: int = 100

# Deep search (iterative interference cancellation): maximum decode passes
# performed on successive residual frames.  Set to 0 to disable deep search.
_DEEP_SEARCH_MAX_PASSES: int = 3

# Common AP tuples (n28a encoding helpers)
_AP_BITS_I3_1 = ((74, 0), (75, 0), (76, 1))   # i3 = 001
_AP_BITS_I3_2 = ((74, 0), (75, 1), (76, 0))   # i3 = 010
_AP_BITS_I3_0 = ((74, 0), (75, 0), (76, 0))   # i3 = 000
_AP_BITS_I3_3 = ((74, 0), (75, 1), (76, 1))   # i3 = 011  (non-standard/hashed calls)
_AP_BITS_I3_4 = ((74, 1), (75, 0), (76, 0))   # i3 = 100  (WWROF contest)
# n28a=2 (CQ): 28-bit MSB-first encoding of 2 = ...00000010
_AP_BITS_N28A_CQ = tuple((i, 0) for i in range(26)) + ((26, 1), (27, 0))   # type: ignore[assignment]
# n28a=0 (DE): all 28 bits zero
_AP_BITS_N28A_DE = tuple((i, 0) for i in range(28))   # type: ignore[assignment]

# Each entry: (pass_name, tuple_of_(bit_index, bit_value) pairs)
# Applied only to the 91 systematic bits (indices 0–90); must satisfy 0 ≤ bit_idx < 91.
_AP_PASSES: tuple[tuple[str, tuple[tuple[int, int], ...]], ...] = (
    # Generic message-type passes (3 known bits each – safe, low false-positive risk)
    ("i3=1",     _AP_BITS_I3_1),
    ("i3=2",     _AP_BITS_I3_2),
    # Non-standard and contest sub-types (3-bit priors – cheap and useful)
    ("i3=3",     _AP_BITS_I3_3),
    ("i3=4",     _AP_BITS_I3_4),
    # CQ + standard type-1 (32 known bits: n28a=CQ, ipa=0, i3=1 – common on FT8 bands)
    ("CQ+i3=1",  _AP_BITS_N28A_CQ + ((28, 0),) + _AP_BITS_I3_1),   # type: ignore[operator]
    # DE + standard type-1 (32 known bits: n28a=DE, ipa=0, i3=1)
    ("DE+i3=1",  _AP_BITS_N28A_DE + ((28, 0),) + _AP_BITS_I3_1),   # type: ignore[operator]
    # Free-text / special sub-types (3 known bits)
    ("i3=0",     _AP_BITS_I3_0),
)


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

    Milestone 6: DFT basis matrix is cached per f0_hz.  When exploring many
    time offsets for the same candidate frequency the basis is identical and
    need only be computed once.
    """

    # Maximum number of distinct f0 values to keep in the basis cache.
    # At 6.25 Hz spacing over a 200–3200 Hz band, there are ≤ 480 possible
    # coarse bins; 64 covers the hot-band portion and most fine-freq variants.
    _CACHE_MAXSIZE: int = 64

    def __init__(self, fs: int = FT8_FS) -> None:
        self.fs = int(fs)
        self.sym_n = int(round(FT8_SYMBOL_DURATION_S * self.fs))  # 1920 @ 12 kHz
        # {f0_hz: basis (8, sym_n) complex128} — insertion-ordered for LRU eviction.
        self._basis_cache: OrderedDict[float, np.ndarray] = OrderedDict()
        # Milestone 6 parallel decode: guard cache mutations so that concurrent
        # _decode_one_candidate calls sharing this extractor don't corrupt the
        # LRU ordering or trigger double-eviction when numpy releases the GIL
        # between cache.get() and cache.__setitem__().
        self._cache_lock = threading.Lock()

    def _get_basis(self, f0_hz: float) -> np.ndarray:
        """Return (cached) DFT basis matrix for the given base frequency.

        Uses a true LRU eviction policy backed by :class:`collections.OrderedDict`:
        a cache hit moves the entry to *most-recently-used* position, and an
        eviction removes the *least-recently-used* (first) entry.

        Thread-safe: a lock guards all cache mutations so that parallel
        :meth:`_decode_one_candidate` calls sharing this extractor do not race
        on the :class:`collections.OrderedDict` during numpy computation
        (which releases the GIL between ``get()`` and ``__setitem__()``).
        The expensive basis computation is performed *outside* the lock so
        that threads can compute different frequencies concurrently.
        """
        # Fast path: check cache under lock and return immediately on hit.
        with self._cache_lock:
            basis = self._basis_cache.get(f0_hz)
            if basis is not None:
                self._basis_cache.move_to_end(f0_hz)
                return basis

        # Cache miss: compute outside the lock (numpy releases GIL; concurrent
        # threads computing different f0 values can run in parallel).
        phase_inc = 2.0 * math.pi * np.array(
            [f0_hz + k * FT8_TONE_SPACING_HZ for k in range(8)]
        ) / float(self.fs)
        t_sym = np.arange(self.sym_n, dtype=np.float64)
        basis_new = np.exp(-1j * np.outer(phase_inc, t_sym))  # (8, sym_n)

        # Re-acquire lock to store.  Another thread may have stored this key
        # while we were computing — in that case return the already-cached
        # entry to preserve object identity (tests rely on `b1 is b2`).
        with self._cache_lock:
            if f0_hz in self._basis_cache:
                # Another thread won the race; use its entry and promote to MRU.
                self._basis_cache.move_to_end(f0_hz)
                return self._basis_cache[f0_hz]
            if len(self._basis_cache) >= self._CACHE_MAXSIZE:
                # Evict the least-recently-used entry.
                self._basis_cache.popitem(last=False)
            self._basis_cache[f0_hz] = basis_new
        return basis_new

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

        # Basis matrix: cached per f0_hz (reused across fine time-search calls).
        basis = self._get_basis(f0_hz)   # (8, sym_n)

        n_start = t0_n
        n_end   = t0_n + FT8_NSYMS * sym_n

        if n_start >= 0 and n_end <= len(x):
            # Fast path: all 79 symbols lie within the frame.
            # Reshape audio into (79, sym_n) and evaluate all DFTs in one matmul:
            #   (79, sym_n) @ (sym_n, 8)  →  (79, 8) complex — a single BLAS call.
            x_matrix = x[n_start:n_end].reshape(FT8_NSYMS, sym_n)
            dfts = x_matrix @ basis.T        # (79, 8) complex
            E79 = np.abs(dfts) ** 2          # |DFT|²  (79, 8) float64
        else:
            # Fallback: per-symbol bounds check for edge-of-buffer cases.
            E79 = np.zeros((FT8_NSYMS, 8), dtype=np.float64)
            for s in range(FT8_NSYMS):
                start = t0_n + s * sym_n
                end   = start + sym_n
                if start < 0 or end > len(x):
                    continue
                dft = basis @ x[start:end]
                E79[s] = (dft * np.conj(dft)).real
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

    N = FT8_NDATASYM   # 58
    E_safe = np.maximum(E, 1e-30)   # (58, 8)

    # Reorder energies so axis-1 indexes gray_value instead of tone index.
    # s2[s, gv] = E_safe[s, gray_fwd[gv]]  — equivalent to the original inner loop
    # "for gv in range(8): s2[gv] = row[gray_fwd[gv]]", but for all 58 symbols
    # at once via numpy fancy indexing (no Python loop).
    s2 = E_safe[:, _GRAY_FWD_NP]   # (58, 8)

    # ── LLRs (ft8_lib ft8_extract_symbol conventions) ────────────────────────
    # Bit 2 (MSB):  max(gv=4..7) − max(gv=0..3)
    llr_b2 = np.max(s2[:, 4:8], axis=1) - np.max(s2[:, 0:4], axis=1)
    # Bit 1:        max(gv=2,3,6,7) − max(gv=0,1,4,5)
    llr_b1 = (np.max(s2[:, [2, 3, 6, 7]], axis=1)
              - np.max(s2[:, [0, 1, 4, 5]], axis=1))
    # Bit 0 (LSB):  max(gv=1,3,5,7) − max(gv=0,2,4,6)
    llr_b0 = (np.max(s2[:, [1, 3, 5, 7]], axis=1)
              - np.max(s2[:, [0, 2, 4, 6]], axis=1))

    # Interleave into (174,): [b2_0, b1_0, b0_0, b2_1, b1_1, b0_1, …]
    llrs = np.empty(N * 3, dtype=np.float64)
    llrs[0::3] = llr_b2
    llrs[1::3] = llr_b1
    llrs[2::3] = llr_b0

    # ── Hard bits via gray inverse ────────────────────────────────────────────
    gv_all = _GRAY_INV_NP[syms]   # (58,) gray values, no Python loop
    hard_bits = np.empty(N * 3, dtype=np.uint8)
    hard_bits[0::3] = (gv_all >> 2) & 1
    hard_bits[1::3] = (gv_all >> 1) & 1
    hard_bits[2::3] = gv_all & 1

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


def _ldpc_check_vec(plain: np.ndarray) -> int:
    """
    Vectorised parity check via H-matrix multiplication.

    Equivalent to _ldpc_check but uses a precomputed dense H matrix and a
    single numpy matmul instead of Python loops — roughly 50× faster.
    """
    syndrome = (_H_LDPC.astype(np.int32) @ plain.astype(np.int32)) % 2
    return int(syndrome.sum())


def ft8_ldpc_decode(
    llrs: np.ndarray,
    *,
    max_iterations: int = 50,
    ap_assignments: Optional[tuple[tuple[int, int], ...]] = None,
) -> tuple[bool, np.ndarray, int, int]:
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

    ap_assignments: optional sequence of (bit_index, bit_value) pairs for
    a priori information.  Each pair boosts codeword[bit_index] by
    ±_AP_LLR_MAGNITUDE before BP begins (positive sign if bit_value=1,
    negative if bit_value=0).  Only indices 0–90 (systematic bits) are
    accepted; this matches the WSJT-X ft8b.f90 AP decode approach.

    Returns (success, payload[0:91], iterations_used, best_parity_errors).
    success=True only when all 83 parity checks pass AND CRC-14 matches.
    best_parity_errors is the minimum number of unsatisfied parity checks seen
    across all iterations (0 means LDPC converged; >0 means it did not).
    """
    codeword = np.asarray(llrs, dtype=np.float64)
    if codeword.shape != (174,):
        raise ValueError(f"Expected (174,) LLRs, got {codeword.shape}")

    # Apply a priori LLR boosts for known structural bits (e.g. i3 field).
    # Only modify a copy so the caller's array is unchanged.
    if ap_assignments:
        codeword = codeword.copy()
        for bit_idx, bit_val in ap_assignments:
            if 0 <= bit_idx < 91:
                codeword[bit_idx] += _AP_LLR_MAGNITUDE if bit_val else -_AP_LLR_MAGNITUDE

    N = 174
    CLAMP = 0.9999
    EPS = 1e-10
    M = len(_BP_Nm)            # 83
    max_row = _BP_MAX_ROW_LEN  # 7

    # tov[n, k]     — C→V message from check _BP_Mn[n][k] to variable n  (174×3)
    tov = np.zeros((N, 3), dtype=np.float64)
    # toc_padded[m, n_idx] — V→C message from variable to check m (83×max_row).
    # Padding positions are initialised to 1.0 (neutral for multiplication).
    toc_padded = np.ones((M, max_row), dtype=np.float64)

    plain = np.zeros(N, dtype=np.uint8)
    best_errors = M
    best_plain = plain.copy()
    iterations_used = max_iterations

    for iteration in range(max_iterations):
        # ── V→C (fully vectorised) ────────────────────────────────────────────
        # Extrinsic belief from variable n towards check m:
        #   Lmn = (codeword[n] + Σ_k tov[n,k]) − tov[n, m_idx]
        total = codeword + tov.sum(axis=1)                                   # (N,)
        extrinsic = total[_BP_EDGES_VN] - tov[_BP_EDGES_VN, _BP_EDGES_MIDX] # (E,)
        toc_flat = np.tanh(-extrinsic * 0.5)                                 # (E,)
        toc_padded[:] = 1.0                           # reset to neutral element
        toc_padded.ravel()[_BP_EDGES_RAVEL] = toc_flat  # scatter into padded matrix

        # ── C→V (log-domain product exclusion, fully vectorised) ─────────────
        # For each check row m and position n_idx, compute the product of all
        # tanh values in that row *excluding* position n_idx.
        #
        # sign(prod_excl[m,i]) = sign(prod_all[m]) · sign(toc[m,i])
        # |prod_excl[m,i]|     = exp( Σ_j log|toc[m,j]| − log|toc[m,i]| )
        #
        # Padding cells hold 1.0: log(1)=0 contributes nothing to the sum,
        # and sign(+1)=+1 is neutral for the product — both safe to include.
        sign_toc = np.where(toc_padded >= 0, 1.0, -1.0)   # (M, max_row)
        sign_toc[~_BP_NM_VALID] = 1.0                      # ensure padding = +1
        log_abs_toc = np.log(np.maximum(np.abs(toc_padded), EPS))  # (M, max_row)
        log_abs_toc[~_BP_NM_VALID] = 0.0                   # padding contributes 0

        sum_log = log_abs_toc.sum(axis=1, keepdims=True)   # (M, 1)
        log_abs_excl = sum_log - log_abs_toc               # (M, max_row)
        abs_excl = np.exp(log_abs_excl)                    # (M, max_row)

        prod_sign = sign_toc.prod(axis=1, keepdims=True)   # (M, 1)
        sign_excl = prod_sign * sign_toc                   # (M, max_row)

        cov_upd = np.clip(sign_excl * abs_excl, -CLAMP, CLAMP)   # (M, max_row)
        cov_upd = -2.0 * np.arctanh(cov_upd)                     # (M, max_row)
        cov_upd[~_BP_NM_VALID] = 0.0                             # zero padding

        # Scatter new C→V messages back into tov (174×3)
        tov[_BP_EDGES_VN, _BP_EDGES_MIDX] = cov_upd.ravel()[_BP_EDGES_RAVEL]

        # ── Hard decision + vectorised parity check ───────────────────────────
        total = codeword + tov.sum(axis=1)
        plain = (total > 0.0).astype(np.uint8)
        errors = _ldpc_check_vec(plain)

        if errors < best_errors:
            best_errors = errors
            best_plain = plain.copy()
            if errors == 0:
                iterations_used = iteration + 1
                break

    plain = best_plain
    payload = plain[:91].copy()

    if best_errors > 0:
        return False, payload, iterations_used, best_errors

    # CRC-14 verification
    msg_bits = payload[:77]
    rx_crc_bits = payload[77:91]
    rx_crc = int(sum(int(b) << (13 - i) for i, b in enumerate(rx_crc_bits)))
    calc_crc = _ft8_crc14(msg_bits)
    return (calc_crc == rx_crc), payload, iterations_used, 0


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

    Milestone 6: vectorised — uses a single np.argmax over all Costas rows
    plus array comparison instead of the former 21-iteration Python loop.
    """
    E79 = np.asarray(E79, dtype=np.float64)
    cos_pos = np.array(FT8_COSTAS_POSITIONS, dtype=np.intp)          # (21,)
    cos_tones = np.array(
        # FT8_COSTAS_TONES has 7 elements (one Costas symbol per 7-position
        # group).  The 21 Costas positions are three identical copies of the
        # same 7-tone pattern, so % 7 cycles correctly: positions 0-6, 7-13,
        # 14-20 all map to FT8_COSTAS_TONES[0..6].
        [FT8_COSTAS_TONES[i % 7] for i in range(21)], dtype=np.int32  # (21,)
    )
    argmax_at_pos = np.argmax(E79[cos_pos], axis=1).astype(np.int32)  # (21,)
    m_normal = int(np.sum(argmax_at_pos == cos_tones))
    m_inv    = int(np.sum(argmax_at_pos == (7 - cos_tones)))
    if m_normal >= m_inv:
        return m_normal, 21, False
    return m_inv, 21, True


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


def _costas_energy_llr_scale(E79: np.ndarray) -> float:
    """
    Estimate a soft LLR scaling factor from the Costas symbol energy ratios.

    Milestone 6 — Soft Costas-energy LLR scaling:
    For each of the 21 Costas symbols the ratio (best_tone_energy /
    second_best_energy) approximates the instantaneous per-symbol SNR.  The
    geometric mean of these ratios serves as a slot-level quality metric.  A
    smooth mapping converts it to a multiplicative factor applied to the
    channel LLRs before LDPC belief propagation, improving convergence under
    multipath fading without changing the variance-24 normalisation target.

    The mapping is intentionally conservative:
      ratio ≤ 1 (noise-dominated)  → scale ≈ 0.7 (mild damping)
      ratio ≈ 4 (typical signal)   → scale ≈ 1.0 (neutral)
      ratio ≥ 16 (strong signal)   → scale ≈ 1.3 (mild boost)

    Returns a float in [0.6, 1.4].
    """
    cos_pos = np.array(FT8_COSTAS_POSITIONS, dtype=np.intp)   # (21,)
    E_cos = np.asarray(E79, dtype=np.float64)[cos_pos]        # (21, 8)
    # Sort descending along tone axis
    E_sorted = np.sort(E_cos, axis=1)[:, ::-1]                # (21, 8) desc
    second_best = E_sorted[:, 1]
    best = E_sorted[:, 0]
    eps = 1e-12
    ratios = best / np.maximum(second_best, eps)               # (21,)
    # Geometric mean in log domain (clamp ratios to ≥1 to stay non-negative)
    log_ratios = np.log(np.maximum(ratios, 1.0))
    geo_mean = float(np.exp(np.mean(log_ratios)))
    # Map: log4(geo_mean) centred at 1, with tanh compression
    # scale = 1 + 0.3 * tanh((log(geo_mean) - log(4)) / log(4))
    x = (math.log(max(geo_mean, 1.0)) - math.log(4.0)) / math.log(4.0)
    scale = 1.0 + 0.3 * math.tanh(x)
    return max(0.6, min(1.4, scale))

@dataclass(frozen=True)
class FT8DecodeResult:
    utc_time: str
    strength_db: float
    frequency_hz: float
    message: str


def format_ft8_message(utc: str, snr_db: float, freq_hz: float, message: str) -> str:
    """
    Format an FT8 decoded message matching the reference decoder output format.

    Output: ``HHMMSS +NN NNNN MESSAGE``

    Parameters
    ----------
    utc      : UTC timestamp string (e.g. ``HH:MM:SS`` or ``HHMMSS``)
    snr_db   : signal-to-noise ratio in dB (rounded to nearest integer)
    freq_hz  : audio frequency in Hz (displayed as integer Hz, matching WSJT-X)
    message  : decoded FT8 message text (e.g. ``CQ W4ABC EM73``)

    Returns
    -------
    str  formatted line ready for terminal or log output
    """
    snr_int = int(round(snr_db))
    freq_int = math.floor(freq_hz + 0.5)
    return f"{utc} {snr_int:+3d} {freq_int:4d} {message}"


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
                if costas_m < _MIN_COSTAS_MATCHES:
                    continue

                # Fine frequency refinement: try sub-bin Hz offsets to handle signals
                # not exactly on the 6.25 Hz grid (transceiver calibration drift ≤ ±3 Hz).
                f0_dec = f0_hz   # refined frequency used for the decode path below
                for _fdf in _FINE_FREQ_OFFSETS_HZ:
                    _E79_try = extractor.extract_all_79(frame, t0_s=t0_s, f0_hz=f0_hz + _fdf)
                    _m_try, _, _ = _costas_score(_E79_try)
                    if _m_try > costas_m:
                        costas_m = _m_try
                        f0_dec = f0_hz + _fdf
                        E79 = _E79_try

                E_payload, hard_syms = ft8_extract_payload_symbols(E79)
                _, ch_llrs = ft8_gray_decode(hard_syms, E_payload)

                # Normalise to variance=24 (ftx_normalize_logl)
                var = float(np.var(ch_llrs))
                if var > 1e-10:
                    ch_llrs = ch_llrs * math.sqrt(24.0 / var)

                # Milestone 6: Soft Costas-energy LLR scaling.
                ch_llrs = ch_llrs * _costas_energy_llr_scale(E79)

                ok, payload, iters, _base_errs = ft8_ldpc_decode(ch_llrs, max_iterations=max_iterations)
                # Milestone 6: Adaptive LDPC for near-miss signals.
                if not ok and _base_errs <= _ADAPTIVE_LDPC_ERROR_THRESHOLD:
                    ok, payload, iters, _base_errs = ft8_ldpc_decode(
                        ch_llrs, max_iterations=_ADAPTIVE_LDPC_MAX_ITERATIONS
                    )
                # If baseline fails with few parity errors (near-miss real signal),
                # try AP passes for common message types.
                if not ok and _base_errs <= _AP_PARITY_ERROR_THRESHOLD:
                    for _ap_name, _ap_bits in _AP_PASSES:
                        ok, payload, iters, _base_errs = ft8_ldpc_decode(
                            ch_llrs, max_iterations=max_iterations,
                            ap_assignments=_ap_bits,
                        )
                        if ok:
                            break
                        if _base_errs <= _ADAPTIVE_LDPC_ERROR_THRESHOLD:
                            ok, payload, iters, _base_errs = ft8_ldpc_decode(
                                ch_llrs,
                                max_iterations=_ADAPTIVE_LDPC_MAX_ITERATIONS,
                                ap_assignments=_ap_bits,
                            )
                            if ok:
                                break
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
                        f"  [DECODE] slot={utc_label} f0={f0_dec:.1f}Hz "
                        f"dt={dt:+.2f}s costas={costas_m}/21 "
                        f"iters={iters} msg='{message}'"
                    )

                results.append(FT8DecodeResult(
                    utc_time=utc_label,
                    strength_db=snr_db,
                    frequency_hz=f0_dec,
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

    # Tolerance (seconds) for treating _t0_utc as exactly on a slot boundary.
    # 1 ms covers accumulated float64 arithmetic drift across many slots.
    _BOUNDARY_TOLERANCE_S: float = 1e-3

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
        while True:
            # Advance _t0_utc to the next slot boundary, discarding any
            # pre-boundary audio.  This ensures every emitted frame starts
            # exactly at a UTC 15-second boundary so that FT8 signals (which
            # also start at those boundaries) appear at t ≈ 0 in the frame,
            # within the sync-search window.
            offset_in_slot = float(self._t0_utc) % self.frame_s
            tol = self._BOUNDARY_TOLERANCE_S
            if offset_in_slot < tol or offset_in_slot > self.frame_s - tol:
                pre_n = 0   # already on (or within tolerance of) a boundary
            else:
                pre_n = int(round((self.frame_s - offset_in_slot) * self.fs))

            if len(self._buf) < pre_n + self.frame_n:
                break

            # Discard samples that precede the slot boundary
            if pre_n > 0:
                self._buf = self._buf[pre_n:]
                self._t0_utc = float(self._t0_utc) + (self.frame_s - offset_in_slot)

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
        dt_step = self.sym_s / 8.0   # 20 ms — finer step improves timing resolution
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


def _ft8_waterfall_sync(
    frame: np.ndarray,
    *,
    fs: int = FT8_FS,
    fmin_hz: float = 200.0,
    fmax_hz: float = 3200.0,
    t0_min_s: float = -0.5,
    t0_max_s: float = 15.0 - FT8_TX_DURATION_S,
    min_costas: int = 7,
    max_results: int = 40,
) -> list[tuple[float, float, int]]:
    """
    Fast coarse sync search using a pre-computed incoherent power spectrogram.

    Implements the ft8_lib ``find_sync`` approach:
      1. Compute a power waterfall at one-symbol-period (160 ms) hop, no overlap.
      2. For every (t0_idx, f0_bin) candidate score by Costas-array argmax matches.
      3. Return candidates sorted by Costas match count descending.

    This covers the full valid FT8 timing window (t0 ∈ [t0_min_s, t0_max_s])
    in O(n_wins × n_f0) time — typically < 5 ms vs the previous ≈ 8 s per slot.

    Parameters
    ----------
    frame      : 1-D float audio at *fs* Hz (should be ≥ 15 s)
    fs         : sample rate (Hz)
    fmin_hz    : lower edge of frequency search band
    fmax_hz    : upper edge of frequency search band
    t0_min_s   : earliest signal start time relative to frame start (s)
    t0_max_s   : latest  signal start time relative to frame start (s)
    min_costas : minimum Costas match count (0–21) required to include result
    max_results: cap on returned candidates

    Returns
    -------
    list of (t0_s, f0_hz, costas_matches) sorted by costas_matches descending.
    ``t0_s``        — coarse signal start time in seconds (160 ms resolution)
    ``f0_hz``       — base frequency of lowest tone (6.25 Hz resolution)
    ``costas_matches`` — number of Costas symbols that matched (0–21)
    """
    sym_n = int(round(FT8_SYMBOL_DURATION_S * fs))   # 1920 at 12 kHz
    df = float(fs) / sym_n                            # 6.25 Hz per bin
    x = np.asarray(frame, dtype=np.float64)
    n_wins = len(x) // sym_n
    if n_wins < FT8_NSYMS:
        return []

    # ── Step 1: Batch FFT — incoherent power spectrogram at 1-symbol hop ─────
    # x_matrix rows are non-overlapping sym_n-sample windows.
    x_matrix = x[: n_wins * sym_n].reshape(n_wins, sym_n)
    spec = np.abs(np.fft.rfft(x_matrix, axis=1)) ** 2   # (n_wins, sym_n//2+1)

    lo_bin = max(0, int(math.floor(fmin_hz / df)))
    hi_bin = min(spec.shape[1] - 8, int(math.ceil(fmax_hz / df)))
    n_f0 = hi_bin - lo_bin
    if n_f0 <= 0:
        return []

    # ── Step 2: Compute argmax tone at every (window, f0) position ───────────
    # For each f0 bin b, the 8 tones occupy spec[:, b+0 .. b+7].
    # We stack 8 shifted slices and take the argmax along the tone axis.
    sub = spec[:, lo_bin: lo_bin + n_f0 + 7]            # (n_wins, n_f0 + 7)
    tone_e = np.stack([sub[:, k: k + n_f0] for k in range(8)], axis=0)  # (8, n_wins, n_f0)
    argmax_tones = np.argmax(tone_e, axis=0)             # (n_wins, n_f0)

    # ── Step 3: Score every (t0_idx, f0_bin) by Costas match count ───────────
    t0_idx_lo = int(math.floor(t0_min_s / FT8_SYMBOL_DURATION_S))
    t0_idx_hi = int(math.ceil(t0_max_s / FT8_SYMBOL_DURATION_S))
    t0_indices = np.arange(t0_idx_lo, t0_idx_hi + 1, dtype=np.int32)  # (T,)
    T = len(t0_indices)

    cos_pos = np.array(FT8_COSTAS_POSITIONS, dtype=np.int32)          # (21,)
    cos_tones = np.array(
        [FT8_COSTAS_TONES[i % 7] for i in range(21)], dtype=np.int32  # (21,)
    )

    # sym_indices[t, c] = t0_indices[t] + cos_pos[c]   shape (T, 21)
    sym_indices = t0_indices[:, np.newaxis] + cos_pos[np.newaxis, :]
    valid = (sym_indices >= 0) & (sym_indices < n_wins)       # (T, 21) bool
    sym_clipped = np.clip(sym_indices, 0, n_wins - 1)

    # argmax_at_syms[t, c, f] = argmax_tones[sym_clipped[t,c], f]   (T, 21, n_f0)
    argmax_at_syms = argmax_tones[sym_clipped, :]

    # matches[t, c, f] = 1 if argmax matches expected Costas tone and index valid
    matches = (argmax_at_syms == cos_tones[np.newaxis, :, np.newaxis])
    matches &= valid[:, :, np.newaxis]

    # scores[t, f] = total Costas matches summed over 21 positions
    scores = matches.sum(axis=1)  # (T, n_f0) int

    # ── Step 4: Collect and return candidates above threshold ─────────────────
    good_t, good_f = np.where(scores >= min_costas)
    if len(good_t) == 0:
        return []

    out = [
        (
            float(t0_indices[int(ti)]) * FT8_SYMBOL_DURATION_S,
            float(lo_bin + int(fi)) * df,
            int(scores[int(ti), int(fi)]),
        )
        for ti, fi in zip(good_t, good_f)
    ]
    out.sort(key=lambda r: -r[2])
    return out[:max_results]


# ═══════════════════════════════════════════════════════════════════════════════
# § 14b  Interference Cancellation Helpers (Milestone 6 — deep search)
# ═══════════════════════════════════════════════════════════════════════════════

def _make_callsign_ap_passes(
    callsign: str,
) -> list[tuple[str, tuple[tuple[int, int], ...]]]:
    """
    Build AP pass tuples that inject a known callsign into LDPC decoding.

    Milestone 6 — Callsign-aware AP passes:
    When the operator has an active QSO partner whose callsign is known, we
    can inject that callsign's 28-bit n28 encoding as a priori bits into extra
    LDPC passes.  This targets the common near-miss scenario where BP almost
    converges but cannot cross the threshold because the callsign bits are
    marginal.

    Parameters
    ----------
    callsign : str  — the DX station callsign (will be packed to n28)

    Returns
    -------
    List of (name, bits) tuples suitable for appending to _AP_PASSES.
    Two passes are generated:
      * callsign as n28a (bits  0-27) + i3=1 — DX is the calling station
      * callsign as n28b (bits 29-56) + i3=1 — DX is the called station
    An empty list is returned if the callsign cannot be packed.
    """
    # Import here to avoid a circular dependency at module level; ft8_encode
    # has no dependency on ft8_decode.
    try:
        from ft8_encode import ft8_pack_callsign
        n28 = ft8_pack_callsign(callsign)
    except Exception:
        return []

    # 28-bit MSB-first decomposition: bit 0 = MSB (bit 27 of n28)
    n28_bits: tuple[tuple[int, int], ...] = tuple(
        (i, int((n28 >> (27 - i)) & 1)) for i in range(28)
    )

    name = callsign.upper()
    # n28a position: bits 0-27, ipa=0 (bit 28)
    bits_n28a: tuple[tuple[int, int], ...] = (
        n28_bits + ((28, 0),) + _AP_BITS_I3_1   # type: ignore[operator]
    )
    # n28b position: bits 29-56, ipb=0 (bit 57)
    n28b_bits: tuple[tuple[int, int], ...] = tuple(
        (29 + i, int((n28 >> (27 - i)) & 1)) for i in range(28)
    )
    bits_n28b: tuple[tuple[int, int], ...] = (
        n28b_bits + ((57, 0),) + _AP_BITS_I3_1  # type: ignore[operator]
    )
    return [
        (f"{name}(n28a)+i3=1", bits_n28a),
        (f"{name}(n28b)+i3=1", bits_n28b),
    ]


def _subtract_decoded_signal(
    frame: np.ndarray,
    *,
    t0_s: float,
    f0_hz: float,
    symbols: np.ndarray,
    fs: int,
    out: "Optional[np.ndarray]" = None,
) -> np.ndarray:
    """
    Reconstruct and subtract a decoded FT8 signal from an audio frame.

    Milestone 6 — Iterative interference cancellation (deep search):
    Given the 79-symbol tone sequence for a successfully decoded signal and
    its time/frequency location in the frame, this function:
      1. Synthesises the signal at unit amplitude (phase-continuous 8-FSK).
      2. Estimates the actual amplitude by comparing the measured Costas-
         symbol energy to the synthesised Costas-symbol energy.
      3. Subtracts the scaled synthesised waveform from the frame.

    Parameters
    ----------
    frame   : 1-D float audio (at ``fs`` Hz)
    t0_s    : signal start time relative to frame start (seconds)
    f0_hz   : base tone frequency (Hz)
    symbols : (79,) uint8 — decoded tone indices (0–7)
    fs      : audio sample rate (Hz)
    out     : optional pre-allocated float64 output buffer of the same shape
              as ``frame``.  When provided the subtraction is performed in-place
              into *out* without an extra full-frame copy.  The caller is
              responsible for initialising *out* with the current residual
              before calling this function.  When *None* (default) a new
              float64 copy is returned.

    Returns
    -------
    np.ndarray  residual frame (same shape as ``frame``, float64)
    """
    from ft8_encode import ft8_symbols_to_audio

    sym_n = int(round(FT8_SYMBOL_DURATION_S * fs))
    t0_n = int(round(t0_s * fs))
    total_n = FT8_NSYMS * sym_n

    # Synthesise the FT8 tone sequence at unit amplitude (ramp suppressed so
    # we can estimate amplitude uniformly across the whole signal).
    synth_f32 = ft8_symbols_to_audio(
        symbols, f0_hz=f0_hz, fs=fs, amplitude=1.0, ramp_samples=0
    )
    synth = synth_f32.astype(np.float64)

    # Estimate signal amplitude from Costas positions (known, reliable energy).
    meas_e = 0.0
    synth_e = 0.0
    for pos in FT8_COSTAS_POSITIONS:
        start_frame = t0_n + pos * sym_n
        end_frame = start_frame + sym_n
        start_synth = pos * sym_n
        end_synth = start_synth + sym_n
        if start_frame < 0 or end_frame > len(frame):
            continue
        meas_e  += float(np.sum(np.asarray(frame[start_frame:end_frame], dtype=np.float64) ** 2))
        synth_e += float(np.sum(synth[start_synth:end_synth] ** 2))

    # Prepare the output buffer.
    if out is None:
        residual = np.asarray(frame, dtype=np.float64).copy()
    else:
        residual = out

    if synth_e < 1e-12:
        # Synthesised Costas energy is effectively zero — this can happen if
        # all Costas positions fall outside the frame boundary (e.g. t0 is
        # near the end of the buffer).  Skip subtraction silently.
        return residual

    amplitude = math.sqrt(meas_e / synth_e)

    # Subtract the scaled synthesised waveform in-place.
    frame_start = max(0, t0_n)
    frame_end   = min(len(residual), t0_n + total_n)
    synth_start = frame_start - t0_n
    synth_end   = synth_start + (frame_end - frame_start)
    if frame_end > frame_start:
        residual[frame_start:frame_end] -= amplitude * synth[synth_start:synth_end]

    return residual


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

    Milestone 6 additions
    ---------------------
    * ``set_dx_callsign(call)`` — injects the active QSO partner's callsign
      into extra LDPC AP passes, improving decode of near-miss messages from
      that station.
    * Deep search (iterative interference cancellation) is enabled by default
      (up to ``_DEEP_SEARCH_MAX_PASSES`` residual passes).  Set
      ``deep_search_passes = 0`` to disable it.
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

        self._q: queue.Queue = queue.Queue(maxsize=200)  # 200 × 100 ms = 20 s buffer
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
        # Milestone 6: use lowered default threshold (5 instead of 7).
        self._min_costas_matches: int = _MIN_COSTAS_MATCHES
        # Milestone 6: number of iterative interference-cancellation passes.
        self.deep_search_passes: int = _DEEP_SEARCH_MAX_PASSES
        # Milestone 6: callsign-aware AP pass tuples (thread-safe read, Tk writes).
        self._dx_ap_passes: list[tuple[str, tuple[tuple[int, int], ...]]] = []

    def set_dx_callsign(self, callsign: Optional[str]) -> None:
        """
        Register the active QSO partner's callsign for callsign-aware AP passes.

        Milestone 6 — Callsign-aware AP passes:
        Calling this with a valid callsign string causes the decoder to inject
        that callsign's n28 bits into additional LDPC belief-propagation passes
        when a baseline decode narrowly fails (best_errors ≤ AP threshold).
        This targets the common near-miss scenario where BP cannot converge
        solely because the partner's callsign bits are marginal.

        Call with ``None`` (or an empty string) to clear the partner context
        (e.g. when the QSO ends or a CQ session is aborted).

        Parameters
        ----------
        callsign : str or None
            DX station callsign (e.g. 'K9XYZ').  Invalid callsigns are silently
            ignored (no AP passes generated).
        """
        if not callsign:
            self._dx_ap_passes = []
            return
        self._dx_ap_passes = _make_callsign_ap_passes(callsign.strip().upper())

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

    def reset_framer(self) -> None:
        """
        Reset the UTC slot framer and drain the audio queue for a clean restart.

        **Must be called before starting a new audio stream** — for example,
        every time the GUI's "Start Audio" button begins a new
        :class:`~digi_input.SoundCardAudioSource` session — to prevent the
        framer from treating new audio as a continuation of a previous (now
        stale) stream.

        Why this is necessary
        ---------------------
        The decoder runs as a long-lived background thread; its
        :class:`UTC15sFramer` accumulates audio and emits UTC-aligned 15-second
        frames.  When the audio stream is stopped and restarted the framer
        retains:

        * ``_t0_utc`` — a past epoch timestamp from the *previous* stream
        * ``_buf``    — leftover audio samples from the same
        * ``_first_frame_emitted`` — True, so the next frame won't be
          marked partial and skipped

        New audio chunks carry fresh ``t0_monotonic`` values (current time)
        but ``_t0_utc`` is only initialised on the very *first* push call
        (when it is ``None``).  Because ``_t0_utc`` is not ``None`` after a
        stop/start cycle, the framer thinks the new audio is a seamless
        continuation of the old stream and assembles frames whose slot
        boundaries are offset from the actual current UTC position.  FT8
        signals in the new audio appear at incorrect time offsets inside those
        frames and the Costas sync search misses them.

        What this method does
        ----------------------
        1. Replaces ``_framer`` with a fresh :class:`UTC15sFramer` instance
           (``_t0_utc = None``, empty buffer, ``_first_frame_emitted = False``).
           The assignment is atomic in CPython; the worker thread picks up the
           new framer on its next iteration.
        2. Drains all queued audio chunks that pre-date this reset.  Old
           chunks would re-initialise ``_t0_utc`` to a stale value, defeating
           the purpose of the framer replacement.
        3. Resets ``_resampler`` to ``None`` so it re-creates on the first
           new chunk (harmless when the sample rate has not changed, avoids
           any edge case if it has).

        After calling this method the first frame emitted by the new stream
        will be marked ``is_partial=True`` and skipped by the worker, which is
        the correct behaviour for any mid-slot stream start.
        """
        # 1. Replace framer atomically (CPython pointer write is atomic).
        self._framer = UTC15sFramer(self.fs_proc)
        # 2. Drain stale queued audio.  Old chunks have timestamps from before
        #    the stop and would set _t0_utc to the wrong epoch on their first
        #    push() into the new framer.
        drained = 0
        while True:
            try:
                self._q.get_nowait()
                drained += 1
            except queue.Empty:
                break
        if self._debug and drained:
            print(
                f"[FT8] reset_framer: drained {drained} stale chunk(s) from queue",
                flush=True,
            )
        # 3. Reset resampler so it re-initialises cleanly on the next chunk.
        self._resampler = None

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

    # ------------------------------------------------------------------
    # Internal decode helpers
    # ------------------------------------------------------------------

    def _decode_one_candidate(
        self,
        frame: np.ndarray,
        t0_coarse: float,
        f0_hz_coarse: float,
        seen_msg: set,
    ) -> "list[tuple[str, float, float, float, np.ndarray, np.ndarray]]":
        """
        Fine-search, LLR extraction, LDPC decode for one sync candidate.

        Returns a list of (message, f0_hz, t0_s, snr_db, payload_91, symbols_79)
        tuples for each successful decode (normally 0 or 1).

        Milestone 6: extracted from _decode_frame to allow parallel execution
        via ThreadPoolExecutor — the frame array is read-only and all state
        is local to this call.
        """
        _T0_MIN = -0.50
        _T0_MAX = 15.0 - FT8_TX_DURATION_S

        # ── Fine time search ──────────────────────────────────────────────────
        # Milestone 6: step is FT8_SYMBOL_DURATION_S * _FINE_DT_FRACTION
        # = 10 ms (was 20 ms), giving 17 tested offsets instead of 9.
        dt_fine = FT8_SYMBOL_DURATION_S * _FINE_DT_FRACTION
        half_sym = FT8_SYMBOL_DURATION_S * 0.5
        best_m = -1
        best_t0 = t0_coarse
        best_E79: Optional[np.ndarray] = None
        for dt in np.arange(-half_sym, half_sym + dt_fine / 2.0, dt_fine):
            t0_try = float(t0_coarse) + float(dt)
            if t0_try < _T0_MIN or t0_try > _T0_MAX:
                continue
            E_try = self._extractor.extract_all_79(
                frame, t0_s=t0_try, f0_hz=f0_hz_coarse
            )
            m, _, _ = _costas_score(E_try)
            if m > best_m:
                best_m = m
                best_t0 = t0_try
                best_E79 = E_try

        if best_E79 is None:
            return []
        t0_s = best_t0
        costas_m = best_m
        E79 = best_E79

        # ── Fine frequency refinement ─────────────────────────────────────────
        # Milestone 6: ±4 Hz with 0.5 Hz sub-steps (was ±3 Hz, 1 Hz steps).
        _best_m = costas_m
        _best_f0 = f0_hz_coarse
        _best_E79 = E79
        for _fdf in _FINE_FREQ_OFFSETS_HZ:
            _E79_try = self._extractor.extract_all_79(
                frame, t0_s=t0_s, f0_hz=f0_hz_coarse + _fdf
            )
            _m_try, _, _ = _costas_score(_E79_try)
            if _m_try > _best_m:
                _best_m = _m_try
                _best_f0 = f0_hz_coarse + _fdf
                _best_E79 = _E79_try
        if _best_m > costas_m and self._debug:
            print(
                f"    [fine-freq] {f0_hz_coarse:.1f} → {_best_f0:.1f} Hz"
                f"  costas {costas_m} → {_best_m}/21",
                flush=True,
            )
        costas_m = _best_m
        f0_hz = _best_f0
        E79 = _best_E79

        if self._debug:
            print(
                f"  [cand] t0={t0_s:+.3f}s  f0={f0_hz:7.1f} Hz"
                f"  costas={costas_m}/21",
                flush=True,
            )
        if costas_m < self._min_costas_matches:
            if self._debug:
                print(
                    f"    -> costas {costas_m} < {self._min_costas_matches}"
                    f" threshold -- skip",
                    flush=True,
                )
            return []

        # ── LLR extraction, normalisation ─────────────────────────────────────
        E_payload, hard_syms = ft8_extract_payload_symbols(E79)
        _, ch_llrs = ft8_gray_decode(hard_syms, E_payload)

        var = float(np.var(ch_llrs))
        if var > 1e-10:
            ch_llrs = ch_llrs * math.sqrt(24.0 / var)

        # Milestone 6: Soft Costas-energy LLR scaling — multiply LLRs by a
        # quality factor derived from the energy contrast at the Costas
        # positions, improving BP convergence under multipath fading.
        ch_llrs = ch_llrs * _costas_energy_llr_scale(E79)

        # ── LDPC + AP passes ──────────────────────────────────────────────────
        ok, payload, iters, best_errors = ft8_ldpc_decode(ch_llrs)
        ap_pass_name = "none"

        if not ok and best_errors <= _AP_PARITY_ERROR_THRESHOLD:
            # Milestone 6: Adaptive LDPC — retry near-miss signals with more
            # iterations before attempting AP passes.
            if best_errors <= _ADAPTIVE_LDPC_ERROR_THRESHOLD:
                ok, payload, iters, best_errors = ft8_ldpc_decode(
                    ch_llrs, max_iterations=_ADAPTIVE_LDPC_MAX_ITERATIONS
                )
                if ok:
                    ap_pass_name = "adaptive-iters"

        if not ok and best_errors <= _AP_PARITY_ERROR_THRESHOLD:
            # Snapshot current DX AP passes (list may be updated from Tk thread).
            dx_passes = list(self._dx_ap_passes)
            all_ap = list(_AP_PASSES) + dx_passes
            for _ap_name, _ap_bits in all_ap:
                ok, payload, iters, best_errors = ft8_ldpc_decode(
                    ch_llrs, ap_assignments=_ap_bits
                )
                if ok:
                    ap_pass_name = _ap_name
                    break
                # Milestone 6: adaptive iterations for near-miss AP decodes.
                if best_errors <= _ADAPTIVE_LDPC_ERROR_THRESHOLD:
                    ok, payload, iters, best_errors = ft8_ldpc_decode(
                        ch_llrs,
                        ap_assignments=_ap_bits,
                        max_iterations=_ADAPTIVE_LDPC_MAX_ITERATIONS,
                    )
                    if ok:
                        ap_pass_name = _ap_name
                        break

        if self._debug:
            if ok:
                ap_tag = f" ap={ap_pass_name}" if ap_pass_name != "none" else ""
                print(f"    [ldpc] OK  ({iters} iters{ap_tag})", flush=True)
            else:
                if best_errors == 0:
                    rx_crc_bits = payload[77:91]
                    rx_crc = int(
                        sum(int(b) << (13 - i)
                            for i, b in enumerate(rx_crc_bits))
                    )
                    calc_crc = _ft8_crc14(payload[:77])
                    print(
                        f"    [ldpc] LDPC OK ({iters} iters) -- CRC mismatch"
                        f"  rx=0x{rx_crc:04X}  calc=0x{calc_crc:04X}",
                        flush=True,
                    )
                else:
                    print(
                        f"    [ldpc] FAIL  {best_errors} parity errors"
                        f"  ({iters} iters)",
                        flush=True,
                    )
        if not ok:
            return []

        message = ft8_unpack_message(payload[:77])
        if self._debug:
            print(f"    [msg]  '{message}'", flush=True)
        if message in seen_msg:
            return []

        # Re-encode to 79-tone symbols for interference cancellation.
        try:
            from ft8_encode import ft8_encode_to_symbols as _enc_syms
            symbols = _enc_syms(message)
        except Exception:
            symbols = np.zeros(FT8_NSYMS, dtype=np.uint8)

        # Compute SNR from payload symbol energy contrast in E79.
        # For each of the 58 payload positions, the dominant (max-energy) bin
        # carries signal + noise; the remaining 7 bins carry noise only.
        # The per-bin noise estimate is the mean energy of the 7 non-dominant
        # bins.  SNR is referenced to a 2500 Hz noise bandwidth (the WSJT-X
        # standard).  The correct normalization factor is:
        #   2500 Hz / (2 × 6.25 Hz) = 200
        # The factor of 2 accounts for the real-valued signal having energy
        # split between positive and negative DFT frequencies.
        # _PAYLOAD_POS_ARRAY is a module-level constant (no per-call allocation).
        _E_pl = E79[_PAYLOAD_POS_ARRAY, :]         # (58, 8) payload energies
        _max_e = np.max(_E_pl, axis=1)             # dominant bin per symbol
        _sum_e = np.sum(_E_pl, axis=1)             # total energy per symbol
        _noise_per_bin = (_sum_e - _max_e) / 7.0  # mean of the other 7 bins
        _avg_noise = float(np.mean(_noise_per_bin))
        _avg_sig = float(np.mean(_max_e)) - _avg_noise  # signal-only power
        _noise_2500 = max(_avg_noise * (2500.0 / (2.0 * FT8_TONE_SPACING_HZ)), 1e-30)
        snr_db = 10.0 * math.log10(max(_avg_sig, 1e-30) / _noise_2500)
        return [(message, f0_hz, t0_s, snr_db, payload[:91].copy(), symbols)]

    def _decode_pass(
        self,
        frame: np.ndarray,
        utc: str,
        seen_msg: set,
    ) -> "list[tuple[str, float, float, float, np.ndarray, np.ndarray]]":
        """
        Run one full waterfall-sync → fine-search → LDPC decode pass.

        Milestone 6: Parallel candidate decode via ThreadPoolExecutor.
        Returns a list of (message, f0_hz, t0_s, snr_db, payload_91, symbols_79)
        for all new decodes found in this pass.
        """
        _T0_MIN = -0.50
        _T0_MAX = 15.0 - FT8_TX_DURATION_S

        coarse = _ft8_waterfall_sync(
            frame,
            fs=self.fs_proc,
            fmin_hz=self.fmin_hz,
            fmax_hz=self.fmax_hz,
            t0_min_s=_T0_MIN,
            t0_max_s=_T0_MAX,
            min_costas=self._min_costas_matches,
            max_results=40,
        )

        if self._debug:
            print(f"[FT8] slot {utc}Z  {len(coarse)} coarse sync candidate(s)",
                  flush=True)

        # De-duplicate by frequency: keep only the best-scoring t0 per f0 bin,
        # then take the top _MAX_SYNC_CANDIDATES for the fine decode loop.
        # Milestone 6: cap raised from 10 → 25.
        seen_f0_bin: set[int] = set()
        sync_cands: list[tuple[float, float, float]] = []
        df = FT8_TONE_SPACING_HZ
        for t0_coarse, f0_hz, cos_m in coarse:
            f0_bin = int(round(f0_hz / df))
            if f0_bin in seen_f0_bin:
                continue
            seen_f0_bin.add(f0_bin)
            score_db = 10.0 * math.log10(max(cos_m / len(FT8_COSTAS_POSITIONS), 1e-6))
            sync_cands.append((t0_coarse, f0_hz, score_db))
            if len(sync_cands) >= _MAX_SYNC_CANDIDATES:
                break

        if self._debug:
            print(
                f"[FT8] slot {utc}Z  {len(sync_cands)} sync candidate(s)",
                flush=True,
            )
            for t0_s, f0_hz, score in sync_cands:
                print(
                    f"  [sync] t0={t0_s:+.3f}s  f0={f0_hz:7.1f} Hz"
                    f"  score={score:+5.1f} dB",
                    flush=True,
                )

        results: list[tuple[str, float, float, float, np.ndarray, np.ndarray]] = []

        # Milestone 6: Parallel candidate decode.
        # Each candidate reads only the immutable frame array; BLAS calls in
        # the matmul / LDPC steps release the GIL, enabling real speedup on
        # multi-core hardware.
        #
        # Thread-safety: Python `set` is not safe for concurrent read/write.
        # Pass a read-only frozenset snapshot of seen_msg to each worker so
        # that already-decoded messages are skipped without any lock contention.
        # Final deduplication against the live seen_msg happens in the
        # collector loop below (on the caller's thread), which is the only
        # place seen_msg is mutated.
        seen_msg_snapshot = frozenset(seen_msg)
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(
                    self._decode_one_candidate, frame, t0c, f0c, seen_msg_snapshot
                )
                for t0c, f0c, _ in sync_cands
            ]
            for future in as_completed(futures):
                try:
                    for item in future.result():
                        msg = item[0]
                        if msg not in seen_msg:
                            seen_msg.add(msg)
                            results.append(item)
                except Exception as exc:
                    if self._debug:
                        print(f"[FT8] candidate decode error: {exc!r}", flush=True)

        return results

    def _decode_frame(self, frame: np.ndarray, utc: str) -> None:
        """Run full decode pipeline on one 15 s frame.

        Sync search follows the ft8_lib ``find_sync`` approach:
          1. Fast waterfall sync  — incoherent spectrogram, covers full
             FT8 timing window in O(n_wins × n_f0) time (< 5 ms).
          2. Fine time search     — ±80 ms at 10 ms steps (Milestone 6:
             was 20 ms) around each coarse candidate.
          3. Fine frequency       — ±4 Hz with ±0.5 Hz sub-steps
             (Milestone 6: was ±3 Hz, 1 Hz steps).
          4. LLR extraction + soft Costas scaling + LDPC + AP passes.
          5. Deep search (Milestone 6) — iterative interference cancellation.
        """
        seen_msg: set[str] = set()

        # Initial decode pass.
        first_pass = self._decode_pass(frame, utc, seen_msg)
        for msg, f0_hz, t0_s, snr_db, payload, _syms in first_pass:
            if self._debug:
                print(
                    f"[FT8] {utc} DECODE f0={f0_hz:.1f}Hz t0={t0_s:+.02f}s "
                    f"snr={snr_db:+.1f}dB '{msg}'",
                    flush=True,
                )
            if self._on_decode is not None:
                self._on_decode(utc, f0_hz, snr_db, msg)

        # Milestone 6: Deep search — iterative interference cancellation.
        # After decoding successfully, subtract each signal's reconstructed
        # waveform from the audio frame and decode the residual to find weaker
        # signals that were masked by co-channel interference.
        #
        # Cumulative approach: all_decoded_signals grows across passes so that
        # each successive residual has *every* previously decoded signal removed,
        # not just those found in the immediately preceding pass.  A single
        # pre-allocated residual buffer is reused to avoid O(N × frame_len)
        # copy overhead.
        all_decoded_signals = list(first_pass)  # cumulative across all passes
        new_pass_signals = list(first_pass)     # only signals new to this pass

        for ds_iter in range(self.deep_search_passes):
            if not new_pass_signals:
                break
            # Build residual in-place: start from the original frame and
            # subtract ALL signals decoded across all passes so far.
            residual = np.asarray(frame, dtype=np.float64).copy()
            for _, f0_hz, t0_s, _, _pay, syms in all_decoded_signals:
                try:
                    _subtract_decoded_signal(
                        residual,
                        t0_s=t0_s,
                        f0_hz=f0_hz,
                        symbols=syms,
                        fs=self.fs_proc,
                        out=residual,   # subtract in-place; no extra copy
                    )
                except Exception as exc:
                    if self._debug:
                        print(
                            f"[FT8] deep-search subtract error: {exc!r}",
                            flush=True,
                        )

            if self._debug:
                print(
                    f"[FT8] {utc} deep-search pass {ds_iter + 1}/"
                    f"{self.deep_search_passes} — decoding residual "
                    f"({len(all_decoded_signals)} signal(s) removed)",
                    flush=True,
                )
            new_decodes = self._decode_pass(residual.astype(np.float32), utc, seen_msg)
            if not new_decodes:
                if self._debug:
                    print(
                        f"[FT8] {utc} deep-search pass {ds_iter + 1}: no new decodes",
                        flush=True,
                    )
                break
            for msg, f0_hz, t0_s, snr_db, payload, syms in new_decodes:
                if self._debug:
                    print(
                        f"[FT8] {utc} DEEP DECODE f0={f0_hz:.1f}Hz t0={t0_s:+.02f}s "
                        f"snr={snr_db:+.1f}dB '{msg}'",
                        flush=True,
                    )
                if self._on_decode is not None:
                    self._on_decode(utc, f0_hz, snr_db, msg)
            new_pass_signals = new_decodes
            all_decoded_signals.extend(new_decodes)  # accumulate for next pass

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
                try:
                    self._decode_frame(frame, utc)
                except Exception as exc:
                    print(
                        f"[FT8] slot {utc}Z decode error: {exc!r}",
                        flush=True,
                    )
