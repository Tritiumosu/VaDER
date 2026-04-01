"""
ft8_encode.py — FT8 transmit encoder for VaDER (Milestone 4).

Implements the complete FT8 encode pipeline, the transmit counterpart to
ft8_decode.py:

  Message string
    → ft8_pack_message()       77 payload bits
    → ft8_append_crc()         91 bits  (msg + 14-bit CRC)
    → ft8_ldpc_encode()       174 bits  (LDPC (174,91) codeword)
    → ft8_codeword_to_tones()  79 tone symbols (incl. Costas sync)
    → ft8_symbols_to_audio()   float32 PCM at target sample rate

All constants and algorithms match the WSJT-X / ft8_lib reference so that
generated transmissions can be decoded by any conforming FT8 receiver.

Key FT8 transmit parameters (WSJT-X / ft8_lib specification):
  Symbol duration   : 160 ms  (1 / 6.25 Hz)
  Tone spacing      : 6.25 Hz
  Symbols per frame : 79  (58 data + 21 Costas sync)
  Costas positions  : symbols 0–6, 36–42, 72–78
  Costas pattern    : {3, 1, 4, 0, 6, 5, 2}
  FEC               : LDPC (174, 91), col-weight 3
  CRC               : 14-bit, polynomial 0x2757
  Transmission time : ~12.64 s (79 × 0.160 s)
  Gray code map     : kFT8_Gray_map = {0,1,3,2,5,6,4,7}  (ft8_lib convention)
"""
from __future__ import annotations

import math
import re
from typing import Optional

import numpy as np

# Import shared decode-side constants so this module is a faithful counterpart.
from ft8_decode import (
    FT8_TONE_SPACING_HZ,
    FT8_SYMBOL_DURATION_S,
    FT8_NSYMS,
    FT8_FS,
    FT8_SYM_SAMPLES,
    FT8_COSTAS_POSITIONS,
    FT8_COSTAS_TONES,
    _FT8_GRAY_MAP,
    _LDPC_CHECKS,
    _ft8_crc14,
)


# ═══════════════════════════════════════════════════════════════════════════════
# § 1  Character tables for callsign packing
# ═══════════════════════════════════════════════════════════════════════════════

# These match ft8_lib constants.c character alphabets exactly.
_C37 = " 0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"  # 37 chars: space + 0-9 + A-Z
_C36 = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"   # 36 chars: 0-9 + A-Z
_C10 = "0123456789"                              # 10 chars: digits only
_C27 = " ABCDEFGHIJKLMNOPQRSTUVWXYZ"            # 27 chars: space + A-Z

# ft8_lib special-token namespace boundaries
_NTOKENS: int = 2_063_592  # number of special tokens (DE, QRZ, CQ variants)
_MAX22:   int = 4_194_304  # 2^22 — hash namespace width

# Maidenhead grid letter set — only A–R are valid for the two-letter
# field designators (18 fields × 18 fields = 324 combinations per hemisphere).
_GRID_LETTERS = "ABCDEFGHIJKLMNOPQR"

# First igrid4 value that is NOT a valid 4-char grid locator
_MAXGRID4: int = 32_400  # = 18 × 18 × 10 × 10


# ═══════════════════════════════════════════════════════════════════════════════
# § 2  LDPC encoding matrix (precomputed at import time)
# ═══════════════════════════════════════════════════════════════════════════════
# The FT8 (174, 91) LDPC parity-check matrix H is NOT lower-triangular in the
# parity columns (columns 91–173), so the simple "XOR each check row" loop
# used in some LDPC encoders gives incorrect results.  We use GF(2) Gaussian
# elimination to compute the exact encoding matrix G_enc (83×91) such that:
#
#     parity = G_enc @ systematic  mod 2          (83 bits)
#     codeword = [systematic | parity]            (174 bits)
#
# Verification: H @ codeword ≡ 0  (mod 2)  for any systematic input.
#
# This matrix is a constant (derived entirely from _LDPC_CHECKS) and is
# computed once at module load time so individual encode calls are fast.

def _build_ldpc_encode_matrix() -> np.ndarray:
    """
    Return the (83 × 91) GF(2) encoding matrix G_enc.

    Algorithm
    ---------
    1. Build H = [H_s | H_p] where H_s is H[:, 0:91] and H_p is H[:, 91:174].
    2. Verify H_p is full-rank (83) over GF(2).
    3. Compute H_p_inv = H_p^{-1} via augmented row reduction.
    4. Return G_enc = (H_p_inv @ H_s) mod 2.

    This satisfies H_p @ G_enc = H_s  (mod 2), which ensures that for any
    systematic bits s: H @ [s | G_enc @ s] = H_s @ s + H_p @ G_enc @ s = 0.
    """
    K, M, N = 91, 83, 174

    # Build binary H matrix
    H = np.zeros((M, N), dtype=np.int32)
    for r, row in enumerate(_LDPC_CHECKS):
        for c in row:
            H[r, c] = 1

    H_s = H[:, :K].copy()   # 83 × 91 systematic partition
    H_p = H[:, K:].copy()   # 83 × 83 parity partition

    # Augmented GF(2) row reduction: [H_p | I] → [I | H_p^{-1}]
    aug = np.hstack([H_p, np.eye(M, dtype=np.int32)])
    for col in range(M):
        # Find pivot row
        pivot = next((r for r in range(col, M) if aug[r, col]), None)
        if pivot is None:
            raise RuntimeError(
                f"LDPC parity matrix H_p is singular at column {col}; "
                "cannot compute encoding matrix."
            )
        aug[[col, pivot]] = aug[[pivot, col]]
        # Eliminate all other rows
        for r in range(M):
            if r != col and aug[r, col]:
                aug[r] = (aug[r] + aug[col]) % 2

    H_p_inv = aug[:, M:] % 2                    # (83 × 83) GF(2) inverse
    G_enc = (H_p_inv @ H_s) % 2                 # (83 × 91) encoding matrix
    return G_enc.astype(np.uint8)


# Precomputed encoding matrix; all subsequent encodes use this constant.
_FT8_ENCODE_MATRIX: np.ndarray = _build_ldpc_encode_matrix()


# ═══════════════════════════════════════════════════════════════════════════════
# § 3  Callsign packing
# ═══════════════════════════════════════════════════════════════════════════════

def validate_callsign(call: str) -> bool:
    """
    Return True if *call* is representable as a standard FT8 28-bit callsign.

    Accepted forms
    --------------
    - Special tokens: 'DE', 'QRZ', 'CQ'
    - CQ with numeric suffix: 'CQ 123'
    - CQ with alpha directional: 'CQ DX', 'CQ EU'
    - Standard ham callsigns (with optional /R or /P suffix):
        0–2 prefix characters followed by one digit followed by 1–3 letters.
        Examples: W4ABC, VK2TIM, K9XYZ, G3ABC, 3D2AG, AA0ABC
    """
    call = call.upper().strip()
    if call in ("DE", "QRZ", "CQ"):
        return True
    if call.startswith("CQ "):
        suffix = call[3:].strip()
        return bool(
            (suffix.isdigit() and len(suffix) <= 3)
            or re.fullmatch(r"[A-Z]{1,4}", suffix)
        )
    # Strip /R or /P portable suffix before structural validation
    base = re.sub(r"/(R|P)$", "", call)
    # Pattern: 0–2 alphanumeric prefix + digit + 1–3 letters
    return bool(re.fullmatch(r"[A-Z0-9]{0,2}[0-9][A-Z]{1,3}", base))


def ft8_pack_callsign(call: str) -> int:
    """
    Pack a callsign string into a 28-bit integer (ft8_lib pack28 equivalent).

    Parameters
    ----------
    call : str
        Callsign to pack.  Examples: 'W4ABC', 'VK2TIM', 'CQ', 'DE', 'QRZ',
        'CQ DX', 'CQ 143'.  Portable variants 'W4ABC/R' and 'W4ABC/P' are
        also accepted (the /R or /P is stripped; the ipa/ipb flag is managed
        by the caller in ft8_pack_message).

    Returns
    -------
    int  28-bit packed value for the n28a / n28b fields in a type-1 message.

    Raises
    ------
    ValueError if the callsign cannot be represented in 28 bits.
    """
    call = call.upper().strip()

    # --- Special tokens ---------------------------------------------------
    if call == "DE":
        return 0
    if call == "QRZ":
        return 1
    if call == "CQ":
        return 2
    if call.startswith("CQ "):
        suffix = call[3:].strip()
        # Numeric directional: CQ 000 – CQ 999
        if suffix.isdigit() and 1 <= len(suffix) <= 3:
            return 3 + int(suffix)
        # Alpha directional: CQ DX, CQ EU, CQ NA (1–4 uppercase letters)
        if re.fullmatch(r"[A-Z]{1,4}", suffix):
            # Encode as 4-char padded string using C27 (space = 0)
            padded = (suffix + "    ")[:4]
            m = 0
            for ch in padded:
                idx = _C27.index(ch) if ch in _C27 else 0
                m = m * 27 + idx
            return 1003 + m
        raise ValueError(f"Cannot pack CQ suffix {suffix!r}")

    # --- Standard callsign -----------------------------------------------
    # Strip portable/rover suffix (ipa/ipb bit is set by the message packer)
    base = re.sub(r"/(R|P)$", "", call)

    # Find the first digit — it separates prefix from suffix
    digit_pos = next((i for i, ch in enumerate(base) if ch.isdigit()), -1)
    if digit_pos < 0:
        raise ValueError(f"No digit found in callsign {call!r}")

    prefix = base[:digit_pos]       # 0, 1, or 2 characters
    digit  = base[digit_pos]        # the numeral
    suffix = base[digit_pos + 1:]   # 1–3 letters

    # Left-pad prefix to 2 chars, right-pad suffix to 3 chars with spaces
    prefix = (" " * (2 - len(prefix))) + prefix
    suffix = (suffix + "   ")[:3]

    c0_s, c1_s = prefix[0], prefix[1]
    c2_s = digit
    c3_s, c4_s, c5_s = suffix[0], suffix[1], suffix[2]

    # Validate each character is in its allowed alphabet
    bad = [
        (c0_s, "c0", _C37), (c1_s, "c1", _C36),
        (c2_s, "c2", _C10),
        (c3_s, "c3", _C27), (c4_s, "c4", _C27), (c5_s, "c5", _C27),
    ]
    for ch, pos, alpha in bad:
        if ch not in alpha:
            raise ValueError(
                f"Character {ch!r} is not valid for position {pos} "
                f"in callsign {call!r}"
            )

    c0 = _C37.index(c0_s)
    c1 = _C36.index(c1_s)
    c2 = _C10.index(c2_s)
    c3 = _C27.index(c3_s)
    c4 = _C27.index(c4_s)
    c5 = _C27.index(c5_s)

    # ft8_lib pack28 encoding formula (c5 = least significant, c0 = most):
    # n = c5 + 27*(c4 + 27*(c3 + 27*(c2 + 10*(c1 + 36*c0))))
    n = c5 + 27 * (c4 + 27 * (c3 + 27 * (c2 + 10 * (c1 + 36 * c0))))
    return _NTOKENS + _MAX22 + n


# ═══════════════════════════════════════════════════════════════════════════════
# § 4  Grid / report packing
# ═══════════════════════════════════════════════════════════════════════════════

def ft8_pack_grid(grid_or_report: str) -> tuple[int, int]:
    """
    Pack a grid locator, SNR report, or special token to (igrid4, ir).

    Accepted values
    ---------------
    - 4-char Maidenhead grid:       'EN52', 'IO91'
    - R-prefixed grid (ir=1):       'REN52'  →  igrid4 for EN52, ir=1
    - Plain SNR report:             '+05', '-10'
    - R-prefixed SNR report (ir=1): 'R+05', 'R-10'
    - Special acknowledgements:     'RRR', 'RR73', '73'
    - Blank (two-callsign only):    ''  →  igrid4=32400, ir=0

    Returns
    -------
    (igrid4 : int, ir : int)
      igrid4 fits in 15 bits (0–32767).
      ir is 0 or 1.
    """
    s = grid_or_report.upper().strip()

    # Blank / two-callsign-only message
    if not s:
        return 32400, 0

    # Special acknowledgement tokens
    if s == "RRR":
        return _MAXGRID4 + 2, 0
    if s == "RR73":
        return _MAXGRID4 + 3, 0
    if s == "73":
        return _MAXGRID4 + 4, 0

    # Detect R-prefix on a report or grid (sets ir=1 flag)
    ir = 0
    if s.startswith("R") and len(s) > 1 and s[1] in ("+", "-", "A", "B", "C",
                                                       "D", "E", "F", "G", "H",
                                                       "I", "J", "K", "L", "M",
                                                       "N", "O", "P", "Q", "R"):
        # Could be R+xx SNR, R-xx SNR, or Rxxxx grid
        if s[1] in ("+", "-"):
            # R-prefixed SNR report
            ir = 1
            s = s[1:]

    # Plain SNR report: +05, -10, etc.
    if len(s) >= 2 and s[0] in ("+", "-"):
        try:
            dd = int(s)
        except ValueError:
            raise ValueError(f"Cannot parse SNR report: {grid_or_report!r}")
        igrid4 = _MAXGRID4 + 35 + dd
        if not (0 <= igrid4 < 0x8000):
            raise ValueError(f"SNR report {dd:+d} dB is out of range")
        return igrid4, ir

    # R-prefixed 4-char grid: REN52 → grid=EN52, ir=1
    if s.startswith("R") and len(s) == 5:
        ir = 1
        s = s[1:]

    # 4-char Maidenhead grid locator: EN52, IO91, etc.
    if len(s) == 4:
        g1_s, g2_s, g3_s, g4_s = s[0], s[1], s[2], s[3]
        if not (
            g1_s in _GRID_LETTERS and g2_s in _GRID_LETTERS
            and g3_s.isdigit() and g4_s.isdigit()
        ):
            raise ValueError(f"Invalid grid locator: {grid_or_report!r}")
        g1 = _GRID_LETTERS.index(g1_s)
        g2 = _GRID_LETTERS.index(g2_s)
        g3 = int(g3_s)
        g4 = int(g4_s)
        # ft8_lib packgrid: igrid4 = (g1 * 18 + g2) * 100 + g3 * 10 + g4
        igrid4 = (g1 * 18 + g2) * 100 + g3 * 10 + g4
        return igrid4, ir

    raise ValueError(f"Cannot parse grid/report: {grid_or_report!r}")


# ═══════════════════════════════════════════════════════════════════════════════
# § 5  Message bit packing
# ═══════════════════════════════════════════════════════════════════════════════

def _int_to_bits(val: int, length: int) -> np.ndarray:
    """Convert integer *val* to an MSB-first bit array of *length* elements."""
    bits = np.zeros(length, dtype=np.uint8)
    for i in range(length - 1, -1, -1):
        bits[length - 1 - i] = (val >> i) & 1
    return bits


def ft8_pack_message(msg: str) -> np.ndarray:
    """
    Pack an FT8 message string into 77 payload bits (type i3=1 / i3=2).

    Supports standard type-1 FT8 messages of the form:
        call1 call2 extra

    where *extra* is one of: a 4-char grid, an SNR report (e.g. '-05'),
    an R-prefixed variant ('R-05', 'REN52'), or a special token
    ('RRR', 'RR73', '73').  *extra* may be absent for a two-callsign message.

    Bit layout (type i3=001, MSB first within each field):
        bits  0–27 : n28a (first callsign, 28 bits)
        bit   28   : ipa  (portable flag for call1: 1 = /R or /P)
        bits 29–56 : n28b (second callsign, 28 bits)
        bit   57   : ipb  (portable flag for call2)
        bit   58   : ir   (R-prefix flag from extra field)
        bits 59–73 : igrid4 (15 bits)
        bits 74–76 : i3 = 001 (message type)

    Parameters
    ----------
    msg : str
        Human-readable FT8 message.  Examples:
          'CQ W4ABC EN52'
          'W4ABC K9XYZ -05'
          'K9XYZ W4ABC R-07'
          'W4ABC K9XYZ RR73'
          'K9XYZ W4ABC 73'

    Returns
    -------
    np.ndarray  shape (77,) dtype uint8, MSB-first bit order.

    Raises
    ------
    ValueError if the message cannot be encoded as a standard type-1 message.
    """
    parts = msg.upper().strip().split()
    if len(parts) < 2:
        raise ValueError(
            f"FT8 message requires at least two callsigns, got: {msg!r}"
        )

    call1 = parts[0]
    call2 = parts[1]
    extra = parts[2] if len(parts) > 2 else ""

    # Detect and strip portable suffixes; the ipa/ipb bits signal them.
    ipa = 1 if call1.endswith(("/R", "/P")) else 0
    ipb = 1 if call2.endswith(("/R", "/P")) else 0

    n28a   = ft8_pack_callsign(call1)
    n28b   = ft8_pack_callsign(call2)
    igrid4, ir = ft8_pack_grid(extra)

    bits = np.zeros(77, dtype=np.uint8)
    bits[0:28]  = _int_to_bits(n28a, 28)
    bits[28]    = ipa
    bits[29:57] = _int_to_bits(n28b, 28)
    bits[57]    = ipb
    bits[58]    = ir
    bits[59:74] = _int_to_bits(igrid4, 15)
    # i3 = 001 (standard message type 1)
    bits[74]    = 0
    bits[75]    = 0
    bits[76]    = 1
    return bits


# ═══════════════════════════════════════════════════════════════════════════════
# § 6  CRC-14 append
# ═══════════════════════════════════════════════════════════════════════════════

def ft8_append_crc(msg_bits: np.ndarray) -> np.ndarray:
    """
    Append a 14-bit CRC to 77 message bits, yielding 91 systematic bits.

    The CRC is computed over the 77 message bits zero-padded to 82 bits,
    using CRC-14 polynomial 0x2757 (matching ft8_lib ftx_compute_crc).

    Parameters
    ----------
    msg_bits : array-like of shape (77,) uint8

    Returns
    -------
    np.ndarray  shape (91,) uint8  — msg_bits + 14 CRC bits (MSB first)
    """
    crc = _ft8_crc14(np.asarray(msg_bits, dtype=np.uint8)[:77])
    crc_bits = _int_to_bits(crc, 14)
    return np.concatenate([np.asarray(msg_bits, dtype=np.uint8)[:77], crc_bits])


# ═══════════════════════════════════════════════════════════════════════════════
# § 7  LDPC (174, 91) encoding
# ═══════════════════════════════════════════════════════════════════════════════

def ft8_ldpc_encode(systematic: np.ndarray) -> np.ndarray:
    """
    LDPC (174, 91) systematic encoding.

    Computes the 83 parity bits from the 91 systematic bits using the
    precomputed GF(2) encoding matrix _FT8_ENCODE_MATRIX:

        parity   = _FT8_ENCODE_MATRIX @ systematic  mod 2   (83 bits)
        codeword = [systematic | parity]                     (174 bits)

    The resulting codeword satisfies H @ codeword = 0 (mod 2) for all 83
    parity-check equations in _LDPC_CHECKS.

    Parameters
    ----------
    systematic : array-like of shape (91,) uint8
        77 message bits followed by 14 CRC bits.

    Returns
    -------
    np.ndarray  shape (174,) dtype uint8
    """
    s = np.asarray(systematic, dtype=np.uint8)
    if s.shape != (91,):
        raise ValueError(f"Expected (91,) systematic bits, got {s.shape}")

    parity   = (_FT8_ENCODE_MATRIX.astype(np.int32) @ s.astype(np.int32)) % 2
    codeword = np.concatenate([s, parity.astype(np.uint8)])
    return codeword


def ft8_ldpc_check(codeword: np.ndarray) -> int:
    """
    Count unsatisfied parity checks for a 174-bit codeword.

    Returns 0 when the codeword is valid (all 83 parity checks pass).
    Any non-zero count indicates a corrupted or incorrectly encoded codeword.
    """
    errors = 0
    c = np.asarray(codeword, dtype=np.uint8)
    for check in _LDPC_CHECKS:
        val = 0
        for j in check:
            val ^= int(c[j])
        if val:
            errors += 1
    return errors


# ═══════════════════════════════════════════════════════════════════════════════
# § 8  Symbol generation  (codeword → 79 tones)
# ═══════════════════════════════════════════════════════════════════════════════

def ft8_codeword_to_tones(codeword: np.ndarray) -> np.ndarray:
    """
    Convert a 174-bit LDPC codeword to 79 FT8 tone symbols.

    Steps (WSJT-X / ft8_lib convention):
      1. Group 174 bits into 58 groups of 3 (MSB first within each group).
      2. Gray-encode each 3-bit value to a tone index 0–7 using the
         ft8_lib kFT8_Gray_map = {0,1,3,2,5,6,4,7}.
      3. Build 79-symbol sequence: insert Costas sync arrays at positions
         0–6, 36–42, 72–78 using the Costas pattern {3,1,4,0,6,5,2};
         place the 58 payload tones at the remaining 58 positions.

    Parameters
    ----------
    codeword : array-like of shape (174,) uint8

    Returns
    -------
    np.ndarray  shape (79,) dtype uint8 — tone indices 0–7
    """
    c = np.asarray(codeword, dtype=np.uint8)
    if c.shape != (174,):
        raise ValueError(f"Expected (174,) codeword, got {c.shape}")

    # Convert 174 bits → 58 Gray-encoded payload tones
    payload_tones = np.zeros(58, dtype=np.uint8)
    for i in range(58):
        b0 = int(c[3 * i])
        b1 = int(c[3 * i + 1])
        b2 = int(c[3 * i + 2])
        gray_val = (b0 << 2) | (b1 << 1) | b2   # MSB-first 3-bit value
        payload_tones[i] = _FT8_GRAY_MAP[gray_val]

    # Assemble 79-symbol sequence (Costas + payload)
    costas_set = set(FT8_COSTAS_POSITIONS)
    symbols = np.zeros(FT8_NSYMS, dtype=np.uint8)
    payload_idx = 0
    for s in range(FT8_NSYMS):
        if s in costas_set:
            # Costas block position: blocks at 0, 36, 72
            if s < 7:
                block_pos = s
            elif s < 43:
                block_pos = s - 36
            else:
                block_pos = s - 72
            symbols[s] = FT8_COSTAS_TONES[block_pos]
        else:
            symbols[s] = payload_tones[payload_idx]
            payload_idx += 1

    return symbols


# ═══════════════════════════════════════════════════════════════════════════════
# § 9  Audio synthesis  (79 tones → float32 PCM)
# ═══════════════════════════════════════════════════════════════════════════════

def ft8_symbols_to_audio(
    symbols: np.ndarray,
    *,
    f0_hz: float = 1500.0,
    fs: int = FT8_FS,
    amplitude: float = 0.5,
    ramp_samples: int = 10,
) -> np.ndarray:
    """
    Synthesize FT8 baseband audio from 79 tone symbols.

    Generates a phase-continuous 8-FSK signal: each of the 79 symbols is a
    sinusoid at f0_hz + tone × 6.25 Hz lasting exactly 160 ms.  The phase
    is maintained continuously across symbol boundaries to minimize spectral
    splatter.  A short cosine ramp at the start and end of the transmission
    suppresses key-click transients.

    Parameters
    ----------
    symbols      : array-like of shape (79,) uint8 — tone indices 0–7
    f0_hz        : float — lowest tone frequency in Hz (default 1500 Hz,
                           a typical FT8 audio offset above the suppressed
                           carrier for USB operation)
    fs           : int   — output sample rate in Hz (default 12 000 Hz)
    amplitude    : float — peak amplitude in [0.0, 1.0] (default 0.5)
    ramp_samples : int   — length of cosine key-click suppression ramp at
                           each end of the transmission (default 10 samples;
                           set to 0 to disable)

    Returns
    -------
    np.ndarray  shape (N,) float32
        N = FT8_NSYMS × round(FT8_SYMBOL_DURATION_S × fs)
    """
    sym_arr = np.asarray(symbols, dtype=np.uint8)
    if sym_arr.shape != (FT8_NSYMS,):
        raise ValueError(f"Expected ({FT8_NSYMS},) symbols, got {sym_arr.shape}")

    sym_n   = int(round(FT8_SYMBOL_DURATION_S * fs))  # samples per symbol
    total_n = FT8_NSYMS * sym_n
    audio   = np.zeros(total_n, dtype=np.float64)

    # Phase-continuous synthesis: carry phase across symbol boundaries
    phase = 0.0
    for s in range(FT8_NSYMS):
        tone  = int(sym_arr[s])
        freq  = f0_hz + tone * FT8_TONE_SPACING_HZ
        omega = 2.0 * math.pi * freq / float(fs)   # rad/sample
        t     = np.arange(sym_n, dtype=np.float64)
        audio[s * sym_n : (s + 1) * sym_n] = np.sin(phase + omega * t)
        # Advance phase to the start of the next symbol (maintains continuity)
        phase = (phase + omega * sym_n) % (2.0 * math.pi)

    audio *= amplitude

    # Cosine ramp to suppress key-clicks at TX start and end
    if ramp_samples > 0 and total_n > 2 * ramp_samples:
        ramp = 0.5 * (1.0 - np.cos(np.linspace(0.0, math.pi, ramp_samples)))
        audio[:ramp_samples]  *= ramp
        audio[-ramp_samples:] *= ramp[::-1]

    return audio.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# § 10  End-to-end encode helpers
# ═══════════════════════════════════════════════════════════════════════════════

def ft8_encode_to_symbols(msg: str) -> np.ndarray:
    """
    Encode a message string to the 79-symbol tone sequence.

    This is the most common entry point for integration testing and for
    custom audio back-ends that do not need PCM at 12 kHz.

    Parameters
    ----------
    msg : str — FT8 message string  (e.g. 'CQ W4ABC EN52')

    Returns
    -------
    np.ndarray  shape (79,) dtype uint8 — tone indices 0–7
    """
    msg_bits = ft8_pack_message(msg)           # 77 bits
    a91      = ft8_append_crc(msg_bits)        # 91 bits (msg + CRC-14)
    codeword = ft8_ldpc_encode(a91)            # 174 bits (LDPC codeword)
    return ft8_codeword_to_tones(codeword)     # 79 tone symbols


def ft8_encode_message(
    msg: str,
    *,
    f0_hz: float = 1500.0,
    fs: int = FT8_FS,
    amplitude: float = 0.5,
) -> np.ndarray:
    """
    Full FT8 encode pipeline: message string → float32 PCM audio.

    Parameters
    ----------
    msg       : str   — FT8 message string  (e.g. 'CQ W4ABC EN52')
    f0_hz     : float — base tone frequency in Hz (default 1500 Hz)
    fs        : int   — output sample rate in Hz (default 12 000 Hz)
    amplitude : float — peak amplitude 0–1 (default 0.5)

    Returns
    -------
    np.ndarray  float32  of  FT8_NSYMS × round(FT8_SYMBOL_DURATION_S × fs)
    samples.
    """
    symbols = ft8_encode_to_symbols(msg)
    return ft8_symbols_to_audio(symbols, f0_hz=f0_hz, fs=fs, amplitude=amplitude)
