"""
Smoke test for the new symbol-extraction pipeline:
  extract_all_79 → ft8_extract_payload_symbols → ft8_deinterleave → ft8_gray_decode

Strategy
--------
1. Synthesise a clean 79-symbol 8-FSK signal at fs=12000 with known tone sequence.
2. Feed it through each stage.
3. Verify that the recovered hard_bits match what we put in.
"""
import math
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from ft8_decode import (
    FT8SymbolEnergyExtractor,
    ft8_extract_payload_symbols,
    ft8_deinterleave,
    ft8_gray_decode,
    FT8_COSTAS_POSITIONS,
    _FT8_GRAY_DECODE,
    _FT8_INTERLEAVE,
    _FT8_DEINTERLEAVE,
)

FS        = 12_000
SYM_S     = 0.160
SPACING   = 6.25
F0        = 1000.0   # base tone frequency
SYM_N     = int(round(SYM_S * FS))  # 1920 samples per symbol

# ── 1. Build known tone sequence (all 79 symbols) ─────────────────────────
rng = np.random.default_rng(42)

# Costas tones (known pattern, no shift, no inversion)
COSTAS = (3, 1, 4, 0, 6, 5, 2)
costas_set = set(FT8_COSTAS_POSITIONS)

# Random payload tones 0..7
payload_tones = rng.integers(0, 8, size=58).tolist()

all_tones = []
payload_idx = 0
for s in range(79):
    if s in costas_set:
        block = s // 36   # 0, 1, or 2
        pos_in_block = s % 36 if block == 0 else (s - 36) % 36 if block == 1 else (s - 72)
        all_tones.append(COSTAS[pos_in_block % 7])
    else:
        all_tones.append(int(payload_tones[payload_idx]))
        payload_idx += 1

assert len(all_tones) == 79

# ── 2. Synthesise the frame ────────────────────────────────────────────────
frame_n = SYM_N * 79
frame   = np.zeros(frame_n, dtype=np.float32)
t_sym   = np.arange(SYM_N, dtype=np.float64) / FS

for s, tone in enumerate(all_tones):
    freq = F0 + tone * SPACING
    frame[s * SYM_N : (s + 1) * SYM_N] = np.cos(2.0 * math.pi * freq * t_sym).astype(np.float32)

# ── 3. Extract all 79 symbol energies ─────────────────────────────────────
extractor = FT8SymbolEnergyExtractor(fs=FS)
E79 = extractor.extract_all_79(frame, t0_s=0.0, f0_hz=F0)

assert E79.shape == (79, 8), f"Expected (79,8) got {E79.shape}"

# Verify hard decisions match input tones for ALL 79 symbols
hard_all = np.argmax(E79, axis=1)
mismatches = np.sum(hard_all != np.array(all_tones))
print(f"[extract_all_79]  hard-decision mismatches vs known tones: {mismatches}/79")
assert mismatches == 0, f"Expected 0 mismatches, got {mismatches}"

# ── 4. Extract payload symbols (no shift, no inversion) ───────────────────
E_payload, hard_syms = ft8_extract_payload_symbols(E79, shift=0, inverted=False)

assert E_payload.shape == (58, 8)
assert hard_syms.shape == (58,)

expected_hard = np.array(payload_tones, dtype=np.int32)
payload_mismatches = np.sum(hard_syms != expected_hard)
print(f"[ft8_extract_payload_symbols]  mismatches vs known payload tones: {payload_mismatches}/58")
assert payload_mismatches == 0

# ── 5. ft8_deinterleave is a pass-through (bit interleaving is done
#       at the 174-bit level inside ft8_gray_decode).  Verify it's identity.
syms_deint, E_deint = ft8_deinterleave(hard_syms, E_payload)

roundtrip_mismatches = np.sum(syms_deint != expected_hard)
print(f"[ft8_deinterleave]  pass-through mismatches: {roundtrip_mismatches}/58")
assert roundtrip_mismatches == 0

# ── 6. Gray decode — verify hard bits and LLRs (after 174-bit de-interleave) ──
# Build expected bits in CHANNEL (interleaved) order from payload tones
gray_table = np.array(_FT8_GRAY_DECODE, dtype=np.int32)
bits_channel = np.empty(174, dtype=np.uint8)
for i, tone in enumerate(payload_tones):
    gv = int(gray_table[tone])
    for b in range(3):
        bits_channel[3 * i + b] = (gv >> (2 - b)) & 1

# Apply the same 174-bit de-interleave that ft8_gray_decode applies internally.
# _FT8_DEINTERLEAVE[src] = dst in channel order, or -1 for erased positions.
# Erased positions (46 parity bits never transmitted) get 0.
expected_bits = np.array([
    bits_channel[_FT8_DEINTERLEAVE[src]] if _FT8_DEINTERLEAVE[src] >= 0 else 0
    for src in range(174)
], dtype=np.uint8)

hard_bits, llrs = ft8_gray_decode(syms_deint, E_deint)

# Only compare non-erased positions (erased positions are filled by the erasure
# solver in ft8_ldpc_decode and may not match raw channel bits)
non_erased = [src for src in range(174) if _FT8_DEINTERLEAVE[src] >= 0]
bit_mismatches = int(np.sum(hard_bits[non_erased] != expected_bits[non_erased]))
print(f"[ft8_gray_decode]  hard_bit mismatches vs expected (non-erased): {bit_mismatches}/{len(non_erased)}")
assert bit_mismatches == 0, f"Expected 0, got {bit_mismatches}"

mean_llr_correct = float(np.mean(np.abs(llrs[non_erased])))
print(f"[ft8_gray_decode]  mean |LLR| (non-erased) = {mean_llr_correct:.2f}  (should be >> 1 for clean signal)")
assert mean_llr_correct > 5.0, f"LLRs unexpectedly weak: mean |LLR| = {mean_llr_correct:.2f}"

# Sign convention: positive LLR → bit=1
llr_sign_errors = int(np.sum((llrs[non_erased] > 0).astype(np.uint8) != hard_bits[non_erased]))
print(f"[ft8_gray_decode]  LLR sign errors vs hard_bits (non-erased): {llr_sign_errors}/{len(non_erased)}")
assert llr_sign_errors == 0

print("\nAll assertions passed — symbol pipeline is correct.")

