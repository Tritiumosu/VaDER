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

# ── 6. Gray decode — verify hard bits and LLRs ──────────────────────────────
# ft8_gray_decode takes 58 payload symbols and produces 174 CHANNEL-ordered bits
# (i.e. bit[3*s+k] = bit k of the Gray-decoded value for symbol s).
# De-interleaving is applied EXTERNALLY after ft8_gray_decode.
#
# Expected CHANNEL bits from known payload tones:
gray_table = np.array(_FT8_GRAY_DECODE, dtype=np.int32)
expected_channel_bits = np.empty(174, dtype=np.uint8)
for i, tone in enumerate(payload_tones):
    gv = int(gray_table[int(tone)])
    expected_channel_bits[3 * i + 0] = (gv >> 2) & 1   # MSB (bit2 of gray value)
    expected_channel_bits[3 * i + 1] = (gv >> 1) & 1   # bit1
    expected_channel_bits[3 * i + 2] = gv & 1           # LSB (bit0)

hard_bits, llrs = ft8_gray_decode(syms_deint, E_deint)

# Compare channel bits (all 174 positions — no erasure at channel level)
bit_mismatches = int(np.sum(hard_bits != expected_channel_bits))
print(f"[ft8_gray_decode]  channel hard_bit mismatches: {bit_mismatches}/174")
assert bit_mismatches == 0, f"Expected 0 channel hard-bit mismatches, got {bit_mismatches}"

mean_llr = float(np.mean(np.abs(llrs)))
print(f"[ft8_gray_decode]  mean |LLR| = {mean_llr:.2f}  (should be >> 1 for clean signal)")
assert mean_llr > 5.0, f"LLRs unexpectedly weak: mean |LLR| = {mean_llr:.2f}"

# Sign convention: positive LLR → bit=1
llr_sign_errors = int(np.sum((llrs > 0).astype(np.uint8) != hard_bits))
print(f"[ft8_gray_decode]  LLR sign errors vs hard_bits: {llr_sign_errors}/174")
assert llr_sign_errors == 0

# ── 7. Verify external de-interleave (as done in FT8ConsoleDecoder) ──────────
# After ft8_gray_decode: channel_llrs (174 channel-ordered)
# De-interleave: codeword_llr[src] = channel_llr[deinterleave[src]] or 0 if erased
deint_map = np.array(_FT8_DEINTERLEAVE, dtype=np.int32)
codeword_llrs = np.zeros(174, dtype=np.float64)
mask = deint_map >= 0
codeword_llrs[mask] = llrs[deint_map[mask]]

non_erased = [i for i in range(174) if deint_map[i] >= 0]
print(f"[de-interleave]  non-erased codeword positions: {len(non_erased)}/174"
      f"  (erased: {174 - len(non_erased)})")
assert len(non_erased) == 128, f"Expected 128 non-erased positions, got {len(non_erased)}"

# Expected codeword bits at non-erased positions
expected_cw_bits = np.array([
    expected_channel_bits[deint_map[src]] for src in non_erased
], dtype=np.uint8)
got_cw_bits = (codeword_llrs[non_erased] > 0).astype(np.uint8)
cw_bit_errors = int(np.sum(got_cw_bits != expected_cw_bits))
print(f"[de-interleave]  codeword bit errors (non-erased): {cw_bit_errors}/128")
assert cw_bit_errors == 0

print("\nAll assertions passed — symbol pipeline is correct.")

