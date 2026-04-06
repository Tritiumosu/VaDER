"""
test_pipeline_e2e.py  — End-to-end pipeline test matching ft8_lib reference.
Tests the full encode -> Gray -> normalise -> BP path without interleaving.

Reference: ft8_lib bp_decode() takes channel LLRs directly (no deinterleave).
The _LDPC_CHECKS matrix uses column indices in transmission order (ft8_lib
kFTX_LDPC_Nm, 0-based), so no permutation is needed between Gray decode and
LDPC decode.
Run:  python test_pipeline_e2e.py
"""
import sys, math, numpy as np
sys.path.insert(0, '.')
from ft8_decode import (
    _LDPC_CHECKS, _FT8_GRAY_DECODE, _ft8_crc14,
    ft8_gray_decode, ft8_ldpc_decode, ft8_unpack_message,
    _ldpc_check,
)

PASS = "PASS"; FAIL = "FAIL"

# ── helpers ────────────────────────────────────────────────────────────────

def build_H():
    H = np.zeros((83, 174), dtype=np.uint8)
    for r, row in enumerate(_LDPC_CHECKS):
        for c in row: H[r, c] = 1
    return H

H = build_H()

def encode_ft8(msg77):
    crc = _ft8_crc14(msg77)
    cb  = np.array([(crc >> (13-i)) & 1 for i in range(14)], dtype=np.uint8)
    sb  = np.concatenate([msg77, cb])
    Hs, Hp = H[:, :91], H[:, 91:]
    rhs = (Hs.astype(np.int32) @ sb.astype(np.int32)) % 2
    aug = np.hstack([Hp.astype(np.uint8), rhs.reshape(83,1).astype(np.uint8)])
    for col in range(83):
        piv = next((r for r in range(col, 83) if aug[r, col]), None)
        if piv is None: raise RuntimeError(f'singular col {col}')
        aug[[col, piv]] = aug[[piv, col]]
        for r in range(83):
            if r != col and aug[r, col]: aug[r] = (aug[r] + aug[col]) % 2
    cw = np.concatenate([sb, aug[:, 83]])
    assert np.all((H.astype(np.int32) @ cw.astype(np.int32)) % 2 == 0)
    return cw

# Forward Gray map  gray_value -> tone
_gray_fwd = [0]*8
for _tone, _gv in enumerate(_FT8_GRAY_DECODE): _gray_fwd[_gv] = _tone

def encode_to_tones(cw):
    """codeword -> 58 Gray-coded tones (ft8_lib convention: no interleaver).
    Codeword bits 3*s, 3*s+1, 3*s+2 are read sequentially for tone s."""
    tones = []
    for s in range(58):
        gv = (int(cw[3*s])<<2) | (int(cw[3*s+1])<<1) | int(cw[3*s+2])
        tones.append(_gray_fwd[gv])
    return np.array(tones, dtype=np.int32)

def tones_to_E(tones, sig=100.0, noise=1.0):
    E = np.full((58, 8), noise, dtype=np.float64)
    for s, t in enumerate(tones): E[s, t] = sig
    return E

def full_pipeline(cw, sig=100.0, noise=1.0):
    """Encode codeword -> simulate channel -> decode. Returns (ok, payload, iters).
    Matches ft8_lib: no interleaver in encoder, no deinterleave before bp_decode."""
    tones = encode_to_tones(cw)
    E = tones_to_E(tones, sig=sig, noise=noise)
    syms = np.argmax(E, axis=1)
    _, ch_llrs = ft8_gray_decode(syms, E)
    # normalise all 174 channel LLRs to variance=24 (ft8_lib ftx_normalize_logl)
    var = float(np.var(ch_llrs))
    if var > 1e-10:
        ch_llrs = ch_llrs * math.sqrt(24.0 / var)
    # Pass channel LLRs directly to LDPC decoder — no deinterleaving needed
    # because _LDPC_CHECKS indices are already in transmission order (ft8_lib
    # kFTX_LDPC_Nm, 0-based).
    return ft8_ldpc_decode(ch_llrs)[:3]

results = []

# ── Test 1: Simple known codeword (non-zero message) ──────────────────────
# Note: all-zeros message is rejected by ft8_lib bp_decode (plain_sum==0 check),
# matching the reference behavior. Use a non-trivial message instead.
msg0 = np.array([1,0,1,0,0,1,1,0]*9 + [1,0,1,0,1], dtype=np.uint8)[:77]
cw0  = encode_ft8(msg0)
ok, pl, it = full_pipeline(cw0)
p = PASS if ok and np.all(pl[:77] == msg0) else FAIL
results.append(p)
print(f"[{p}] Known non-zero codeword  ok={ok} iters={it}")

# ── Test 2: Direct codeword LLRs (ft8_lib convention, no interleaver) ─────────
# Pass LLRs directly corresponding to codeword bit values. Positions 128..173
# are set to 0 (erased/uninformative); the BP decoder infers them from the checks.
rng = np.random.default_rng(42)
msg1 = rng.integers(0, 2, size=77, dtype=np.uint8)
cw1  = encode_ft8(msg1)
cw_llrs = np.where(cw1==0, -10.0, 10.0).astype(np.float64)
cw_llrs[128:] = 0.0
ok2, pl2, it2, _ = ft8_ldpc_decode(cw_llrs)
p = PASS if ok2 and np.all(pl2[:77] == msg1) else FAIL
results.append(p)
print(f"[{p}] Direct codeword LLRs (erased 128-173)  ok={ok2} iters={it2}")

# ── Test 3: 20 random messages, perfect channel ─────────────────────────────
pass_count = 0
fail_seeds = []
for seed in range(20):
    rng2 = np.random.default_rng(seed + 200)
    msg  = rng2.integers(0, 2, size=77, dtype=np.uint8)
    cw   = encode_ft8(msg)
    ok3, pl3, it3 = full_pipeline(cw)
    if ok3 and np.all(pl3[:77] == msg):
        pass_count += 1
    else:
        fail_seeds.append(seed)
p = PASS if pass_count == 20 else FAIL
results.append(p)
print(f"[{p}] 20 random messages, perfect channel: {pass_count}/20"
      + (f"  failed seeds={fail_seeds}" if fail_seeds else ""))

# ── Test 4: 10 random messages, noisy channel (SNR ~10 dB) ─────────────────
pass_count_n = 0
for seed in range(10):
    rng3 = np.random.default_rng(seed + 300)
    msg  = rng3.integers(0, 2, size=77, dtype=np.uint8)
    cw   = encode_ft8(msg)
    ok4, pl4, it4 = full_pipeline(cw, sig=10.0, noise=1.0)
    if ok4 and np.all(pl4[:77] == msg):
        pass_count_n += 1
p = PASS if pass_count_n >= 8 else FAIL   # allow 2 failures at low SNR
results.append(p)
print(f"[{p}] 10 random messages, SNR ~10 dB: {pass_count_n}/10")

# ── Test 5: CRC round-trip ──────────────────────────────────────────────────
msg5 = np.array([1,0,1,1,0,0,1,0]*9 + [1,0,1,0,1], dtype=np.uint8)[:77]
crc5 = _ft8_crc14(msg5)
cw5  = encode_ft8(msg5)
crc_from_cw = sum(int(b)<<(13-i) for i,b in enumerate(cw5[77:91]))
p = PASS if crc5 == crc_from_cw else FAIL
results.append(p)
print(f"[{p}] CRC round-trip: calc=0x{crc5:04X} from_cw=0x{crc_from_cw:04X}")

# ── Summary ────────────────────────────────────────────────────────────────
n_pass = results.count(PASS)
print(f"\n{n_pass}/{len(results)} tests passed")


def test_all_pass():
    assert n_pass == len(results), f"{n_pass}/{len(results)} tests passed"


if __name__ == "__main__":
    sys.exit(0 if n_pass == len(results) else 1)

