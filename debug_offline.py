
import numpy as np
import math
from ft8_decode import ft8_gray_decode, ft8_ldpc_decode, _FT8_DEINTERLEAVE, ft8_unpack_message

def run_debug():
    # From log: 23:15:30 (Score ~18dB)
    # payload_syms (interleaved)
    sym_str = "0202100020121125323573404621320556673564252577330504233031"

    if len(sym_str) != 58:
        print(f"Error: string length {len(sym_str)}")
        return

    syms = np.array([int(c) for c in sym_str], dtype=np.int32)
    print(f"Symbols: {syms}")

    # Create dummy energies: Strong confidence in the hard decision
    # E=100 for the symbol, 1 for others
    E_interleaved = np.ones((58, 8), dtype=np.float64)
    for s, t in enumerate(syms):
        E_interleaved[s, t] = 100.0

    # 1. Gray Decode
    hard_bits_ch, channel_llrs = ft8_gray_decode(syms, E_interleaved)

    # 2. De-interleave
    codeword_llrs = np.zeros(174, dtype=np.float64)
    deint_map = np.array(_FT8_DEINTERLEAVE, dtype=np.int32)
    mask = (deint_map >= 0)
    codeword_llrs[mask] = channel_llrs[deint_map[mask]]

    # Debug: Check recovered bits
    cw_hard = (codeword_llrs < 0).astype(int) # Wait. Positive LLR -> bit 1?
    # ft8_gray_decode says: positive -> bit 1.
    # So hard bit is 1 if LLR > 0.
    cw_hard = (codeword_llrs > 0).astype(int)

    print(f"Codeword bits (first 20): {cw_hard[:20]}")

    # 3. LDPC Decode
    # Try Normal
    ok, payload, iters, _ = ft8_ldpc_decode(codeword_llrs, max_iterations=200)
    print(f"Normal Decode: OK={ok} Iters={iters}")
    if ok:
        msg = ft8_unpack_message(payload[:77])
        print(f"Message: {msg}")

    # Try LLR Sign Flip
    print("\n--- Trying Sign Flip ---")
    ok, payload, iters, _ = ft8_ldpc_decode(-codeword_llrs, max_iterations=200)
    print(f"Inverted Decode: OK={ok} Iters={iters}")
    if ok:
        msg = ft8_unpack_message(payload[:77])
        print(f"Message: {msg}")

    # Try Bit Reversal in Gray (LSB first vs MSB first)
    # Current ft8_gray_decode does:
    # bit0 (MSB of gray) = ...
    # bit2 (LSB of gray) = ...
    # This maps Symbol index 0 -> bits 0,1,2.
    # What if it should be bits 2,1,0?

    # Let's manual-swap the channel LLRs
    # reshape to (58, 3)
    llrs_rs = channel_llrs.reshape(58, 3)
    # Swap columns 0 and 2
    llrs_swapped = llrs_rs[:, ::-1].reshape(174)

    codeword_llrs_sw = np.zeros(174, dtype=np.float64)
    codeword_llrs_sw[mask] = llrs_swapped[deint_map[mask]]

    print("\n--- Trying Bit Order Swap (LSB-first) ---")
    ok, payload, iters, _ = ft8_ldpc_decode(codeword_llrs_sw, max_iterations=200)
    print(f"Swap Decode: OK={ok} Iters={iters}")
    if ok:
        msg = ft8_unpack_message(payload[:77])
        print(f"Message: {msg}")

     # Try Bit Order Swap AND Invert
    print("\n--- Trying Bit Order Swap + Invert ---")
    ok, payload, iters, _ = ft8_ldpc_decode(-codeword_llrs_sw, max_iterations=200)
    print(f"Swap+Inv Decode: OK={ok} Iters={iters}")
    if ok:
        msg = ft8_unpack_message(payload[:77])
        print(f"Message: {msg}")

if __name__ == "__main__":
    run_debug()

