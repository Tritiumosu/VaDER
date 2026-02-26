# FT8 LDPC Matrix Fix Summary

## Issue
The FT8 LDPC parity-check matrix `_LDPC_CHECKS` in `ft8_decode.py` had an incorrect number of rows:
- **Before**: 84 rows × 174 columns (each row with 91 entries)
- **Expected**: 83 rows × 174 columns (for an (174, 91) LDPC code with 83 parity checks)

## Root Cause
The matrix had an extra spurious first row that included all columns 0-90:
```python
(  0, 1, 2, 3, 4, 5, ..., 88, 89, 90),  # ← This row was incorrect and removed
```

## Fix Applied
Removed the incorrect first row from the `_LDPC_CHECKS` tuple in `ft8_decode.py` at line ~1209.

## Verification
After the fix, the matrix now has the correct structure:

### Matrix Dimensions ✓
- **Shape**: (83, 174) — 83 parity check equations, 174 codeword bits
- **Rank**: 83 (full rank)
- **Row weight**: 91 (each check equation involves 91 bits)

### Systematic Code Structure ✓
- **Free columns** (systematic bits): Columns 0-90 (91 bits = 77 message + 14 CRC)
- **Pivot columns** (parity bits): Columns 91-173 (83 bits)
- **Parity submatrix rank**: H[:,91:174] has full rank 83 (invertible)

### Test Results ✓
All LDPC decoder tests pass:
- ✓ Test 1: All-zeros codeword
- ✓ Test 2: Random valid codeword
- ✓ Test 3: CRC-valid codeword (full LDPC+CRC round-trip)
- ✓ Test 4: Single bit error correction
- ✓ Test 5: Large LLR handling (no numerical overflow)
- ✓ Test 6: CRC-14 self-consistency

## Reference Specification
The FT8 LDPC code is a **(174, 91)** code:
- **n = 174**: codeword length (channel bits)
- **k = 91**: message length (77 message bits + 14 CRC bits)
- **m = 83**: number of parity checks (n - k = 174 - 91 = 83)

This implementation uses a systematic form where the first 91 bits are the message+CRC and the last 83 bits are parity, which simplifies encoding and decoding.

## Files Modified
1. `ft8_decode.py`:
   - Fixed `_LDPC_CHECKS` matrix (removed spurious row 0)
   - Updated comments to reflect systematic code structure

## Date
Fixed: February 25, 2026

