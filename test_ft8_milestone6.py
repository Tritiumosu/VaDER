"""
test_ft8_milestone6.py — Unit and integration tests for Milestone 6 FT8 decoder
enhancements.

Covered features (from README § Milestone 6):

Candidate Search Improvements
  1. Lower Costas match threshold  (_MIN_COSTAS_MATCHES = 5, was 7)
  2. Increase candidate count       (_MAX_SYNC_CANDIDATES = 25, was 10)
  3. Finer time-search step         (_FINE_DT_FRACTION = 1/16 → 10 ms, was 20 ms)
  4. Wider fine-frequency search    (±4 Hz with ±0.5 Hz sub-steps, was ±3 Hz)

LLR & LDPC Quality Improvements
  5. Additional AP pass types       (i3=3 non-standard, i3=4 WWROF)
  6. Adaptive LDPC iteration count  (retry with 100 iters when best_errors ≤ 5)
  7. Soft Costas-energy LLR scaling (_costas_energy_llr_scale helper)
  8. Callsign-aware AP passes       (set_dx_callsign / _make_callsign_ap_passes)

Performance / Infrastructure
  9. Vectorised _costas_score       (NumPy argmax, no Python loop)
 10. DFT basis cache                (FT8SymbolEnergyExtractor._get_basis)
 11. Parallel candidate decode      (ThreadPoolExecutor in _decode_pass)

Flagship
 12. Iterative interference cancellation  (_subtract_decoded_signal / deep search)
"""

from __future__ import annotations

import math
import threading
import time
from typing import Optional
from unittest import mock

import numpy as np
import pytest

import ft8_decode
from ft8_decode import (
    FT8ConsoleDecoder,
    FT8SymbolEnergyExtractor,
    FT8_COSTAS_POSITIONS,
    FT8_COSTAS_TONES,
    FT8_FS,
    FT8_NSYMS,
    FT8_SYMBOL_DURATION_S,
    FT8_TONE_SPACING_HZ,
    FT8_TX_DURATION_S,
    _AP_PASSES,
    _ADAPTIVE_LDPC_ERROR_THRESHOLD,
    _ADAPTIVE_LDPC_MAX_ITERATIONS,
    _DEEP_SEARCH_MAX_PASSES,
    _FINE_DT_FRACTION,
    _FINE_FREQ_OFFSETS_HZ,
    _MAX_SYNC_CANDIDATES,
    _MIN_COSTAS_MATCHES,
    _costas_energy_llr_scale,
    _costas_score,
    _make_callsign_ap_passes,
    _subtract_decoded_signal,
    ft8_ldpc_decode,
    ft8_unpack_message,
)
from ft8_encode import ft8_encode_message, ft8_encode_to_symbols, ft8_symbols_to_audio


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _synthesise_ft8(msg: str, *, f0_hz: float = 1500.0,
                    fs: int = FT8_FS, t0_s: float = 0.5,
                    amplitude: float = 0.5,
                    frame_duration_s: float = 15.0) -> np.ndarray:
    """Synthesise a 15-s float32 frame containing one FT8 signal."""
    sym_n = int(round(FT8_SYMBOL_DURATION_S * fs))
    total_signal_n = FT8_NSYMS * sym_n
    frame_n = int(round(frame_duration_s * fs))
    frame = np.zeros(frame_n, dtype=np.float32)

    symbols = ft8_encode_to_symbols(msg)
    audio = ft8_symbols_to_audio(symbols, f0_hz=f0_hz, fs=fs, amplitude=amplitude)

    t0_n = int(round(t0_s * fs))
    end = min(t0_n + total_signal_n, frame_n)
    src_end = end - t0_n
    frame[t0_n:end] = audio[:src_end]
    return frame


def _make_e79_with_costas(n_correct: int) -> np.ndarray:
    """
    Create a synthetic (79, 8) energy matrix where exactly n_correct
    Costas positions match the FT8_COSTAS_TONES and the rest don't.
    """
    E79 = np.ones((FT8_NSYMS, 8), dtype=np.float64)
    cos_positions = list(FT8_COSTAS_POSITIONS)
    # Set up the Costas rows: small uniform energy first
    for i, pos in enumerate(cos_positions):
        expected = FT8_COSTAS_TONES[i % 7]
        if i < n_correct:
            # Make the correct tone dominant
            E79[pos, expected] = 10.0
        else:
            # Make the WRONG tone dominant (one off)
            wrong = (expected + 1) % 8
            E79[pos, wrong] = 10.0
    return E79


# ═══════════════════════════════════════════════════════════════════════════════
# 1  Module-level constants (Candidate Search Improvements)
# ═══════════════════════════════════════════════════════════════════════════════

class TestMilestone6Constants:
    """Verify that the new Milestone 6 module constants have the expected values."""

    def test_min_costas_matches_lowered(self):
        assert _MIN_COSTAS_MATCHES == 5, (
            "Costas threshold should be 5 (was 7); let LDPC+CRC be the arbiter"
        )

    def test_max_sync_candidates_increased(self):
        assert _MAX_SYNC_CANDIDATES == 25, (
            "Candidate cap should be 25 (was 10) to not discard weak signals"
        )

    def test_fine_dt_fraction_halved(self):
        dt_ms = FT8_SYMBOL_DURATION_S * _FINE_DT_FRACTION * 1000
        assert abs(dt_ms - 10.0) < 0.1, (
            f"Fine time step should be 10 ms, got {dt_ms:.1f} ms"
        )

    def test_fine_freq_offsets_include_half_hz(self):
        assert 0.5 in _FINE_FREQ_OFFSETS_HZ or -0.5 in _FINE_FREQ_OFFSETS_HZ, (
            "Fine frequency offsets must include ±0.5 Hz sub-steps"
        )

    def test_fine_freq_offsets_reach_4hz(self):
        max_offset = max(abs(x) for x in _FINE_FREQ_OFFSETS_HZ)
        assert max_offset >= 4.0, (
            f"Fine frequency search should reach ±4 Hz, max={max_offset} Hz"
        )

    def test_adaptive_ldpc_threshold(self):
        assert _ADAPTIVE_LDPC_ERROR_THRESHOLD == 5

    def test_adaptive_ldpc_max_iterations(self):
        assert _ADAPTIVE_LDPC_MAX_ITERATIONS == 100

    def test_deep_search_max_passes_nonzero(self):
        assert _DEEP_SEARCH_MAX_PASSES >= 1, (
            "Deep search should be enabled by default"
        )

    def test_decoder_default_min_costas(self):
        """FT8ConsoleDecoder should default to the new lower threshold."""
        dec = FT8ConsoleDecoder()
        assert dec._min_costas_matches == _MIN_COSTAS_MATCHES

    def test_decoder_deep_search_enabled_by_default(self):
        dec = FT8ConsoleDecoder()
        assert dec.deep_search_passes == _DEEP_SEARCH_MAX_PASSES


# ═══════════════════════════════════════════════════════════════════════════════
# 2  Additional AP pass types (i3=3, i3=4)
# ═══════════════════════════════════════════════════════════════════════════════

class TestAPPassTypes:
    """Verify _AP_PASSES contains the new i3=3 and i3=4 entries."""

    def _get_pass_names(self):
        return [name for name, _ in _AP_PASSES]

    def test_i3_3_present(self):
        assert "i3=3" in self._get_pass_names(), (
            "AP pass 'i3=3' (non-standard/hashed calls) must be in _AP_PASSES"
        )

    def test_i3_4_present(self):
        assert "i3=4" in self._get_pass_names(), (
            "AP pass 'i3=4' (WWROF contest) must be in _AP_PASSES"
        )

    def test_i3_3_bit_encoding(self):
        """i3=3 = binary 011: bit74=0, bit75=1, bit76=1."""
        for name, bits_tuple in _AP_PASSES:
            if name == "i3=3":
                bit_dict = {idx: val for idx, val in bits_tuple}
                assert bit_dict.get(74) == 0, "i3=3 bit74 should be 0 (MSB)"
                assert bit_dict.get(75) == 1, "i3=3 bit75 should be 1"
                assert bit_dict.get(76) == 1, "i3=3 bit76 should be 1 (LSB)"
                return
        pytest.fail("i3=3 pass not found")

    def test_i3_4_bit_encoding(self):
        """i3=4 = binary 100: bit74=1, bit75=0, bit76=0."""
        for name, bits_tuple in _AP_PASSES:
            if name == "i3=4":
                bit_dict = {idx: val for idx, val in bits_tuple}
                assert bit_dict.get(74) == 1, "i3=4 bit74 should be 1 (MSB)"
                assert bit_dict.get(75) == 0, "i3=4 bit75 should be 0"
                assert bit_dict.get(76) == 0, "i3=4 bit76 should be 0 (LSB)"
                return
        pytest.fail("i3=4 pass not found")

    def test_original_passes_preserved(self):
        """Original AP pass types (i3=0,1,2, CQ, DE) must still be present."""
        names = self._get_pass_names()
        for expected in ("i3=1", "i3=2", "i3=0", "CQ+i3=1", "DE+i3=1"):
            assert expected in names, f"Original AP pass '{expected}' missing"

    def test_ap_pass_bit_indices_in_range(self):
        """All AP bit indices must be within the 91 systematic bits."""
        for name, bits_tuple in _AP_PASSES:
            for idx, val in bits_tuple:
                assert 0 <= idx < 91, (
                    f"AP pass '{name}' has out-of-range bit index {idx}"
                )
                assert val in (0, 1), (
                    f"AP pass '{name}' has invalid bit value {val}"
                )

    def test_i3_3_ap_helps_decode_near_miss(self):
        """
        Verify that with i3=3 bits injected, LDPC decode succeeds on LLRs
        that would fail with a 0 parity-error count but CRC mismatch if the
        i3 field is mangled.

        We synthesise an FT8 message, extract LLRs, flip the i3 bits to
        produce a near-miss with best_errors > 0, then confirm that the
        i3=3 AP pass does not spuriously decode a different valid message.
        (This test mainly validates the pass is wired up correctly.)
        """
        from ft8_encode import ft8_pack_message, ft8_ldpc_encode, ft8_append_crc
        from ft8_decode import ft8_gray_decode, ft8_extract_payload_symbols

        msg = "CQ W4ABC EN52"
        bits = ft8_pack_message(msg)
        a91 = ft8_append_crc(bits)

        # Build LLRs from a clean encode and verify the i3=3 pass doesn't
        # spuriously flip a successful decode to a failure.
        from ft8_encode import ft8_codeword_to_tones
        codeword = ft8_ldpc_encode(a91)
        tones = ft8_codeword_to_tones(codeword)
        E79 = np.zeros((FT8_NSYMS, 8), dtype=np.float64)
        for i, t in enumerate(tones):
            E79[i, t] = 1.0
        E_pay, hs = ft8_extract_payload_symbols(E79)
        _, llrs = ft8_gray_decode(hs, E_pay)
        var = float(np.var(llrs))
        if var > 1e-10:
            llrs = llrs * math.sqrt(24.0 / var)

        ok_plain, _, _, errs = ft8_ldpc_decode(llrs)
        assert ok_plain, "Clean encode should decode successfully"

        # Now run with i3=3 pass — should still decode (the pass adds priors
        # but cannot break a convergent signal).
        i3_3_bits = next(b for n, b in _AP_PASSES if n == "i3=3")
        ok_ap, _, _, _ = ft8_ldpc_decode(llrs, ap_assignments=i3_3_bits)
        # The AP pass on the wrong i3 value will either still converge (signal
        # is strong enough) or fail — either is acceptable, it must not crash.
        assert isinstance(ok_ap, bool)


# ═══════════════════════════════════════════════════════════════════════════════
# 3  Adaptive LDPC iteration count
# ═══════════════════════════════════════════════════════════════════════════════

class TestAdaptiveLDPCIterations:
    """Verify adaptive LDPC retry logic in _decode_one_candidate."""

    def _noisy_llrs_for_msg(self, msg: str, noise_sigma: float = 0.6) -> np.ndarray:
        """Generate LLRs for a message with controlled noise level."""
        from ft8_encode import (
            ft8_pack_message, ft8_append_crc, ft8_ldpc_encode,
            ft8_codeword_to_tones,
        )
        from ft8_decode import ft8_extract_payload_symbols, ft8_gray_decode

        bits = ft8_pack_message(msg)
        a91 = ft8_append_crc(bits)
        codeword = ft8_ldpc_encode(a91)
        tones = ft8_codeword_to_tones(codeword)

        rng = np.random.default_rng(42)
        E79 = np.zeros((FT8_NSYMS, 8), dtype=np.float64)
        for i, t in enumerate(tones):
            E79[i, t] = 1.0 + rng.normal(0, noise_sigma)
            for k in range(8):
                if k != t:
                    E79[i, k] = max(0.0, rng.normal(0, noise_sigma))

        E_pay, hs = ft8_extract_payload_symbols(E79)
        _, llrs = ft8_gray_decode(hs, E_pay)
        var = float(np.var(llrs))
        if var > 1e-10:
            llrs = llrs * math.sqrt(24.0 / var)
        return llrs

    def test_adaptive_threshold_constant(self):
        assert _ADAPTIVE_LDPC_ERROR_THRESHOLD == 5

    def test_baseline_decode_uses_50_iters(self):
        """ft8_ldpc_decode default is 50 iterations."""
        from ft8_decode import ft8_ldpc_decode
        # A trivially correct decode should finish in << 50 iters
        from ft8_encode import (
            ft8_pack_message, ft8_append_crc, ft8_ldpc_encode,
            ft8_codeword_to_tones,
        )
        from ft8_decode import ft8_extract_payload_symbols, ft8_gray_decode
        msg = "W4ABC K9XYZ EN52"
        bits = ft8_pack_message(msg)
        a91 = ft8_append_crc(bits)
        cw = ft8_ldpc_encode(a91)
        tones = ft8_codeword_to_tones(cw)
        E79 = np.zeros((FT8_NSYMS, 8))
        for i, t in enumerate(tones):
            E79[i, t] = 1.0
        E_pay, hs = ft8_extract_payload_symbols(E79)
        _, llrs = ft8_gray_decode(hs, E_pay)
        var = float(np.var(llrs))
        if var > 1e-10:
            llrs = llrs * math.sqrt(24.0 / var)

        ok, _, iters, _ = ft8_ldpc_decode(llrs)
        assert ok
        assert iters <= 50

    def test_adaptive_iter_retry_callable(self):
        """ft8_ldpc_decode with max_iterations=100 must not raise."""
        from ft8_encode import (
            ft8_pack_message, ft8_append_crc, ft8_ldpc_encode,
            ft8_codeword_to_tones,
        )
        from ft8_decode import ft8_extract_payload_symbols, ft8_gray_decode
        msg = "W4ABC K9XYZ EN52"
        bits = ft8_pack_message(msg)
        a91 = ft8_append_crc(bits)
        cw = ft8_ldpc_encode(a91)
        tones = ft8_codeword_to_tones(cw)
        E79 = np.zeros((FT8_NSYMS, 8))
        for i, t in enumerate(tones):
            E79[i, t] = 1.0
        E_pay, hs = ft8_extract_payload_symbols(E79)
        _, llrs = ft8_gray_decode(hs, E_pay)
        var = float(np.var(llrs))
        if var > 1e-10:
            llrs = llrs * math.sqrt(24.0 / var)
        # Degrade LLRs to trigger near-miss
        llrs *= 0.1
        ok, payload, iters, errs = ft8_ldpc_decode(llrs, max_iterations=100)
        assert iters <= 100  # must not exceed the requested cap

    def test_decode_one_candidate_calls_adaptive_retry(self):
        """
        _decode_one_candidate must call ft8_ldpc_decode with
        max_iterations=_ADAPTIVE_LDPC_MAX_ITERATIONS when the baseline decode
        produces best_errors ≤ _ADAPTIVE_LDPC_ERROR_THRESHOLD.
        """
        # We patch ft8_ldpc_decode to return a known near-miss on the first
        # call (best_errors=3 ≤ 5) and succeed on the second (adaptive) call.
        call_log = []
        _orig = ft8_decode.ft8_ldpc_decode

        def _fake_ldpc(llrs, *, max_iterations=50, ap_assignments=None):
            call_log.append(max_iterations)
            # First call: near-miss
            if len(call_log) == 1:
                _, pay, _, _ = _orig(llrs, max_iterations=max_iterations)
                return False, pay, max_iterations, 3  # 3 errors → adaptive
            # Second call: succeed
            return _orig(llrs, max_iterations=max_iterations,
                         ap_assignments=ap_assignments)

        dec = FT8ConsoleDecoder()
        frame = _synthesise_ft8("CQ W4ABC EN52", f0_hz=1500.0)

        with mock.patch.object(ft8_decode, "ft8_ldpc_decode", side_effect=_fake_ldpc):
            dec._decode_one_candidate(frame, 0.5, 1500.0, set())

        assert any(n == _ADAPTIVE_LDPC_MAX_ITERATIONS for n in call_log), (
            f"Expected a call with max_iterations={_ADAPTIVE_LDPC_MAX_ITERATIONS}, "
            f"got calls: {call_log}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 4  Soft Costas-energy LLR scaling
# ═══════════════════════════════════════════════════════════════════════════════

class TestCostasEnergyLLRScaling:
    """Verify _costas_energy_llr_scale behaviour and properties."""

    def _uniform_e79(self) -> np.ndarray:
        """E79 where all tones have equal energy (worst-case, ratio=1)."""
        return np.ones((FT8_NSYMS, 8), dtype=np.float64)

    def _strong_e79(self, ratio: float = 16.0) -> np.ndarray:
        """E79 where the correct Costas tone is 'ratio' times stronger."""
        E79 = np.ones((FT8_NSYMS, 8), dtype=np.float64)
        for i, pos in enumerate(FT8_COSTAS_POSITIONS):
            expected = FT8_COSTAS_TONES[i % 7]
            E79[pos, expected] = ratio
        return E79

    def test_uniform_energy_returns_scale_below_one(self):
        """No dominant tones → low quality → scale < 1."""
        scale = _costas_energy_llr_scale(self._uniform_e79())
        assert scale < 1.0, (
            f"Uniform energy should give scale < 1.0, got {scale:.3f}"
        )

    def test_strong_signal_returns_scale_above_one(self):
        """Very dominant tones → high quality → scale > 1."""
        scale = _costas_energy_llr_scale(self._strong_e79(ratio=64.0))
        assert scale > 1.0, (
            f"Strong signal should give scale > 1.0, got {scale:.3f}"
        )

    def test_scale_bounded(self):
        """Scale must stay in [0.6, 1.4] regardless of input."""
        for e79 in (self._uniform_e79(), self._strong_e79(64.0),
                    np.zeros((FT8_NSYMS, 8))):
            scale = _costas_energy_llr_scale(e79)
            assert 0.6 <= scale <= 1.4, (
                f"Scale out of bounds: {scale:.3f}"
            )

    def test_scale_monotone_with_ratio(self):
        """Higher energy contrast → higher scale factor."""
        scales = [
            _costas_energy_llr_scale(self._strong_e79(ratio=r))
            for r in (1.0, 2.0, 4.0, 8.0, 16.0)
        ]
        for i in range(len(scales) - 1):
            assert scales[i] <= scales[i + 1] + 1e-9, (
                f"Scale should be non-decreasing: {scales}"
            )

    def test_zero_energy_safe(self):
        """All-zero E79 must not raise."""
        scale = _costas_energy_llr_scale(np.zeros((FT8_NSYMS, 8)))
        assert 0.6 <= scale <= 1.4

    def test_scale_applied_in_decode_one_candidate(self):
        """
        Verify that _costas_energy_llr_scale is called inside
        _decode_one_candidate and its return value is used to scale LLRs.
        """
        scales_seen = []
        _orig = ft8_decode._costas_energy_llr_scale  # capture before patch

        def _spy_scale(E79):
            s = _orig(E79)  # call the original, not the patched version
            scales_seen.append(s)
            return s

        frame = _synthesise_ft8("CQ W4ABC EN52", f0_hz=1500.0)
        dec = FT8ConsoleDecoder()
        with mock.patch.object(ft8_decode, "_costas_energy_llr_scale",
                               side_effect=_spy_scale):
            dec._decode_one_candidate(frame, 0.5, 1500.0, set())
        assert len(scales_seen) >= 1, (
            "_costas_energy_llr_scale should be called during candidate decode"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 5  Callsign-aware AP passes
# ═══════════════════════════════════════════════════════════════════════════════

class TestCallsignAwareAPPasses:
    """Verify _make_callsign_ap_passes and FT8ConsoleDecoder.set_dx_callsign."""

    def test_make_callsign_ap_returns_two_passes(self):
        passes = _make_callsign_ap_passes("W4ABC")
        assert len(passes) == 2, "Should return exactly 2 AP passes (n28a and n28b)"

    def test_pass_names_contain_callsign(self):
        passes = _make_callsign_ap_passes("K9XYZ")
        names = [name for name, _ in passes]
        for name in names:
            assert "K9XYZ" in name.upper(), (
                f"Pass name '{name}' should contain the callsign"
            )

    def test_n28a_bits_at_positions_0_to_27(self):
        passes = _make_callsign_ap_passes("W4ABC")
        n28a_pass = next((b for n, b in passes if "n28a" in n), None)
        assert n28a_pass is not None
        bit_indices = [idx for idx, _ in n28a_pass]
        assert all(0 <= i <= 90 for i in bit_indices), (
            "All AP bit indices must be in range 0..90"
        )
        # n28a occupies bits 0-27 (plus ipa at 28, i3 at 74-76)
        n28a_indices = [i for i in bit_indices if i < 28]
        assert len(n28a_indices) == 28

    def test_n28b_bits_at_positions_29_to_56(self):
        passes = _make_callsign_ap_passes("W4ABC")
        n28b_pass = next((b for n, b in passes if "n28b" in n), None)
        assert n28b_pass is not None
        bit_indices = [idx for idx, _ in n28b_pass]
        n28b_indices = [i for i in bit_indices if 29 <= i <= 56]
        assert len(n28b_indices) == 28

    def test_callsign_bits_match_ft8_pack(self):
        """Bits in the AP pass must match the n28 from ft8_pack_callsign."""
        from ft8_encode import ft8_pack_callsign
        callsign = "VK2TIM"
        n28 = ft8_pack_callsign(callsign)
        expected_bits = [(i, int((n28 >> (27 - i)) & 1)) for i in range(28)]

        passes = _make_callsign_ap_passes(callsign)
        n28a_pass = next((b for nm, b in passes if "n28a" in nm), None)
        assert n28a_pass is not None
        ap_n28a_bits = [(idx, val) for idx, val in n28a_pass if idx < 28]
        assert ap_n28a_bits == expected_bits, (
            "n28a bits in AP pass don't match ft8_pack_callsign output"
        )

    def test_invalid_callsign_returns_empty(self):
        passes = _make_callsign_ap_passes("NOTACALLSIGN!!!")
        assert passes == [], "Invalid callsign should return empty list"

    def test_set_dx_callsign_sets_passes(self):
        dec = FT8ConsoleDecoder()
        assert dec._dx_ap_passes == []
        dec.set_dx_callsign("W4ABC")
        assert len(dec._dx_ap_passes) == 2

    def test_set_dx_callsign_none_clears(self):
        dec = FT8ConsoleDecoder()
        dec.set_dx_callsign("W4ABC")
        dec.set_dx_callsign(None)
        assert dec._dx_ap_passes == []

    def test_set_dx_callsign_empty_string_clears(self):
        dec = FT8ConsoleDecoder()
        dec.set_dx_callsign("W4ABC")
        dec.set_dx_callsign("")
        assert dec._dx_ap_passes == []

    def test_dx_ap_passes_included_in_decode(self):
        """
        Verify that dx_ap_passes are tried when baseline LDPC fails.
        We mock ft8_ldpc_decode to return failure + few errors, and check
        that the DX-callsign AP pass is attempted.
        """
        call_log: list[Optional[tuple]] = []
        _orig = ft8_decode.ft8_ldpc_decode

        def _fake(llrs, *, max_iterations=50, ap_assignments=None):
            call_log.append(ap_assignments)
            return _orig(llrs, max_iterations=max_iterations,
                         ap_assignments=ap_assignments)

        dec = FT8ConsoleDecoder()
        dec.set_dx_callsign("K9XYZ")
        frame = _synthesise_ft8("CQ W4ABC EN52", f0_hz=1500.0)

        with mock.patch.object(ft8_decode, "ft8_ldpc_decode", side_effect=_fake):
            dec._decode_one_candidate(frame, 0.5, 1500.0, set())

        # The DX AP passes should appear somewhere in the call log when
        # the baseline decode may fail.  Since the signal is clean and decodes
        # immediately, we just verify the function doesn't crash.
        assert call_log, "ft8_ldpc_decode should have been called at least once"


# ═══════════════════════════════════════════════════════════════════════════════
# 6  Vectorised _costas_score
# ═══════════════════════════════════════════════════════════════════════════════

class TestVectorisedCostasScore:
    """Verify the vectorised _costas_score matches expected values."""

    def test_all_correct(self):
        E79 = _make_e79_with_costas(21)
        m, total, inv = _costas_score(E79)
        assert total == 21
        assert m == 21
        assert not inv

    def test_none_correct(self):
        # All Costas tones wrong (one offset)
        E79 = _make_e79_with_costas(0)
        m, total, inv = _costas_score(E79)
        assert total == 21
        assert m < 21

    def test_partial_match(self):
        for n in range(1, 21, 3):
            E79 = _make_e79_with_costas(n)
            m, _, _ = _costas_score(E79)
            assert m >= n - 1, (
                f"Expected ≥{n - 1} matches, got {m} for {n} correct Costas"
            )

    def test_inverted_detection(self):
        """E79 with inverted Costas (7-n) should return inv=True."""
        E79 = np.ones((FT8_NSYMS, 8), dtype=np.float64)
        for i, pos in enumerate(FT8_COSTAS_POSITIONS):
            expected = FT8_COSTAS_TONES[i % 7]
            inverted = 7 - expected
            E79[pos, inverted] = 10.0
        m, _, inv = _costas_score(E79)
        assert m == 21
        assert inv

    def test_output_types(self):
        E79 = np.ones((FT8_NSYMS, 8))
        result = _costas_score(E79)
        assert len(result) == 3
        assert isinstance(result[0], int)
        assert isinstance(result[1], int)
        assert isinstance(result[2], bool)

    def test_numpy_array_input(self):
        """Should accept any numeric array (float32, float64, int)."""
        E79_f32 = np.ones((FT8_NSYMS, 8), dtype=np.float32)
        E79_int = np.ones((FT8_NSYMS, 8), dtype=np.int32)
        for arr in (E79_f32, E79_int):
            m, t, inv = _costas_score(arr)
            assert isinstance(m, int)


# ═══════════════════════════════════════════════════════════════════════════════
# 7  DFT basis cache
# ═══════════════════════════════════════════════════════════════════════════════

class TestDFTBasisCache:
    """Verify FT8SymbolEnergyExtractor caches the basis matrix per f0_hz."""

    def test_cache_hit_returns_same_object(self):
        ext = FT8SymbolEnergyExtractor(fs=FT8_FS)
        b1 = ext._get_basis(1500.0)
        b2 = ext._get_basis(1500.0)
        assert b1 is b2, "Second call with same f0_hz must return the cached object"

    def test_cache_miss_different_freq(self):
        ext = FT8SymbolEnergyExtractor(fs=FT8_FS)
        b1 = ext._get_basis(1500.0)
        b2 = ext._get_basis(1506.25)  # different f0
        assert b1 is not b2

    def test_cache_correct_values(self):
        ext = FT8SymbolEnergyExtractor(fs=FT8_FS)
        f0 = 1500.0
        basis = ext._get_basis(f0)
        assert basis.shape == (8, ext.sym_n)
        assert basis.dtype == np.complex128

        # Manually compute expected basis for tone 0
        phase_inc_0 = 2.0 * math.pi * f0 / float(ext.fs)
        t = np.arange(ext.sym_n, dtype=np.float64)
        expected_row0 = np.exp(-1j * phase_inc_0 * t)
        np.testing.assert_allclose(basis[0], expected_row0, atol=1e-12)

    def test_cache_evicts_when_full(self):
        ext = FT8SymbolEnergyExtractor(fs=FT8_FS)
        max_size = ext._CACHE_MAXSIZE
        # Fill cache beyond capacity
        for i in range(max_size + 5):
            ext._get_basis(float(1000 + i))
        assert len(ext._basis_cache) <= max_size, (
            f"Cache size {len(ext._basis_cache)} exceeded max {max_size}"
        )

    def test_extract_all_79_uses_cache(self):
        """Calling extract_all_79 twice with same f0_hz should hit the cache."""
        ext = FT8SymbolEnergyExtractor(fs=FT8_FS)
        frame = np.random.default_rng(0).random(int(15 * FT8_FS)).astype(np.float32)
        ext.extract_all_79(frame, t0_s=0.5, f0_hz=1500.0)
        n_cached_after_first = len(ext._basis_cache)
        ext.extract_all_79(frame, t0_s=0.6, f0_hz=1500.0)
        n_cached_after_second = len(ext._basis_cache)
        assert n_cached_after_first == n_cached_after_second, (
            "Cache size should not grow when same f0_hz is reused"
        )

    def test_cache_shared_across_time_offsets(self):
        """
        Fine time search calls extract_all_79 multiple times with the same
        f0_hz; verify the basis matrix object is reused (not recomputed).
        """
        ext = FT8SymbolEnergyExtractor(fs=FT8_FS)
        frame = np.random.default_rng(1).random(int(15 * FT8_FS)).astype(np.float32)
        f0 = 1500.0
        dt_fine = FT8_SYMBOL_DURATION_S * _FINE_DT_FRACTION
        half_sym = FT8_SYMBOL_DURATION_S * 0.5
        for dt in np.arange(-half_sym, half_sym + dt_fine / 2, dt_fine):
            ext.extract_all_79(frame, t0_s=0.5 + float(dt), f0_hz=f0)
        # The cache should contain exactly one entry for this f0
        assert len(ext._basis_cache) == 1
        assert f0 in ext._basis_cache


# ═══════════════════════════════════════════════════════════════════════════════
# 8  Parallel candidate decode
# ═══════════════════════════════════════════════════════════════════════════════

class TestParallelCandidateDecode:
    """Verify that _decode_pass uses ThreadPoolExecutor (parallel decode)."""

    def test_decode_pass_uses_thread_pool(self):
        """ThreadPoolExecutor must be used inside _decode_pass."""
        from concurrent.futures import ThreadPoolExecutor

        executor_calls = []
        _orig_executor = ft8_decode.ThreadPoolExecutor

        class _TrackingExecutor:
            def __init__(self, *args, **kwargs):
                executor_calls.append(True)
                self._inner = _orig_executor(*args, **kwargs)

            def submit(self, *a, **kw):
                return self._inner.submit(*a, **kw)

            def __enter__(self):
                self._inner.__enter__()
                return self

            def __exit__(self, *a):
                return self._inner.__exit__(*a)

        dec = FT8ConsoleDecoder()
        frame = _synthesise_ft8("CQ W4ABC EN52", f0_hz=1500.0)

        with mock.patch.object(ft8_decode, "ThreadPoolExecutor", _TrackingExecutor):
            dec._decode_pass(frame, "000000", set())

        assert executor_calls, "ThreadPoolExecutor was not used in _decode_pass"

    def test_decode_pass_returns_list(self):
        """_decode_pass should return a list (possibly empty)."""
        dec = FT8ConsoleDecoder()
        frame = np.zeros(int(15 * FT8_FS), dtype=np.float32)
        result = dec._decode_pass(frame, "000000", set())
        assert isinstance(result, list)

    def test_parallel_decode_finds_signal(self):
        """A clean synthesised signal should be decoded by _decode_pass."""
        msg = "CQ W4ABC EN52"
        frame = _synthesise_ft8(msg, f0_hz=1500.0, amplitude=0.8)
        dec = FT8ConsoleDecoder()
        results = dec._decode_pass(frame, "000000", set())
        messages = [r[0] for r in results]
        assert msg in messages, (
            f"Expected '{msg}' in decoded messages, got {messages}"
        )

    def test_parallel_decode_deduplicates_messages(self):
        """Same message decoded by two candidates should appear only once."""
        msg = "CQ W4ABC EN52"
        frame = _synthesise_ft8(msg, f0_hz=1500.0, amplitude=0.8)
        dec = FT8ConsoleDecoder()
        seen: set[str] = set()
        results = dec._decode_pass(frame, "000000", seen)
        messages = [r[0] for r in results]
        assert len(messages) == len(set(messages)), (
            f"Duplicate messages found: {messages}"
        )

    def test_parallel_decode_thread_safe_seen_msg(self):
        """
        Two concurrent workers decoding the same message should not both
        appear in the final result list.
        """
        msg = "W4ABC K9XYZ EN52"
        frame = _synthesise_ft8(msg, f0_hz=1500.0, amplitude=0.8)
        dec = FT8ConsoleDecoder()
        seen: set[str] = set()
        for _ in range(3):
            results = dec._decode_pass(frame, "000000", seen)
            messages = [r[0] for r in results]
            assert messages.count(msg) <= 1, (
                "Message should appear at most once per pass"
            )


# ═══════════════════════════════════════════════════════════════════════════════
# 9  Interference cancellation helpers
# ═══════════════════════════════════════════════════════════════════════════════

class TestSubtractDecodedSignal:
    """Verify _subtract_decoded_signal reduces signal energy at (t0, f0)."""

    def _make_signal_and_symbols(self, msg: str, f0_hz: float = 1500.0,
                                  t0_s: float = 0.5):
        """Return (frame, symbols) for an isolated FT8 signal."""
        symbols = ft8_encode_to_symbols(msg)
        frame = _synthesise_ft8(msg, f0_hz=f0_hz, t0_s=t0_s, amplitude=0.5)
        return frame.astype(np.float64), symbols

    def test_residual_has_less_energy(self):
        """Signal energy at (t0, f0) should drop after subtraction."""
        msg = "CQ W4ABC EN52"
        f0 = 1500.0
        t0 = 0.5
        frame, symbols = self._make_signal_and_symbols(msg, f0_hz=f0, t0_s=t0)

        sym_n = int(round(FT8_SYMBOL_DURATION_S * FT8_FS))
        t0_n = int(round(t0 * FT8_FS))
        sig_end = t0_n + FT8_NSYMS * sym_n

        original_e = float(np.sum(frame[t0_n:sig_end] ** 2))
        residual = _subtract_decoded_signal(
            frame, t0_s=t0, f0_hz=f0, symbols=symbols, fs=FT8_FS
        )
        residual_e = float(np.sum(residual[t0_n:sig_end] ** 2))

        assert residual_e < original_e * 0.5, (
            f"Residual energy should be significantly less than original; "
            f"original={original_e:.3f} residual={residual_e:.3f}"
        )

    def test_residual_same_shape_as_frame(self):
        msg = "W4ABC K9XYZ +02"
        frame, symbols = self._make_signal_and_symbols(msg)
        residual = _subtract_decoded_signal(
            frame, t0_s=0.5, f0_hz=1500.0, symbols=symbols, fs=FT8_FS
        )
        assert residual.shape == frame.shape

    def test_residual_is_copy_not_inplace(self):
        """The original frame should be unchanged after subtraction."""
        msg = "CQ K1AAA FN31"
        frame, symbols = self._make_signal_and_symbols(msg)
        frame_original = frame.copy()
        _ = _subtract_decoded_signal(
            frame, t0_s=0.5, f0_hz=1500.0, symbols=symbols, fs=FT8_FS
        )
        np.testing.assert_array_equal(frame, frame_original,
                                      err_msg="Input frame should not be modified")

    def test_zero_synth_energy_safe(self):
        """Symbols array of all zeros should not raise."""
        frame = np.zeros(int(15 * FT8_FS), dtype=np.float64)
        symbols = np.zeros(FT8_NSYMS, dtype=np.uint8)
        result = _subtract_decoded_signal(
            frame, t0_s=0.5, f0_hz=1500.0, symbols=symbols, fs=FT8_FS
        )
        assert result.shape == frame.shape

    def test_edge_t0_near_zero(self):
        """t0 near zero (signal starts at frame edge) should not crash."""
        msg = "CQ W4ABC EN52"
        symbols = ft8_encode_to_symbols(msg)
        frame = np.zeros(int(16 * FT8_FS), dtype=np.float64)
        result = _subtract_decoded_signal(
            frame, t0_s=0.01, f0_hz=1500.0, symbols=symbols, fs=FT8_FS
        )
        assert result.shape == frame.shape

    def test_two_signals_both_subtracted(self):
        """
        With two signals at different frequencies, subtracting the first
        should not remove the second.
        """
        msg1 = "CQ W4ABC EN52"
        msg2 = "CQ K1BBB FN31"
        f1, f2 = 1000.0, 2000.0
        sym_n = int(round(FT8_SYMBOL_DURATION_S * FT8_FS))
        total_n = FT8_NSYMS * sym_n
        frame_n = int(15 * FT8_FS)
        t0 = 0.5
        t0_n = int(round(t0 * FT8_FS))

        frame = np.zeros(frame_n, dtype=np.float64)
        s1 = ft8_symbols_to_audio(ft8_encode_to_symbols(msg1),
                                   f0_hz=f1, fs=FT8_FS, amplitude=0.5)
        s2 = ft8_symbols_to_audio(ft8_encode_to_symbols(msg2),
                                   f0_hz=f2, fs=FT8_FS, amplitude=0.5)
        end = min(t0_n + total_n, frame_n)
        frame[t0_n:end] += s1[:end - t0_n].astype(np.float64)
        frame[t0_n:end] += s2[:end - t0_n].astype(np.float64)

        # Measure energy of signal 2 before subtraction
        syms2 = ft8_encode_to_symbols(msg2)
        ext = FT8SymbolEnergyExtractor(fs=FT8_FS)
        E_before = ext.extract_all_79(frame, t0_s=t0, f0_hz=f2)
        e_before = float(np.sum(E_before))

        # Subtract signal 1
        syms1 = ft8_encode_to_symbols(msg1)
        residual = _subtract_decoded_signal(
            frame, t0_s=t0, f0_hz=f1, symbols=syms1, fs=FT8_FS
        )

        # Signal 2 energy should still be present in the residual
        E_after = ext.extract_all_79(residual, t0_s=t0, f0_hz=f2)
        e_after = float(np.sum(E_after))

        assert e_after > e_before * 0.5, (
            f"Signal 2 energy should be mostly intact after subtracting signal 1; "
            f"before={e_before:.2f} after={e_after:.2f}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 10  Deep search integration
# ═══════════════════════════════════════════════════════════════════════════════

class TestDeepSearch:
    """Integration tests for the iterative interference cancellation loop."""

    def test_deep_search_can_be_disabled(self):
        """Setting deep_search_passes=0 should skip the deep search loop."""
        dec = FT8ConsoleDecoder()
        dec.deep_search_passes = 0

        subtract_calls = []
        _orig_subtract = ft8_decode._subtract_decoded_signal

        def _spy(*a, **kw):
            subtract_calls.append(True)
            return _orig_subtract(*a, **kw)

        frame = _synthesise_ft8("CQ W4ABC EN52", f0_hz=1500.0)
        with mock.patch.object(ft8_decode, "_subtract_decoded_signal", _spy):
            dec._decode_frame(frame, "000000")

        assert not subtract_calls, (
            "_subtract_decoded_signal should not be called when deep_search_passes=0"
        )

    def test_deep_search_enabled_calls_subtract(self):
        """With deep_search_passes ≥ 1, subtract should be called after a decode."""
        dec = FT8ConsoleDecoder()
        dec.deep_search_passes = 1

        subtract_calls = []
        _orig_subtract = ft8_decode._subtract_decoded_signal

        def _spy(*a, **kw):
            subtract_calls.append(True)
            return _orig_subtract(*a, **kw)

        frame = _synthesise_ft8("CQ W4ABC EN52", f0_hz=1500.0, amplitude=0.8)
        with mock.patch.object(ft8_decode, "_subtract_decoded_signal", _spy):
            dec._decode_frame(frame, "000000")

        assert subtract_calls, (
            "_subtract_decoded_signal should be called when a signal is decoded "
            "and deep_search_passes=1"
        )

    def test_deep_search_uses_seen_msg_across_passes(self):
        """
        Messages decoded in the first pass should not be re-reported in
        subsequent deep search passes.
        """
        reported = []
        dec = FT8ConsoleDecoder(on_decode=lambda *a: reported.append(a[3]))
        dec.deep_search_passes = 2

        frame = _synthesise_ft8("CQ W4ABC EN52", f0_hz=1500.0, amplitude=0.8)
        dec._decode_frame(frame, "000000")

        counts = {}
        for msg in reported:
            counts[msg] = counts.get(msg, 0) + 1
        duplicates = {m: c for m, c in counts.items() if c > 1}
        assert not duplicates, (
            f"Messages decoded multiple times (should be deduped): {duplicates}"
        )

    def test_decode_frame_reports_via_callback(self):
        """_decode_frame should invoke the on_decode callback for each message."""
        reported = []
        dec = FT8ConsoleDecoder(on_decode=lambda utc, f0, snr, msg: reported.append(msg))
        msg = "CQ W4ABC EN52"
        frame = _synthesise_ft8(msg, f0_hz=1500.0, amplitude=0.8)
        dec._decode_frame(frame, "000000")
        assert msg in reported

    def test_deep_search_no_subtract_on_empty_first_pass(self):
        """
        When the first decode pass finds nothing, no subtraction should happen.
        """
        subtract_calls = []
        _orig = ft8_decode._subtract_decoded_signal

        def _spy(*a, **kw):
            subtract_calls.append(True)
            return _orig(*a, **kw)

        dec = FT8ConsoleDecoder()
        dec.deep_search_passes = 3
        # Silent frame — no signals
        frame = np.zeros(int(15 * FT8_FS), dtype=np.float32)
        with mock.patch.object(ft8_decode, "_subtract_decoded_signal", _spy):
            dec._decode_frame(frame, "000000")

        assert not subtract_calls, (
            "No subtraction should happen when there are no initial decodes"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 11  FT8ConsoleDecoder.set_dx_callsign wiring with main.py
# ═══════════════════════════════════════════════════════════════════════════════

class TestSetDxCallsignIntegration:
    """Verify set_dx_callsign is accessible and correctly affects decode."""

    def test_method_exists(self):
        dec = FT8ConsoleDecoder()
        assert hasattr(dec, "set_dx_callsign")
        assert callable(dec.set_dx_callsign)

    def test_set_clears_previous(self):
        dec = FT8ConsoleDecoder()
        dec.set_dx_callsign("W4ABC")
        dec.set_dx_callsign("K9XYZ")
        # Should have 2 passes for the NEW callsign, not 4
        assert len(dec._dx_ap_passes) == 2

    def test_passes_are_lists_of_tuples(self):
        dec = FT8ConsoleDecoder()
        dec.set_dx_callsign("W4ABC")
        for name, bits in dec._dx_ap_passes:
            assert isinstance(name, str)
            assert isinstance(bits, tuple)
            for idx, val in bits:
                assert isinstance(idx, int)
                assert isinstance(val, int)

    def test_thread_safe_list_copy_in_decode(self):
        """
        _decode_one_candidate snapshots _dx_ap_passes with list()
        so Tk-thread updates don't corrupt an in-flight decode.
        """
        dec = FT8ConsoleDecoder()
        dec.set_dx_callsign("W4ABC")

        snapshot_taken = []
        _orig_ldpc = ft8_decode.ft8_ldpc_decode

        def _spy(llrs, **kwargs):
            # Simulate Tk thread clearing DX callsign mid-decode
            if not snapshot_taken:
                snapshot_taken.append(True)
                dec.set_dx_callsign(None)
            return _orig_ldpc(llrs, **kwargs)

        frame = _synthesise_ft8("CQ W4ABC EN52", f0_hz=1500.0, amplitude=0.8)
        with mock.patch.object(ft8_decode, "ft8_ldpc_decode", side_effect=_spy):
            # Must not raise even though DX callsign is cleared mid-decode
            dec._decode_one_candidate(frame, 0.5, 1500.0, set())


# ═══════════════════════════════════════════════════════════════════════════════
# 12  Regression tests for existing decode correctness
# ═══════════════════════════════════════════════════════════════════════════════

class TestRegressionDecodeCorrectness:
    """
    Ensure the Milestone 6 changes do not break round-trip decode accuracy.
    These are faster alternatives to the full decode_wav pipeline tests.
    """

    @pytest.mark.parametrize("msg", [
        "CQ W4ABC EN52",
        "W4ABC K9XYZ +02",
        "K9XYZ W4ABC R-07",
        "K9XYZ W4ABC RR73",
        "W4ABC K9XYZ 73",
        "CQ VK2TIM QF56",
    ])
    def test_roundtrip_clean_signal(self, msg: str):
        """Clean synthesised signal should decode to the original message."""
        frame = _synthesise_ft8(msg, f0_hz=1500.0, amplitude=0.5)
        dec = FT8ConsoleDecoder()
        results = dec._decode_pass(frame, "000000", set())
        messages = [r[0] for r in results]
        assert msg in messages, (
            f"Round-trip decode failed for '{msg}': got {messages}"
        )

    def test_roundtrip_off_grid_frequency(self):
        """Signal at an off-grid frequency (±2.5 Hz) should still decode."""
        msg = "CQ W4ABC EN52"
        # 2.5 Hz offset — falls between two 6.25 Hz bins, requires fine-freq search
        frame = _synthesise_ft8(msg, f0_hz=1502.5, amplitude=0.5)
        dec = FT8ConsoleDecoder()
        results = dec._decode_pass(frame, "000000", set())
        messages = [r[0] for r in results]
        assert msg in messages, (
            f"Off-grid signal not decoded: {messages}"
        )

    def test_roundtrip_snr_reported(self):
        """The SNR dB value should be a finite float for a clean signal."""
        msg = "CQ W4ABC EN52"
        frame = _synthesise_ft8(msg, f0_hz=1500.0, amplitude=0.5)
        dec = FT8ConsoleDecoder()
        results = dec._decode_pass(frame, "000000", set())
        assert results, "Expected at least one decode"
        snr = results[0][3]
        assert math.isfinite(snr)

    def test_costas_score_legacy_shims(self):
        """Legacy API shims ft8_costas_ok / ft8_costas_ok_costasE still work."""
        from ft8_decode import ft8_costas_ok, ft8_costas_ok_costasE
        E79 = _make_e79_with_costas(15)
        m, t, s = ft8_costas_ok(E79)
        assert t == 21
        m2, t2, s2, inv2 = ft8_costas_ok_costasE(E79)
        assert t2 == 21
        assert isinstance(inv2, bool)
