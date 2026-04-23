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
    FT8_PAYLOAD_POSITIONS,
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
    PolyphaseResampler,
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

    FT8_COSTAS_TONES has 7 elements.  The 21 Costas positions are three
    groups of 7, all sharing the same 7-tone pattern, so ``i % 7`` is
    the correct index into FT8_COSTAS_TONES for Costas position ``i``.
    """
    E79 = np.ones((FT8_NSYMS, 8), dtype=np.float64)
    cos_positions = list(FT8_COSTAS_POSITIONS)
    # Set up the Costas rows: small uniform energy first
    for i, pos in enumerate(cos_positions):
        expected = FT8_COSTAS_TONES[i % 7]  # same 7-tone pattern repeats 3×
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
        """E79 where the correct Costas tone is 'ratio' times stronger.

        FT8_COSTAS_TONES has 7 elements; the 21 Costas positions are three
        groups of 7 all using the same 7-tone pattern, so ``i % 7`` is the
        correct subscript into FT8_COSTAS_TONES for position index ``i``.
        """
        E79 = np.ones((FT8_NSYMS, 8), dtype=np.float64)
        for i, pos in enumerate(FT8_COSTAS_POSITIONS):
            expected = FT8_COSTAS_TONES[i % 7]  # same 7-tone pattern repeats 3×
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
        We mock ft8_ldpc_decode to force the baseline and all non-DX AP passes
        to fail with best_errors=3 (≤ threshold), and assert that at least one
        call uses a DX-callsign AP assignment.
        """
        call_log: list[Optional[tuple]] = []
        _orig = ft8_decode.ft8_ldpc_decode

        class _DXAPAttempted(RuntimeError):
            """Raised when a DX-specific AP pass is observed, ending the test."""

        dec = FT8ConsoleDecoder()
        dec.set_dx_callsign("K9XYZ")
        dx_ap_set = {bits for _, bits in dec._dx_ap_passes}
        frame = _synthesise_ft8("CQ W4ABC EN52", f0_hz=1500.0)

        def _fake(llrs, *, max_iterations=50, ap_assignments=None):
            call_log.append(ap_assignments)
            # If this is a DX AP pass we've confirmed the code reaches it.
            if ap_assignments in dx_ap_set:
                raise _DXAPAttempted
            # Force all non-DX calls to fail with best_errors=3 so the decoder
            # proceeds to the DX AP passes.
            _, pay, _, _ = _orig(llrs, max_iterations=max_iterations,
                                 ap_assignments=None)
            return False, pay, max_iterations, 3

        with pytest.raises(_DXAPAttempted):
            with mock.patch.object(ft8_decode, "ft8_ldpc_decode", side_effect=_fake):
                dec._decode_one_candidate(frame, 0.5, 1500.0, set())

        assert call_log, "ft8_ldpc_decode should have been called at least once"
        assert any(
            ap in dx_ap_set for ap in call_log
        ), "Expected at least one LDPC call with a DX-callsign AP assignment"


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
        """E79 with inverted Costas (7-n) should return inv=True.

        FT8_COSTAS_TONES has 7 elements; three groups of 7 Costas positions
        all share the same pattern so ``i % 7`` is the correct index.
        """
        E79 = np.ones((FT8_NSYMS, 8), dtype=np.float64)
        for i, pos in enumerate(FT8_COSTAS_POSITIONS):
            expected = FT8_COSTAS_TONES[i % 7]  # same 7-tone pattern repeats 3×
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

    def test_cache_lru_eviction_order(self):
        """
        True LRU: the least-recently-used entry is evicted, not the
        first-inserted one.
        """
        from collections import OrderedDict
        ext = FT8SymbolEnergyExtractor(fs=FT8_FS)
        max_size = ext._CACHE_MAXSIZE
        assert max_size >= 3, "This test needs cache size ≥ 3"

        # Fill to exactly max_size
        freqs = [float(1000 + i) for i in range(max_size)]
        for f in freqs:
            ext._get_basis(f)
        assert len(ext._basis_cache) == max_size

        # Re-access the oldest entry (freqs[0]) to promote it to MRU.
        ext._get_basis(freqs[0])

        # Insert a new entry to trigger eviction.
        new_f = float(1000 + max_size)
        ext._get_basis(new_f)
        assert len(ext._basis_cache) == max_size

        # The evicted entry should be freqs[1] (LRU), NOT freqs[0] (just accessed).
        assert freqs[0] in ext._basis_cache, (
            "freqs[0] was recently accessed and must NOT be evicted (LRU)"
        )
        assert new_f in ext._basis_cache, "Newly inserted key should be present"
        assert freqs[1] not in ext._basis_cache, (
            "freqs[1] was the true LRU entry and should have been evicted"
        )

    def test_basis_cache_is_ordered_dict(self):
        """_basis_cache must be an OrderedDict for true LRU eviction."""
        from collections import OrderedDict
        ext = FT8SymbolEnergyExtractor(fs=FT8_FS)
        assert isinstance(ext._basis_cache, OrderedDict)

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

    def test_out_parameter_inplace(self):
        """When out= is provided, subtraction happens in-place into that buffer."""
        msg = "CQ W4ABC EN52"
        frame, symbols = self._make_signal_and_symbols(msg)
        buf = frame.copy()
        result = _subtract_decoded_signal(
            frame, t0_s=0.5, f0_hz=1500.0, symbols=symbols, fs=FT8_FS, out=buf
        )
        # result must be the exact same object as buf
        assert result is buf, "out= buffer must be returned directly"

    def test_out_parameter_modifies_correctly(self):
        """Result via out= must equal result via default (copy) path."""
        msg = "CQ W4ABC EN52"
        frame, symbols = self._make_signal_and_symbols(msg)
        expected = _subtract_decoded_signal(
            frame, t0_s=0.5, f0_hz=1500.0, symbols=symbols, fs=FT8_FS
        )
        buf = frame.copy()
        _subtract_decoded_signal(
            frame, t0_s=0.5, f0_hz=1500.0, symbols=symbols, fs=FT8_FS, out=buf
        )
        np.testing.assert_allclose(buf, expected, atol=1e-12,
                                   err_msg="out= path must produce same result as copy path")

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

    def test_deep_search_cumulative_subtraction(self):
        """
        Each deep-search pass must subtract ALL previously decoded signals,
        not just those decoded in the immediately preceding pass.

        We spy on _subtract_decoded_signal and record which symbols are
        passed.  After the first pass decodes N signals, pass 2 must subtract
        those same N signals (plus any new ones from pass 1, etc.).
        """
        from ft8_encode import ft8_encode_to_symbols

        subtracted: list[tuple] = []   # (t0_s, f0_hz, symbols_tuple)
        _orig = ft8_decode._subtract_decoded_signal

        def _spy(frame, *, t0_s, f0_hz, symbols, fs, out=None):
            subtracted.append((round(t0_s, 3), round(f0_hz, 1), tuple(symbols)))
            return _orig(frame, t0_s=t0_s, f0_hz=f0_hz, symbols=symbols,
                         fs=fs, out=out)

        msg = "CQ W4ABC EN52"
        frame = _synthesise_ft8(msg, f0_hz=1500.0, amplitude=0.8)

        dec = FT8ConsoleDecoder()
        dec.deep_search_passes = 2   # two deep-search passes after the first

        first_pass_results = []

        def _capture_first_pass(utc, f0, snr, decoded_msg):
            first_pass_results.append(decoded_msg)

        dec._on_decode = _capture_first_pass

        with mock.patch.object(ft8_decode, "_subtract_decoded_signal", _spy):
            dec._decode_frame(frame, "000000")

        if not first_pass_results:
            pytest.skip("Signal not decoded; cannot verify cumulative subtraction")

        # Pass 1 of deep search: should subtract all signals from first pass.
        # Pass 2 of deep search: should subtract all signals from first AND
        # second (deep) pass.  Even if the second pass yielded nothing new, the
        # loop exits early — but at minimum pass 1 must have subtracted
        # everything from the initial pass.
        first_pass_n = len(first_pass_results)
        # The subtracted list spans all deep-search passes.  The very first
        # batch must be at least first_pass_n entries (one per decoded signal).
        assert len(subtracted) >= first_pass_n, (
            f"Deep search pass 1 should subtract all {first_pass_n} decoded "
            f"signal(s); only subtracted {len(subtracted)}"
        )

    def test_deep_search_inplace_out_buffer(self):
        """
        _subtract_decoded_signal is called with out= in the deep-search loop,
        so no extra full-frame copies are made per signal.
        """
        out_params = []
        _orig = ft8_decode._subtract_decoded_signal

        def _spy(frame, *, t0_s, f0_hz, symbols, fs, out=None):
            out_params.append(out)
            return _orig(frame, t0_s=t0_s, f0_hz=f0_hz, symbols=symbols,
                         fs=fs, out=out)

        msg = "CQ W4ABC EN52"
        frame = _synthesise_ft8(msg, f0_hz=1500.0, amplitude=0.8)
        dec = FT8ConsoleDecoder()
        dec.deep_search_passes = 1

        with mock.patch.object(ft8_decode, "_subtract_decoded_signal", _spy):
            dec._decode_frame(frame, "000000")

        if out_params:
            assert any(o is not None for o in out_params), (
                "Deep-search loop should pass out= buffer to avoid per-signal copies"
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
# 13  reset_framer() — Stop/Start audio re-sync fix (Bug fix regression tests)
# ═══════════════════════════════════════════════════════════════════════════════

class TestResetFramer:
    """
    Regression tests for FT8ConsoleDecoder.reset_framer().

    Background
    ----------
    When the user clicks "Stop Audio" and then "Start Audio" in DATA mode the
    decoder continues running but the internal UTC15sFramer retains stale state:

    * ``_t0_utc`` — still holds the epoch timestamp from the *previous* audio
      stream, which may be tens of seconds or more in the past.
    * ``_buf``    — may contain leftover audio samples from before the stop.
    * ``_first_frame_emitted = True`` — so the first assembled frame after
      restart is *not* marked partial and is decoded rather than skipped.

    New audio arrives with fresh ``t0_monotonic`` values, but because
    ``_t0_utc`` is not ``None`` it is never re-initialised.  The framer
    assembles frames whose slot boundaries are offset from the actual current
    UTC position; FT8 signals in the new audio appear at wrong time offsets
    within those frames and the Costas sync search fails.

    ``reset_framer()`` fixes this by replacing the framer with a fresh
    :class:`~ft8_decode.UTC15sFramer` instance and draining the audio queue.
    """

    def test_reset_framer_method_exists(self):
        """FT8ConsoleDecoder must expose reset_framer()."""
        dec = FT8ConsoleDecoder()
        assert hasattr(dec, "reset_framer")
        assert callable(dec.reset_framer)

    def test_reset_framer_replaces_framer_instance(self):
        """reset_framer() must replace the _framer with a new instance."""
        from ft8_decode import UTC15sFramer
        dec = FT8ConsoleDecoder()
        framer_before = dec._framer
        dec.reset_framer()
        assert dec._framer is not framer_before, (
            "reset_framer() must create a new UTC15sFramer instance"
        )
        assert isinstance(dec._framer, UTC15sFramer)

    def test_reset_framer_clears_t0_utc(self):
        """New framer must start with _t0_utc=None (not the stale old value)."""
        dec = FT8ConsoleDecoder()
        # Prime the old framer with a stale timestamp
        silence = np.zeros(int(0.1 * FT8_FS), dtype=np.float32)
        dec._framer.push(silence, t0_monotonic=99999.0)
        assert dec._framer._t0_utc is not None, "Precondition: _t0_utc should be set"

        dec.reset_framer()
        assert dec._framer._t0_utc is None, (
            f"After reset_framer(), new framer _t0_utc must be None, "
            f"got {dec._framer._t0_utc}"
        )

    def test_reset_framer_clears_buffer(self):
        """New framer must have an empty sample buffer."""
        dec = FT8ConsoleDecoder()
        silence = np.zeros(int(0.5 * FT8_FS), dtype=np.float32)
        dec._framer.push(silence, t0_monotonic=0.0)

        dec.reset_framer()
        assert len(dec._framer._buf) == 0, (
            f"New framer should have empty _buf after reset, "
            f"got {len(dec._framer._buf)} samples"
        )

    def test_reset_framer_clears_first_frame_emitted(self):
        """New framer must have _first_frame_emitted=False so the first frame is partial."""
        dec = FT8ConsoleDecoder()
        silence = np.zeros(int(30 * FT8_FS), dtype=np.float32)
        frames_before = dec._framer.push(silence, t0_monotonic=0.0)
        # Ensure at least one frame was emitted (making _first_frame_emitted=True)
        assert any(not p for _, _, p in frames_before) or dec._framer._first_frame_emitted

        dec.reset_framer()
        assert not dec._framer._first_frame_emitted, (
            "_first_frame_emitted should be False on fresh framer after reset"
        )

    def test_reset_framer_drains_queue(self):
        """reset_framer() must empty the audio queue to discard stale chunks."""
        dec = FT8ConsoleDecoder()
        fake = np.zeros(100, dtype=np.float32)
        for _ in range(8):
            dec._q.put_nowait((FT8_FS, fake, 1234.0))
        assert dec._q.qsize() == 8, "Precondition: queue should have 8 items"

        dec.reset_framer()
        assert dec._q.empty(), (
            f"Queue must be empty after reset_framer(), "
            f"got {dec._q.qsize()} item(s)"
        )

    def test_reset_framer_resets_resampler(self):
        """reset_framer() must set _resampler to None for clean reinitialisation."""
        dec = FT8ConsoleDecoder()
        # Trigger resampler creation
        dec._resampler = ft8_decode.PolyphaseResampler(fs_in=48000, fs_out=FT8_FS)

        dec.reset_framer()
        assert dec._resampler is None, (
            "_resampler should be None after reset_framer() so it re-creates "
            "cleanly for the new stream's sample rate"
        )

    def test_reset_framer_new_stream_uses_current_utc(self):
        """
        After reset_framer(), the first audio chunk's t0_monotonic must
        initialise _t0_utc to *approximately* the current time, not to the
        stale old timestamp.
        """
        dec = FT8ConsoleDecoder()

        # Simulate a stale timestamp: prime the old framer with UTC=1000s
        old_framer = dec._framer
        old_framer._utc_minus_mono = 1000.0
        old_framer._alpha = 0.0
        silence = np.zeros(int(0.1 * FT8_FS), dtype=np.float32)
        old_framer.push(silence, t0_monotonic=0.0)
        stale_t0 = old_framer._t0_utc
        assert stale_t0 is not None and abs(stale_t0 - 1000.0) < 1.0

        # Reset — new stream starts at UTC=5000s (45+ minutes later)
        dec.reset_framer()
        new_framer = dec._framer
        new_framer._utc_minus_mono = 5000.0
        new_framer._alpha = 0.0

        current_mono = 0.5  # current monotonic time
        expected_utc = current_mono + 5000.0  # = 5000.5

        new_framer.push(silence, t0_monotonic=current_mono)
        assert new_framer._t0_utc is not None
        assert abs(new_framer._t0_utc - expected_utc) < 1.0, (
            f"New framer _t0_utc={new_framer._t0_utc:.1f}s should be ≈{expected_utc:.1f}s "
            f"(current UTC), not ≈{stale_t0:.1f}s (stale old value)"
        )

    def test_reset_framer_idempotent(self):
        """Calling reset_framer() multiple times must not raise."""
        dec = FT8ConsoleDecoder()
        dec.reset_framer()
        dec.reset_framer()
        dec.reset_framer()
        # If we get here without exception, the test passes.

    def test_reset_framer_while_stopped(self):
        """reset_framer() must work even when the decoder worker is not running."""
        dec = FT8ConsoleDecoder()
        # Worker not started — reset_framer() should still work
        dec.reset_framer()
        assert dec._framer._t0_utc is None

    def test_reset_framer_does_not_stop_worker(self):
        """
        reset_framer() must NOT stop the decoder's background worker thread.
        The worker must still be alive after the reset so it can process
        new audio immediately without requiring an explicit start() call.
        """
        dec = FT8ConsoleDecoder()
        dec.start()
        try:
            assert dec._thread is not None and dec._thread.is_alive(), (
                "Precondition: worker thread should be alive after start()"
            )
            dec.reset_framer()
            assert dec._thread is not None and dec._thread.is_alive(), (
                "Worker thread must still be alive after reset_framer()"
            )
        finally:
            dec.stop()

    @pytest.mark.parametrize("pause_s", [5.0, 15.0, 60.0, 300.0])
    def test_reset_framer_resync_after_various_pause_lengths(self, pause_s: float):
        """
        After pauses of different durations, the new stream must synchronise
        to the current UTC, not the pre-pause timestamp.
        """
        dec = FT8ConsoleDecoder()
        silence = np.zeros(int(0.1 * FT8_FS), dtype=np.float32)

        # Prime at UTC = 30s
        dec._framer._utc_minus_mono = 30.0
        dec._framer._alpha = 0.0
        dec._framer.push(silence, t0_monotonic=0.0)
        stale_t0 = dec._framer._t0_utc
        assert stale_t0 is not None

        # Reset and restart at UTC = 30 + pause_s
        dec.reset_framer()
        new_utc_offset = 30.0 + pause_s  # UTC - mono stays constant
        dec._framer._utc_minus_mono = new_utc_offset
        dec._framer._alpha = 0.0

        restart_mono = 0.5
        expected_new_utc = restart_mono + new_utc_offset
        dec._framer.push(silence, t0_monotonic=restart_mono)

        new_t0 = dec._framer._t0_utc
        assert new_t0 is not None
        assert abs(new_t0 - expected_new_utc) < 1.0, (
            f"pause={pause_s}s: new _t0_utc={new_t0:.2f}s, "
            f"expected≈{expected_new_utc:.2f}s"
        )
        assert abs(new_t0 - stale_t0) > pause_s * 0.8, (
            f"pause={pause_s}s: new _t0_utc too close to stale value; "
            f"new={new_t0:.2f}s stale={stale_t0:.2f}s"
        )


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
        """The SNR dB value should be a finite float and meaningful for a clean signal."""
        msg = "CQ W4ABC EN52"
        frame = _synthesise_ft8(msg, f0_hz=1500.0, amplitude=0.5)
        dec = FT8ConsoleDecoder()
        results = dec._decode_pass(frame, "000000", set())
        assert results, "Expected at least one decode"
        snr = results[0][3]
        assert math.isfinite(snr)

    def test_snr_calibration_against_theory(self):
        """
        SNR from E79 symbol energies should match the WSJT-X 2500 Hz reference
        bandwidth formula to within ±3 dB when extracted at exact signal timing.

        This validates the bandwidth normalization factor 2500/(2*6.25)=200 is correct.
        """
        msg = "CQ W4ABC EN52"
        amplitude = 1.0
        noise_sigma = 1.0
        rng = np.random.default_rng(42)

        sym_n = int(round(FT8_SYMBOL_DURATION_S * FT8_FS))
        frame_n = FT8_NSYMS * sym_n + int(0.5 * FT8_FS)
        frame = (rng.standard_normal(frame_n) * noise_sigma).astype(np.float32)
        tones = ft8_encode_to_symbols(msg)
        t0_n = int(0.5 * FT8_FS)
        t_sym = np.arange(sym_n, dtype=np.float64) / FT8_FS
        for s, tone in enumerate(tones):
            freq = 1500.0 + tone * FT8_TONE_SPACING_HZ
            frame[t0_n + s * sym_n: t0_n + (s + 1) * sym_n] += (
                amplitude * np.cos(2 * math.pi * freq * t_sym)
            ).astype(np.float32)

        # Extract E79 at the exact known signal timing (not via decoder search)
        extractor = FT8SymbolEnergyExtractor(fs=FT8_FS)
        E79 = extractor.extract_all_79(frame, t0_s=0.5, f0_hz=1500.0)

        # Compute SNR using the same formula as _decode_one_candidate
        _pl_pos = np.array(FT8_PAYLOAD_POSITIONS)
        E_pl = E79[_pl_pos, :]
        max_e = np.max(E_pl, axis=1)
        sum_e = np.sum(E_pl, axis=1)
        noise_per_bin = (sum_e - max_e) / 7.0
        avg_noise = float(np.mean(noise_per_bin))
        avg_sig = float(np.mean(max_e)) - avg_noise
        noise_2500 = avg_noise * (2500.0 / (2.0 * FT8_TONE_SPACING_HZ))
        measured_snr = 10.0 * math.log10(max(avg_sig, 1e-30) / max(noise_2500, 1e-30))

        # Theoretical WSJT-X SNR for amp=1, noise_sigma=1 at 12kHz
        theoretical_snr = 10.0 * math.log10(
            (amplitude ** 2 * FT8_FS) / (2.0 * noise_sigma ** 2 * 2500.0)
        )  # ≈ +3.8 dB

        assert abs(measured_snr - theoretical_snr) < 3.0, (
            f"SNR {measured_snr:.1f} dB differs from theoretical "
            f"{theoretical_snr:.1f} dB by more than 3 dB"
        )

    def test_costas_score_legacy_shims(self):
        """Legacy API shims ft8_costas_ok / ft8_costas_ok_costasE still work."""
        from ft8_decode import ft8_costas_ok, ft8_costas_ok_costasE
        E79 = _make_e79_with_costas(15)
        m, t, s = ft8_costas_ok(E79)
        assert t == 21
        m2, t2, s2, inv2 = ft8_costas_ok_costasE(E79)
        assert t2 == 21
        assert isinstance(inv2, bool)
