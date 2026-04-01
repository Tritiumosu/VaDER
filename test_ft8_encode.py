"""
test_ft8_encode.py — pytest suite for ft8_encode.py and ft8_qso.py (Milestone 4).

Tests are organised into these classes / groups:

  TestCallsignPacking     — ft8_pack_callsign round-trips and edge cases
  TestCallsignValidation  — validate_callsign positive and negative cases
  TestGridPacking         — ft8_pack_grid for locators, reports, and special tokens
  TestMessagePacking      — ft8_pack_message bit layout for various message types
  TestCrc14               — ft8_append_crc matches known CRC values
  TestLdpcEncode          — ft8_ldpc_encode satisfies all parity checks
  TestSymbolGeneration    — ft8_codeword_to_tones Costas positions and tone range
  TestAudioSynthesis      — ft8_symbols_to_audio length, amplitude, and frequency
  TestEncodeDecodeRoundTrip — encode a message → audio → full decode pipeline
  TestOperatorConfig      — OperatorConfig validation and properties
  TestMessageComposition  — compose_* helper functions
  TestReceivedMessage     — ReceivedMessage parser correctness
  TestQsoStateMachine     — Ft8QsoManager CQ and reply-to-CQ workflows

All tests run without physical hardware; no audio devices or serial ports needed.
"""
from __future__ import annotations

import math
import sys
import os
import importlib
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

# --- Module imports ---------------------------------------------------------
from ft8_encode import (
    validate_callsign,
    ft8_pack_callsign,
    ft8_pack_grid,
    ft8_pack_message,
    ft8_append_crc,
    ft8_ldpc_encode,
    ft8_ldpc_check,
    ft8_codeword_to_tones,
    ft8_symbols_to_audio,
    ft8_encode_to_symbols,
    ft8_encode_message,
    _FT8_ENCODE_MATRIX,
)
from ft8_qso import (
    QsoState,
    OperatorConfig,
    compose_cq,
    compose_reply,
    compose_exchange,
    compose_rrr,
    compose_rr73,
    compose_73,
    ReceivedMessage,
    Ft8QsoManager,
)
from ft8_decode import (
    FT8_COSTAS_POSITIONS,
    FT8_COSTAS_TONES,
    FT8_NSYMS,
    FT8_FS,
    FT8_SYMBOL_DURATION_S,
    FT8_PAYLOAD_POSITIONS,
    FT8SymbolEnergyExtractor,
    ft8_extract_payload_symbols,
    ft8_gray_decode,
    ft8_ldpc_decode,
    ft8_unpack_message,
    _unpack_callsign_28,
    _unpack_grid,
    _ft8_crc14,
)


# ═══════════════════════════════════════════════════════════════════════════════
# § 1  Callsign packing
# ═══════════════════════════════════════════════════════════════════════════════

class TestCallsignPacking(unittest.TestCase):
    """ft8_pack_callsign must produce values that _unpack_callsign_28 reverses."""

    def _round_trip(self, call: str) -> None:
        """Pack *call*, then unpack and verify we get the same string back."""
        packed   = ft8_pack_callsign(call)
        unpacked = _unpack_callsign_28(packed)
        self.assertEqual(unpacked, call.upper().strip(),
                         f"Round-trip failed for {call!r}: got {unpacked!r}")

    # Known numeric values from test_msg_unpack.py reference implementation
    def test_w4abc_known_value(self):
        self.assertEqual(ft8_pack_callsign("W4ABC"), 12635974)

    def test_k9xyz_known_value(self):
        self.assertEqual(ft8_pack_callsign("K9XYZ"), 10389840)

    def test_vk2tim_known_value(self):
        self.assertEqual(ft8_pack_callsign("VK2TIM"), 236996858)

    def test_g3abc_known_value(self):
        self.assertEqual(ft8_pack_callsign("G3ABC"), 9467011)

    def test_de_special_token(self):
        self.assertEqual(ft8_pack_callsign("DE"), 0)

    def test_qrz_special_token(self):
        self.assertEqual(ft8_pack_callsign("QRZ"), 1)

    def test_cq_special_token(self):
        self.assertEqual(ft8_pack_callsign("CQ"), 2)

    def test_cq_numeric_suffix(self):
        n = ft8_pack_callsign("CQ 143")
        self.assertEqual(n, 3 + 143)

    def test_round_trip_w4abc(self):
        self._round_trip("W4ABC")

    def test_round_trip_vk2tim(self):
        self._round_trip("VK2TIM")

    def test_round_trip_k9xyz(self):
        self._round_trip("K9XYZ")

    def test_round_trip_g3abc(self):
        self._round_trip("G3ABC")

    def test_round_trip_aa0abc(self):
        self._round_trip("AA0ABC")

    def test_lowercase_normalised(self):
        """Lowercase input must produce the same result as uppercase."""
        self.assertEqual(ft8_pack_callsign("w4abc"), ft8_pack_callsign("W4ABC"))

    def test_portable_suffix_stripped(self):
        """Pack with /R should give same base value as without (ipa bit is separate)."""
        self.assertEqual(ft8_pack_callsign("W4ABC/R"), ft8_pack_callsign("W4ABC"))

    def test_invalid_callsign_raises(self):
        with self.assertRaises(ValueError):
            ft8_pack_callsign("TOOLONGCALL123")


# ═══════════════════════════════════════════════════════════════════════════════
# § 2  Callsign validation
# ═══════════════════════════════════════════════════════════════════════════════

class TestCallsignValidation(unittest.TestCase):

    def _valid(self, call: str) -> None:
        self.assertTrue(validate_callsign(call), f"Expected {call!r} to be valid")

    def _invalid(self, call: str) -> None:
        self.assertFalse(validate_callsign(call), f"Expected {call!r} to be invalid")

    def test_w4abc(self):      self._valid("W4ABC")
    def test_k9xyz(self):      self._valid("K9XYZ")
    def test_vk2tim(self):     self._valid("VK2TIM")
    def test_g3abc(self):      self._valid("G3ABC")
    def test_aa0abc(self):     self._valid("AA0ABC")
    def test_de_token(self):   self._valid("DE")
    def test_qrz_token(self):  self._valid("QRZ")
    def test_cq_token(self):   self._valid("CQ")
    def test_cq_dx(self):      self._valid("CQ DX")
    def test_cq_eu(self):      self._valid("CQ EU")
    def test_portable_r(self): self._valid("W4ABC/R")
    def test_portable_p(self): self._valid("W4ABC/P")

    def test_no_digit(self):        self._invalid("WABC")
    def test_empty(self):           self._invalid("")
    def test_too_long_suffix(self): self._invalid("W4ABCD")  # 4-letter suffix
    def test_starts_with_digit(self):
        # Digit-only "callsign" has no letters — ambiguous; we expect it invalid
        # because the regex requires at least 1 letter in the suffix
        self._invalid("4")


# ═══════════════════════════════════════════════════════════════════════════════
# § 3  Grid packing
# ═══════════════════════════════════════════════════════════════════════════════

class TestGridPacking(unittest.TestCase):
    """ft8_pack_grid must produce (igrid4, ir) that _unpack_grid reverses."""

    def _round_trip(self, grid_str: str, expected_text: str | None = None) -> None:
        """Pack then unpack and verify the result."""
        igrid4, ir = ft8_pack_grid(grid_str)
        unpacked   = _unpack_grid(igrid4, ir)
        expected   = expected_text if expected_text is not None else grid_str.upper()
        self.assertEqual(unpacked, expected,
                         f"Round-trip failed for {grid_str!r}: got {unpacked!r}")

    def test_en52(self):   self._round_trip("EN52")
    def test_em73(self):   self._round_trip("EM73")
    def test_io91(self):   self._round_trip("IO91")
    def test_qf56(self):   self._round_trip("QF56")

    def test_rrr(self):    self._round_trip("RRR")
    def test_rr73(self):   self._round_trip("RR73")
    def test_73(self):     self._round_trip("73")

    def test_snr_minus5(self):
        igrid4, ir = ft8_pack_grid("-05")
        self.assertEqual(ir, 0)
        self.assertEqual(_unpack_grid(igrid4, ir), "-05")

    def test_snr_plus12(self):
        igrid4, ir = ft8_pack_grid("+12")
        self.assertEqual(ir, 0)
        self.assertEqual(_unpack_grid(igrid4, ir), "+12")

    def test_r_snr_minus7(self):
        igrid4, ir = ft8_pack_grid("R-07")
        self.assertEqual(ir, 1)
        self.assertEqual(_unpack_grid(igrid4, ir), "R-07")

    def test_blank(self):
        igrid4, ir = ft8_pack_grid("")
        self.assertEqual(igrid4, 32400)
        self.assertEqual(ir, 0)

    def test_known_en52_value(self):
        igrid4, ir = ft8_pack_grid("EN52")
        self.assertEqual(igrid4, 8552)
        self.assertEqual(ir, 0)

    def test_invalid_grid_raises(self):
        with self.assertRaises(ValueError):
            ft8_pack_grid("ZZ99")   # Z is not a valid grid letter (A-R only)

    def test_invalid_report_raises(self):
        with self.assertRaises(ValueError):
            ft8_pack_grid("+999")   # way out of range


# ═══════════════════════════════════════════════════════════════════════════════
# § 4  Message packing
# ═══════════════════════════════════════════════════════════════════════════════

class TestMessagePacking(unittest.TestCase):
    """ft8_pack_message must produce 77-bit arrays with correct field layout."""

    def _bits_to_int(self, bits: np.ndarray, start: int, length: int) -> int:
        val = 0
        for i in range(length):
            val = (val << 1) | int(bits[start + i])
        return val

    def test_output_shape_and_dtype(self):
        bits = ft8_pack_message("CQ W4ABC EN52")
        self.assertEqual(bits.shape, (77,))
        self.assertTrue(np.issubdtype(bits.dtype, np.unsignedinteger))

    def test_i3_field_is_type1(self):
        """Bits 74-76 must encode i3 = 001 for a standard type-1 message."""
        bits = ft8_pack_message("W4ABC K9XYZ -05")
        i3 = self._bits_to_int(bits, 74, 3)
        self.assertEqual(i3, 1)

    def test_n28a_field_cq(self):
        """First callsign CQ should encode n28a = 2."""
        bits  = ft8_pack_message("CQ W4ABC EN52")
        n28a  = self._bits_to_int(bits, 0, 28)
        self.assertEqual(n28a, 2)

    def test_n28b_field_w4abc(self):
        """Second callsign W4ABC should encode to its known 28-bit value."""
        bits = ft8_pack_message("CQ W4ABC EN52")
        n28b = self._bits_to_int(bits, 29, 28)
        self.assertEqual(n28b, ft8_pack_callsign("W4ABC"))

    def test_igrid4_en52(self):
        """Grid field should encode EN52 = 8552."""
        bits   = ft8_pack_message("CQ W4ABC EN52")
        igrid4 = self._bits_to_int(bits, 59, 15)
        self.assertEqual(igrid4, 8552)

    def test_ipa_zero_for_non_portable(self):
        bits = ft8_pack_message("W4ABC K9XYZ -05")
        self.assertEqual(int(bits[28]), 0)

    def test_ipb_zero_for_non_portable(self):
        bits = ft8_pack_message("W4ABC K9XYZ -05")
        self.assertEqual(int(bits[57]), 0)

    def test_ir_zero_for_plain_snr(self):
        bits = ft8_pack_message("W4ABC K9XYZ -05")
        self.assertEqual(int(bits[58]), 0)

    def test_ir_one_for_r_snr(self):
        bits = ft8_pack_message("K9XYZ W4ABC R-07")
        self.assertEqual(int(bits[58]), 1)

    def test_rrr_message_packs(self):
        bits = ft8_pack_message("W4ABC K9XYZ RRR")
        self.assertEqual(bits.shape, (77,))

    def test_rr73_message_packs(self):
        bits = ft8_pack_message("W4ABC K9XYZ RR73")
        self.assertEqual(bits.shape, (77,))

    def test_73_message_packs(self):
        bits = ft8_pack_message("K9XYZ W4ABC 73")
        self.assertEqual(bits.shape, (77,))

    def test_too_few_parts_raises(self):
        with self.assertRaises(ValueError):
            ft8_pack_message("W4ABC")


# ═══════════════════════════════════════════════════════════════════════════════
# § 5  CRC-14
# ═══════════════════════════════════════════════════════════════════════════════

class TestCrc14(unittest.TestCase):

    def test_append_crc_length(self):
        bits = ft8_pack_message("CQ W4ABC EN52")
        a91  = ft8_append_crc(bits)
        self.assertEqual(a91.shape, (91,))

    def test_crc_bits_are_binary(self):
        bits = ft8_pack_message("CQ W4ABC EN52")
        a91  = ft8_append_crc(bits)
        unique = set(int(b) for b in a91[77:])
        self.assertTrue(unique.issubset({0, 1}))

    def test_crc_matches_ft8_crc14(self):
        """CRC appended by ft8_append_crc must match _ft8_crc14 on the same bits."""
        bits = ft8_pack_message("W4ABC K9XYZ -05")
        a91  = ft8_append_crc(bits)
        crc_from_append = int(np.dot(a91[77:], [1 << (13 - i) for i in range(14)]))
        crc_direct      = _ft8_crc14(bits)
        self.assertEqual(crc_from_append, crc_direct)

    def test_different_messages_give_different_crc(self):
        bits1 = ft8_pack_message("CQ W4ABC EN52")
        bits2 = ft8_pack_message("CQ K9XYZ EM73")
        crc1  = _ft8_crc14(bits1)
        crc2  = _ft8_crc14(bits2)
        self.assertNotEqual(crc1, crc2)


# ═══════════════════════════════════════════════════════════════════════════════
# § 6  LDPC encoding
# ═══════════════════════════════════════════════════════════════════════════════

class TestLdpcEncode(unittest.TestCase):

    def _random_systematic(self, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return rng.integers(0, 2, size=91, dtype=np.uint8)

    def test_output_shape(self):
        s  = self._random_systematic()
        cw = ft8_ldpc_encode(s)
        self.assertEqual(cw.shape, (174,))

    def test_systematic_bits_preserved(self):
        s  = self._random_systematic(42)
        cw = ft8_ldpc_encode(s)
        np.testing.assert_array_equal(cw[:91], s)

    def test_parity_checks_satisfied_random(self):
        """All 83 parity checks must be satisfied for any random systematic bits."""
        for seed in range(10):
            s   = self._random_systematic(seed)
            cw  = ft8_ldpc_encode(s)
            err = ft8_ldpc_check(cw)
            self.assertEqual(err, 0,
                             f"Parity error with seed={seed}: {err} checks failed")

    def test_parity_checks_satisfied_real_message(self):
        """Parity checks must pass for a real FT8 message payload."""
        for msg in ("CQ W4ABC EN52", "W4ABC K9XYZ -05", "K9XYZ W4ABC RR73"):
            bits = ft8_pack_message(msg)
            a91  = ft8_append_crc(bits)
            cw   = ft8_ldpc_encode(a91)
            err  = ft8_ldpc_check(cw)
            self.assertEqual(err, 0, f"Parity error for {msg!r}: {err} failed")

    def test_encode_matrix_shape(self):
        self.assertEqual(_FT8_ENCODE_MATRIX.shape, (83, 91))

    def test_wrong_input_length_raises(self):
        with self.assertRaises(ValueError):
            ft8_ldpc_encode(np.zeros(90, dtype=np.uint8))


# ═══════════════════════════════════════════════════════════════════════════════
# § 7  Symbol generation
# ═══════════════════════════════════════════════════════════════════════════════

class TestSymbolGeneration(unittest.TestCase):

    def _symbols_for(self, msg: str) -> np.ndarray:
        bits = ft8_pack_message(msg)
        a91  = ft8_append_crc(bits)
        cw   = ft8_ldpc_encode(a91)
        return ft8_codeword_to_tones(cw)

    def test_output_shape(self):
        sym = self._symbols_for("CQ W4ABC EN52")
        self.assertEqual(sym.shape, (FT8_NSYMS,))

    def test_tone_values_in_range(self):
        sym = self._symbols_for("W4ABC K9XYZ -05")
        self.assertTrue(np.all(sym >= 0))
        self.assertTrue(np.all(sym <= 7))

    def test_costas_block_1_correct(self):
        """Symbols 0–6 must match the Costas pattern {3,1,4,0,6,5,2}."""
        sym = self._symbols_for("CQ W4ABC EN52")
        for i in range(7):
            self.assertEqual(int(sym[i]), FT8_COSTAS_TONES[i],
                             f"Costas block 1 mismatch at position {i}")

    def test_costas_block_2_correct(self):
        """Symbols 36–42 must also match the Costas pattern."""
        sym = self._symbols_for("CQ W4ABC EN52")
        for i in range(7):
            self.assertEqual(int(sym[36 + i]), FT8_COSTAS_TONES[i],
                             f"Costas block 2 mismatch at position {36+i}")

    def test_costas_block_3_correct(self):
        """Symbols 72–78 must also match the Costas pattern."""
        sym = self._symbols_for("CQ W4ABC EN52")
        for i in range(7):
            self.assertEqual(int(sym[72 + i]), FT8_COSTAS_TONES[i],
                             f"Costas block 3 mismatch at position {72+i}")

    def test_wrong_input_length_raises(self):
        with self.assertRaises(ValueError):
            ft8_codeword_to_tones(np.zeros(173, dtype=np.uint8))

    def test_costas_all_21_correct(self):
        """All 21 Costas symbols must be correct across all three blocks."""
        sym = self._symbols_for("K9XYZ W4ABC RR73")
        for pos in FT8_COSTAS_POSITIONS:
            if pos < 7:
                expected = FT8_COSTAS_TONES[pos]
            elif pos < 43:
                expected = FT8_COSTAS_TONES[pos - 36]
            else:
                expected = FT8_COSTAS_TONES[pos - 72]
            self.assertEqual(int(sym[pos]), expected,
                             f"Costas mismatch at symbol position {pos}")


# ═══════════════════════════════════════════════════════════════════════════════
# § 8  Audio synthesis
# ═══════════════════════════════════════════════════════════════════════════════

class TestAudioSynthesis(unittest.TestCase):

    def setUp(self) -> None:
        self.symbols = ft8_encode_to_symbols("CQ W4ABC EN52")

    def test_output_length(self):
        """Audio must be exactly FT8_NSYMS * sym_samples long."""
        audio    = ft8_symbols_to_audio(self.symbols, fs=FT8_FS)
        sym_n    = int(round(FT8_SYMBOL_DURATION_S * FT8_FS))
        expected = FT8_NSYMS * sym_n
        self.assertEqual(len(audio), expected)

    def test_output_dtype_float32(self):
        audio = ft8_symbols_to_audio(self.symbols)
        self.assertEqual(audio.dtype, np.float32)

    def test_amplitude_does_not_exceed_limit(self):
        """Peak amplitude must not exceed the requested value."""
        amp   = 0.5
        audio = ft8_symbols_to_audio(self.symbols, amplitude=amp, ramp_samples=0)
        self.assertLessEqual(float(np.max(np.abs(audio))), amp + 1e-6)

    def test_ramp_starts_near_zero(self):
        """With key-click suppression the first sample should be very small."""
        audio = ft8_symbols_to_audio(self.symbols, amplitude=1.0, ramp_samples=64)
        self.assertAlmostEqual(float(audio[0]), 0.0, places=4)

    def test_wrong_symbol_count_raises(self):
        with self.assertRaises(ValueError):
            ft8_symbols_to_audio(np.zeros(78, dtype=np.uint8))

    def test_non_default_sample_rate(self):
        """Audio generated at 48 kHz must be proportionally longer."""
        fs_48k = 48_000
        audio  = ft8_symbols_to_audio(self.symbols, fs=fs_48k)
        sym_n  = int(round(FT8_SYMBOL_DURATION_S * fs_48k))
        self.assertEqual(len(audio), FT8_NSYMS * sym_n)

    def test_dominant_frequency_is_correct(self):
        """
        The first symbol uses Costas tone 3 → frequency = f0 + 3 * 6.25 Hz.
        An FFT peak should fall near this frequency.
        """
        f0    = 1000.0
        fs    = FT8_FS
        sym_n = int(round(FT8_SYMBOL_DURATION_S * fs))
        audio = ft8_symbols_to_audio(self.symbols, f0_hz=f0, fs=fs,
                                     amplitude=1.0, ramp_samples=0)
        # Take FFT of just the first symbol
        first_sym = audio[:sym_n].astype(np.float64)
        fft_mag   = np.abs(np.fft.rfft(first_sym))
        freqs     = np.fft.rfftfreq(sym_n, d=1.0 / fs)
        peak_freq = freqs[np.argmax(fft_mag)]
        # Costas tone 3 → f0 + 3 * 6.25 = 1018.75 Hz
        expected_f = f0 + FT8_COSTAS_TONES[0] * 6.25
        self.assertAlmostEqual(peak_freq, expected_f, delta=7.0,
                               msg=f"FFT peak {peak_freq:.2f} Hz not near {expected_f:.2f} Hz")


# ═══════════════════════════════════════════════════════════════════════════════
# § 9  Encode → Decode round-trip (FT8 compliance)
# ═══════════════════════════════════════════════════════════════════════════════

class TestEncodeDecodeRoundTrip(unittest.TestCase):
    """
    Verifies that messages encoded by ft8_encode.py are correctly decoded by
    the existing ft8_decode.py pipeline — the key FT8 compliance check.
    """

    def _decode_audio(self, audio: np.ndarray, f0: float) -> str | None:
        """
        Run the full receive pipeline on *audio* and return the decoded message
        string, or None if decoding fails.
        """
        # Pad to a 15-second frame starting at t=0
        frame = np.zeros(int(15 * FT8_FS), dtype=np.float32)
        n = min(len(audio), len(frame))
        frame[:n] = audio[:n]

        extractor = FT8SymbolEnergyExtractor(fs=FT8_FS)
        E79 = extractor.extract_all_79(frame, t0_s=0.0, f0_hz=f0)

        # Hard-decision symbols for all 79 positions
        hard_all = np.argmax(E79, axis=1).astype(np.int32)

        # Extract payload energies and hard symbols
        E_payload, _  = ft8_extract_payload_symbols(E79, shift=0, inverted=False)
        hard_payload  = hard_all[list(FT8_PAYLOAD_POSITIONS)]

        hard_bits, llrs = ft8_gray_decode(hard_payload, E_payload)
        success, payload, _iters, _errs = ft8_ldpc_decode(llrs)
        if not success:
            return None
        return ft8_unpack_message(payload[:77])

    def _assert_round_trip(self, msg: str) -> None:
        symbols    = ft8_encode_to_symbols(msg)
        f0         = 1000.0
        audio      = ft8_symbols_to_audio(symbols, f0_hz=f0, fs=FT8_FS,
                                          amplitude=0.5, ramp_samples=0)
        decoded    = self._decode_audio(audio, f0)
        self.assertEqual(decoded, msg,
                         f"Round-trip mismatch for {msg!r}: decoded {decoded!r}")

    def test_cq_message(self):
        self._assert_round_trip("CQ W4ABC EN52")

    def test_cq_dx_callsign(self):
        self._assert_round_trip("CQ VK2TIM QF56")

    def test_reply_with_negative_snr(self):
        self._assert_round_trip("W4ABC K9XYZ -05")

    def test_reply_with_positive_snr(self):
        self._assert_round_trip("W4ABC K9XYZ +12")

    def test_exchange_r_prefix(self):
        self._assert_round_trip("K9XYZ W4ABC R-07")

    def test_rr73_message(self):
        self._assert_round_trip("W4ABC K9XYZ RR73")

    def test_rrr_message(self):
        self._assert_round_trip("W4ABC K9XYZ RRR")

    def test_73_message(self):
        self._assert_round_trip("K9XYZ W4ABC 73")

    def test_full_encode_message_helper(self):
        """ft8_encode_message should produce audio that decodes correctly."""
        msg   = "CQ K9XYZ EM73"
        f0    = 1500.0
        audio = ft8_encode_message(msg, f0_hz=f0, fs=FT8_FS)
        self.assertEqual(audio.dtype, np.float32)
        decoded = self._decode_audio(audio, f0)
        self.assertEqual(decoded, msg)


# ═══════════════════════════════════════════════════════════════════════════════
# § 10  OperatorConfig
# ═══════════════════════════════════════════════════════════════════════════════

class TestOperatorConfig(unittest.TestCase):

    def test_valid_callsign_accepted(self):
        op = OperatorConfig()
        op.callsign = "W4ABC"
        self.assertEqual(op.callsign, "W4ABC")

    def test_callsign_normalised_to_uppercase(self):
        op = OperatorConfig()
        op.callsign = "w4abc"
        self.assertEqual(op.callsign, "W4ABC")

    def test_invalid_callsign_raises(self):
        op = OperatorConfig()
        with self.assertRaises(ValueError):
            op.callsign = "INVALID"

    def test_valid_grid_accepted(self):
        op = OperatorConfig()
        op.grid = "en52"
        self.assertEqual(op.grid, "EN52")

    def test_invalid_grid_raises(self):
        op = OperatorConfig()
        with self.assertRaises(ValueError):
            op.grid = "ZZ99"   # Z not in A-R

    def test_empty_grid_accepted(self):
        op = OperatorConfig()
        op.grid = ""           # blank is allowed (not yet configured)
        self.assertEqual(op.grid, "")

    def test_is_configured_false_when_empty(self):
        op = OperatorConfig()
        self.assertFalse(op.is_configured())

    def test_is_configured_false_when_only_callsign(self):
        op = OperatorConfig(callsign="W4ABC")
        self.assertFalse(op.is_configured())

    def test_is_configured_true_when_both_set(self):
        op = OperatorConfig(callsign="W4ABC", grid="EN52")
        self.assertTrue(op.is_configured())

    def test_constructor_with_both(self):
        op = OperatorConfig(callsign="K9XYZ", grid="EM73")
        self.assertEqual(op.callsign, "K9XYZ")
        self.assertEqual(op.grid, "EM73")

    def test_repr(self):
        op = OperatorConfig(callsign="W4ABC", grid="EN52")
        r  = repr(op)
        self.assertIn("W4ABC", r)
        self.assertIn("EN52", r)


# ═══════════════════════════════════════════════════════════════════════════════
# § 11  Message composition helpers
# ═══════════════════════════════════════════════════════════════════════════════

class TestMessageComposition(unittest.TestCase):

    def test_compose_cq(self):
        self.assertEqual(compose_cq("W4ABC", "EN52"), "CQ W4ABC EN52")

    def test_compose_cq_uppercase(self):
        self.assertEqual(compose_cq("w4abc", "en52"), "CQ W4ABC EN52")

    def test_compose_reply_negative_snr(self):
        self.assertEqual(compose_reply("W4ABC", "K9XYZ", -5), "W4ABC K9XYZ -05")

    def test_compose_reply_positive_snr(self):
        self.assertEqual(compose_reply("W4ABC", "K9XYZ", +12), "W4ABC K9XYZ +12")

    def test_compose_exchange(self):
        self.assertEqual(compose_exchange("K9XYZ", "W4ABC", -7), "K9XYZ W4ABC R-07")

    def test_compose_rrr(self):
        self.assertEqual(compose_rrr("W4ABC", "K9XYZ"), "W4ABC K9XYZ RRR")

    def test_compose_rr73(self):
        self.assertEqual(compose_rr73("W4ABC", "K9XYZ"), "W4ABC K9XYZ RR73")

    def test_compose_73(self):
        self.assertEqual(compose_73("K9XYZ", "W4ABC"), "K9XYZ W4ABC 73")


# ═══════════════════════════════════════════════════════════════════════════════
# § 12  ReceivedMessage parser
# ═══════════════════════════════════════════════════════════════════════════════

class TestReceivedMessage(unittest.TestCase):

    def test_cq_message_parsed(self):
        rx = ReceivedMessage("CQ W4ABC EN52")
        self.assertEqual(rx.call1, "CQ")
        self.assertEqual(rx.call2, "W4ABC")
        self.assertEqual(rx.extra, "EN52")
        self.assertTrue(rx.is_cq)

    def test_snr_report_parsed(self):
        rx = ReceivedMessage("W4ABC K9XYZ -05")
        self.assertFalse(rx.is_cq)
        self.assertEqual(rx.snr_db, -5)

    def test_r_snr_report_parsed(self):
        rx = ReceivedMessage("K9XYZ W4ABC R-07")
        self.assertTrue(rx.is_r_report)
        self.assertEqual(rx.snr_db, -7)

    def test_rrr_flag(self):
        rx = ReceivedMessage("W4ABC K9XYZ RRR")
        self.assertTrue(rx.is_rrr)
        self.assertFalse(rx.is_rr73)
        self.assertFalse(rx.is_73)

    def test_rr73_flag(self):
        rx = ReceivedMessage("W4ABC K9XYZ RR73")
        self.assertFalse(rx.is_rrr)
        self.assertTrue(rx.is_rr73)

    def test_73_flag(self):
        rx = ReceivedMessage("K9XYZ W4ABC 73")
        self.assertTrue(rx.is_73)

    def test_is_addressed_to(self):
        rx = ReceivedMessage("W4ABC K9XYZ -05")
        self.assertTrue(rx.is_addressed_to("W4ABC"))
        self.assertFalse(rx.is_addressed_to("K9XYZ"))

    def test_is_from(self):
        rx = ReceivedMessage("W4ABC K9XYZ -05")
        self.assertTrue(rx.is_from("K9XYZ"))
        self.assertFalse(rx.is_from("W4ABC"))

    def test_case_insensitive_matching(self):
        rx = ReceivedMessage("W4ABC K9XYZ -05")
        self.assertTrue(rx.is_addressed_to("w4abc"))
        self.assertTrue(rx.is_from("k9xyz"))

    def test_two_field_message(self):
        rx = ReceivedMessage("W4ABC K9XYZ")
        self.assertEqual(rx.call1, "W4ABC")
        self.assertEqual(rx.call2, "K9XYZ")
        self.assertEqual(rx.extra, "")


# ═══════════════════════════════════════════════════════════════════════════════
# § 13  QSO state machine
# ═══════════════════════════════════════════════════════════════════════════════

class TestQsoStateMachine(unittest.TestCase):

    def _make_manager(self, call: str = "W4ABC", grid: str = "EN52") -> Ft8QsoManager:
        op = OperatorConfig(callsign=call, grid=grid)
        return Ft8QsoManager(op)

    # -- Initial state --------------------------------------------------------

    def test_initial_state_is_idle(self):
        mgr = self._make_manager()
        self.assertEqual(mgr.state, QsoState.IDLE)

    def test_no_tx_queued_initially(self):
        mgr = self._make_manager()
        self.assertIsNone(mgr.get_queued_tx())

    # -- Calling CQ -----------------------------------------------------------

    def test_start_cq_produces_correct_message(self):
        mgr = self._make_manager("W4ABC", "EN52")
        msg = mgr.start_cq()
        self.assertEqual(msg, "CQ W4ABC EN52")

    def test_start_cq_sets_state(self):
        mgr = self._make_manager()
        mgr.start_cq()
        self.assertEqual(mgr.state, QsoState.CQ_SENT)

    def test_start_cq_queues_tx(self):
        mgr = self._make_manager("W4ABC", "EN52")
        mgr.start_cq()
        self.assertEqual(mgr.get_queued_tx(), "CQ W4ABC EN52")

    def test_cq_without_callsign_raises(self):
        mgr = Ft8QsoManager(OperatorConfig())
        with self.assertRaises(RuntimeError):
            mgr.start_cq()

    def test_cq_without_grid_raises(self):
        op = OperatorConfig()
        op.callsign = "W4ABC"
        mgr = Ft8QsoManager(op)
        with self.assertRaises(RuntimeError):
            mgr.start_cq()

    def test_advance_after_cq_produces_exchange(self):
        """After CQ, a reply from K9XYZ should trigger our exchange message."""
        mgr = self._make_manager("W4ABC", "EN52")
        mgr.start_cq()
        msg = mgr.advance("W4ABC K9XYZ -05", snr_db=0)
        self.assertIsNotNone(msg)
        self.assertIn("K9XYZ", msg)
        self.assertIn("W4ABC", msg)
        self.assertIn("R", msg)   # exchange has R-prefix
        self.assertEqual(mgr.state, QsoState.EXCHANGE_SENT)

    def test_advance_after_exchange_produces_73(self):
        """After our exchange, RR73 from DX should trigger our 73."""
        mgr = self._make_manager("W4ABC", "EN52")
        mgr.start_cq()
        mgr.advance("W4ABC K9XYZ -05", snr_db=0)
        # K9XYZ addresses W4ABC (our call is first, DX is second)
        msg = mgr.advance("W4ABC K9XYZ RR73")
        self.assertEqual(msg, "K9XYZ W4ABC 73")
        self.assertEqual(mgr.state, QsoState.COMPLETE)

    def test_advance_after_exchange_with_rrr_produces_73(self):
        """RRR (not RR73) should also trigger our 73 response."""
        mgr = self._make_manager("W4ABC", "EN52")
        mgr.start_cq()
        mgr.advance("W4ABC K9XYZ -05")
        # K9XYZ addresses W4ABC (our call is first, DX is second)
        msg = mgr.advance("W4ABC K9XYZ RRR")
        self.assertEqual(msg, "K9XYZ W4ABC 73")
        self.assertEqual(mgr.state, QsoState.COMPLETE)

    # -- Answering a CQ -------------------------------------------------------

    def test_start_from_received_cq(self):
        """Selecting a CQ message should produce a reply."""
        mgr = self._make_manager("K9XYZ", "EM73")
        msg = mgr.start_from_received("CQ W4ABC EN52", snr_db=-5)
        self.assertEqual(msg, "W4ABC K9XYZ -05")
        self.assertEqual(mgr.state, QsoState.REPLY_SENT)

    def test_advance_from_reply_sent_on_exchange(self):
        """After replying to a CQ, receiving their exchange should trigger RR73."""
        mgr = self._make_manager("K9XYZ", "EM73")
        mgr.start_from_received("CQ W4ABC EN52", snr_db=-5)
        msg = mgr.advance("K9XYZ W4ABC R-07")
        self.assertEqual(msg, "W4ABC K9XYZ RR73")
        self.assertEqual(mgr.state, QsoState.RRR_SENT)

    def test_advance_from_rrr_sent_on_73(self):
        """After sending RR73, receiving their 73 should complete the QSO."""
        mgr = self._make_manager("K9XYZ", "EM73")
        mgr.start_from_received("CQ W4ABC EN52", snr_db=-5)
        mgr.advance("K9XYZ W4ABC R-07")
        # W4ABC addresses K9XYZ (our call first, their call second)
        mgr.advance("K9XYZ W4ABC 73")
        self.assertEqual(mgr.state, QsoState.COMPLETE)

    # -- Utility methods ------------------------------------------------------

    def test_is_active_after_cq(self):
        mgr = self._make_manager()
        mgr.start_cq()
        self.assertTrue(mgr.is_active)

    def test_is_active_false_when_idle(self):
        self.assertFalse(self._make_manager().is_active)

    def test_is_active_false_when_complete(self):
        mgr = self._make_manager("W4ABC", "EN52")
        mgr.start_cq()
        mgr.advance("W4ABC K9XYZ -05")
        # K9XYZ addresses W4ABC (our call is first)
        mgr.advance("W4ABC K9XYZ RR73")
        self.assertFalse(mgr.is_active)

    def test_abort_clears_state(self):
        mgr = self._make_manager()
        mgr.start_cq()
        mgr.abort()
        self.assertEqual(mgr.state, QsoState.ABORTED)
        self.assertIsNone(mgr.get_queued_tx())
        self.assertFalse(mgr.is_active)

    def test_reset_returns_to_idle(self):
        mgr = self._make_manager()
        mgr.start_cq()
        mgr.reset()
        self.assertEqual(mgr.state, QsoState.IDLE)
        self.assertIsNone(mgr.get_queued_tx())

    def test_get_queued_symbols_returns_ndarray(self):
        mgr = self._make_manager("W4ABC", "EN52")
        mgr.start_cq()
        sym = mgr.get_queued_symbols()
        self.assertIsNotNone(sym)
        self.assertEqual(sym.shape, (FT8_NSYMS,))

    def test_get_queued_symbols_none_when_idle(self):
        mgr = self._make_manager()
        self.assertIsNone(mgr.get_queued_symbols())

    def test_dx_call_tracked(self):
        mgr = self._make_manager("W4ABC", "EN52")
        mgr.start_cq()
        mgr.advance("W4ABC K9XYZ -05")
        self.assertEqual(mgr.dx_call, "K9XYZ")

    def test_message_not_addressed_to_us_ignored(self):
        """A message addressed to someone else should not advance our state."""
        mgr = self._make_manager("W4ABC", "EN52")
        mgr.start_cq()
        result = mgr.advance("VK2TIM G3ABC -10")   # not to us
        self.assertIsNone(result)
        self.assertEqual(mgr.state, QsoState.CQ_SENT)  # unchanged

    def test_clear_tx(self):
        mgr = self._make_manager("W4ABC", "EN52")
        mgr.start_cq()
        mgr.clear_tx()
        self.assertIsNone(mgr.get_queued_tx())


# ═══════════════════════════════════════════════════════════════════════════════
# § 14  AppConfig operator fields (integration with main.py)
# ═══════════════════════════════════════════════════════════════════════════════

class TestAppConfigOperator(unittest.TestCase):
    """Verify the callsign / grid fields added to AppConfig in main.py."""

    @classmethod
    def setUpClass(cls) -> None:
        """Import main.py once with a proper stub environment."""
        import types
        import unittest.mock as mock

        # Mirrors the _import_main_no_gui() pattern from test_main_gui_mode.py
        tk_stub = types.ModuleType("tkinter")
        tk_stub.Tk           = mock.MagicMock()
        tk_stub.Toplevel     = mock.MagicMock()
        tk_stub.Frame        = mock.MagicMock()
        tk_stub.LabelFrame   = mock.MagicMock()
        tk_stub.Label        = mock.MagicMock()
        tk_stub.Button       = mock.MagicMock()
        tk_stub.Entry        = mock.MagicMock()
        tk_stub.Text         = mock.MagicMock()
        tk_stub.Spinbox      = mock.MagicMock()
        tk_stub.StringVar    = mock.MagicMock
        tk_stub.IntVar       = mock.MagicMock
        tk_stub.NORMAL       = "normal"
        tk_stub.DISABLED     = "disabled"
        tk_stub.LEFT         = "left"
        tk_stub.RIGHT        = "right"
        tk_stub.X            = "x"
        tk_stub.Y            = "y"
        tk_stub.BOTH         = "both"
        tk_stub.END          = "end"
        tk_stub.SUNKEN       = "sunken"
        tk_stub.RAISED       = "raised"
        tk_stub.GROOVE       = "groove"
        tk_stub.Misc         = object

        ttk_stub          = types.ModuleType("tkinter.ttk")
        ttk_stub.Combobox = mock.MagicMock()
        ttk_stub.Progressbar = mock.MagicMock()
        ttk_stub.Scrollbar = mock.MagicMock()

        mb_stub           = types.ModuleType("tkinter.messagebox")
        mb_stub.showerror = mock.MagicMock()

        sd_stub = types.ModuleType("sounddevice")
        sd_stub.query_devices   = mock.MagicMock(return_value=[])
        sd_stub.query_hostapis  = mock.MagicMock(return_value=[])
        sd_stub.check_input_settings  = mock.MagicMock(side_effect=Exception)
        sd_stub.check_output_settings = mock.MagicMock(side_effect=Exception)

        serial_stub            = types.ModuleType("serial")
        serial_stub.STOPBITS_ONE            = 1
        serial_stub.STOPBITS_ONE_POINT_FIVE = 1.5
        serial_stub.STOPBITS_TWO            = 2
        serial_tools    = types.ModuleType("serial.tools")
        serial_tools_lp = types.ModuleType("serial.tools.list_ports")
        serial_tools_lp.comports = mock.MagicMock(return_value=[])

        import sys
        sys.modules.setdefault("tkinter",               tk_stub)
        sys.modules.setdefault("tkinter.ttk",            ttk_stub)
        sys.modules.setdefault("tkinter.messagebox",     mb_stub)
        sys.modules.setdefault("sounddevice",            sd_stub)
        sys.modules.setdefault("serial",                 serial_stub)
        sys.modules.setdefault("serial.tools",           serial_tools)
        sys.modules.setdefault("serial.tools.list_ports", serial_tools_lp)

        import importlib
        if "main" in sys.modules:
            del sys.modules["main"]
        import main as m
        cls._cfg_cls = m.AppConfig

    def setUp(self) -> None:
        import tempfile
        self._tmp = tempfile.NamedTemporaryFile(suffix=".cfg", delete=False)
        self._tmp.close()

    def tearDown(self) -> None:
        try:
            os.unlink(self._tmp.name)
        except OSError:
            pass

    def test_default_callsign_is_empty(self):
        cfg = self._cfg_cls(path=self._tmp.name)
        self.assertEqual(cfg.operator_callsign, "")

    def test_default_grid_is_empty(self):
        cfg = self._cfg_cls(path=self._tmp.name)
        self.assertEqual(cfg.operator_grid, "")

    def test_save_and_reload_callsign(self):
        cfg = self._cfg_cls(path=self._tmp.name)
        cfg.save_operator("W4ABC", "EN52")
        cfg2 = self._cfg_cls(path=self._tmp.name)
        self.assertEqual(cfg2.operator_callsign, "W4ABC")
        self.assertEqual(cfg2.operator_grid, "EN52")

    def test_callsign_saved_uppercase(self):
        cfg = self._cfg_cls(path=self._tmp.name)
        cfg.save_operator("w4abc", "en52")
        cfg2 = self._cfg_cls(path=self._tmp.name)
        self.assertEqual(cfg2.operator_callsign, "W4ABC")
        self.assertEqual(cfg2.operator_grid, "EN52")


if __name__ == "__main__":
    unittest.main()
