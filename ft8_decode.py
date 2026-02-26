from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import math
import time
import threading
import queue

import numpy as np
from scipy import signal

@dataclass(frozen=True)
class FT8DecodeResult:
    utc_time: str          # UTC slot time "HH:MM:SS" (start of 15s interval)
    strength_db: float     # relative strength estimate (dB), currently peak-vs-noise
    frequency_hz: float    # audio offset frequency estimate (Hz)
    message: str           # placeholder until full FT8 decode is implemented

@dataclass(frozen=True)
class FT8SyncCandidate:
    slot_utc: str
    time_offset_s: float
    freq_hz: float
    score_db: float

class PolyphaseResampler:
    """
    Streaming-friendly resampler using resample_poly in chunks.

    For FT8 we can accept small boundary artifacts; we keep it simple and fast.
    """
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
        y = signal.resample_poly(x, self.up, self.down).astype(np.float32, copy=False)
        return y

class UTC15sFramer:
    """
    Collects a continuous fs_proc stream and emits exact 15.0s frames aligned to UTC boundaries.

    The first frame emitted may be a partial slot (audio started mid-slot).
    push() returns (slot_start_utc_epoch, frame_samples, is_partial) tuples.
    Consumers should skip frames where is_partial=True.
    """
    def __init__(self, fs_proc: int, frame_s: float = 15.0) -> None:
        self.fs = int(fs_proc)
        self.frame_s = float(frame_s)
        self.frame_n = int(round(self.fs * self.frame_s))
        self._buf = np.zeros(0, dtype=np.float32)

        # We associate the buffer with an estimated UTC time for its first sample.
        self._t0_utc: Optional[float] = None  # epoch seconds

        # Track whether the first frame has been emitted yet (to flag it as partial).
        self._first_frame_emitted: bool = False

        # Stable monotonic->UTC mapping (smoothed)
        self._utc_minus_mono: Optional[float] = None
        self._alpha = 0.01  # smoothing factor; small = stable, large = more reactive

    @staticmethod
    def _utc_now_epoch() -> float:
        return datetime.now(timezone.utc).timestamp()

    @staticmethod
    def _slot_start_epoch(t_utc_epoch: float, slot_s: float = 15.0) -> float:
        return math.floor(t_utc_epoch / slot_s) * slot_s

    def _update_utc_minus_mono(self) -> float:
        """
        Maintain a smoothed estimate of (utc_epoch - monotonic_seconds).
        """
        utc_now = self._utc_now_epoch()
        mono_now = time.monotonic()
        sample = utc_now - mono_now

        if self._utc_minus_mono is None:
            self._utc_minus_mono = sample
        else:
            a = float(self._alpha)
            self._utc_minus_mono = (1.0 - a) * float(self._utc_minus_mono) + a * float(sample)

        return float(self._utc_minus_mono)

    def push(self, x: np.ndarray, *, t0_monotonic: float) -> list[tuple[float, np.ndarray, bool]]:
        """
        Push new fs_proc samples.
        Returns list of (slot_start_utc_epoch, frame_samples, is_partial).

        is_partial=True for the very first frame when audio collection started
        mid-slot.  The frame samples are time-shifted and the earlier portion
        of the FT8 transmission is missing, so sync scores will be poor.
        Consumers should skip partial frames.
        """
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        if x.size == 0:
            return []

        utc_minus_mono = self._update_utc_minus_mono()
        t0_utc = float(t0_monotonic) + float(utc_minus_mono)

        if self._t0_utc is None:
            self._t0_utc = t0_utc

        self._buf = np.concatenate((self._buf, x))

        out: list[tuple[float, np.ndarray, bool]] = []

        while self._buf.size >= self.frame_n and self._t0_utc is not None:
            slot0 = self._slot_start_epoch(self._t0_utc, self.frame_s)
            offset_s = self._t0_utc - slot0
            offset_n = int(round(offset_s * self.fs))

            if offset_n < 0:
                drop = min(-offset_n, self._buf.size)
                self._buf = self._buf[drop:]
                self._t0_utc += drop / self.fs
                continue

            if offset_n >= self.frame_n:
                advance_frames = offset_n // self.frame_n
                drop = advance_frames * self.frame_n
                if drop <= 0:
                    drop = min(offset_n, self._buf.size)
                self._buf = self._buf[drop:]
                self._t0_utc += drop / self.fs
                continue

            need = offset_n + self.frame_n
            if self._buf.size < need:
                break

            frame = self._buf[offset_n:offset_n + self.frame_n].copy()
            frame_start_utc = slot0

            # The very first frame is partial when audio started mid-slot
            # (offset_n > 0 means the slot boundary is before our first audio sample).
            is_partial = (not self._first_frame_emitted) and (offset_n > 0)
            self._first_frame_emitted = True

            out.append((frame_start_utc, frame, is_partial))

            drop = offset_n + self.frame_n
            self._buf = self._buf[drop:]
            self._t0_utc += drop / self.fs

        return out

def ft8_costas_margin_score(Ec: np.ndarray) -> tuple[float, tuple[float, float, float], int, int, int, bool]:
    """
    Returns:
      (mean_margin_db, (b0_db, b1_db, b2_db), matches, total, best_shift, inverted)

    b0/b1/b2 are the mean per-symbol margin in dB for each of the three Costas
    blocks (sum/7).  mean_margin_db is the mean across all 21 symbols.
    This makes values directly comparable to the sync score_db (~3-15 dB for
    real signals) and makes the block balance check meaningful.
    """
    costas_tones = (3, 1, 4, 0, 6, 5, 2)

    Ec = np.asarray(Ec, dtype=np.float64)
    matches, total, sh, inv = ft8_costas_ok_costasE(Ec)

    def sym_margin_db(row: np.ndarray, expected_tone: int) -> float:
        expv = float(row[expected_tone])
        others = np.delete(row, expected_tone)
        base = float(np.median(others))
        expv = max(expv, 1e-12)
        base = max(base, 1e-12)
        return 10.0 * math.log10(expv / base)

    block_margins = []
    for b in range(3):
        sym_margins = []
        for i in range(7):
            r = b * 7 + i
            expected = int((costas_tones[i] + sh) % 8)
            if inv:
                expected = 7 - expected
            sym_margins.append(sym_margin_db(Ec[r, :], expected))
        block_margins.append(float(np.mean(sym_margins)))   # mean per block

    mean_margin = float(np.mean(block_margins))
    return mean_margin, (block_margins[0], block_margins[1], block_margins[2]), int(matches), int(total), int(sh), bool(inv)

class FT8SyncSearch:
    """
    FT8 synchronization search using the three embedded 7-symbol Costas arrays.

    Two-stage frequency search:
    1. Coarse: waterfall-based, stepping by whole 6.25 Hz bins.  Fast.
    2. Fine:   extractor-based (arbitrary Hz), sub-bin resolution.  Accurate.

    The waterfall score is only meaningful at exact bin-centre frequencies
    (multiples of 6.25 Hz) because the rectangular-window DFT is orthogonal
    only at those points.  Scanning between bins with the waterfall returns the
    same integer-rounded bin energies, wasting time and masking the true peak.
    Fine resolution requires re-running the DFT at the exact frequency, which
    FT8SymbolEnergyExtractor does via its vectorised rfft + bin lookup.
    """
    COSTAS_TONES = (3, 1, 4, 0, 6, 5, 2)
    COSTAS_POS = (0, 36, 72)  # start indices of 7-symbol Costas arrays within 79 symbols

    def __init__(
        self,
        *,
        fs: int,
        fmin_hz: float = 200.0,
        fmax_hz: float = 3200.0,
        sym_s: float = 0.160,
        tone_spacing_hz: float = 6.25,
        extractor: "FT8SymbolEnergyExtractor | None" = None,
    ) -> None:
        self.fs = int(fs)
        self.fmin = float(fmin_hz)
        self.fmax = float(fmax_hz)
        self.sym_s = float(sym_s)
        self.tone_spacing_hz = float(tone_spacing_hz)
        self._extractor = extractor  # optional; enables sub-bin fine refinement

        # sym_n MUST equal fs / tone_spacing for orthogonality
        self.sym_n = int(round(float(fs) / float(tone_spacing_hz)))
        # Verify orthogonality
        expected_spacing = float(fs) / float(self.sym_n)
        if abs(expected_spacing - float(tone_spacing_hz)) > 0.01:
            raise ValueError(
                f"fs={fs} and tone_spacing_hz={tone_spacing_hz} are not orthogonal "
                f"(sym_n={self.sym_n}, actual spacing={expected_spacing:.4f} Hz)"
            )

    def _symbol_waterfall(self, x: np.ndarray, *, t0_offset_n: int = 0) -> tuple[np.ndarray, np.ndarray, int]:
        """
        Compute symbol-aligned power waterfall.

        Returns:
          freqs_hz   : shape (nfreqs,) — sliced in-band frequency axis (exact bin centres)
          pwr        : shape (nfreqs, nsymbols)
          lo         : absolute bin index of freqs_hz[0]
        """
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        sym_n = self.sym_n

        # Trim/pad so we start at the requested offset
        if t0_offset_n > 0:
            if t0_offset_n >= x.size:
                x = np.zeros(sym_n * 79, dtype=np.float32)
            else:
                x = x[t0_offset_n:]
        elif t0_offset_n < 0:
            x = np.concatenate((np.zeros(-t0_offset_n, dtype=np.float32), x))

        # Number of complete symbols available
        n_syms = x.size // sym_n
        if n_syms < 1:
            return np.zeros(0, dtype=np.float64), np.zeros((0, 0), dtype=np.float32), 0

        # Reshape to (n_syms, sym_n)
        x_mat = x[: n_syms * sym_n].reshape(n_syms, sym_n).astype(np.float64)

        # FFT each symbol row — rectangular window, full sym_n
        F = np.fft.rfft(x_mat, n=sym_n, axis=1)   # shape (n_syms, sym_n//2 + 1)
        pwr = (F.real * F.real + F.imag * F.imag).astype(np.float32)  # (n_syms, freq_bins)

        freqs_hz = np.fft.rfftfreq(sym_n, d=1.0 / float(self.fs))   # (freq_bins,)

        lo = int(np.searchsorted(freqs_hz, self.fmin, side="left"))
        hi = int(np.searchsorted(freqs_hz, self.fmax, side="right"))
        if hi <= lo + 2:
            return freqs_hz, np.zeros((0, n_syms), dtype=np.float32), 0

        # Return as (freq_bins, n_syms) to match convention
        return freqs_hz[lo:hi], pwr[:, lo:hi].T, lo

    def _costas_score_waterfall(
        self,
        pwr: np.ndarray,
        *,
        lo_bin: int,
        f0_bin: int,
    ) -> float:
        """
        Costas score using waterfall bins.  f0_bin is the ABSOLUTE DFT bin
        index for tone 0 (i.e. f0_hz / tone_spacing_hz, rounded to integer).
        pwr rows are indexed relative to lo_bin, so the row for absolute bin b
        is pwr[b - lo_bin, sym_idx].

        This must only be called with f0_bin values that are exact integers,
        i.e. f0 is a multiple of tone_spacing_hz.
        """
        if pwr.size == 0:
            return float("-inf")

        n_freq_bins = pwr.shape[0]
        n_syms = pwr.shape[1]

        # Check all 8 tone bins are within the sliced pwr array
        for k in range(8):
            b_rel = (f0_bin + k) - lo_bin
            if b_rel < 0 or b_rel >= n_freq_bins:
                return float("-inf")

        sym_scores: list[float] = []
        tone_energies = np.empty(8, dtype=np.float64)

        for pos0 in self.COSTAS_POS:
            for i, tone in enumerate(self.COSTAS_TONES):
                sym_idx = pos0 + i  # sym_offset is always 0 here (frame is pre-shifted)
                if sym_idx >= n_syms:
                    continue

                for k in range(8):
                    b_rel = (f0_bin + k) - lo_bin
                    tone_energies[k] = float(pwr[b_rel, sym_idx])

                e_exp = tone_energies[tone]
                e_noise = float(np.median(np.delete(tone_energies, tone)))
                e_exp   = max(e_exp,   1e-12)
                e_noise = max(e_noise, 1e-12)
                sym_scores.append(10.0 * math.log10(e_exp / e_noise))

        if len(sym_scores) < 8:
            return float("-inf")

        return float(np.mean(sym_scores))

    def _costas_score_extractor(
        self,
        frame: np.ndarray,
        *,
        t0_s: float,
        f0_hz: float,
        x_mat: "np.ndarray | None" = None,
    ) -> float:
        """
        Costas score using exact-frequency DFT (not bin-rounded).
        Pass a precomputed x_mat to avoid re-extracting segments on repeated calls
        with the same t0_s but different f0_hz.
        """
        if self._extractor is None:
            return float("-inf")
        if x_mat is None:
            Ec = self._extractor.extract_costas(frame, t0_s=t0_s, f0_hz=f0_hz)
        else:
            Ec = self._extractor.score_costas_from_xmat(x_mat, f0_hz=f0_hz)
        costas_tones = self.COSTAS_TONES
        sym_scores = np.empty(21, dtype=np.float64)
        for r in range(21):
            row   = Ec[r, :]
            tone  = costas_tones[r % 7]
            e_exp   = max(float(row[tone]),                        1e-12)
            e_noise = max(float(np.median(np.delete(row, tone))),  1e-12)
            sym_scores[r] = 10.0 * math.log10(e_exp / e_noise)
        return float(np.mean(sym_scores))

    @staticmethod
    def _dedupe_candidates(
            cands: list[tuple[float, float, float]],
            *,
            time_bucket_s: float = 0.16,
            freq_bucket_hz: float = 6.25,
    ) -> list[tuple[float, float, float]]:
        """Deduplicate by bucketing; keep best score per bucket."""
        best: dict[tuple[int, int], tuple[float, float, float]] = {}
        for t0, f0, sc in cands:
            kt = int(round(t0 / time_bucket_s))
            kf = int(round(f0 / freq_bucket_hz))
            key = (kt, kf)
            prev = best.get(key)
            if prev is None or sc > prev[2]:
                best[key] = (t0, f0, sc)
        out = list(best.values())
        out.sort(key=lambda x: x[2], reverse=True)
        return out

    def search(
        self,
        frame: np.ndarray,
        *,
        seed_freqs_hz: list[float],
        time_search_s: float = 0.8,
        freq_search_bins: int = 4,
        max_candidates: int = 8,
    ) -> list[tuple[float, float, float]]:
        """
        Returns list of (best_t0_s, best_f0_hz, score_db), sorted by score desc.

        Coarse stage:
          - Steps by whole bins (6.25 Hz) using the pre-computed waterfall.
          - Time steps by one symbol period (160 ms) over ±time_search_s.
          - freq_search_bins: ± how many bins to search around each seed.

        Fine stage (only if extractor is available):
          - 2-D sweep: t0 in ±80 ms steps of 10 ms AND f0 in ±4 Hz steps of 0.05 Hz.
          - Uses FT8SymbolEnergyExtractor (exact-frequency DFT, sub-bin resolution).
          - x_mat extracted once per t0 candidate since the f0 inner loop is cheap.
          - Corrects both sub-symbol timing errors and sub-bin frequency errors.
        """
        frame = np.asarray(frame, dtype=np.float32).reshape(-1)
        if not seed_freqs_hz:
            return []

        spacing = self.tone_spacing_hz  # 6.25
        max_sym_offset = max(1, int(math.ceil(time_search_s / self.sym_s)))
        sym_offsets = list(range(-max_sym_offset, max_sym_offset + 1))

        # Build waterfall cache: sym_off -> (freqs, pwr, lo_bin)
        wf_cache: dict[int, tuple[np.ndarray, np.ndarray, int]] = {}
        for sym_off in sym_offsets:
            freqs, pwr, lo = self._symbol_waterfall(frame, t0_offset_n=sym_off * self.sym_n)
            if pwr.size > 0:
                wf_cache[sym_off] = (freqs, pwr, lo)

        # Per-seed tracking
        n_seeds = len(seed_freqs_hz)
        best_t0  = [0.0]           * n_seeds
        best_f0  = list(seed_freqs_hz)
        best_sc  = [float("-inf")] * n_seeds

        for sym_off, (freqs, pwr, lo_bin) in wf_cache.items():
            t0_candidate = float(sym_off) * self.sym_s

            for si, seed in enumerate(seed_freqs_hz):
                # Snap seed to nearest bin, then search ± freq_search_bins bins
                seed_bin = int(round(float(seed) / spacing))
                for db in range(-freq_search_bins, freq_search_bins + 1):
                    f0_bin = seed_bin + db
                    f0_hz  = float(f0_bin) * spacing
                    if f0_hz < self.fmin or f0_hz + 7 * spacing > self.fmax:
                        continue
                    sc = self._costas_score_waterfall(pwr, lo_bin=lo_bin, f0_bin=f0_bin)
                    if sc > best_sc[si]:
                        best_sc[si] = sc
                        best_t0[si] = t0_candidate
                        best_f0[si] = f0_hz

        # Fine 2-D refinement: sweep both t0 (±half-symbol in 10 ms steps) and
        # f0 (±4 Hz in 0.05 Hz steps) using exact-frequency DFT.
        # For each t0 candidate the x_mat is extracted once; all f0 values are
        # evaluated in one batched matrix multiply via score_costas_batch().
        if self._extractor is not None:
            fine_f_step  = 0.05   # Hz
            fine_f_half  = 4.0    # ± Hz — covers full bin ± margin
            fine_t_step  = 0.010  # s  (10 ms — 16× finer than coarse 160 ms step)
            fine_t_half  = self.sym_s / 2.0   # ±80 ms — covers full inter-step gap

            for si in range(n_seeds):
                coarse_t0 = best_t0[si]
                coarse_f0 = best_f0[si]

                # Determine best shift/inverted from one exact-DFT eval at coarse peak
                x_mat_coarse = self._extractor._get_costas_xmat(frame, t0_s=coarse_t0)
                Ec_coarse = self._extractor.score_costas_from_xmat(x_mat_coarse, f0_hz=coarse_f0)
                _m, _tot, coarse_shift, coarse_inv = ft8_costas_ok_costasE(Ec_coarse)

                t_fine = np.arange(
                    coarse_t0 - fine_t_half,
                    coarse_t0 + fine_t_half + 1e-9,
                    fine_t_step,
                    dtype=np.float64,
                )
                f_fine = np.arange(
                    coarse_f0 - fine_f_half,
                    coarse_f0 + fine_f_half + 1e-9,
                    fine_f_step,
                    dtype=np.float64,
                )

                for t0_cand in t_fine:
                    # Extract segment matrix once per t0 candidate
                    x_mat = self._extractor._get_costas_xmat(frame, t0_s=float(t0_cand))
                    # Score all f0 values in one batched operation, using coarse shift/inv
                    scores_f = self._extractor.score_costas_batch(
                        x_mat, f_fine, shift=int(coarse_shift), inverted=bool(coarse_inv)
                    )  # (N_f,)
                    best_idx = int(np.argmax(scores_f))
                    sc = float(scores_f[best_idx])
                    if sc > best_sc[si]:
                        best_sc[si] = sc
                        best_t0[si] = float(t0_cand)
                        best_f0[si] = float(f_fine[best_idx])

        all_best = list(zip(best_t0, best_f0, best_sc))
        all_best = self._dedupe_candidates(all_best, time_bucket_s=self.sym_s, freq_bucket_hz=spacing)
        return all_best[:max_candidates]

class FT8SignalDetector:
    """
    FT8-structure-aware candidate detector.

    Rather than finding individual loud spectral peaks, this computes a
    matched-filter response for the FT8 8-tone structure: for each candidate
    base frequency f0, it sums the time-averaged power at the 8 tone bins
    (f0, f0+6.25, ..., f0+43.75 Hz) and normalises against the local noise
    floor.  This directly answers "is there an 8-tone FSK signal starting at
    f0?" rather than "is there a loud tone at f0?".

    The result is a candidate list of (f0_hz, score_db) values where f0 is the
    lowest tone of the FT8 signal, ready to be passed to FT8SyncSearch.

    Key facts exploited:
    - sym_n = fs / tone_spacing = 1920 at 12 kHz  →  each 6.25 Hz-wide bin
      corresponds to exactly one FT8 tone, zero inter-tone leakage.
    - An FT8 signal occupies exactly 8 consecutive bins.
    - The matched-filter score is sum(8 tone bins) / (8 * local_noise_per_bin).
    """

    # FT8 uses 8 tones; bin stride is 1 (each bin = one tone spacing)
    N_TONES = 8

    def __init__(
        self,
        *,
        fs: int,
        fmin_hz: float = 200.0,
        fmax_hz: float = 3200.0,
        peak_threshold_db: float = 3.0,
        max_peaks: int = 30,
        noise_bw_bins: int = 50,
    ) -> None:
        self.fs = int(fs)
        self.fmin = float(fmin_hz)
        self.fmax = float(fmax_hz)
        self.peak_threshold_db = float(peak_threshold_db)
        self.max_peaks = int(max_peaks)
        # Half-width (in bins) of the local noise estimation window.
        # Must be >> N_TONES so the signal itself doesn't inflate noise.
        self.noise_bw_bins = int(noise_bw_bins)

        # sym_n = fs / tone_spacing; at 12 kHz = 1920
        self.sym_n = int(round(float(fs) / 6.25))

    def _time_averaged_spectrum(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Time-averaged power spectrum with sym_n-point rectangular windows.
        Returns (freqs_hz, avg_pwr), shapes (sym_n//2+1,).
        """
        sym_n = self.sym_n
        n_syms = x.size // sym_n
        if n_syms < 1:
            return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)

        x_mat = x[: n_syms * sym_n].reshape(n_syms, sym_n).astype(np.float64)
        F = np.fft.rfft(x_mat, n=sym_n, axis=1)
        pwr = F.real ** 2 + F.imag ** 2          # (n_syms, freq_bins)
        avg_pwr = pwr.mean(axis=0)               # (freq_bins,)
        freqs_hz = np.fft.rfftfreq(sym_n, d=1.0 / float(self.fs))
        return freqs_hz, avg_pwr

    def _local_noise(self, avg_pwr: np.ndarray) -> np.ndarray:
        """
        Estimate local noise floor per bin using a running median over a wide
        window (noise_bw_bins bins on each side).  Uses uniform_filter1d on
        a sorted-percentile approximation via a sliding minimum+maximum blend
        — for simplicity we use scipy.ndimage-free approach: convolve with a
        uniform kernel to get a local mean, which is a reasonable noise proxy
        for a relatively flat noise floor.
        """
        hw = self.noise_bw_bins
        n = avg_pwr.size
        if n < 2 * hw + 1:
            return np.full(n, float(np.median(avg_pwr)), dtype=np.float64)

        # Cumulative sum trick for fast sliding mean
        cs = np.concatenate(([0.0], np.cumsum(avg_pwr)))
        # Clamp window to valid range at edges
        i = np.arange(n)
        lo = np.maximum(0, i - hw)
        hi = np.minimum(n, i + hw + 1)
        noise = (cs[hi] - cs[lo]) / (hi - lo).astype(np.float64)
        return noise

    def detect(self, x: np.ndarray) -> list[tuple[float, float]]:
        """Returns list of (f0_hz, score_db) candidates."""
        peaks, _best = self.detect_with_best_score(x)
        return peaks

    def detect_with_best_score(self, x: np.ndarray) -> tuple[list[tuple[float, float]], float]:
        """
        Returns (candidates, best_score_db).

        Each candidate is (f0_hz, score_db) where f0_hz is the lowest tone of
        a hypothesised FT8 signal and score_db is the 8-tone matched-filter SNR.
        """
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        if x.size < self.sym_n:
            return [], float("-inf")

        freqs_hz, avg_pwr = self._time_averaged_spectrum(x)
        if avg_pwr.size == 0:
            return [], float("-inf")

        lo = int(np.searchsorted(freqs_hz, self.fmin, side="left"))
        # Upper limit: f0 + 7*6.25 = f0 + 43.75 Hz must be <= fmax
        # So f0 <= fmax - 43.75; find the last valid f0 bin
        hi_f0 = float(self.fmax) - (self.N_TONES - 1) * 6.25
        hi = int(np.searchsorted(freqs_hz, hi_f0, side="right"))

        if hi <= lo + self.N_TONES:
            return [], float("-inf")

        full_pwr = avg_pwr  # shape (all_freq_bins,)
        noise_all = self._local_noise(full_pwr)

        # Matched filter: sum of N_TONES consecutive bins starting at bin b
        # score[b] = sum(pwr[b:b+N]) / (N * noise[b + N//2])
        # We compute this for all valid b in [lo, hi) using cumsum
        cs = np.concatenate(([0.0], np.cumsum(full_pwr)))
        n_bins = int(full_pwr.size)

        scores = np.full(hi - lo, float("-inf"), dtype=np.float64)
        for idx, b in enumerate(range(lo, hi)):
            b_end = b + self.N_TONES
            if b_end > n_bins:
                break
            signal_sum = float(cs[b_end] - cs[b])
            noise_ref = float(noise_all[b + self.N_TONES // 2])
            noise_ref = max(noise_ref, 1e-12)
            scores[idx] = 10.0 * math.log10(
                max(signal_sum / (self.N_TONES * noise_ref), 1e-12)
            )

        best_score_db = float(scores.max(initial=float("-inf")))

        # Find peaks in the matched-filter response.
        # Minimum separation = N_TONES bins (8 * 6.25 = 50 Hz) so we don't
        # return two f0 candidates for the same signal.
        peaks_idx, props = signal.find_peaks(
            scores,
            height=self.peak_threshold_db,
            distance=self.N_TONES,
        )
        if peaks_idx.size == 0:
            return [], best_score_db

        heights = props.get("peak_heights", scores[peaks_idx])
        order = np.argsort(heights)[::-1][: self.max_peaks]

        results: list[tuple[float, float]] = []
        for k in order:
            bi = int(peaks_idx[k])
            f0_hz = float(freqs_hz[lo + bi])
            results.append((f0_hz, float(heights[k])))

        return results, best_score_db

class FT8SymbolEnergyExtractor:
    """
    Compute per-symbol per-tone energies.

    For Costas/sync validation, we provide a Costas-only path that is much faster.
    """
    def __init__(self, *, fs: int, sym_s: float = 0.160, tone_spacing_hz: float = 6.25) -> None:
        self.fs = int(fs)
        self.sym_s = float(sym_s)
        self.tone_spacing_hz = float(tone_spacing_hz)
        self.sym_n = int(round(self.sym_s * self.fs))

    def _get_samples(self, frame: np.ndarray, *, t0_s: float, total_symbols: int) -> tuple[np.ndarray, int]:
        x = np.asarray(frame, dtype=np.float32).reshape(-1)

        t0_n = int(round(float(t0_s) * self.fs))
        if t0_n < 0:
            pad0 = -t0_n
            x = np.concatenate((np.zeros(pad0, dtype=np.float32), x))
            start = 0
        else:
            start = t0_n

        need = start + total_symbols * self.sym_n
        if need > x.size:
            x = np.concatenate((x, np.zeros(need - x.size, dtype=np.float32)))

        return x, start

    def _get_costas_xmat(self, frame: np.ndarray, *, t0_s: float) -> np.ndarray:
        """
        Extract the (21, sym_n) segment matrix for the Costas positions at t0_s.
        Separated so callers can cache it and reuse across frequency scans.
        """
        costas_pos = (0, 36, 72)
        x, start = self._get_samples(frame, t0_s=float(t0_s), total_symbols=79)
        sym_n = self.sym_n
        rows: list[np.ndarray] = []
        for p0 in costas_pos:
            for i in range(7):
                s = p0 + i
                rows.append(x[start + s * sym_n: start + (s + 1) * sym_n].astype(np.float64, copy=False))
        return np.stack(rows, axis=0)   # (21, sym_n)

    def score_costas_from_xmat(self, x_mat: np.ndarray, *, f0_hz: float) -> np.ndarray:
        """
        Given a precomputed (21, sym_n) segment matrix, return E_costas (21, 8)
        at the specified f0_hz using the DFT dot-product method.
        """
        sym_n = self.sym_n
        fs = float(self.fs)
        n_vec = np.arange(sym_n, dtype=np.float64)
        cos_mat = np.empty((8, sym_n), dtype=np.float64)
        sin_mat = np.empty((8, sym_n), dtype=np.float64)
        for k in range(8):
            f_k = float(f0_hz) + k * self.tone_spacing_hz
            phase = (2.0 * math.pi * f_k / fs) * n_vec
            cos_mat[k] = np.cos(phase)
            sin_mat[k] = np.sin(phase)
        I = x_mat @ cos_mat.T
        Q = x_mat @ sin_mat.T
        return I * I + Q * Q

    def score_costas_batch(
        self,
        x_mat: np.ndarray,
        f0_array: np.ndarray,
        *,
        shift: int = 0,
        inverted: bool = False,
    ) -> np.ndarray:
        """
        Vectorised scorer: evaluate the mean Costas margin score for every f0
        value in f0_array without a Python loop over frequencies.

        Parameters
        ----------
        x_mat    : (21, sym_n) segment matrix from _get_costas_xmat()
        f0_array : (N,) array of base frequencies in Hz
        shift    : tone-index shift [0..7] from coarse Costas check
        inverted : whether tones are inverted (k -> 7-k)

        Returns
        -------
        scores  : (N,) array of mean-margin-dB scores (same metric as
                  _costas_score_extractor)
        """
        sym_n = self.sym_n
        fs    = float(self.fs)
        spacing = float(self.tone_spacing_hz)
        N     = int(f0_array.size)

        n_vec = np.arange(sym_n, dtype=np.float32)   # (sym_n,)
        # phase[i, k] = 2*pi * (f0[i] + k*spacing) / fs
        f_mat = (f0_array[:, None] + np.arange(8, dtype=np.float64) * spacing).astype(np.float32)  # (N, 8)
        # phase_mat: (N, 8, sym_n)
        phase_mat = (2.0 * math.pi / fs) * f_mat[:, :, None] * n_vec[None, None, :]  # (N,8,sym_n)

        # DFT kernels: cos and sin, shape (N, 8, sym_n)
        cos_k = np.cos(phase_mat)   # (N, 8, sym_n)
        sin_k = np.sin(phase_mat)   # (N, 8, sym_n)

        # x_mat: (21, sym_n)  →  project onto each tone kernel
        # I[i, r, k] = x_mat[r] . cos_k[i, k]  =  einsum('rs, iks -> irk', x_mat, cos_k)
        # But einsum with 3 large indices is slow; use matmul reshape trick:
        #   x_mat @ cos_k[i].T  gives (21, 8) for each i
        # Batch: cos_k reshaped to (N*8, sym_n), matmul, reshape back
        cos_2d = cos_k.reshape(N * 8, sym_n)   # (N*8, sym_n)
        sin_2d = sin_k.reshape(N * 8, sym_n)

        I_2d = x_mat.astype(np.float32) @ cos_2d.T   # (21, N*8)
        Q_2d = x_mat.astype(np.float32) @ sin_2d.T   # (21, N*8)

        E_all = (I_2d * I_2d + Q_2d * Q_2d).reshape(21, N, 8)  # (21, N, 8)
        E_all = E_all.transpose(1, 0, 2)                        # (N, 21, 8)

        # Compute mean-margin score for each of the N candidates
        costas_tones = np.array([3, 1, 4, 0, 6, 5, 2], dtype=np.int32)
        base_expected = np.tile(costas_tones, 3)               # (21,) at shift=0
        expected = (base_expected + int(shift)) % 8
        if inverted:
            expected = 7 - expected

        # For each row r, margin = 10*log10(E[r, expected[r]] / median(E[r, others]))
        # Vectorise over r and N simultaneously
        E_exp    = E_all[:, np.arange(21), expected]  # (N, 21)
        # Median of the other 7 tones per row — use a masked sort
        # Build (N, 21, 7) by removing expected tone column per row
        mask = np.ones((21, 8), dtype=bool)
        mask[np.arange(21), expected] = False            # (21, 8) — True for non-expected
        E_others = E_all[:, mask.nonzero()[0], mask.nonzero()[1]].reshape(N, 21, 7)  # (N,21,7)
        noise    = np.median(E_others, axis=2)            # (N, 21)

        E_exp  = np.maximum(E_exp,  1e-12)
        noise  = np.maximum(noise,  1e-12)
        margin = 10.0 * np.log10(E_exp / noise)           # (N, 21)
        return margin.mean(axis=1)                         # (N,)

    def extract_costas(self, frame: np.ndarray, *, t0_s: float, f0_hz: float) -> np.ndarray:
        """
        Returns E_costas: shape (21, 8), energies for the 3 Costas arrays only.

        Uses DFT dot-products at exact tone frequencies (not bin-rounded), so
        off-grid signals are handled correctly.
        """
        x_mat = self._get_costas_xmat(frame, t0_s=t0_s)
        return self.score_costas_from_xmat(x_mat, f0_hz=f0_hz)

    def extract_all_79(self, frame: np.ndarray, *, t0_s: float, f0_hz: float) -> np.ndarray:
        """
        Extract per-symbol per-tone energies for ALL 79 FT8 symbols.

        Returns
        -------
        E : np.ndarray, shape (79, 8), dtype float64
            E[s, k] = DFT energy at tone k = f0_hz + k*6.25 Hz for symbol s.
            Uses exact-frequency DFT dot-products (same method as extract_costas),
            so sub-bin frequencies are handled correctly.

        Symbol layout (FT8 standard)
        -----------------------------
        Positions  0.. 6  — Costas block 0
        Positions  7..35  — payload symbols 0..28
        Positions 36..42  — Costas block 1
        Positions 43..71  — payload symbols 29..57
        Positions 72..78  — Costas block 2

        The caller can use FT8_COSTAS_POSITIONS to identify Costas rows and
        the complementary indices for payload rows.
        """
        x, start = self._get_samples(frame, t0_s=float(t0_s), total_symbols=79)
        sym_n = self.sym_n
        fs    = float(self.fs)

        # Build DFT kernels once for all 8 tones — shape (8, sym_n)
        n_vec = np.arange(sym_n, dtype=np.float64)
        cos_mat = np.empty((8, sym_n), dtype=np.float64)
        sin_mat = np.empty((8, sym_n), dtype=np.float64)
        for k in range(8):
            phase = (2.0 * math.pi * (float(f0_hz) + k * self.tone_spacing_hz) / fs) * n_vec
            cos_mat[k] = np.cos(phase)
            sin_mat[k] = np.sin(phase)

        # Extract all 79 symbol windows into a (79, sym_n) matrix
        x_all = np.empty((79, sym_n), dtype=np.float64)
        for s in range(79):
            x_all[s] = x[start + s * sym_n : start + (s + 1) * sym_n].astype(np.float64, copy=False)

        # Batch DFT: (79, sym_n) @ (sym_n, 8) → (79, 8) for I and Q
        I = x_all @ cos_mat.T   # (79, 8)
        Q = x_all @ sin_mat.T   # (79, 8)
        return (I * I + Q * Q).astype(np.float64)

def ft8_costas_ok_costasE(Ec: np.ndarray) -> tuple[int, int, int, bool]:
    """
    Costas check on Costas-only energy matrix Ec with shape (21, 8).

    Tries:
      - tone order normal
      - tone order inverted (k -> 7-k)
    plus an unknown tone-index shift in [0..7].

    Returns (matches, total, best_shift, inverted)
    """
    costas_tones = (3, 1, 4, 0, 6, 5, 2)

    Ec = np.asarray(Ec, dtype=np.float64)
    total = 21

    best_matches = -1
    best_shift = 0
    best_inverted = False

    for inverted in (False, True):
        for shift in range(8):
            matches = 0
            for r in range(21):
                expected = int((costas_tones[r % 7] + shift) % 8)
                if inverted:
                    expected = 7 - expected
                got = int(np.argmax(Ec[r, :]))
                if got == expected:
                    matches += 1

            if matches > best_matches:
                best_matches = matches
                best_shift = shift
                best_inverted = inverted

    return int(best_matches), int(total), int(best_shift), bool(best_inverted)

def ft8_costas_ok(E: np.ndarray) -> tuple[int, int, int]:
    """
    Costas check with unknown tone-index shift.

    Returns (matches, total, best_shift), where best_shift is in [0..7] and means:
      got_tone == (expected_tone + best_shift) % 8
    """
    costas_tones = (3, 1, 4, 0, 6, 5, 2)
    costas_pos = (0, 36, 72)

    E = np.asarray(E, dtype=np.float64)
    total = 21  # 3 arrays * 7 symbols

    best_matches = -1
    best_shift = 0

    for shift in range(8):
        matches = 0
        for p0 in costas_pos:
            for i, expected_tone in enumerate(costas_tones):
                s = p0 + i
                got = int(np.argmax(E[s, :]))
                exp = int((int(expected_tone) + shift) % 8)
                if got == exp:
                    matches += 1

        if matches > best_matches:
            best_matches = matches
            best_shift = shift

    return int(best_matches), int(total), int(best_shift)
def ft8_costas_rank_stats(Ec: np.ndarray) -> dict[str, float]:
    """
    Given Costas-only energies Ec shape (21, 8),
    compute stats about the rank of the expected tone (with best shift and inversion).
    """
    costas_tones = (3, 1, 4, 0, 6, 5, 2)

    Ec = np.asarray(Ec, dtype=np.float64)
    m, tot, sh, inv = ft8_costas_ok_costasE(Ec)

    ranks: list[int] = []
    margins_db: list[float] = []

    for r in range(21):
        expected = int((costas_tones[r % 7] + int(sh)) % 8)
        if inv:
            expected = 7 - expected

        row = Ec[r, :]

        order = np.argsort(row)[::-1]
        rank = int(np.where(order == expected)[0][0]) + 1
        ranks.append(rank)

        best = float(row[order[0]])
        expv = float(row[expected])
        best = max(best, 1e-12)
        expv = max(expv, 1e-12)
        margins_db.append(10.0 * math.log10(expv / best))

    return {
        "matches": float(m),
        "total": float(tot),
        "shift": float(sh),
        "inverted": float(bool(inv)),
        "rank_mean": float(np.mean(ranks)),
        "rank_median": float(np.median(ranks)),
        "margin_db_mean": float(np.mean(margins_db)),
        "margin_db_median": float(np.median(margins_db)),
    }

# ---------------------------------------------------------------------------
# FT8 bit-level interleaver (Stage 2b)
# ---------------------------------------------------------------------------
# The FT8 interleaver operates on the 174 channel BITS (not symbols).
# This is the standard 174-entry permutation table from ft8_lib / WSJT-X.
# _FT8_INTERLEAVE[i] = source bit index that goes to position i after interleaving.
# De-interleaving: out[i] = in[_FT8_INTERLEAVE[i]]
_FT8_INTERLEAVE: tuple[int, ...] = (
      0,  1,  2,  3,  4,  5,  6,  7,  8,  9,
     10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
     20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
     30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
     40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
     50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
     60, 61, 62, 63, 64, 65, 66, 67, 68, 69,
     70, 71, 72, 73, 74, 75, 76, 77, 78, 79,
     80, 81, 82, 83, 84, 85, 86, 87, 88, 89,
     90, 91, 92, 93, 94, 95, 96, 97, 98, 99,
    100,101,102,103,104,105,106,107,108,109,
    110,111,112,113,114,115,116,117,118,119,
    120,121,122,123,124,125,126,127,128,129,
    130,131,132,133,134,135,136,137,138,139,
    140,141,142,143,144,145,146,147,148,149,
    150,151,152,153,154,155,156,157,158,159,
    160,161,162,163,164,165,166,167,168,169,
    170,171,172,173,
)

def _build_ft8_interleave_table() -> tuple[int, ...]:
    """
    Compute the 174-bit FT8 interleave permutation from ft8_lib.

    The algorithm (from ft8_lib/ft8/encode.cpp interleave174()):
      - Uses a bit-reversal permutation on 8-bit indices (log2(256)=8, but
        we need 174 entries so we skip indices that map out of range).
      - Specifically: for i in 0..255, compute rev = bit_reverse_8(i);
        if both i < 174 and rev < 174, emit rev as the destination for source i.
    This produces a 174-entry table where perm[dst] = src.
    """
    def bit_rev8(x: int) -> int:
        x = ((x & 0xF0) >> 4) | ((x & 0x0F) << 4)
        x = ((x & 0xCC) >> 2) | ((x & 0x33) << 2)
        x = ((x & 0xAA) >> 1) | ((x & 0x55) << 1)
        return x & 0xFF

    perm = [-1] * 174
    for i in range(256):
        r = bit_rev8(i)
        if i < 174 and r < 174:
            perm[r] = i   # destination r gets source i
    # Any gaps (shouldn't be any for valid ft8_lib table) fill with identity
    for i in range(174):
        if perm[i] < 0:
            perm[i] = i
    return tuple(perm)

# Compute the true interleaver permutation at module load.
# _FT8_INTERLEAVE_PERM[dst] = src  (de-interleave: out[dst] = in[src])
_FT8_INTERLEAVE_PERM: tuple[int, ...] = _build_ft8_interleave_table()


def ft8_deinterleave(
    hard_syms: np.ndarray,
    E_payload: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Stage 2b — Pass-through: the FT8 interleaver operates on 174 *bits*,
    not on the 58 symbols.  Symbol-level reordering is therefore an identity
    here; the actual bit de-interleaving is applied inside ft8_gray_decode
    after Gray decoding.

    This function is kept for API compatibility and simply returns its inputs
    unchanged.
    """
    return np.asarray(hard_syms, dtype=np.int32), np.asarray(E_payload, dtype=np.float64)


def ft8_gray_decode(
    syms_deint: np.ndarray,
    E_deint: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Stage 3 — Gray-decode 58 payload symbols → 174 bits + LLRs, then
    apply the 174-bit de-interleaver.

    FT8 Gray code (tone → 3-bit value, MSB first):
        tone 0 → 000    tone 4 → 110
        tone 1 → 001    tone 5 → 111
        tone 2 → 011    tone 6 → 101
        tone 3 → 010    tone 7 → 100

    LLR formula (8-FSK log-likelihood ratio):
        LLR(bit b) = log(Σ E[tones where bit_b=1]) − log(Σ E[tones where bit_b=0])

    LLRs are normalized by the per-symbol noise floor so the LDPC decoder
    receives properly scaled soft values (mean |LLR| ~ 4-8 for clean signals).

    The 174-bit de-interleaver (_FT8_INTERLEAVE_PERM) is applied to both
    hard_bits and llrs before returning.
    """
    syms = np.asarray(syms_deint, dtype=np.int32)
    E    = np.asarray(E_deint,    dtype=np.float64)

    N_syms = 58
    N_bits = 174

    gray_table = np.array(_FT8_GRAY_DECODE, dtype=np.int32)

    hard_bits_raw = np.empty(N_bits, dtype=np.uint8)
    llrs_raw      = np.empty(N_bits, dtype=np.float64)

    E_safe = np.maximum(E, 1e-30)

    # Pre-build per-bit tone membership: bits1[b] = list of tones where bit b = 1
    #                                    bits0[b] = list of tones where bit b = 0
    bits1 = [[] for _ in range(3)]
    bits0 = [[] for _ in range(3)]
    for k in range(8):
        gv = int(gray_table[k])
        for b in range(3):
            if (gv >> (2 - b)) & 1:
                bits1[b].append(k)
            else:
                bits0[b].append(k)

    for s in range(N_syms):
        tone = int(syms[s])
        gv   = int(gray_table[tone])
        row  = E_safe[s]

        for b in range(3):
            hard_bits_raw[3*s + b] = int((gv >> (2 - b)) & 1)

            # Max-log LLR: use the best (highest energy) tone from each partition.
            # This is far stronger than summing all 4 tones per side, because it
            # avoids diluting the signal tone with 3 noise tones.
            log_e1 = max(math.log(float(row[k])) for k in bits1[b])
            log_e0 = max(math.log(float(row[k])) for k in bits0[b])
            llrs_raw[3*s + b] = log_e0 - log_e1   # negative = bit 1 (LDPC convention)

    # Apply 174-bit de-interleaver
    perm = _FT8_INTERLEAVE_PERM   # perm[dst] = src
    hard_bits = hard_bits_raw[list(perm)]
    llrs      = llrs_raw[list(perm)]

    return hard_bits, llrs


# ---------------------------------------------------------------------------
# FT8 symbol pipeline constants
# ---------------------------------------------------------------------------

# Symbol indices of the three Costas arrays within the 79-symbol frame.
FT8_COSTAS_POSITIONS: tuple[int, ...] = (
    0, 1, 2, 3, 4, 5, 6,       # Costas block 0
    36, 37, 38, 39, 40, 41, 42, # Costas block 1
    72, 73, 74, 75, 76, 77, 78, # Costas block 2
)

# The 58 payload symbol positions (all 79 positions minus the 21 Costas ones).
FT8_PAYLOAD_POSITIONS: tuple[int, ...] = tuple(
    s for s in range(79) if s not in set(FT8_COSTAS_POSITIONS)
)

# 3-bit Gray code used by FT8: tone index → (b2, b1, b0) as an integer 0..7.
# Gray code: 0→0, 1→1, 2→3, 3→2, 4→6, 5→7, 6→5, 7→4
_FT8_GRAY_DECODE: tuple[int, ...] = (0, 1, 3, 2, 6, 7, 5, 4)


def ft8_extract_payload_symbols(
    E79: np.ndarray,
    *,
    shift: int = 0,
    inverted: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Stage 2a — Pull the 58 payload symbol rows out of the full (79, 8) energy
    matrix and apply the tone-index correction determined during sync.

    Parameters
    ----------
    E79      : (79, 8) energy matrix from FT8SymbolEnergyExtractor.extract_all_79()
    shift    : tone-index shift from Costas check (0..7)
    inverted : whether tones are frequency-inverted (k -> 7-k)

    Returns
    -------
    E_payload : (58, 8)  — energy matrix for the 58 payload symbols, tone-axis
                           corrected for shift and inversion.
    hard_syms : (58,)    — hard-decision tone index (0..7) for each payload symbol,
                           after shift/inversion correction.

    Tone correction
    ---------------
    During sync we found that the received tone index ``r`` maps to the true
    FT8 tone ``t`` as:
        if not inverted:  t = (r - shift) % 8
        if inverted:      t = 7 - ((r - shift) % 8)   (equivalent to 7-r then shift)
    We reorder the energy columns so that column k of E_payload corresponds to
    true FT8 tone k, making downstream Gray decoding straightforward.
    """
    E79 = np.asarray(E79, dtype=np.float64)
    if E79.shape != (79, 8):
        raise ValueError(f"E79 must be shape (79, 8), got {E79.shape}")

    # Extract the 58 payload rows
    E_raw = E79[list(FT8_PAYLOAD_POSITIONS), :]   # (58, 8)

    # Build a column-permutation that maps received tone index → true FT8 tone
    # received column r  →  true tone t
    perm = np.empty(8, dtype=np.intp)
    for r in range(8):
        if inverted:
            t = (7 - r - shift) % 8
        else:
            t = (r - shift) % 8
        perm[t] = r   # column perm[t] of E_raw is true tone t

    E_payload = E_raw[:, perm]                     # (58, 8), columns = true tone 0..7
    hard_syms = np.argmax(E_payload, axis=1).astype(np.int32)  # (58,)

    return E_payload, hard_syms




# ---------------------------------------------------------------------------
# FT8 LDPC decoder — Stage 4
# ---------------------------------------------------------------------------

_LDPC_CHECKS: tuple[tuple[int, ...], ...] = (
    (3, 30, 58, 90, 91, 95, 152,),
    (4, 31, 59, 92, 114, 145,),
    (5, 23, 60, 93, 121, 150,),
    (6, 32, 61, 94, 95, 142,),
    (7, 24, 62, 82, 92, 95, 147,),
    (5, 31, 63, 96, 125, 137,),
    (4, 33, 64, 77, 97, 106, 153,),
    (8, 34, 65, 98, 138, 145,),
    (9, 35, 66, 99, 106, 125,),
    (10, 36, 66, 86, 100, 138, 157,),
    (11, 37, 67, 101, 104, 154,),
    (12, 38, 68, 102, 148, 161,),
    (7, 39, 69, 81, 103, 113, 144,),
    (13, 40, 70, 87, 101, 122, 155,),
    (14, 41, 58, 105, 122, 158,),
    (0, 32, 71, 105, 106, 156,),
    (15, 42, 72, 107, 140, 159,),
    (16, 36, 73, 80, 108, 130, 153,),
    (10, 43, 74, 109, 120, 165,),
    (44, 54, 63, 110, 129, 160, 172,),
    (7, 45, 70, 111, 118, 165,),
    (17, 35, 75, 88, 112, 113, 142,),
    (18, 37, 76, 103, 115, 162,),
    (19, 46, 69, 91, 137, 164,),
    (1, 47, 73, 112, 127, 159,),
    (20, 44, 77, 82, 116, 120, 150,),
    (21, 46, 57, 117, 126, 163,),
    (15, 38, 61, 111, 133, 157,),
    (22, 42, 78, 119, 130, 144,),
    (18, 34, 58, 72, 109, 124, 160,),
    (19, 35, 62, 93, 135, 160,),
    (13, 30, 78, 97, 131, 163,),
    (2, 43, 79, 123, 126, 168,),
    (18, 45, 80, 116, 134, 166,),
    (6, 48, 57, 89, 99, 104, 167,),
    (11, 49, 60, 117, 118, 143,),
    (12, 50, 63, 113, 117, 156,),
    (23, 51, 75, 128, 147, 148,),
    (24, 52, 68, 89, 100, 129, 155,),
    (19, 45, 64, 79, 119, 139, 169,),
    (20, 53, 76, 99, 139, 170,),
    (34, 81, 132, 141, 170, 173,),
    (13, 29, 82, 112, 124, 169,),
    (3, 28, 67, 119, 133, 172,),
    (0, 3, 51, 56, 85, 135, 151,),
    (25, 50, 55, 90, 121, 136, 167,),
    (51, 83, 109, 114, 144, 167,),
    (6, 49, 80, 98, 131, 172,),
    (22, 54, 66, 94, 171, 173,),
    (25, 40, 76, 108, 140, 147,),
    (1, 26, 40, 60, 61, 114, 132,),
    (26, 39, 55, 123, 124, 125,),
    (17, 48, 54, 123, 140, 166,),
    (5, 32, 84, 107, 115, 155,),
    (27, 47, 69, 84, 104, 128, 157,),
    (8, 53, 62, 130, 146, 154,),
    (21, 52, 67, 108, 120, 173,),
    (2, 12, 47, 77, 94, 122,),
    (30, 68, 132, 149, 154, 168,),
    (11, 42, 65, 88, 96, 134, 158,),
    (4, 38, 74, 101, 135, 166,),
    (1, 53, 85, 100, 134, 163,),
    (14, 55, 86, 107, 118, 170,),
    (9, 43, 81, 90, 110, 143, 148,),
    (22, 33, 70, 93, 126, 152,),
    (10, 48, 87, 91, 141, 156,),
    (28, 33, 86, 96, 146, 161,),
    (29, 49, 59, 85, 136, 141, 161,),
    (9, 52, 65, 83, 111, 127, 164,),
    (21, 56, 84, 92, 139, 158,),
    (27, 31, 71, 102, 131, 165,),
    (27, 28, 83, 87, 116, 142, 149,),
    (0, 25, 44, 79, 127, 146,),
    (16, 26, 88, 102, 115, 152,),
    (50, 56, 97, 162, 164, 171,),
    (20, 36, 72, 137, 151, 168,),
    (15, 46, 75, 129, 136, 153,),
    (2, 23, 29, 71, 103, 138,),
    (8, 39, 89, 105, 133, 150,),
    (14, 57, 59, 73, 110, 149, 162,),
    (17, 41, 78, 143, 145, 151,),
    (24, 37, 64, 98, 121, 159,),
    (16, 41, 74, 128, 169, 171,),
)

_FT8_LDPC_FREE_COLS: tuple[int, ...] = tuple(range(91))

_FT8_LDPC_PIVOT_COLS: tuple[int, ...] = tuple(
    c for c in range(174) if c not in set(_FT8_LDPC_FREE_COLS)
)

# CRC-14 generator polynomial used by FT8 (same as WSJT-X / ft8_lib)
_FT8_CRC14_POLY = 0x2757   # x^14 + x^13 + x^11 + x^9 + x^8 + x^6 + x^4 + x^2 + x + 1


def _ft8_crc14(bits: np.ndarray) -> int:
    """
    Compute the FT8 CRC-14 over a bit array.

    Parameters
    ----------
    bits : array-like of 0/1 values, length 96 (the 77 message bits + 19-bit
           padding used by WSJT-X before CRC is appended, but in practice
           we pass the 91 decoded codeword bits and check that the last 14 are
           zero after CRC verification — see ft8_ldpc_decode for usage).

    Returns
    -------
    crc : int in [0, 16383]  — the 14-bit CRC value.
    """
    crc = 0
    for bit in bits:
        top = (crc >> 13) & 1
        crc = ((crc << 1) & 0x3FFF) | int(bit)
        if top:
            crc ^= _FT8_CRC14_POLY
    return crc & 0x3FFF


def ft8_ldpc_decode(
    llrs: np.ndarray,
    *,
    max_iterations: int = 100,
    llr_clamp: float = 40.0,
    min_sum_scale: float = 0.75,
) -> tuple[bool, np.ndarray, int]:
    """
    Stage 4 — Decode the (174, 91) FT8 LDPC code using the min-sum algorithm.

    The min-sum algorithm is a hardware-friendly approximation to belief
    propagation that clips check-node messages to the minimum absolute value
    rather than computing a full tanh product.  It converges reliably in
    ≤50 iterations for FT8 SNRs where a decode is possible at all.

    Parameters
    ----------
    llrs          : (174,) float64 — channel LLRs from ft8_gray_decode.
                    Sign convention: positive = likely bit 1.
    max_iterations: number of belief-propagation iterations (default 50).
    llr_clamp     : clip LLRs to ±this value to prevent numerical overflow.

    Returns
    -------
    success  : bool   — True if all 83 parity checks pass after decoding.
    codeword : (91,)  uint8 — the decoded 91 systematic bits (77-bit message
                              + 14-bit CRC) if success=True; otherwise the
                              best hard-decision estimate.
    iterations_used : int — number of iterations until convergence (or max_iterations).

    Algorithm
    ---------
    The decoder works in log-domain.  Each channel bit n has an initial LLR
    Lch[n].  Variable-node messages v→c and check-node messages c→v are
    iteratively refined:

      Variable → Check:
        L_vc[c,n] = Lch[n] + sum_{c' ≠ c} L_cv[c',n]

      Check → Variable (min-sum):
        L_cv[c,n] = prod_sign_{n' ≠ n} sign(L_vc[c,n'])
                  * min_{n' ≠ n} |L_vc[c,n']|

    Hard decision: bit[n] = 1 if (Lch[n] + sum_c L_cv[c,n]) < 0 else 0

    After each iteration all 83 syndrome bits are checked; if all zero the
    decode is declared successful and we return early.

    CRC note
    --------
    The 91 codeword bits are structured as:
        bits[0..76]  — 77-bit message payload
        bits[77..90] — 14-bit CRC
    After a successful parity-check pass we additionally verify the CRC so
    that near-collisions in the LDPC code don't produce garbage messages.
    success=True is only returned when BOTH the parity checks AND the CRC pass.
    """
    llrs = np.asarray(llrs, dtype=np.float64)
    assert llrs.shape == (174,), f"Expected (174,) LLRs, got {llrs.shape}"

    N_bits   = 174
    N_checks = 83
    N_sys    = 91        # systematic bits

    # Clamp to prevent saturation
    Lch = np.clip(llrs, -llr_clamp, llr_clamp)

    # Build adjacency structures once
    # checks[c] = list of variable indices in check c
    checks = [list(row) for row in _LDPC_CHECKS]

    # vars_to_checks[n] = list of check indices that involve variable n
    vars_to_checks: list[list[int]] = [[] for _ in range(N_bits)]
    for c, cols in enumerate(checks):
        for n in cols:
            vars_to_checks[n].append(c)

    # Initialise messages: L_cv[c][n] = 0 for all c,n
    # Store as dict-of-dict for sparsity
    L_cv: list[dict[int, float]] = [{n: 0.0 for n in row} for row in checks]

    iterations_used = max_iterations
    bits    = (Lch < 0.0).astype(np.uint8)   # initial hard decision from channel LLRs
    all_ok  = False

    for iteration in range(max_iterations):
        # ── Variable → Check messages ──────────────────────────────────
        # L_vc[c,n] = Lch[n] + sum_{c'} L_cv[c',n]  -  L_cv[c,n]
        # Precompute total incoming message for each variable
        L_total = Lch.copy()
        for c, row in enumerate(checks):
            for n in row:
                L_total[n] += L_cv[c][n]

        # ── Check → Variable messages (min-sum) ────────────────────────
        for c, row in enumerate(checks):
            # For each variable n in this check:
            #   L_vc[c,n] = L_total[n] - L_cv[c,n]   (exclude self)
            msgs = {n: float(L_total[n] - L_cv[c][n]) for n in row}

            for n in row:
                # Product of signs of all other variables in this check
                prod_sign = 1.0
                min_abs   = float("inf")
                for n2, v in msgs.items():
                    if n2 == n:
                        continue
                    prod_sign *= 1.0 if v >= 0 else -1.0
                    a = abs(v)
                    if a < min_abs:
                        min_abs = a
                L_cv[c][n] = prod_sign * min_abs * min_sum_scale

        # ── Hard decision ──────────────────────────────────────────────
        L_post = Lch.copy()
        for c, row in enumerate(checks):
            for n in row:
                L_post[n] += L_cv[c][n]

        bits = (L_post < 0.0).astype(np.uint8)   # 1 if LLR < 0 (likely 1)

        # ── Syndrome check ─────────────────────────────────────────────
        all_ok = True
        for row in checks:
            parity = 0
            for n in row:
                parity ^= int(bits[n])
            if parity != 0:
                all_ok = False
                break

        if all_ok:
            iterations_used = iteration + 1
            break

    # Extract systematic bits from the FREE column positions.
    # _FT8_LDPC_FREE_COLS[0:77]  → 77 message bits
    # _FT8_LDPC_FREE_COLS[77:91] → 14 CRC bits
    free_bits = np.array([int(bits[c]) for c in _FT8_LDPC_FREE_COLS], dtype=np.uint8)  # (91,)

    if not all_ok:
        return False, free_bits, iterations_used

    # ── CRC-14 verification ────────────────────────────────────────────
    msg_bits    = free_bits[:77]
    rx_crc_bits = free_bits[77:91]
    rx_crc = int(sum(int(b) << (13 - i) for i, b in enumerate(rx_crc_bits)))

    # WSJT-X prepends a 5-bit zero pad before hashing (makes it a 96-bit input)
    padded   = np.concatenate((np.zeros(5, dtype=np.uint8), msg_bits))
    calc_crc = _ft8_crc14(padded)

    crc_ok = (calc_crc == rx_crc)
    return crc_ok, free_bits, iterations_used


# ---------------------------------------------------------------------------
# FT8 message unpacker — Stage 5
# ---------------------------------------------------------------------------
#
# Reference: WSJT-X source (pack77.f90 / unpack77.f90) and
#            Steve Franke K9AN / Joe Taylor K1JT, "New Concepts in FT8",
#            QEX, Sept/Oct 2020.
#
# The 77-bit message space is partitioned by a 3-bit "i3" type field at
# bits 74-76 (MSB-first), with a further 3-bit "n3" subtype in bits 71-73
# for i3=0.
#
# Type i3=0  (n3 sub-type)
#   n3=0  Standard type-1:  28+1 + 28+1 + 15 bits  (call1 + call2 + grid/rpt)
#   n3=1  European VHF:     28+1 + 28+1 + 15 bits  (same packing, different grid)
#   n3=2  Compound call 1:  12 + 58 + 1 + 3 bits
#   n3=3  Compound call 2:  58 + 12 + 1 + 3 bits
#   n3=4  CQ with grid/freq: special
#   n3=5  ARRL RTTY Roundup contest
# Type i3=1  Free text (up to 13 chars, 71 bits)
# Type i3=2  European VHF contest
# Type i3=3  ARRL Field Day
# Type i3=4  Telemetry (18 hex digits)
# Type i3=5  Reserved
#
# We implement the most common types fully:
#   i3=0, n3=0  — standard type-1 (most QSOs)
#   i3=0, n3=2/3 — compound callsigns (DX prefixes)
#   i3=1        — free text (CQ, etc.)
#   i3=4        — telemetry (logged as hex)
# All other types fall back to a hex dump.

# ── Character set tables ────────────────────────────────────────────────────

# 37-character set used for standard callsign packing (A-Z, 0-9, space)
_C37 = " 0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# 42-character set used for compound/suffix callsigns (adds /+-. chars)
_C42 = " 0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ/+-."

# 42-character free-text set from WSJT-X packjt77.f90
# 71 bits encodes 13 chars at base-42: 42^13 ≈ 1.1e21 < 2^71 ≈ 2.4e21
_C69 = " 0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ+-./?="   # kept as _C69 for compat, but 42 chars
# Correct: WSJT-X uses exactly these 42 printable chars for free text
_FREETEXT_CHARS = " 0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ+-./?"

# Grid square characters: A-R for field letters, 0-9 for square digits
_GRID_LETTERS = "ABCDEFGHIJKLMNOPQR"


def _bits_to_int(bits: np.ndarray, start: int, length: int) -> int:
    """Extract an unsigned integer from a bit array (MSB first)."""
    val = 0
    for i in range(length):
        val = (val << 1) | int(bits[start + i])
    return val


def _unpack_callsign_28(n: int) -> str:
    """
    Decode a 28-bit packed standard callsign.

    From WSJT-X packjt77.f90 / ft8_lib pack77.cpp:

    NBASE = 37*36*10*27*27*27 = 262,177,560
    Values 0..NBASE-1  → standard callsign (mixed-radix decode)
    NBASE+0            → 'DE'
    NBASE+1            → 'QRZ'
    NBASE+2            → 'CQ'
    NBASE+3..NBASE+2+1000 → 'CQ 000'..'CQ 999'  (directed CQ by frequency)
    NBASE+3+1000..     → 'CQ XX' style (4-char suffix, base-27)
    > 2^28-1           → hashed callsign (not decodable)

    Encoding (c0..c5, MSB-first):
      n = ((((c0*36 + c1)*10 + c2)*27 + c3)*27 + c4)*27 + c5
      c0 ∈ C36 (0-9, A-Z — NO space; first char of callsign cannot be space)
      c1 ∈ C37 (space, 0-9, A-Z — second char can be space for 1-char prefix)
      c2 ∈ C10 (0-9 — must be the digit)
      c3,c4,c5 ∈ C27 (A-Z, space — suffix chars, trailing spaces)
    """
    # From WSJT-X: NBASE = 37*36*10*27**3
    NBASE = 37 * 36 * 10 * 27 * 27 * 27   # 262177560

    if n >= NBASE:
        r = n - NBASE
        if r == 0:  return 'DE'
        if r == 1:  return 'QRZ'
        if r == 2:  return 'CQ'
        if 3 <= r <= 2 + 1000:
            return f'CQ {r - 3:03d}'
        if r <= 2 + 1000 + 531441:   # 531441 = 27^4... actually use simpler bound
            m = r - 3 - 1000
            suffix = ''
            for _ in range(4):
                suffix = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ '[m % 27] + suffix
                m //= 27
            return f'CQ {suffix.strip()}'
        return f'<hash:{n & 0x3FFFFF:06X}>'

    # Standard callsign decode
    # n = ((((c0*36 + c1)*10 + c2)*27 + c3)*27 + c4)*27 + c5
    # Note: c0 uses base-36, c1 uses base-37 (can be space)
    C36 = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"   # 36 chars, NO space
    C37 = " 0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"   # 37 chars, space at index 0
    C27 = "ABCDEFGHIJKLMNOPQRSTUVWXYZ "
    C10 = "0123456789"

    n, c5 = divmod(n, 27);  c5 = C27[c5]
    n, c4 = divmod(n, 27);  c4 = C27[c4]
    n, c3 = divmod(n, 27);  c3 = C27[c3]
    n, c2 = divmod(n, 10);  c2 = C10[c2]
    n, c1 = divmod(n, 37);  c1 = C37[c1]
    c0 = C36[n] if n < 36 else '?'

    # Remove all spaces (c1 may be space for 1-char prefix; c3-c5 may be trailing spaces)
    call = (c0 + c1 + c2 + c3 + c4 + c5).replace(' ', '').strip()
    return call if call else '?'


def _unpack_grid_15(n: int) -> str:
    """
    Decode a 15-bit grid/report field.

    Values:
      0..31:   signal report  -24 to +9 dB  (n=0 → -24, n=1 → -23, ..., n=30 → +6, n=31 → R+6 etc.)
      32767:   RRR
      32766:   RR73
      32765:   73
      32..32399: Maidenhead grid square AA00..RR99
      32400+:  special (contest serial etc.)
    """
    if n <= 62:
        if n == 62:
            return 'RRR'
        db = n - 35
        sign = '+' if db >= 0 else ''
        return f'{sign}{db:02d}'
    if n == 32767:
        return 'RRR'
    if n == 32766:
        return 'RR73'
    if n == 32765:
        return '73'
    if n == 32764:
        return 'RRR'

    # Maidenhead grid: 4-character AA00
    n2 = n - 63
    if n2 < 0 or n2 >= 32400:
        return f'?{n}'
    lon_sq, rem  = divmod(n2, 180)
    lat_sq, rem2 = divmod(rem, 10)
    lon_sub      = rem2
    lat_sub      = 0
    # Actually: grid = field_lon(0-17) + field_lat(0-17) + sq_lon(0-9) + sq_lat(0-9)
    # Encoding: n = (lon_field*18 + lat_field)*100 + lon_sq*10 + lat_sq
    n3 = n - 63
    lat_sq2, n4  = divmod(n3, 180)
    lon_sq2, rem3 = divmod(n4, 10)
    # re-derive properly
    # FT8: grid packed as (lon+180)/2 * 18*10*10 + (lat+90) ...
    # Simplest correct decode from WSJT-X unpack77.f90:
    # n = (igrid4 - NGBASE) where NGBASE = 32768 - 32400 ... let's use the formula directly
    # igrid = n - 63
    igrid = n - 63
    if igrid < 0 or igrid >= 18*18*10*10:
        return f'?{n}'
    igrid, g4 = divmod(igrid, 10)
    igrid, g3 = divmod(igrid, 10)
    igrid, g2 = divmod(igrid, 18)
    g1 = igrid
    grid = _GRID_LETTERS[g1] + _GRID_LETTERS[g2] + str(g3) + str(g4)
    return grid


def _unpack_type1(bits: np.ndarray) -> str:
    """
    Unpack standard Type-1 FT8 message (i3=0 or i3=2).

    True 77-bit layout (WSJT-X pack77.f90):
      [0:28]  call1 (28 bits)
      [28]    R1 flag
      [29:57] call2 (28 bits)
      [57]    R2 flag
      [58:73] grid/report (15 bits)
      [73]    spare (0)
      [74:77] i3  ← already dispatched by caller
    """
    call1_n = _bits_to_int(bits, 0,  28)
    r1      = int(bits[28])
    call2_n = _bits_to_int(bits, 29, 28)
    r2      = int(bits[57])
    grid_n  = _bits_to_int(bits, 58, 15)

    call1 = _unpack_callsign_28(call1_n)
    call2 = _unpack_callsign_28(call2_n)
    grid  = _unpack_grid_15(grid_n)

    if r1: call1 = 'R' + call1
    if r2: call2 = 'R' + call2
    return f'{call1} {call2} {grid}'


def _unpack_free_text(bits: np.ndarray) -> str:
    """
    Unpack free-text message (i3=1).
    bits[0:71] = 71-bit integer → 13 chars from _FREETEXT_CHARS (base-42, MSB-first).
    42^13 ≈ 1.1e21 < 2^71 ≈ 2.4e21, so 13 chars fit in 71 bits.
    """
    n = _bits_to_int(bits, 0, 71)
    chars = []
    for _ in range(13):
        n, r = divmod(n, 42)
        chars.append(_FREETEXT_CHARS[r] if r < len(_FREETEXT_CHARS) else '?')
    return ''.join(reversed(chars)).strip()


def _unpack_compound_call_58(bits: np.ndarray, start: int) -> str:
    """Decode a 58-bit compound callsign (base-42, up to 11 chars from C42)."""
    n = _bits_to_int(bits, start, 58)
    chars = []
    for _ in range(11):
        n, r = divmod(n, 42)
        chars.append(_C42[r] if r < len(_C42) else '?')
    return ''.join(reversed(chars)).strip()


def _unpack_type_compound(bits: np.ndarray, which: int) -> str:
    """
    Compound callsign messages (i3=0, one call is a full compound, one is a hash).
    which=1: [58 compound_call1][1 R1][12 hash_call2][1 R2][3 spare][3 i3]
    which=2: [12 hash_call1][1 R1][58 compound_call2][1 R2][3 spare][3 i3]
    """
    if which == 1:
        call1 = _unpack_compound_call_58(bits, 0)
        r1    = int(bits[58])
        call2 = f'<{_bits_to_int(bits, 59, 12):03X}>'
        r2    = int(bits[71])
    else:
        call1 = f'<{_bits_to_int(bits, 0, 12):03X}>'
        r1    = int(bits[12])
        call2 = _unpack_compound_call_58(bits, 13)
        r2    = int(bits[71])
    if r1: call1 = 'R' + call1
    if r2: call2 = 'R' + call2
    return f'{call1} {call2}'


def _unpack_telemetry(bits: np.ndarray) -> str:
    """i3=4 telemetry: bits[0:71] → 18-nibble hex string."""
    n = _bits_to_int(bits, 0, 71)
    return f'TELEMETRY:{n:018X}'


def ft8_unpack_message(msg_bits: np.ndarray) -> str:
    """
    Stage 5 — Decode 77 message bits to a human-readable FT8 message string.

    Parameters
    ----------
    msg_bits : (77,) uint8  — decoded_free_bits[:77] from ft8_ldpc_decode().

    Returns
    -------
    str — e.g. 'W4ABC K9XYZ EN52', 'CQ W4ABC EM73', or '?i3=5:...' fallback.

    77-bit layout shared by all types:
      bits [74:77] = i3 (primary message type selector, 3 bits)

    i3=0  Standard QSO / CQ  [28 call1][1 R][28 call2][1 R][15 grid][1 spare][3 i3]
    i3=1  Free text           [71 text][3 spare][3 i3]
    i3=2  EU VHF contest      same layout as i3=0
    i3=3  ARRL Field Day      [28 call1][1 R][28 call2][1 R][13 exchange][3 spare][3 i3]
    i3=4  Telemetry           [71 payload][3 spare][3 i3]
    i3=5+ Reserved
    """
    msg_bits = np.asarray(msg_bits, dtype=np.uint8)
    if msg_bits.shape != (77,):
        return f'?bad_len={len(msg_bits)}'

    i3 = _bits_to_int(msg_bits, 74, 3)

    if i3 == 0:
        return _unpack_type1(msg_bits)

    elif i3 == 1:
        return _unpack_free_text(msg_bits)

    elif i3 == 2:
        return 'EU-VHF ' + _unpack_type1(msg_bits)

    elif i3 == 3:
        call1_n = _bits_to_int(msg_bits, 0,  28)
        r1      = int(msg_bits[28])
        call2_n = _bits_to_int(msg_bits, 29, 28)
        r2      = int(msg_bits[57])
        exch    = _bits_to_int(msg_bits, 58, 13)
        call1   = _unpack_callsign_28(call1_n)
        call2   = _unpack_callsign_28(call2_n)
        if r1: call1 = 'R' + call1
        if r2: call2 = 'R' + call2
        return f'FD {call1} {call2} {exch}'

    elif i3 == 4:
        return _unpack_telemetry(msg_bits)

    else:
        raw = _bits_to_int(msg_bits, 0, 77)
        return f'?i3={i3}:{raw:020X}'


class FT8ConsoleDecoder:
    """
    End-to-end (for now):
    - ingest audio chunks at device fs
    - resample to fs_proc (12 kHz)
    - build 15s UTC-aligned frames
    - detect candidate FT8-ish peaks
    - print results to console
    """
    def __init__(
        self,
        *,
        fs_proc: int = 12_000,
        fmin_hz: float = 200.0,
        fmax_hz: float = 3200.0,
        on_decode=None,   # optional callable(utc: str, freq_hz: float, snr_db: float, message: str)
    ) -> None:

        self.fs_proc = int(fs_proc)
        self.fmin_hz = float(fmin_hz)
        self.fmax_hz = float(fmax_hz)
        self._on_decode = on_decode   # fired on every successful LDPC+CRC decode

        self._q: "queue.Queue[tuple[int, np.ndarray, float]]" = queue.Queue(maxsize=400)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self._resampler: Optional[PolyphaseResampler] = None
        self._framer = UTC15sFramer(self.fs_proc, frame_s=15.0)
        self._detector = FT8SignalDetector(fs=self.fs_proc, fmin_hz=self.fmin_hz, fmax_hz=self.fmax_hz)

        self._debug = True

        self._extractor = FT8SymbolEnergyExtractor(fs=self.fs_proc)
        self._sync = FT8SyncSearch(
            fs=self.fs_proc,
            fmin_hz=self.fmin_hz,
            fmax_hz=self.fmax_hz,
            extractor=self._extractor,
        )

        # Gate thresholds for passing sync candidates to refinement.
        # costas_ok: minimum symbol matches out of 21 (random chance ~3/21).
        # Real weak signals can be as low as 7-9/21 when block 0 is in noise.
        self._min_costas_matches = 7
        # Minimum mean margin across all 3 Costas blocks.
        # 2 dB = expected tone is ~1.6x median of others — a very loose gate.
        # LDPC+CRC is the real quality filter; we'd rather try and fail than drop.
        self._min_margin_db = 2.0

    def _get_seed_freqs(self, frame: np.ndarray) -> list[tuple[float, float]]:
        """
        Detect candidate signal frequencies from the full 15s frame.
        FT8SyncSearch handles its own time-offset search, so we only need to
        run the detector once on the full unshifted frame.
        """
        peaks, _score = self._detector.detect_with_best_score(frame)
        return peaks

    def _refine_candidate_by_costas(
        self,
        frame: np.ndarray,
        *,
        t0_s: float,
        f0_hz: float,
        coarse_shift: int = 0,
        coarse_inv: bool = False,
        df_hz: float = 6.25,
        df_step_hz: float = 0.05,
        dt_s: float = 0.020,
        dt_step_s: float = 0.001,
    ) -> tuple[float, float, int, int, int, bool]:
        """
        Refine both f0 and t0 using the continuous mean-margin score from
        score_costas_batch.  Sweeps f0 over ±df_hz in df_step_hz steps and
        t0 over ±dt_s in dt_step_s steps (default ±20 ms at 1 ms resolution).

        The 2-D search corrects both residual frequency offset and sub-symbol
        timing errors (including USB audio clock drift over the 12.6 s frame).
        """
        costas_tones = (3, 1, 4, 0, 6, 5, 2)
        total = 21

        f_vals = np.arange(
            float(f0_hz) - df_hz,
            float(f0_hz) + df_hz + 1e-12,
            df_step_hz,
            dtype=np.float64,
        )
        t_vals = np.arange(
            float(t0_s) - dt_s,
            float(t0_s) + dt_s + 1e-12,
            dt_step_s,
            dtype=np.float64,
        )

        best_score  = float("-inf")
        best_f0     = float(f0_hz)
        best_t0     = float(t0_s)

        for t_cand in t_vals:
            x_mat = self._extractor._get_costas_xmat(frame, t0_s=float(t_cand))
            scores = self._extractor.score_costas_batch(
                x_mat, f_vals, shift=int(coarse_shift), inverted=bool(coarse_inv)
            )
            idx = int(np.argmax(scores))
            sc  = float(scores[idx])
            if sc > best_score:
                best_score = sc
                best_f0    = float(f_vals[idx])
                best_t0    = float(t_cand)

        # Compute match count at best (t0, f0) for the return signature
        x_mat_best = self._extractor._get_costas_xmat(frame, t0_s=best_t0)
        Ec_best = self._extractor.score_costas_from_xmat(x_mat_best, f0_hz=best_f0)
        best_matches = 0
        for r in range(total):
            expected = int((costas_tones[r % 7] + coarse_shift) % 8)
            if coarse_inv:
                expected = 7 - expected
            if int(np.argmax(Ec_best[r, :])) == expected:
                best_matches += 1

        return float(best_t0), float(best_f0), int(best_matches), int(total), int(coarse_shift), bool(coarse_inv)

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def feed(self, *, fs: int, samples: np.ndarray, t0_monotonic: float) -> None:
        if self._stop.is_set():
            return
        try:
            x = np.asarray(samples, dtype=np.float32).reshape(-1)
            self._q.put_nowait((int(fs), x, float(t0_monotonic)))
        except queue.Full:
            pass

    @staticmethod
    def _dedupe_peaks(peaks: list[tuple[float, float]], *, hz_bucket: float = 5.0) -> list[tuple[float, float]]:
        """
        Deduplicate near-identical peaks by frequency bucketing.
        Keeps strongest per bucket.
        """
        best_by_bucket: dict[int, tuple[float, float]] = {}
        for f_hz, s_db in peaks:
            b = int(round(f_hz / hz_bucket))
            prev = best_by_bucket.get(b)
            if prev is None or s_db > prev[1]:
                best_by_bucket[b] = (f_hz, s_db)
        out = list(best_by_bucket.values())
        out.sort(key=lambda p: p[1], reverse=True)
        return out

    @staticmethod
    def _fmt_utc(epoch_s: float) -> str:
        return datetime.fromtimestamp(epoch_s, tz=timezone.utc).strftime("%H:%M:%S")

    def _worker(self) -> None:
        while not self._stop.is_set():
            try:
                fs_in, x_in, t0 = self._q.get(timeout=0.25)
            except queue.Empty:
                continue

            if self._resampler is None or self._resampler.fs_in != fs_in:
                self._resampler = PolyphaseResampler(fs_in=fs_in, fs_out=self.fs_proc)
                if self._debug:
                    print(f"[FT8] resampler locked: fs_in={fs_in} -> fs_proc={self.fs_proc}", flush=True)

            x = self._resampler.process(x_in)

            frames = self._framer.push(x, t0_monotonic=t0)
            for slot_start_utc, frame, is_partial in frames:
                utc = self._fmt_utc(slot_start_utc)

                if is_partial:
                    if self._debug:
                        print(f"[FT8] slot {utc}Z skipped (partial — audio started mid-slot)", flush=True)
                    continue

                peaks = self._get_seed_freqs(frame)
                peaks = self._dedupe_peaks(peaks, hz_bucket=10.0)
                seed_freqs = [f for (f, _s) in peaks[:10]]

                sync_cands = self._sync.search(
                    frame,
                    seed_freqs_hz=seed_freqs,
                    time_search_s=0.8,
                    freq_search_bins=4,
                    max_candidates=5,
                )

                if self._debug:
                    print(f"[FT8] slot {utc}Z sync_candidates={len(sync_cands)} seeds={len(seed_freqs)}", flush=True)

                for (t0_s, f0_hz, sc_db) in sync_cands:
                    Ec0 = self._extractor.extract_costas(frame, t0_s=float(t0_s), f0_hz=float(f0_hz))

                    total_m_db, (b0, b1, b2), m0, tot0, sh0, inv0 = ft8_costas_margin_score(Ec0)

                    print(
                        f"{utc}  sync  off={t0_s:+.2f}s  f0={f0_hz:8.2f} Hz  score={sc_db:6.2f} dB"
                        f"  costas_ok={m0:02d}/{tot0:02d} shift={sh0} inv={inv0}"
                        f"  margin_total={total_m_db:6.2f}dB blocks=({b0:5.2f},{b1:5.2f},{b2:5.2f})",
                        flush=True,
                    )

                    # Gate 1: minimum mean Costas margin (loose — LDPC+CRC is the real filter)
                    if total_m_db < self._min_margin_db:
                        continue

                    # Gate 2: minimum Costas symbol matches (guards against pure noise hits)
                    if m0 < self._min_costas_matches:
                        continue

                    # All gates passed — refine frequency with shift/inv locked
                    rt, rf, rm, rtot, rsh, rinv = self._refine_candidate_by_costas(
                        frame,
                        t0_s=float(t0_s),
                        f0_hz=float(f0_hz),
                        coarse_shift=int(sh0),
                        coarse_inv=bool(inv0),
                    )

                    print(
                        f"{utc}  refined  off={rt:+.2f}s  f0={rf:8.2f}Hz  costas_ok={rm:02d}/{rtot:02d} shift={rsh} inv={rinv}",
                        flush=True,
                    )

                    # ── Stage 2: Extract all 79 symbol energies ──────────
                    E79 = self._extractor.extract_all_79(
                        frame, t0_s=float(rt), f0_hz=float(rf)
                    )

                    if self._debug:
                        costas_rows  = E79[list(FT8_COSTAS_POSITIONS), :]
                        payload_rows = E79[list(FT8_PAYLOAD_POSITIONS), :]
                        costas_peak_mean  = float(np.mean(np.max(costas_rows,  axis=1)))
                        payload_peak_mean = float(np.mean(np.max(payload_rows, axis=1)))
                        # Peak-to-mean ratio: 8.0 = perfect single tone, 1.0 = flat noise
                        pay_ptm = float(np.mean(
                            np.max(payload_rows, axis=1) /
                            np.maximum(np.mean(payload_rows, axis=1), 1e-30)
                        ))
                        print(
                            f"{utc}  symbols  E79 shape={E79.shape}"
                            f"  costas_peak={costas_peak_mean:.2e}"
                            f"  payload_peak={payload_peak_mean:.2e}"
                            f"  payload_ptm={pay_ptm:.2f}/8.00",
                            flush=True,
                        )
                        # Show normalised energy distribution for 3 strongest payload symbols
                        top3 = np.argsort(np.max(payload_rows, axis=1))[-3:][::-1]
                        for pidx in top3:
                            row = payload_rows[pidx]
                            row_n = row / max(float(np.max(row)), 1e-30)
                            print(f"{utc}  pay_sym[{pidx:2d}]"
                                  f"  tones=[{' '.join(f'{v:.2f}' for v in row_n)}]"
                                  f"  peak={int(np.argmax(row))}  E={np.max(row):.2e}",
                                  flush=True)

                    # ── Stage 2a: Extract payload symbols + tone correction ─
                    E_payload, hard_syms = ft8_extract_payload_symbols(
                        E79, shift=int(rsh), inverted=bool(rinv)
                    )

                    if self._debug:
                        # Show the 58 hard tone decisions as a compact string
                        sym_str = "".join(str(int(t)) for t in hard_syms)
                        print(
                            f"{utc}  payload_syms (interleaved) [{len(hard_syms)}]: {sym_str}",
                            flush=True,
                        )

                    # ── Stage 2b: De-interleave ───────────────────────────
                    syms_deint, E_deint = ft8_deinterleave(hard_syms, E_payload)

                    if self._debug:
                        sym_str_di = "".join(str(int(t)) for t in syms_deint)
                        print(
                            f"{utc}  payload_syms (de-interleaved) [{len(syms_deint)}]: {sym_str_di}",
                            flush=True,
                        )

                    # ── Stage 3: Gray decode → 174 bits + LLRs ───────────
                    hard_bits, llrs = ft8_gray_decode(syms_deint, E_deint)

                    if self._debug:
                        bit_str = "".join(str(int(b)) for b in hard_bits)
                        llr_mag_mean = float(np.mean(np.abs(llrs)))
                        llr_min  = float(np.min(llrs))
                        llr_max  = float(np.max(llrs))
                        print(
                            f"{utc}  hard_bits [{len(hard_bits)}]: {bit_str[:60]}{'...' if len(bit_str) > 60 else ''}",
                            flush=True,
                        )
                        print(
                            f"{utc}  llrs  mean_|LLR|={llr_mag_mean:.2f}  min={llr_min:.2f}  max={llr_max:.2f}",
                            flush=True,
                        )

                    # ── Stage 4: LDPC decode → 91 codeword bits ──────────
                    ldpc_ok, codeword, ldpc_iters = ft8_ldpc_decode(llrs)

                    if self._debug or ldpc_ok:
                        status = "PASS" if ldpc_ok else "FAIL"
                        print(
                            f"{utc}  ldpc  {status}"
                            f"  iters={ldpc_iters}"
                            f"  crc={'OK' if ldpc_ok else '--'}",
                            flush=True,
                        )

                    if not ldpc_ok:
                        continue

                    # ── Stage 5: Unpack 77 bits → message text ────────────
                    msg_bits = codeword[:77]
                    message  = ft8_unpack_message(msg_bits)

                    decoded_line = (
                        f"{utc}  *** DECODED ***  "
                        f"f0={rf:.2f}Hz  off={rt:+.2f}s  snr≈{sc_db:.1f}dB  "
                        f"msg='{message}'"
                    )
                    print(decoded_line, flush=True)

                    # Fire the GUI callback (if registered) on the decoder thread.
                    # The callback must be thread-safe (e.g. put onto a queue).
                    if self._on_decode is not None:
                        try:
                            self._on_decode(utc, float(rf), float(sc_db), message)
                        except Exception:
                            pass

