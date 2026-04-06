"""
offline_test.py — Run the VADER FT8 decoder against a recorded WAV file.

Usage:
    python offline_test.py [wavfile] [--slot N]

The WAV is fed through the identical pipeline used by main.py
(PolyphaseResampler → UTC15sFramer → FT8ConsoleDecoder worker logic),
driven synchronously so every slot is processed and all debug output
is printed to stdout.

Results are also written to offline_test_results.txt for comparison
across runs.
"""

import sys
import os
import wave
import math
import time
import numpy as np
from datetime import datetime, timezone

# ── import decoder components ──────────────────────────────────────────────
from ft8_decode import (
    PolyphaseResampler,
    UTC15sFramer,
    FT8ConsoleDecoder,
    FT8SignalDetector,
    FT8SymbolEnergyExtractor,
    FT8SyncSearch,
    ft8_costas_margin_score,
    ft8_extract_payload_symbols,
    ft8_gray_decode,
    ft8_ldpc_decode,
    ft8_unpack_message,
    FT8_COSTAS_POSITIONS,
    FT8_PAYLOAD_POSITIONS,
    _FT8_BIT_REV7_TABLE,
)


# ──────────────────────────────────────────────────────────────────────────
# Offline frame processor — mirrors FT8ConsoleDecoder._worker() exactly
# but driven from a WAV file with a fake monotonic clock.
# ──────────────────────────────────────────────────────────────────────────

def process_wav(wav_path: str, slot_filter: int | None = None) -> list[str]:
    """
    Decode a WAV file slot by slot.

    Parameters
    ----------
    wav_path    : path to the WAV file
    slot_filter : if set, only process slot number N (0-based); None = all slots

    Returns
    -------
    List of decoded message strings ("UTC  freq  msg").
    """

    # ── Read WAV ───────────────────────────────────────────────────────────
    with wave.open(wav_path, 'rb') as wf:
        n_channels  = wf.getnchannels()
        sample_width = wf.getsampwidth()
        fs_in       = wf.getframerate()
        n_frames    = wf.getnframes()
        raw         = wf.readframes(n_frames)

    # Convert to float32, mix to mono if stereo
    dtype = np.int16 if sample_width == 2 else np.int32
    audio = np.frombuffer(raw, dtype=dtype).astype(np.float32)
    if n_channels == 2:
        audio = audio.reshape(-1, 2).mean(axis=1)
    # Normalise to ±1
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio /= peak

    duration_s = len(audio) / fs_in
    print(f"WAV: {os.path.basename(wav_path)}")
    print(f"     {fs_in} Hz  {n_channels}ch  {sample_width*8}-bit  "
          f"{len(audio)} samples  {duration_s:.2f}s  (~{duration_s/15:.1f} slots)")
    print()

    # ── Set up pipeline ────────────────────────────────────────────────────
    FS_PROC   = 12_000
    FMIN      = 200.0
    FMAX      = 3200.0

    resampler = PolyphaseResampler(fs_in=fs_in, fs_out=FS_PROC)
    # Use a fixed fake UTC epoch aligned to a 15s boundary so slot 0 is clean.
    # We pin t=0 of the audio to the start of a UTC 15s slot.
    FAKE_UTC_EPOCH = math.floor(time.time() / 15) * 15  # snap to last 15s boundary
    framer    = UTC15sFramer(FS_PROC, frame_s=15.0)

    extractor = FT8SymbolEnergyExtractor(fs=FS_PROC)
    detector  = FT8SignalDetector(fs=FS_PROC, fmin_hz=FMIN, fmax_hz=FMAX)
    sync      = FT8SyncSearch(fs=FS_PROC, fmin_hz=FMIN, fmax_hz=FMAX, extractor=extractor)

    MIN_COSTAS_MATCHES = 7
    MIN_MARGIN_DB      = 2.0
    _FT8_FRAME_S       = 79 * 0.160   # 12.64 s
    _T0_MIN            = -0.50
    _T0_MAX            = 15.0 - _FT8_FRAME_S

    decodes: list[str] = []
    results_lines: list[str] = []

    def log(s: str):
        print(s, flush=True)
        results_lines.append(s)

    # ── Push audio in 0.1s chunks ──────────────────────────────────────────
    chunk_n_in  = int(fs_in * 0.1)
    pos         = 0
    chunk_idx   = 0

    while pos < len(audio):
        chunk = audio[pos: pos + chunk_n_in]
        pos  += chunk_n_in

        # Fake monotonic time: chunk_idx * 0.1s offset from our fake UTC epoch
        t0_mono  = FAKE_UTC_EPOCH + chunk_idx * 0.1
        chunk_idx += 1

        # Resample to 12 kHz
        x = resampler.process(chunk)

        # Frame into 15s UTC-aligned slots
        frames = framer.push(x, t0_monotonic=t0_mono)

        for slot_idx_local, (slot_start_utc, frame, is_partial) in enumerate(frames):
            utc = datetime.fromtimestamp(slot_start_utc, tz=timezone.utc).strftime("%H:%M:%S")

            if is_partial:
                log(f"[FT8] slot {utc}Z skipped (partial)")
                continue

            # ── Slot counter (rough) ───────────────────────────────────────
            slot_num = int((slot_start_utc - FAKE_UTC_EPOCH) / 15)
            if slot_filter is not None and slot_num != slot_filter:
                continue

            # ── Signal detection ───────────────────────────────────────────
            peaks, _ = detector.detect_with_best_score(frame)
            # Dedupe by 10 Hz bucket, keep top 10
            best_by_bucket: dict[int, tuple[float,float]] = {}
            for f_hz, s_db in peaks:
                b = int(round(f_hz / 10.0))
                if b not in best_by_bucket or s_db > best_by_bucket[b][1]:
                    best_by_bucket[b] = (f_hz, s_db)
            peaks_dd = sorted(best_by_bucket.values(), key=lambda p: p[1], reverse=True)
            seed_freqs = [f for f, _ in peaks_dd[:10]]

            sync_cands = sync.search(
                frame,
                seed_freqs_hz=seed_freqs,
                time_search_s=0.5,
                freq_search_bins=4,
                max_candidates=5,
            )

            log(f"[FT8] slot {utc}Z  slot#{slot_num}  "
                f"sync_candidates={len(sync_cands)}  seeds={len(seed_freqs)}")

            for (t0_s, f0_hz, sc_db) in sync_cands:
                if t0_s < _T0_MIN or t0_s > _T0_MAX:
                    log(f"  {utc}  skip  off={t0_s:+.2f}s  f0={f0_hz:.2f}Hz  (out of window)")
                    continue

                Ec0 = extractor.extract_costas(frame, t0_s=float(t0_s), f0_hz=float(f0_hz))
                total_m_db, (b0, b1, b2), m0, tot0, sh0, inv0 = ft8_costas_margin_score(Ec0)

                log(f"  {utc}  sync  off={t0_s:+.2f}s  f0={f0_hz:8.2f}Hz  score={sc_db:6.2f}dB"
                    f"  costas={m0:02d}/{tot0}  shift={sh0}  inv={inv0}"
                    f"  margin={total_m_db:6.2f}dB  blocks=({b0:.2f},{b1:.2f},{b2:.2f})")

                if total_m_db < MIN_MARGIN_DB:
                    log(f"  {utc}  gate FAIL: margin {total_m_db:.2f} < {MIN_MARGIN_DB}")
                    continue
                if m0 < MIN_COSTAS_MATCHES:
                    log(f"  {utc}  gate FAIL: costas {m0} < {MIN_COSTAS_MATCHES}")
                    continue

                # ── Refine ─────────────────────────────────────────────────
                f_vals = np.arange(f0_hz - 6.25, f0_hz + 6.25 + 1e-9, 0.05)
                t_vals = np.arange(t0_s  - 0.020, t0_s  + 0.020 + 1e-9, 0.001)
                best_sc, best_f, best_t = float('-inf'), f0_hz, t0_s
                for tc in t_vals:
                    xmat = extractor._get_costas_xmat(frame, t0_s=float(tc))
                    scores = extractor.score_costas_batch(xmat, f_vals, shift=int(sh0), inverted=bool(inv0))
                    idx = int(np.argmax(scores))
                    if float(scores[idx]) > best_sc:
                        best_sc = float(scores[idx])
                        best_f  = float(f_vals[idx])
                        best_t  = float(tc)

                xmat_best = extractor._get_costas_xmat(frame, t0_s=best_t)
                Ec_best   = extractor.score_costas_from_xmat(xmat_best, f0_hz=best_f)
                costas_tones = (3,1,4,0,6,5,2)
                rm = sum(
                    1 for r in range(21)
                    if np.argmax(Ec_best[r]) == ((costas_tones[r%7] + sh0) % 8
                                                  if not inv0 else 7 - (costas_tones[r%7] + sh0) % 8)
                )
                log(f"  {utc}  refined  off={best_t:+.2f}s  f0={best_f:.2f}Hz  costas={rm}/21")

                # ── Extract all 79 symbols ─────────────────────────────────
                E79 = extractor.extract_all_79(frame, t0_s=best_t, f0_hz=best_f)

                costas_rows  = E79[list(FT8_COSTAS_POSITIONS), :]
                payload_rows = E79[list(FT8_PAYLOAD_POSITIONS), :]
                cp_mean = float(np.mean(np.max(costas_rows,  axis=1)))
                pp_mean = float(np.mean(np.max(payload_rows, axis=1)))
                ptm     = float(np.mean(
                    np.max(payload_rows, axis=1) /
                    np.maximum(np.mean(payload_rows, axis=1), 1e-30)
                ))
                log(f"  {utc}  symbols  costas_peak={cp_mean:.2e}"
                    f"  payload_peak={pp_mean:.2e}  ptm={ptm:.2f}/8.00")

                # ── Stage 2a: payload symbols (no permutation) ─────────────
                E_payload, hard_syms = ft8_extract_payload_symbols(E79)
                sym_str = "".join(str(int(t)) for t in hard_syms)
                log(f"  {utc}  payload_syms [{len(hard_syms)}]: {sym_str}")

                # ── Stage 3: Gray decode ────────────────────────────────────
                syms = np.argmax(E_payload, axis=1)
                _, channel_llrs = ft8_gray_decode(syms, E_payload)

                # ── Stage 3b: Normalise (ft8_lib ftx_normalize_logl) ────────
                mean_llr = float(np.mean(channel_llrs))
                var_llr  = float(np.var(channel_llrs))
                if var_llr > 1e-2:
                    norm_f = math.sqrt(24.0 / var_llr)
                    channel_llrs = (channel_llrs - mean_llr) * norm_f
                log(f"  {utc}  llrs (norm) mean_|LLR|={float(np.mean(np.abs(channel_llrs))):.2f}"
                    f"  mean={float(np.mean(channel_llrs)):+.4f}  var={float(np.var(channel_llrs)):.2f}")

                # ── Stage 2b: De-interleave ────────────────────────────────
                codeword_llrs = np.zeros(174, dtype=np.float64)
                for i in range(174):
                    codeword_llrs[_FT8_BIT_REV7_TABLE[i]] = channel_llrs[i]

                # ── Stage 4: LDPC ──────────────────────────────────────────
                ldpc_ok, codeword, ldpc_iters = ft8_ldpc_decode(codeword_llrs)
                status = "PASS" if ldpc_ok else "FAIL"
                log(f"  {utc}  ldpc {status}  iters={ldpc_iters}  crc={'OK' if ldpc_ok else '--'}")

                if not ldpc_ok:
                    continue

                # ── Stage 5: Unpack message ────────────────────────────────
                message = ft8_unpack_message(codeword[:77])
                result  = f"  {utc}  *** DECODED ***  f0={best_f:.1f}Hz  off={best_t:+.2f}s  snr~{sc_db:.1f}dB  '{message}'"
                log(result)
                decodes.append(result)

    log("")
    log(f"=== SUMMARY: {len(decodes)} decode(s) ===")
    for d in decodes:
        log(d)

    # Write results file
    out_path = "offline_test_results.txt"
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(results_lines) + "\n")
    print(f"\nResults written to {out_path}")

    return decodes


# ──────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    wav = "live_ft8_audio_traffic.wav"
    slot_filter = None

    args = sys.argv[1:]
    if args and not args[0].startswith("--"):
        wav = args.pop(0)
    if "--slot" in args:
        idx = args.index("--slot")
        slot_filter = int(args[idx + 1])

    if not os.path.exists(wav):
        print(f"ERROR: {wav} not found.")
        sys.exit(1)

    process_wav(wav, slot_filter=slot_filter)

