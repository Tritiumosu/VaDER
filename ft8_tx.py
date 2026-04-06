"""
ft8_tx.py — Manual-assisted FT8 TX orchestration for VaDER (Milestone 4).

This module implements the transmit coordinator for VaDER's manual-assisted
FT8 workflow.  The operator selects or composes a message, arms TX for the
next valid FT8 slot, and the coordinator handles slot timing, CAT PTT control,
audio output, and teardown — all in a safe, deterministic way.

TX State Machine
----------------
    IDLE
      │  arm()
      ▼
    ARMED  ──────────────── cancel() ──► CANCELED
      │  slot arrives
      ▼
    TX_PREP  (pre-key guard, open audio device)
      │  ready
      ▼
    TX_ACTIVE  (PTT on → play tones → PTT off)
      │  completed cleanly
      ▼
    COMPLETE
      │  arm() again or reset()
      ▼
    IDLE

Any unhandled exception during TX_PREP / TX_ACTIVE transitions to ERROR.
cancel() is accepted from ARMED.
reset() returns ERROR / COMPLETE / CANCELED → IDLE.

Guardrails (checked at arm())
------------------------------
- CAT (radio) must be connected.
- TX output audio device index must be >= 0.
- Operator callsign + grid must be valid (delegated to OperatorConfig).
- No other TX job may already be ARMED or TX_ACTIVE.

Timing
------
PRE_KEY_S  : seconds before slot boundary to open audio output and key PTT.
POST_KEY_S : seconds after FT8 tones end to hold PTT before unkey.
Both are module-level constants, easily changed for testing.

PTT safety
----------
PTT is always unkeyed in a ``finally`` block regardless of errors.
If the radio CAT connection is lost during TX, the code continues to
unkey (or at worst logs the error) and transitions to ERROR rather than
leaving PTT stuck.
"""
from __future__ import annotations

import logging
import platform
import threading
import time
from enum import Enum, auto
from typing import Callable, Optional

import numpy as np

from ft8_encode import ft8_encode_message, FT8_FS
from ft8_ntp import Ft8SlotTimer, default_slot_timer
from ft8_qso import OperatorConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Timing constants (all in seconds)
# ---------------------------------------------------------------------------

#: How far before the next slot boundary to key PTT and open audio.
#: 200 ms gives enough time for the rig to fully enter TX mode and for
#: the soundcard driver to fill its first buffer.
PRE_KEY_S: float = 0.20

#: How long to hold PTT after the last audio sample has played.
#: 50 ms keeps the rig keyed while the audio buffer drains.
POST_KEY_S: float = 0.05

#: Maximum drift allowed between the intended slot start and actual wall
#: time.  If the slot is already more than this many seconds past its
#: boundary when the worker thread wakes up, the TX is aborted as "missed".
MISSED_SLOT_THRESHOLD_S: float = 0.5

#: After the pre-key sleep, seconds_to_next_slot() should be approximately
#: PRE_KEY_S (≈ 0.2 s).  If the returned value exceeds this buffer, the slot
#: boundary has already been crossed and we need to check for a missed-slot
#: condition.  Set to 1 s to give a comfortable margin above normal jitter
#: while still catching any overrun well before the next slot starts.
SLOT_OVERRUN_DETECTION_BUFFER_S: float = 1.0

#: Pause inserted before retrying audio playback after a transient
#: PortAudioError on Windows.  USB audio codecs used in ham radio
#: transceivers need time to switch from RX to TX mode after PTT is keyed;
#: 200 ms is sufficient for this settling period.
USB_AUDIO_SWITCH_DELAY_S: float = 0.20

# ---------------------------------------------------------------------------
# TX audio format constants
# ---------------------------------------------------------------------------

#: Output sample rate sent to the soundcard for every TX transmission.
#: 48 000 Hz is universally supported by USB audio codecs used in ham radio
#: transceivers and avoids the ``paInvalidSampleRate`` (PaErrorCode -9997)
#: error that occurs when 12 000 Hz (FT8_FS) is requested directly.
TX_OUTPUT_SAMPLE_RATE: int = 48_000

#: Bit depth (dtype) sent to the soundcard.  16-bit PCM is the lowest common
#: denominator for USB audio class devices and avoids driver-level format
#: conversion that can introduce latency or fail outright on some hardware.
TX_OUTPUT_DTYPE: str = "int16"


# ═══════════════════════════════════════════════════════════════════════════════
# § 0  Module-level audio helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _find_wasapi_output_device(sd, device_index: Optional[int]) -> Optional[int]:
    """
    On Windows, return the WASAPI output-device index that corresponds to
    the physical device at *device_index*.

    The same physical soundcard often appears under multiple host APIs
    (WDM-KS, WASAPI, MME, …).  If the caller-supplied *device_index* points
    to a WDM-KS device, opening a stream through it may trigger
    ``paUnanticipatedHostError`` (PaErrorCode -9999) when the driver rejects
    the requested stream configuration.  This helper locates the equivalent
    WASAPI device so the caller can retry.

    Returns the WASAPI device index on success, or ``None`` if no WASAPI
    host API or no matching device is found.
    """
    try:
        # Locate the WASAPI host API
        wasapi_api_idx: Optional[int] = None
        for api in sd.query_hostapis():
            if "wasapi" in api.get("name", "").lower():
                wasapi_api_idx = api["index"]
                break

        if wasapi_api_idx is None:
            return None

        # Determine the name of the requested device so we can match it
        target_name: Optional[str] = None
        if device_index is not None and device_index >= 0:
            try:
                target_name = sd.query_devices(device_index).get("name", "")
            except Exception:
                pass

        # Collect WASAPI output devices
        all_devices = sd.query_devices()
        if not isinstance(all_devices, list):
            # Single-device systems return a dict — wrap it
            all_devices = [all_devices]

        wasapi_out_devices = [
            d for d in all_devices
            if d.get("hostapi") == wasapi_api_idx
            and d.get("max_output_channels", 0) > 0
        ]

        if not wasapi_out_devices:
            return None

        # If we have a target name, try to find the best match
        if target_name:
            # Exact name match first
            for d in wasapi_out_devices:
                if d.get("name", "") == target_name:
                    return d["index"]
            # Partial name match: sounddevice sometimes appends the host-API
            # name in parentheses (e.g. "USB Audio Device (WDM-KS)").
            # Strip the parenthesised suffix so we can match just the base
            # device name ("USB Audio Device") against the WASAPI entry.
            # The check is bidirectional: the stripped WDM-KS name may be
            # longer than the WASAPI name (e.g. "USB Audio Device" vs
            # "USB Audio"), so we match if either is contained in the other.
            target_stripped = target_name.split("(")[0].strip().lower()
            if target_stripped:
                for d in wasapi_out_devices:
                    candidate_name = d.get("name", "").lower()
                    candidate_stripped = candidate_name.split("(")[0].strip()
                    if target_stripped in candidate_name or (
                        candidate_stripped and candidate_stripped in target_stripped
                    ):
                        return d["index"]

        # Fall back to the default WASAPI output device
        default_out = sd.query_hostapis(wasapi_api_idx).get("default_output_device", -1)
        if default_out is not None and default_out >= 0:
            return default_out

        # Last resort: first available WASAPI output device
        return wasapi_out_devices[0]["index"]

    except (KeyError, IndexError, TypeError, AttributeError, ValueError) as exc:
        # This is a best-effort helper — if the sounddevice API returns
        # unexpected data structures we catch the common dict/list access
        # errors and fall back to returning None so the caller can surface
        # the original PortAudioError rather than an unrelated lookup error.
        logger.debug("_find_wasapi_output_device: lookup failed (%s)", exc)
        return None
    except Exception as exc:  # noqa: BLE001  (sounddevice may raise custom types)
        logger.warning("_find_wasapi_output_device: unexpected error (%s)", exc)
        return None


def _find_mme_output_device(sd, device_index: Optional[int]) -> Optional[int]:
    """
    On Windows, return the MME output-device index that corresponds to
    the physical device at *device_index*.

    MME (MultiMedia Extensions) is the oldest Windows audio API and is
    supported by virtually all audio devices.  This function is used as a
    secondary fallback (after WASAPI) when the WDM-KS host API fails with
    ``paUnanticipatedHostError`` (-9999) and no WASAPI equivalent can be
    found.  Unlike WDM-KS, MME routes audio through the Windows audio
    mixer (wdmaud.sys) and does not attempt kernel-level sample-rate
    negotiation via ``KSPROPERTY_AUDIO_SAMPLING_FREQ``.

    Returns the MME device index on success, or ``None`` if no MME host
    API or no matching device is found.
    """
    try:
        # Locate the MME host API
        mme_api_idx: Optional[int] = None
        for api in sd.query_hostapis():
            if api.get("name", "").lower() == "mme":
                mme_api_idx = api["index"]
                break

        if mme_api_idx is None:
            return None

        # Determine the name of the requested device so we can match it
        target_name: Optional[str] = None
        if device_index is not None and device_index >= 0:
            try:
                target_name = sd.query_devices(device_index).get("name", "")
            except Exception:
                pass

        # Collect MME output devices
        all_devices = sd.query_devices()
        if not isinstance(all_devices, list):
            all_devices = [all_devices]

        mme_out_devices = [
            d for d in all_devices
            if d.get("hostapi") == mme_api_idx
            and d.get("max_output_channels", 0) > 0
        ]

        if not mme_out_devices:
            return None

        # If we have a target name, try to find the best match
        if target_name:
            # Exact name match first
            for d in mme_out_devices:
                if d.get("name", "") == target_name:
                    return d["index"]
            # Partial name match: MME device names may have a numeric prefix
            # (e.g. "5- USB Audio CODEC") while the WDM-KS name is plain
            # ("USB Audio CODEC").  Strip leading digits and the
            # host-API suffix in parentheses from both sides before comparing.
            def _normalize_device_name(name: str) -> str:
                if not isinstance(name, str):
                    return ""
                # Drop any host-API suffix in parentheses.
                base = name.split("(")[0]
                base = base.lstrip()
                # Strip leading digits and common delimiter characters.
                i = 0
                while i < len(base) and (base[i].isdigit() or base[i] in "-. "):
                    i += 1
                return base[i:].strip().lower()

            target_stripped = _normalize_device_name(target_name)
            if target_stripped:
                for d in mme_out_devices:
                    candidate_stripped = _normalize_device_name(d.get("name", ""))
                    if candidate_stripped and (
                        target_stripped == candidate_stripped
                        or target_stripped in candidate_stripped
                        or candidate_stripped in target_stripped
                    ):
                        return d["index"]

        # Fall back to the default MME output device
        default_out = sd.query_hostapis(mme_api_idx).get("default_output_device", -1)
        if default_out is not None and default_out >= 0:
            return default_out

        # Last resort: first available MME output device
        return mme_out_devices[0]["index"]

    except (KeyError, IndexError, TypeError, AttributeError, ValueError) as exc:
        logger.debug("_find_mme_output_device: lookup failed (%s)", exc)
        return None
    except Exception as exc:  # noqa: BLE001
        logger.warning("_find_mme_output_device: unexpected error (%s)", exc)
        return None


def _is_wdm_ks_device(sd, device_index: Optional[int]) -> bool:
    """
    Return True if the device at *device_index* is exposed through the
    WDM-KS (Windows Driver Model – Kernel Streaming) host API.

    WDM-KS requires kernel-level exclusive access and frequently fails with
    ``KSPROPERTY_AUDIO_SAMPLING_FREQ`` (prop_id=10) / ``ERROR_NOT_FOUND``
    (GLE=0x490) errors on USB audio interfaces.  Use this helper to detect
    WDM-KS devices proactively so that a more compatible host API (WASAPI)
    can be selected before the stream is opened.

    Returns False whenever the check cannot be performed (``device_index``
    is None or negative, the device info is unavailable, or any unexpected
    error occurs).
    """
    try:
        if device_index is None or device_index < 0:
            return False
        device_info = sd.query_devices(device_index)
        host_api_idx = device_info.get("hostapi")
        if host_api_idx is None:
            return False
        host_api_info = sd.query_hostapis(host_api_idx)
        api_name = host_api_info.get("name", "").lower()
        # PortAudio reports this host API as "Windows WDM-KS"; match the
        # canonical hyphenated form to avoid false positives.
        return "wdm-ks" in api_name
    except (KeyError, IndexError, TypeError, AttributeError, ValueError) as exc:
        logger.debug("_is_wdm_ks_device: lookup failed (%s)", exc)
        return False
    except Exception as exc:  # noqa: BLE001
        logger.warning("_is_wdm_ks_device: unexpected error (%s)", exc)
        return False


def _resample_audio(
    audio: "np.ndarray",
    from_fs: int,
    to_fs: int,
) -> "tuple[np.ndarray, int]":
    """
    Resample *audio* from *from_fs* to *to_fs* Hz using polyphase filtering.

    Returns ``(resampled_audio, to_fs)`` on success, or ``(audio, from_fs)``
    if ``scipy`` is unavailable or resampling fails.
    """
    if from_fs == to_fs:
        return audio, from_fs
    try:
        from math import gcd
        from scipy.signal import resample_poly
        g = gcd(to_fs, from_fs)
        resampled = resample_poly(
            audio, to_fs // g, from_fs // g
        ).astype(np.float32)
        return resampled, to_fs
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "_resample_audio: resample %d → %d Hz failed (%s) — "
            "using original rate",
            from_fs, to_fs, exc,
        )
        return audio, from_fs


def _to_int16(audio: "np.ndarray") -> "np.ndarray":
    """
    Convert a float32 audio array (values in [-1.0, 1.0]) to int16 PCM.

    Clips the signal before conversion to prevent wrap-around distortion on
    samples that exceed the normalised range.  The result dtype is
    ``numpy.int16``.
    """
    clipped = np.clip(audio, -1.0, 1.0)
    return (clipped * 32767.0).astype(np.int16)


# ═══════════════════════════════════════════════════════════════════════════════
# § 1  TX State
# ═══════════════════════════════════════════════════════════════════════════════

class TxState(Enum):
    """Lifecycle states for a single FT8 transmit job."""
    IDLE     = auto()  # No job scheduled
    ARMED    = auto()  # Job queued; waiting for the next slot
    TX_PREP  = auto()  # Pre-key interval: opening audio, about to key PTT
    TX_ACTIVE = auto() # PTT on, audio playing
    COMPLETE = auto()  # Job finished successfully
    ERROR    = auto()  # Job ended with an error; see last_error
    CANCELED = auto()  # Operator canceled before slot started


# ═══════════════════════════════════════════════════════════════════════════════
# § 2  TX Job
# ═══════════════════════════════════════════════════════════════════════════════

class TxJob:
    """
    Immutable description of a single TX request.

    Parameters
    ----------
    message      : str — FT8 message text to encode and transmit.
    f0_hz        : float — base tone frequency in Hz (default 1500 Hz).
    audio_device : int | None — sounddevice output device index.
                   None / -1 = default system output.
    amplitude    : float — peak amplitude 0.0–1.0 (default 0.5).
    """

    def __init__(
        self,
        message: str,
        *,
        f0_hz: float = 1500.0,
        audio_device: Optional[int] = None,
        amplitude: float = 0.5,
    ) -> None:
        self.message      = message.strip()
        self.f0_hz        = float(f0_hz)
        self.audio_device = audio_device
        self.amplitude    = float(amplitude)

    def __repr__(self) -> str:
        return (
            f"TxJob(message={self.message!r}, f0_hz={self.f0_hz}, "
            f"audio_device={self.audio_device})"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# § 3  TX Coordinator
# ═══════════════════════════════════════════════════════════════════════════════

class Ft8TxCoordinator:
    """
    Orchestrates a manual-assisted FT8 transmit sequence.

    Responsibilities
    ----------------
    • Accept an armed TX job from the GUI.
    • Sleep until the next FT8 slot boundary (minus PRE_KEY_S).
    • Open the audio output device and key CAT PTT.
    • Play the pre-generated FT8 tone audio.
    • Unkey PTT in a guaranteed finally path.
    • Emit event callbacks so the GUI can update status and countdown labels.

    Usage
    -----
    ::
        radio   = Yaesu991AControl(...)
        timer   = Ft8SlotTimer()
        coord   = Ft8TxCoordinator(radio=radio, slot_timer=timer)
        coord.on_state_change = lambda state, msg: print(state, msg)

        job = TxJob("CQ W4ABC EN52")
        coord.arm(job)   # schedules for next slot
        # ... operator can call coord.cancel() any time before TX_PREP
        # GUI updates via on_state_change callback
    """

    def __init__(
        self,
        radio=None,
        slot_timer: Optional[Ft8SlotTimer] = None,
        *,
        pre_key_s: float = PRE_KEY_S,
        post_key_s: float = POST_KEY_S,
        missed_threshold_s: float = MISSED_SLOT_THRESHOLD_S,
    ) -> None:
        """
        Parameters
        ----------
        radio          : Yaesu991AControl (or duck-type) with .ptt_on() /
                         .ptt_off() / .is_connected().  May be None for unit
                         testing without hardware.
        slot_timer     : Ft8SlotTimer instance; defaults to the module-level
                         default_slot_timer if None.
        pre_key_s      : seconds before slot boundary to key PTT.
        post_key_s     : seconds to hold PTT after audio ends.
        missed_threshold_s : abort if slot start is this many seconds late.
        """
        self._radio             = radio
        self._timer: Ft8SlotTimer = (
            slot_timer if slot_timer is not None else default_slot_timer
        )
        self._pre_key_s         = float(pre_key_s)
        self._post_key_s        = float(post_key_s)
        self._missed_s          = float(missed_threshold_s)

        self._lock              = threading.Lock()
        self._state             = TxState.IDLE
        self._current_job: Optional[TxJob] = None
        self._worker: Optional[threading.Thread] = None
        self._cancel_event      = threading.Event()
        self._last_error: Optional[str] = None

        # Optional callbacks — set by the GUI or tests
        # Signature: (new_state: TxState, message: str) -> None
        self.on_state_change: Optional[Callable[[TxState, str], None]] = None

    # ── Properties ─────────────────────────────────────────────────────────

    @property
    def state(self) -> TxState:
        """Current TX state (thread-safe snapshot)."""
        with self._lock:
            return self._state

    @property
    def last_error(self) -> Optional[str]:
        """Human-readable description of the last error, or None."""
        with self._lock:
            return self._last_error

    @property
    def current_job(self) -> Optional[TxJob]:
        """The currently armed/active TX job, or None."""
        with self._lock:
            return self._current_job

    # ── Public API ──────────────────────────────────────────────────────────

    def arm(self, job: TxJob) -> None:
        """
        Arm a TX job for the next valid FT8 slot.

        Validates pre-conditions and starts a background worker thread that
        waits for the slot boundary, then executes the TX sequence.

        Parameters
        ----------
        job : TxJob — the message/audio parameters to transmit.

        Raises
        ------
        RuntimeError  if any pre-condition is not met (CAT not connected,
                      invalid audio device, another TX job already active).
        """
        with self._lock:
            self._validate_arm(job)   # raises RuntimeError on failure
            self._state        = TxState.ARMED
            self._current_job  = job
            self._last_error   = None
            self._cancel_event.clear()

        self._notify(TxState.ARMED, f"Armed: {job.message!r} — waiting for next slot")

        t = threading.Thread(target=self._worker_main, daemon=True, name="Ft8TxWorker")
        with self._lock:
            self._worker = t
        t.start()

    def cancel(self) -> bool:
        """
        Request cancellation of the current TX job.

        Safe to call from any thread.  Only effective when state is ARMED;
        once TX_PREP or TX_ACTIVE has started the job cannot be interrupted
        (the FT8 transmission window is too short to safely stop mid-tone).

        Returns
        -------
        bool — True if cancellation was accepted; False if state is not ARMED.
        """
        with self._lock:
            if self._state != TxState.ARMED:
                return False
            self._cancel_event.set()
        # The worker thread will detect the event and transition to CANCELED.
        return True

    def reset(self) -> None:
        """
        Reset from a terminal state (COMPLETE / ERROR / CANCELED) to IDLE.

        Must not be called while ARMED, TX_PREP, or TX_ACTIVE — those
        states are managed exclusively by the worker thread.
        """
        with self._lock:
            if self._state in (
                TxState.COMPLETE, TxState.ERROR, TxState.CANCELED
            ):
                self._state       = TxState.IDLE
                self._current_job = None
                self._last_error  = None
                self._worker      = None

    def seconds_to_next_slot(self) -> float:
        """Convenience: seconds until the next FT8 slot boundary."""
        return self._timer.seconds_to_next_slot()

    # ── Internal worker ────────────────────────────────────────────────────

    def _worker_main(self) -> None:
        """
        Background thread entry point.

        Waits for the slot boundary, then drives the PTT + audio + PTT-off
        sequence.  Always unkeys PTT in a finally block.
        """
        job: TxJob
        with self._lock:
            job = self._current_job  # type: ignore[assignment]

        ptt_keyed = False
        try:
            # ── Pre-phase: Pre-generate audio during ARMED period ─────────
            # Generate the FT8 audio waveform immediately — the message is
            # already fixed at arm() time so there is no reason to defer
            # encoding to TX_PREP.  Doing it here:
            #   • surfaces encoding errors before PTT is keyed,
            #   • eliminates encoding latency from the critical TX window.
            audio = ft8_encode_message(
                job.message,
                f0_hz=job.f0_hz,
                fs=FT8_FS,
                amplitude=job.amplitude,
            )
            logger.debug(
                "_worker_main: audio pre-generated (%d samples) during ARMED phase",
                len(audio),
            )

            # ── Phase 1: Wait for slot boundary - PRE_KEY_S ──────────────
            wait = self._timer.seconds_to_next_slot() - self._pre_key_s
            if wait > 0:
                # Break the sleep into 50 ms intervals to check for cancel
                deadline = time.monotonic() + wait
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    if self._cancel_event.wait(timeout=min(0.05, remaining)):
                        # Operator canceled
                        self._transition(TxState.CANCELED, "Canceled by operator")
                        return

            # ── Phase 2: Check for missed slot ───────────────────────────
            slot_overrun = self._timer.seconds_to_next_slot()
            # After the pre-key sleep, seconds_to_next_slot() should be
            # approximately PRE_KEY_S (i.e. the slot is still a fraction of a
            # second away).  If the returned value is significantly larger than
            # that, the slot boundary has already been crossed — either the OS
            # scheduler woke us late or wall-clock jitter pushed us past the
            # boundary.  In that case compute how many seconds have elapsed
            # since the boundary and abort if it exceeds the threshold.
            #
            # Heuristic: any value > pre_key_s + SLOT_OVERRUN_DETECTION_BUFFER_S
            # means the boundary just passed (slot_remaining ≈ slot_duration − elapsed).
            if slot_overrun > self._pre_key_s + SLOT_OVERRUN_DETECTION_BUFFER_S:
                elapsed_past = 15.0 - slot_overrun
                if elapsed_past > self._missed_s:
                    msg = (
                        f"Missed TX slot by {elapsed_past:.2f} s "
                        f"(threshold {self._missed_s:.2f} s) — aborted"
                    )
                    logger.warning(msg)
                    with self._lock:
                        self._last_error = msg
                    self._transition(TxState.ERROR, msg)
                    return

            # ── Phase 3: TX_PREP ─────────────────────────────────────────
            self._transition(TxState.TX_PREP, "Preparing TX…")

            # Audio was already generated during the ARMED phase (see
            # pre-phase above), so TX_PREP is now just a state-change marker.

            # ── Phase 4: Key PTT ─────────────────────────────────────────
            self._transition(TxState.TX_ACTIVE, f"TX active: {job.message!r}")
            self._ptt_on()
            ptt_keyed = True

            # Wait until the exact slot boundary before playing audio
            # (PTT was keyed early to give the rig time to enter TX mode)
            slot_remaining = self._timer.seconds_to_next_slot()
            if 0 < slot_remaining < self._pre_key_s + 0.1:
                time.sleep(slot_remaining)

            # ── Phase 5: Play audio ──────────────────────────────────────
            self._play_audio(audio, device=job.audio_device)

            # ── Phase 6: Post-key hold ───────────────────────────────────
            if self._post_key_s > 0:
                time.sleep(self._post_key_s)

            self._transition(TxState.COMPLETE, f"TX complete: {job.message!r}")

        except Exception as exc:  # noqa: BLE001 — intentional broad catch in TX path
            msg = f"TX error: {exc}"
            logger.exception("Ft8TxCoordinator: unhandled exception in worker")
            with self._lock:
                self._last_error = msg
            self._transition(TxState.ERROR, msg)

        finally:
            # ALWAYS unkey PTT — this is the guaranteed teardown path.
            if ptt_keyed:
                try:
                    self._ptt_off()
                except Exception as exc:  # noqa: BLE001
                    logger.error("PTT unkey failed in finally: %s", exc)

    # ── PTT helpers ────────────────────────────────────────────────────────

    def _ptt_on(self) -> None:
        """Key PTT via CAT.  Logs error but does not raise."""
        if self._radio is None:
            logger.debug("_ptt_on: no radio object (testing mode)")
            return
        try:
            self._radio.ptt_on()
            logger.info("PTT ON")
        except Exception as exc:  # noqa: BLE001
            logger.error("PTT ON failed: %s", exc)
            raise

    def _ptt_off(self) -> None:
        """Unkey PTT via CAT.  Always attempts even after prior errors."""
        if self._radio is None:
            logger.debug("_ptt_off: no radio object (testing mode)")
            return
        try:
            self._radio.ptt_off()
            logger.info("PTT OFF")
        except Exception as exc:  # noqa: BLE001
            logger.error("PTT OFF failed: %s", exc)
            # Do not re-raise — we are in a finally path

    # ── Audio helper ───────────────────────────────────────────────────────

    def _play_audio(
        self,
        audio: np.ndarray,
        *,
        device: Optional[int] = None,
    ) -> None:
        """
        Play ``audio`` (float32 array at FT8_FS Hz) to the specified output
        device, blocking until playback completes.

        If sounddevice is unavailable, logs a warning and returns immediately
        (this allows unit tests to run without a soundcard).

        The audio is always resampled to TX_OUTPUT_SAMPLE_RATE (48 000 Hz) and
        converted to TX_OUTPUT_DTYPE (int16, 16-bit PCM) before opening the
        stream.  This fixed format is universally supported by USB audio codecs
        used in ham radio transceivers and avoids ``paInvalidSampleRate``
        (PaErrorCode -9997) errors that occur when the FT8 native rate
        (12 000 Hz) or float32 data are passed directly.
        """
        try:
            import sounddevice as sd  # local import — optional dependency
        except ImportError:
            logger.warning(
                "_play_audio: sounddevice not installed — audio output skipped"
            )
            return

        # ── On Windows, proactively swap WDM-KS device for WASAPI ───────────
        # WDM-KS requires kernel-level exclusive access and frequently fails
        # with KSPROPERTY_AUDIO_SAMPLING_FREQ (prop_id=10) / ERROR_NOT_FOUND
        # (GLE=0x490) errors on USB audio interfaces.  If the configured
        # device is a WDM-KS device, replace it with the WASAPI equivalent
        # before attempting to open any stream — this eliminates the error
        # entirely rather than waiting for the stream to fail and retrying.
        #
        # Resolve the device index to check: use the caller-supplied index if
        # valid; otherwise look up the system default output.  This ensures
        # the swap also fires when device=None/negative and the default system
        # output happens to be a WDM-KS device.
        effective_device = device
        _proactive_swap_occurred = False
        resolved_device: Optional[int] = (
            device if (device is not None and device >= 0) else None
        )
        if platform.system() == "Windows":
            if resolved_device is None:
                try:
                    resolved_device = sd.default.device[1]
                except Exception:
                    pass
            if _is_wdm_ks_device(sd, resolved_device):
                alt_candidate = _find_wasapi_output_device(sd, resolved_device)
                alt_api_name = "WASAPI"
                if alt_candidate is None:
                    # WASAPI unavailable — try MME as a secondary fallback.
                    # MME routes through the Windows audio mixer and avoids
                    # the kernel-level KSPROPERTY_AUDIO_SAMPLING_FREQ query
                    # that causes WDM-KS to fail with ERROR_NOT_FOUND (0x490)
                    # on many USB audio codecs used in ham radio transceivers.
                    alt_candidate = _find_mme_output_device(sd, resolved_device)
                    alt_api_name = "MME"
                if alt_candidate is not None:
                    logger.info(
                        "_play_audio: device %s is WDM-KS — proactively using "
                        "%s device %d instead",
                        resolved_device, alt_api_name, alt_candidate,
                    )
                    _proactive_swap_occurred = True
                    effective_device = alt_candidate

        dev_kwarg: dict = (
            {} if (effective_device is None or effective_device < 0)
            else {"device": effective_device}
        )

        # ── Prepare fixed-format audio buffer ─────────────────────────────
        # Always resample to TX_OUTPUT_SAMPLE_RATE (48 000 Hz) and convert to
        # TX_OUTPUT_DTYPE (int16).  This fixed format is universally supported
        # by USB audio codecs and avoids driver-level format-negotiation errors
        # (paInvalidSampleRate / paUnanticipatedHostError) that occur when the
        # FT8 native rate (12 000 Hz) or float32 data are passed directly.
        play_audio_f32, play_fs = _resample_audio(audio, FT8_FS, TX_OUTPUT_SAMPLE_RATE)
        if play_fs != TX_OUTPUT_SAMPLE_RATE:
            logger.error(
                "_play_audio: resampling failed to produce required output rate: "
                "expected %d Hz, got %d Hz; refusing playback",
                TX_OUTPUT_SAMPLE_RATE, play_fs,
            )
            raise RuntimeError(
                f"_play_audio requires {TX_OUTPUT_SAMPLE_RATE} Hz output, "
                f"but resampler returned {play_fs} Hz"
            )
        play_audio = _to_int16(play_audio_f32)
        logger.debug(
            "_play_audio: prepared %d samples at %d Hz, dtype=%s for device %s",
            len(play_audio), play_fs, TX_OUTPUT_DTYPE, effective_device,
        )

        try:
            sd.play(play_audio, samplerate=play_fs, **dev_kwarg)
            sd.wait()
        except sd.PortAudioError as exc:
            logger.error("sounddevice PortAudio error during TX: %s", exc)
            # Reactive fallback chain (Windows only):
            # 1. If a different WASAPI device is available (reactive swap not
            #    yet tried), attempt it in sounddevice's default WASAPI mode
            #    (exclusive), then retry in shared mode if that also fails.
            #    sounddevice does not pass WasapiSettings here, so the first
            #    attempt uses PortAudio's WASAPI default (exclusive mode).
            # 2. If the proactive WDM-KS swap already selected this WASAPI
            #    device (_proactive_swap_occurred, wasapi_dev == effective_device),
            #    the first WASAPI attempt already failed; go straight to shared
            #    mode to bypass the KSPROPERTY_AUDIO_SAMPLING_FREQ IOCTL.
            # 3. Try MME as a last-resort alternative audio path.
            # 4. Delay-retry the same device once for transient errors.
            if platform.system() == "Windows":
                wasapi_dev = _find_wasapi_output_device(sd, resolved_device)
                if wasapi_dev is not None and wasapi_dev != effective_device:
                    logger.info(
                        "_play_audio: PortAudioError on device %s — retrying "
                        "with WASAPI device %d",
                        effective_device, wasapi_dev,
                    )
                    # play_audio is already at TX_OUTPUT_SAMPLE_RATE / int16;
                    # no re-resampling needed for the fallback device.
                    try:
                        sd.play(
                            play_audio,
                            samplerate=play_fs,
                            device=wasapi_dev,
                        )
                        sd.wait()
                        return
                    except sd.PortAudioError as exc2:
                        logger.error(
                            "_play_audio: WASAPI fallback also failed: %s", exc2
                        )
                        # First WASAPI attempt failed — retry in shared mode.
                        # Some devices reject the default (exclusive) WASAPI
                        # stream, but accept shared-mode WASAPI where audio is
                        # routed through the Windows Audio Session API mixer.
                        # WasapiSettings was added in sounddevice 0.4.0; skip
                        # the shared-mode step on older versions that lack it.
                        # If shared mode also fails, fall through to the MME
                        # and delay-retry paths below rather than raising
                        # immediately — the device may still respond to those.
                        if hasattr(sd, "WasapiSettings"):
                            logger.info(
                                "_play_audio: retrying WASAPI device %d in "
                                "shared mode",
                                wasapi_dev,
                            )
                            try:
                                ws = sd.WasapiSettings(exclusive=False)
                                sd.play(
                                    play_audio,
                                    samplerate=play_fs,
                                    device=wasapi_dev,
                                    extra_settings=ws,
                                )
                                sd.wait()
                                return
                            except sd.PortAudioError as exc_shared:
                                logger.error(
                                    "_play_audio: WASAPI shared mode also "
                                    "failed: %s",
                                    exc_shared,
                                )
                                # Fall through to MME / delay-retry below.
                        # Fall through to MME / delay-retry — all WASAPI
                        # attempts for this device failed.
                elif _proactive_swap_occurred and wasapi_dev is not None and hasattr(sd, "WasapiSettings"):
                    # NOTE: `_proactive_swap_occurred` guards this branch so
                    # it only fires when a WDM-KS device was detected and
                    # swapped to WASAPI proactively.  Without this guard the
                    # branch would also fire when the user directly selected a
                    # WASAPI device (wasapi_dev == effective_device but no swap
                    # occurred), resulting in an unexpected extra play attempt
                    # and a misleading log message.
                    #
                    # The proactive WDM-KS swap selected this WASAPI device and
                    # the first play attempt (sounddevice's default WASAPI mode,
                    # which is exclusive) just failed.  The default WASAPI mode
                    # can still trigger the same KSPROPERTY_AUDIO_SAMPLING_FREQ
                    # IOCTL (prop_id=10) as WDM-KS on some USB audio codecs.
                    # WASAPI shared mode routes audio through the Windows Audio
                    # Session API mixer and avoids that IOCTL entirely.  Try
                    # shared mode before falling back to MME or the delay-retry.
                    logger.info(
                        "_play_audio: proactive WASAPI device %d also failed "
                        "— retrying in shared mode",
                        wasapi_dev,
                    )
                    try:
                        ws = sd.WasapiSettings(exclusive=False)
                        sd.play(
                            play_audio,
                            samplerate=play_fs,
                            device=wasapi_dev,
                            extra_settings=ws,
                        )
                        sd.wait()
                        return
                    except sd.PortAudioError as exc_shared:
                        logger.error(
                            "_play_audio: WASAPI shared mode also failed on "
                            "device %d: %s",
                            wasapi_dev, exc_shared,
                        )
                        # Fall through to MME / delay-retry below.
                # No WASAPI alternative succeeded — try MME as a last-resort
                # alternative output device. Some USB audio codecs for ham radio
                # transceivers are only exposed under WDM-KS and MME (not WASAPI);
                # MME routes through the Windows audio mixer and does not use
                # kernel-level sample-rate IOCTLs, so it avoids the
                # KSPROPERTY_AUDIO_SAMPLING_FREQ error that causes WDM-KS to fail
                # permanently on these devices. If no different alternative device
                # (WASAPI or MME) is available, we fall back to a transient
                # delay-retry on the same device below.
                mme_dev = _find_mme_output_device(sd, resolved_device)
                if mme_dev is not None and mme_dev != effective_device:
                    logger.info(
                        "_play_audio: PortAudioError on device %s — retrying "
                        "with MME device %d",
                        effective_device, mme_dev,
                    )
                    # play_audio is already at TX_OUTPUT_SAMPLE_RATE / int16.
                    try:
                        sd.play(
                            play_audio,
                            samplerate=play_fs,
                            device=mme_dev,
                        )
                        sd.wait()
                        return
                    except sd.PortAudioError as exc_mme:
                        logger.error(
                            "_play_audio: MME fallback also failed: %s", exc_mme
                        )
                        raise RuntimeError(
                            f"Audio output failed: {exc_mme}"
                        ) from exc_mme
                # No different alternative device (WASAPI or MME) is available.
                # USB audio codecs in ham radio transceivers often need a short
                # settling time to switch from RX to TX mode after PTT is keyed.
                # A single retry after a brief pause recovers from this transient
                # condition without requiring a different audio device.
                logger.info(
                    "_play_audio: retrying after 200 ms pause for device %s "
                    "(transient stream-start error)",
                    effective_device,
                )
                time.sleep(USB_AUDIO_SWITCH_DELAY_S)
                try:
                    sd.play(play_audio, samplerate=play_fs, **dev_kwarg)
                    sd.wait()
                    return
                except sd.PortAudioError as exc_retry:
                    raise RuntimeError(f"Audio output failed: {exc_retry}") from exc_retry
            raise RuntimeError(f"Audio output failed: {exc}") from exc

    # ── State management ───────────────────────────────────────────────────

    def _transition(self, new_state: TxState, message: str) -> None:
        """Thread-safe state transition + callback dispatch."""
        with self._lock:
            self._state = new_state
        logger.info("TxState → %s: %s", new_state.name, message)
        self._notify(new_state, message)

    def _notify(self, state: TxState, message: str) -> None:
        """Invoke on_state_change callback (catches exceptions so TX isn't blocked)."""
        cb = self.on_state_change
        if cb is None:
            return
        try:
            cb(state, message)
        except Exception as exc:  # noqa: BLE001
            logger.warning("on_state_change callback raised: %s", exc)

    # ── Validation ─────────────────────────────────────────────────────────

    def _validate_arm(self, job: TxJob) -> None:
        """
        Check all pre-conditions before accepting an arm() request.

        Called with self._lock held.

        Raises
        ------
        RuntimeError  with a human-readable reason for each failure.
        """
        # No overlapping TX jobs
        if self._state in (TxState.ARMED, TxState.TX_PREP, TxState.TX_ACTIVE):
            raise RuntimeError(
                f"TX already in progress (state={self._state.name}).  "
                "Cancel the current job or wait for it to complete."
            )

        # CAT must be connected
        if self._radio is not None and not self._radio.is_connected():
            raise RuntimeError(
                "CAT not connected — cannot arm TX.  "
                "Connect to the radio first."
            )

        # Message must be non-empty
        if not job.message:
            raise RuntimeError("TX message is empty — cannot arm TX.")

        # Audio device must be specified
        if job.audio_device is not None and job.audio_device < 0:
            raise RuntimeError(
                "No TX audio output device configured (device index < 0).  "
                "Select a TX audio device in Settings before arming TX."
            )


# ═══════════════════════════════════════════════════════════════════════════════
# § 4  Operator validator helper
# ═══════════════════════════════════════════════════════════════════════════════

def validate_operator(callsign: str, grid: str) -> tuple[bool, str]:
    """
    Validate operator callsign and grid locator without raising.

    Returns
    -------
    (True, "")  when both are valid.
    (False, reason)  when validation fails.
    """
    try:
        op = OperatorConfig(callsign=callsign, grid=grid)
        if not op.is_configured():
            return False, "Callsign and grid are required."
        return True, ""
    except ValueError as exc:
        return False, str(exc)
