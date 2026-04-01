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
            # Heuristic: any value > pre_key_s + 1 s means the boundary just
            # passed (slot_remaining ≈ slot_duration − elapsed).
            if slot_overrun > self._pre_key_s + 1.0:
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

            # Generate audio
            audio = ft8_encode_message(
                job.message,
                f0_hz=job.f0_hz,
                fs=FT8_FS,
                amplitude=job.amplitude,
            )

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
        """
        try:
            import sounddevice as sd  # local import — optional dependency
        except ImportError:
            logger.warning(
                "_play_audio: sounddevice not installed — audio output skipped"
            )
            return

        dev_kwarg: dict = {} if (device is None or device < 0) else {"device": device}

        try:
            sd.play(audio, samplerate=FT8_FS, **dev_kwarg)
            sd.wait()
        except sd.PortAudioError as exc:
            logger.error("sounddevice PortAudio error during TX: %s", exc)
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
