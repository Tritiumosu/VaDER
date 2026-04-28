"""
ft8_qso.py — FT8 QSO state machine and message composition for VaDER (Milestone 4).

This module provides:
  - OperatorConfig  : stores and validates the operator's callsign and grid
  - Message composition helpers (compose_cq, compose_reply, …)
  - ReceivedMessage : parses a decoded FT8 message string into typed fields
  - QsoState        : enum tracking the lifecycle of a single FT8 QSO
  - Ft8QsoManager   : state machine that selects responses and advances QSO state

Standard FT8 QSO sequence (WSJT-X convention)
----------------------------------------------
Station A (CQ caller)        Station B (answering)
──────────────────────────────────────────────────
1.  CQ W4ABC EN52
                              2.  W4ABC K9XYZ -05
3.  K9XYZ W4ABC R-07
                              4.  W4ABC K9XYZ RR73
5.  K9XYZ W4ABC 73

By convention Station A uses even UTC slots (0 s, 30 s) and Station B uses
odd slots (15 s, 45 s).  Slot parity is advisory; Ft8QsoManager exposes
helpers to query the current slot but does not enforce any timing.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Optional

import numpy as np

from ft8_encode import validate_callsign, ft8_encode_to_symbols
from ft8_ntp import Ft8SlotTimer, default_slot_timer

# Maidenhead grid locator pattern: two letters (A-R) followed by two digits.
# Used to validate the extra field of CQ messages (e.g. ``CQ W4ABC EN52``).
_GRID_PATTERN = re.compile(r"^[A-R]{2}[0-9]{2}$")


# ═══════════════════════════════════════════════════════════════════════════════
# § 1  QSO lifecycle states
# ═══════════════════════════════════════════════════════════════════════════════

class QsoState(Enum):
    """Lifecycle states for a single FT8 QSO exchange."""
    IDLE          = auto()  # No active QSO — ready to call CQ or reply
    CQ_SENT       = auto()  # We sent CQ; waiting for a reply from any station
    REPLY_SENT    = auto()  # We replied to a CQ; waiting for their exchange
    EXCHANGE_SENT = auto()  # We sent our exchange (R+SNR); waiting for RRR/RR73
    RRR_SENT      = auto()  # We sent RR73; waiting for the final 73
    COMPLETE      = auto()  # QSO completed — 73 exchanged
    ABORTED       = auto()  # QSO abandoned (timeout, manual cancel, etc.)


# ═══════════════════════════════════════════════════════════════════════════════
# § 1b  QSO record (for future contact logging — Milestone 5)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class QsoRecord:
    """
    Immutable snapshot of a completed QSO for contact logging (Milestone 5).

    Populated by ``Ft8QsoManager.build_record()`` after the exchange reaches
    ``QsoState.COMPLETE``, or constructed directly for voice-mode contacts.
    Intended for ADIF logging.  The dataclass is ``frozen=True`` so instances
    are truly immutable once created — safe to cache, hash, or pass across
    thread boundaries.

    Attributes
    ----------
    our_call    : str              — Operator callsign (e.g. 'W4ABC')
    dx_call     : str              — DX station callsign (e.g. 'K9XYZ')
    freq_mhz    : float            — Operating frequency in MHz (0.0 if unknown)
    band        : str              — Band label (e.g. '20m') or '' if not resolved
    mode        : str              — Operating mode, e.g. 'FT8' or 'SSB'
    time_on     : datetime         — UTC start time
    rst_sent    : str              — RST/signal report sent (e.g. '+00', '-07', '59')
    rst_rcvd    : str              — RST/signal report received (e.g. '-05', '59')
    initiated   : str              — 'CQ' if we called CQ, 'REPLY' if we answered
    dx_grid     : str              — DX station Maidenhead grid (may be empty)
    my_grid     : str              — Our Maidenhead grid locator (may be empty)
    tx_pwr_w    : float            — TX power in watts (0.0 if unknown)
    comment     : str              — Optional free-form comment
    """
    our_call:  str
    dx_call:   str
    freq_mhz:  float = 0.0
    band:      str   = ""
    mode:      str   = "FT8"
    time_on:   datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    rst_sent:  str   = "+00"
    rst_rcvd:  str   = "+00"
    initiated: str   = "CQ"      # 'CQ' or 'REPLY'
    dx_grid:   str   = ""        # DX station's Maidenhead grid (from CQ message)
    my_grid:   str   = ""        # Our Maidenhead grid
    tx_pwr_w:  float = 0.0       # TX power in watts
    comment:   str   = ""        # Optional comment

    def adif_date(self) -> str:
        """Return the QSO date formatted for ADIF: YYYYMMDD."""
        return self.time_on.strftime("%Y%m%d")

    def adif_time(self) -> str:
        """Return the QSO time formatted for ADIF: HHMMSS."""
        return self.time_on.strftime("%H%M%S")


# ═══════════════════════════════════════════════════════════════════════════════
# § 2  Operator configuration
# ═══════════════════════════════════════════════════════════════════════════════

class OperatorConfig:
    """
    Holds the operator's callsign and Maidenhead grid locator.

    Performs basic validation on assignment to catch common mistakes before
    any message encoding is attempted.

    Usage
    -----
    op = OperatorConfig()
    op.callsign = 'W4ABC'     # raises ValueError if invalid
    op.grid     = 'EN52'      # raises ValueError if not 4-char Maidenhead
    assert op.is_configured()
    """

    def __init__(self, callsign: str = "", grid: str = "") -> None:
        self._callsign: str = ""
        self._grid: str = ""
        if callsign:
            self.callsign = callsign
        if grid:
            self.grid = grid

    # -- Properties --------------------------------------------------------

    @property
    def callsign(self) -> str:
        """Operator callsign (uppercase, validated)."""
        return self._callsign

    @callsign.setter
    def callsign(self, value: str) -> None:
        """Set operator callsign after ft8_lib format validation."""
        call = value.upper().strip()
        if not validate_callsign(call):
            raise ValueError(
                f"Invalid callsign {value!r}.  "
                "Expected a standard ham callsign (e.g. W4ABC, VK2TIM, K9XYZ)."
            )
        self._callsign = call

    @property
    def grid(self) -> str:
        """4-character Maidenhead grid locator (uppercase, validated)."""
        return self._grid

    @grid.setter
    def grid(self, value: str) -> None:
        """Set grid locator after 4-char Maidenhead format validation."""
        g = value.upper().strip()
        if g and not re.fullmatch(r"[A-R]{2}[0-9]{2}", g):
            raise ValueError(
                f"Invalid grid locator {value!r}.  "
                "Must be a 4-character Maidenhead locator (e.g. EN52, IO91, QF56)."
            )
        self._grid = g

    # -- Helpers -----------------------------------------------------------

    def is_configured(self) -> bool:
        """Return True when both callsign and grid are non-empty."""
        return bool(self._callsign and self._grid)

    def __repr__(self) -> str:
        return (
            f"OperatorConfig(callsign={self._callsign!r}, grid={self._grid!r})"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# § 3  Standard message composition
# ═══════════════════════════════════════════════════════════════════════════════

def compose_cq(our_call: str, our_grid: str) -> str:
    """
    Compose a CQ message.

    Example
    -------
    >>> compose_cq('W4ABC', 'EN52')
    'CQ W4ABC EN52'
    """
    return f"CQ {our_call.upper()} {our_grid.upper()}"


def compose_reply(dx_call: str, our_call: str, snr_db: int) -> str:
    """
    Compose a reply to a received CQ (Station B's opening message).

    Includes the received SNR of the DX station's signal so they know the
    link quality from their end.

    Example
    -------
    >>> compose_reply('W4ABC', 'K9XYZ', -5)
    'W4ABC K9XYZ -05'
    """
    return f"{dx_call.upper()} {our_call.upper()} {snr_db:+03d}"


def compose_exchange(dx_call: str, our_call: str, snr_db: int) -> str:
    """
    Compose the exchange message with 'R' acknowledgement prefix.

    This is Station A's response to Station B's reply; the 'R' confirms
    receipt and is followed by our own SNR measurement.

    Example
    -------
    >>> compose_exchange('K9XYZ', 'W4ABC', -7)
    'K9XYZ W4ABC R-07'
    """
    return f"{dx_call.upper()} {our_call.upper()} R{snr_db:+03d}"


def compose_rrr(dx_call: str, our_call: str) -> str:
    """
    Compose an RRR acknowledgement message.

    Example
    -------
    >>> compose_rrr('W4ABC', 'K9XYZ')
    'W4ABC K9XYZ RRR'
    """
    return f"{dx_call.upper()} {our_call.upper()} RRR"


def compose_rr73(dx_call: str, our_call: str) -> str:
    """
    Compose an RR73 message (roger + farewell combined, saves one exchange).

    Example
    -------
    >>> compose_rr73('W4ABC', 'K9XYZ')
    'W4ABC K9XYZ RR73'
    """
    return f"{dx_call.upper()} {our_call.upper()} RR73"


def compose_73(dx_call: str, our_call: str) -> str:
    """
    Compose a 73 (farewell / QSO complete) message.

    Example
    -------
    >>> compose_73('K9XYZ', 'W4ABC')
    'K9XYZ W4ABC 73'
    """
    return f"{dx_call.upper()} {our_call.upper()} 73"


# ═══════════════════════════════════════════════════════════════════════════════
# § 4  Received-message parser
# ═══════════════════════════════════════════════════════════════════════════════

class ReceivedMessage:
    """
    Parsed representation of a decoded FT8 message.

    Splits the raw string into typed fields so that the QSO state machine
    can react to the message without repeated string parsing.

    Attributes
    ----------
    raw          : str  — original decoded message string
    call1        : str  — first callsign field  (the addressed station)
    call2        : str  — second callsign field  (the transmitting station)
    extra        : str  — third field (grid, SNR, RRR, RR73, 73, or '')
    snr_db       : int | None  — numeric SNR if extra is a dB report
    is_cq        : bool — True when call1 is 'CQ' or 'QRZ'
    is_rrr       : bool — True when extra == 'RRR'
    is_rr73      : bool — True when extra == 'RR73'
    is_73        : bool — True when extra == '73'
    is_r_report  : bool — True when extra is an R-prefixed SNR (e.g. 'R-07')
    """

    def __init__(self, raw: str) -> None:
        self.raw   = raw.strip()
        parts      = self.raw.upper().split()
        self.call1 = parts[0] if len(parts) > 0 else ""
        self.call2 = parts[1] if len(parts) > 1 else ""
        self.extra = parts[2] if len(parts) > 2 else ""

        self.is_cq   = self.call1 in ("CQ", "QRZ")
        self.is_rrr  = self.extra == "RRR"
        self.is_rr73 = self.extra == "RR73"
        self.is_73   = self.extra == "73"
        self.is_r_report = bool(
            len(self.extra) > 1
            and self.extra[0] == "R"
            and self.extra[1] in ("+", "-")
        )

        self.snr_db: Optional[int] = None
        if self.extra and self.extra[0] in ("+", "-"):
            try:
                self.snr_db = int(self.extra)
            except ValueError:
                pass
        elif self.is_r_report:
            try:
                self.snr_db = int(self.extra[1:])
            except ValueError:
                pass

    def is_addressed_to(self, callsign: str) -> bool:
        """Return True when call1 matches *callsign* (case-insensitive)."""
        return self.call1.upper() == callsign.upper()

    def is_from(self, callsign: str) -> bool:
        """Return True when call2 matches *callsign* (case-insensitive)."""
        return self.call2.upper() == callsign.upper()

    def __repr__(self) -> str:
        return f"ReceivedMessage({self.raw!r})"


# ═══════════════════════════════════════════════════════════════════════════════
# § 5  QSO manager
# ═══════════════════════════════════════════════════════════════════════════════

class Ft8QsoManager:
    """
    Manages a single FT8 QSO exchange from initial contact to 73.

    The manager tracks state, composes the correct next message, and exposes
    the encoded 79-symbol tone sequence ready for audio synthesis.

    Calling CQ (Station A)
    ----------------------
    manager = Ft8QsoManager(operator)
    manager.start_cq()                           # queued_tx = 'CQ W4ABC EN52'
    manager.advance('W4ABC K9XYZ -05')           # queued_tx = 'K9XYZ W4ABC R+00'
    manager.advance('K9XYZ W4ABC RR73')          # queued_tx = 'K9XYZ W4ABC 73'
    assert manager.state == QsoState.COMPLETE

    Answering a CQ (Station B)
    --------------------------
    manager = Ft8QsoManager(operator)
    manager.start_from_received('CQ W4ABC EN52', snr_db=-5)
    # queued_tx = 'W4ABC K9XYZ -05'
    manager.advance('K9XYZ W4ABC R-07')          # queued_tx = 'W4ABC K9XYZ RR73'
    manager.advance('W4ABC K9XYZ 73')
    assert manager.state == QsoState.COMPLETE
    """

    def __init__(
        self,
        operator: OperatorConfig,
        slot_timer: Optional[Ft8SlotTimer] = None,
    ) -> None:
        self.operator: OperatorConfig = operator
        self.state: QsoState = QsoState.IDLE
        self._queued_tx: Optional[str] = None
        self._dx_call:   Optional[str] = None
        # SNR reported to us by the DX station (e.g. their measurement of our
        # signal, included in their reply/exchange message).
        self._rx_snr:    Optional[int] = None
        # SNR we reported to the DX station in our most recent exchange message
        # (our measurement of their signal).  Used as rst_sent in build_record().
        self._tx_snr:    Optional[int] = None
        # Maidenhead grid extracted from the DX station's CQ message, if present.
        self._dx_grid:   Optional[str] = None
        # UTC timestamp captured when the QSO session starts (start_cq /
        # start_from_received).  Used as QSO time_on in build_record() so the
        # logged start time matches when the exchange actually began rather
        # than when the record was assembled.
        self._time_on_utc: Optional[datetime] = None
        # NTP-backed slot timer; each manager gets its own instance unless the
        # caller injects one (e.g. for testing or for sharing a pre-synced timer
        # across multiple managers in the same session).
        self._slot_timer: Ft8SlotTimer = (
            slot_timer if slot_timer is not None else Ft8SlotTimer()
        )

    # ── Public API ─────────────────────────────────────────────────────────

    @property
    def dx_callsign(self) -> Optional[str]:
        """
        Return the active DX partner's callsign, or ``None`` if not yet known.

        Populated once the QSO manager receives the first reply that identifies
        the remote station (via :meth:`start_from_received` or
        :meth:`advance`).  Read-only — use the state machine methods to update
        QSO state rather than mutating this directly.
        """
        return self._dx_call

    def start_cq(self) -> str:
        """
        Compose a CQ message and queue it for the next TX slot.

        Returns
        -------
        str  The CQ message text (e.g. 'CQ W4ABC EN52').

        Raises
        ------
        RuntimeError if the operator callsign is not configured.
        """
        self._require_callsign()
        if not self.operator.grid:
            raise RuntimeError(
                "Operator grid not configured.  "
                "Set operator.grid before calling start_cq()."
            )
        msg = compose_cq(self.operator.callsign, self.operator.grid)
        self._set_tx(msg)
        self.state         = QsoState.CQ_SENT
        self._dx_call      = None
        self._time_on_utc  = datetime.now(timezone.utc)
        return msg

    def start_from_received(
        self,
        received_msg: str,
        *,
        snr_db: int = 0,
    ) -> Optional[str]:
        """
        Select a received message and queue the appropriate response.

        Call this when the operator clicks on a decoded FT8 line to begin a
        QSO.  The *snr_db* value should be the SNR reported by the decoder
        for that message so it can be included in the response.

        Parameters
        ----------
        received_msg : str  — raw decoded FT8 message string
        snr_db       : int  — signal SNR of the received message (dB)

        Returns
        -------
        str | None — the message queued for TX, or None if we cannot respond.
        """
        self._require_callsign()
        rx = ReceivedMessage(received_msg)

        if rx.is_cq:
            # Reply to any CQ
            self._dx_call     = rx.call2
            self._rx_snr      = rx.snr_db
            self._time_on_utc = datetime.now(timezone.utc)
            # Capture their grid from the CQ message (extra field), if present
            self._dx_grid = rx.extra if (rx.extra and _GRID_PATTERN.match(rx.extra)) else None
            msg = compose_reply(rx.call2, self.operator.callsign, snr_db)
            self._tx_snr = snr_db  # SNR we reported in our reply
            self._set_tx(msg)
            self.state = QsoState.REPLY_SENT
            return msg

        if rx.is_addressed_to(self.operator.callsign):
            # Someone is already calling us — delegate to advance()
            return self.advance(received_msg, snr_db=snr_db)

        return None  # Message not addressed to us

    def advance(
        self,
        received_msg: str,
        *,
        snr_db: int = 0,
    ) -> Optional[str]:
        """
        Process a received message and advance the QSO state machine.

        Parameters
        ----------
        received_msg : str  — decoded FT8 message from the DX station
        snr_db       : int  — SNR of the received message (dB); used when we
                              need to include a signal report in our reply

        Returns
        -------
        str | None — next TX message, or None when the QSO is complete.
        """
        self._require_callsign()
        rx  = ReceivedMessage(received_msg)
        msg: Optional[str] = None

        if self.state == QsoState.CQ_SENT:
            # Waiting for any station to reply to our CQ
            if rx.is_addressed_to(self.operator.callsign) and rx.call2:
                self._dx_call = rx.call2
                self._rx_snr  = rx.snr_db
                msg = compose_exchange(rx.call2, self.operator.callsign, snr_db)
                self._tx_snr  = snr_db  # SNR we reported to the DX station
                self._set_tx(msg)
                self.state = QsoState.EXCHANGE_SENT

        elif self.state == QsoState.REPLY_SENT:
            # We replied to a CQ; waiting for DX to send us an exchange
            if (
                self._dx_call
                and rx.is_addressed_to(self.operator.callsign)
                and rx.is_from(self._dx_call)
            ):
                msg = compose_rr73(self._dx_call, self.operator.callsign)
                self._set_tx(msg)
                self.state = QsoState.RRR_SENT

        elif self.state == QsoState.EXCHANGE_SENT:
            # We sent an exchange; waiting for RRR or RR73 from the locked DX
            if (
                self._dx_call
                and rx.is_addressed_to(self.operator.callsign)
                and rx.is_from(self._dx_call)
                and (rx.is_rrr or rx.is_rr73)
            ):
                msg = compose_73(self._dx_call, self.operator.callsign)
                self._set_tx(msg)
                self.state = QsoState.COMPLETE

        elif self.state == QsoState.RRR_SENT:
            # We sent RR73; optionally wait for their confirming 73
            if (
                self._dx_call
                and rx.is_addressed_to(self.operator.callsign)
                and rx.is_from(self._dx_call)
                and rx.is_73
            ):
                self._queued_tx = None
                self.state = QsoState.COMPLETE

        return msg

    def get_queued_tx(self) -> Optional[str]:
        """Return the message text currently queued for transmission."""
        return self._queued_tx

    def get_queued_symbols(self) -> Optional[np.ndarray]:
        """
        Return the 79-symbol tone sequence for the queued TX message.

        Returns None if no message is queued.  The caller is responsible for
        audio synthesis (see ft8_encode.ft8_symbols_to_audio).
        """
        if self._queued_tx is None:
            return None
        return ft8_encode_to_symbols(self._queued_tx)

    def clear_tx(self) -> None:
        """Clear the queued TX message (call after the transmission completes)."""
        self._queued_tx = None

    def abort(self) -> None:
        """Abort the current QSO and return to IDLE."""
        self.state         = QsoState.ABORTED
        self._queued_tx    = None
        self._dx_call      = None
        self._rx_snr       = None
        self._tx_snr       = None
        self._dx_grid      = None
        self._time_on_utc  = None

    def reset(self) -> None:
        """Reset the manager to IDLE state, ready for a new QSO."""
        self.state         = QsoState.IDLE
        self._queued_tx    = None
        self._dx_call      = None
        self._rx_snr       = None
        self._tx_snr       = None
        self._dx_grid      = None
        self._time_on_utc  = None

    @property
    def is_active(self) -> bool:
        """True when a QSO is in progress (not IDLE, COMPLETE, or ABORTED)."""
        return self.state not in (
            QsoState.IDLE, QsoState.COMPLETE, QsoState.ABORTED
        )

    @property
    def dx_call(self) -> Optional[str]:
        """Callsign of the DX station we are working, or None."""
        return self._dx_call

    @property
    def dx_grid(self) -> Optional[str]:
        """
        Maidenhead grid of the DX station, or ``None`` if not yet known.

        Populated from the ``extra`` field of a received CQ message when the
        DX station includes their grid (standard FT8 CQ format: ``CQ CALL GRID``).
        """
        return self._dx_grid

    def build_record(
        self,
        *,
        freq_mhz: float = 0.0,
        band: str = "",
        snr_sent: Optional[int] = None,
        initiated: str = "CQ",
        my_grid: str = "",
    ) -> "QsoRecord":
        """
        Build a :class:`QsoRecord` snapshot from the current QSO state.

        Should only be called after the QSO reaches ``QsoState.COMPLETE``.
        Raises ``RuntimeError`` if the QSO is not complete or has no DX call.

        Parameters
        ----------
        freq_mhz : float        — Operating frequency in MHz (pass from radio poll).
        band     : str          — Band label (e.g. '20m'); caller resolves from freq.
        snr_sent : int | None   — Signal report we sent to the DX station (dB).
                                  When *None* (default), the value tracked internally
                                  by the state machine (``_tx_snr``) is used.
        initiated: str          — ``'CQ'`` if we called CQ, ``'REPLY'`` if we answered.
        my_grid  : str          — Our Maidenhead grid locator (MY_GRIDSQUARE).
        """
        if self.state not in (QsoState.COMPLETE,):
            raise RuntimeError(
                f"QSO is not complete (state={self.state.name}); "
                "cannot build a contact record."
            )
        if not self._dx_call:
            raise RuntimeError("No DX callsign recorded; cannot build a contact record.")
        rst_rcvd = f"{self._rx_snr:+03d}" if self._rx_snr is not None else "+00"
        # Use caller-supplied snr_sent if provided; fall back to the value the
        # state machine tracked from compose_exchange / compose_reply.
        effective_sent = snr_sent if snr_sent is not None else (self._tx_snr or 0)
        rst_sent = f"{effective_sent:+03d}"
        # Use the timestamp captured when the session started (start_cq /
        # start_from_received), falling back to now() only as a safety net.
        time_on = self._time_on_utc if self._time_on_utc is not None else datetime.now(timezone.utc)
        return QsoRecord(
            our_call  = self.operator.callsign,
            dx_call   = self._dx_call,
            freq_mhz  = freq_mhz,
            band      = band,
            time_on   = time_on,
            rst_sent  = rst_sent,
            rst_rcvd  = rst_rcvd,
            initiated = initiated,
            dx_grid   = self._dx_grid or "",
            my_grid   = my_grid,
        )

    # ── Timeslot helpers ──────────────────────────────────────────────────

    def seconds_to_next_slot(self) -> float:
        """
        Return the number of seconds until the start of the next FT8 UTC slot.

        FT8 slots begin at :00, :15, :30, and :45 of each UTC minute.
        Delegates to the Ft8SlotTimer so the result uses NTP-corrected time
        when a sync has been performed, otherwise falls back to local system
        time.
        """
        return self._slot_timer.seconds_to_next_slot()

    def current_slot_parity(self) -> int:
        """
        Return 0 for even slots (:00, :30) or 1 for odd slots (:15, :45).

        By WSJT-X convention Station A (CQ caller) transmits in even slots
        and Station B (answering station) transmits in odd slots.
        Delegates to the Ft8SlotTimer for NTP-corrected accuracy.
        """
        return self._slot_timer.current_slot_parity()

    def next_slot_utc(self):
        """
        Return a UTC datetime for the start of the next FT8 slot boundary.

        Useful for displaying the scheduled TX time to the operator or for
        passing to threading.Timer / asyncio.call_at.
        """
        return self._slot_timer.next_slot_utc()

    # ── Private helpers ───────────────────────────────────────────────────

    def _set_tx(self, msg: str) -> None:
        """Queue a message for transmission."""
        self._queued_tx = msg

    def _require_callsign(self) -> None:
        if not self.operator.callsign:
            raise RuntimeError(
                "Operator callsign is not configured.  "
                "Set operator.callsign before starting a QSO."
            )
