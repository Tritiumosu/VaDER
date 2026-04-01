"""
ft8_ntp.py — NTP-disciplined time source for FT8 TX slot scheduling (VaDER).

FT8 transmissions must be aligned to 15-second UTC boundaries.  Even a
fraction of a second of clock error causes the transmitted signal to fall
outside the decoder's sync-search window, resulting in a missed decode.
This module provides an NTP-corrected clock and slot-timing helpers so that
VaDER can schedule transmissions precisely.

Architecture
------------
NtpTimeSync
    Queries one or more NTP servers (NIST, Cloudflare, Google, and the NTP
    Pool are included as public defaults).  On success it stores the measured
    clock offset between the local system clock and the authoritative NTP
    time.  All subsequent time reads apply that offset to datetime.now(UTC),
    delivering an NTP-corrected timestamp without any further network round
    trips.

    If every configured server is unreachable or returns an error the object
    falls back silently to the local system clock and logs a warning.  The
    application can always check is_synced to know which source is active.

Ft8SlotTimer
    Wraps an NtpTimeSync instance and exposes the slot-scheduling primitives
    needed by Ft8QsoManager and the transmit path:
      • utc_now()            — NTP-corrected UTC datetime
      • seconds_to_next_slot() — seconds until the next 15 s boundary
      • current_slot_index() — slot number within the minute (0–3)
      • current_slot_parity() — 0 = even slot (A), 1 = odd slot (B)
      • next_slot_utc()      — UTC datetime of the next slot boundary

Public defaults
---------------
DEFAULT_NTP_SERVERS  — ordered list of public NTP servers tried on sync():
    1. time.nist.gov   (NIST Internet Time Service)
    2. time.cloudflare.com
    3. time.google.com
    4. pool.ntp.org    (NTP Pool Project)
"""
from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Callable, Optional

try:
    import ntplib
    _NTPLIB_AVAILABLE = True
except ImportError:                          # graceful fallback if not installed
    _NTPLIB_AVAILABLE = False

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public default NTP server list
# ---------------------------------------------------------------------------
# Ordered by reliability / preference:
#   1. NIST (U.S. National Institute of Standards and Technology)
#   2. Cloudflare — anycast, low latency
#   3. Google     — anycast, high availability
#   4. NTP Pool   — community-run, geographically diverse
DEFAULT_NTP_SERVERS: list[str] = [
    "time.nist.gov",
    "time.cloudflare.com",
    "time.google.com",
    "pool.ntp.org",
]

# How long to wait for a single NTP server before trying the next one (seconds)
_DEFAULT_TIMEOUT_S: float = 3.0

# FT8 slot duration — 15 s per WSJT-X specification
FT8_SLOT_DURATION_S: float = 15.0


# ═══════════════════════════════════════════════════════════════════════════════
# § 1  NtpTimeSync
# ═══════════════════════════════════════════════════════════════════════════════

class NtpTimeSync:
    """
    Multi-server NTP clock with offset caching and automatic fallback.

    Usage
    -----
    sync = NtpTimeSync()           # uses DEFAULT_NTP_SERVERS
    sync.sync()                    # query NTP; stores offset
    now = sync.utc_now()           # NTP-corrected UTC datetime

    # Custom servers:
    sync = NtpTimeSync(servers=['time.nist.gov', 'pool.ntp.org'])

    # Check sync status:
    if sync.is_synced:
        print(f"Clock offset: {sync.offset_s:+.3f} s")
    else:
        print("No NTP sync — using local system clock")

    Thread safety
    -------------
    sync() and utc_now() are safe to call from multiple threads.  A lock
    protects the stored offset so a background re-sync cannot corrupt a
    concurrent read.
    """

    def __init__(
        self,
        servers: Optional[list[str]] = None,
        timeout_s: float = _DEFAULT_TIMEOUT_S,
        ntp_version: int = 3,
    ) -> None:
        """
        Parameters
        ----------
        servers    : list of NTP hostname strings to try in order.
                     Defaults to DEFAULT_NTP_SERVERS.
        timeout_s  : per-server socket timeout in seconds (default 3.0).
        ntp_version: NTP protocol version to request (default 3; use 4 for
                     RFC 5905 — most public servers accept both).
        """
        self._servers:     list[str] = list(servers or DEFAULT_NTP_SERVERS)
        self._timeout_s:   float     = float(timeout_s)
        self._ntp_version: int       = int(ntp_version)

        self._lock:         threading.Lock    = threading.Lock()
        self._offset_s:     Optional[float]  = None   # system_time + offset = NTP time
        self._last_sync_ts: Optional[float]  = None   # monotonic timestamp of last sync
        self._sync_server:  Optional[str]    = None   # server that provided the offset

    # ── Properties ─────────────────────────────────────────────────────────

    @property
    def is_synced(self) -> bool:
        """True when a successful NTP sync has been performed."""
        return self._offset_s is not None

    @property
    def offset_s(self) -> Optional[float]:
        """
        Measured clock offset in seconds: NTP_time = system_time + offset_s.

        Positive offset means the local clock is behind NTP time.
        None when no sync has been performed or all servers failed.
        """
        with self._lock:
            return self._offset_s

    @property
    def last_sync_utc(self) -> Optional[datetime]:
        """UTC datetime of the most recent successful sync, or None."""
        with self._lock:
            if self._last_sync_ts is None:
                return None
            # Convert monotonic → wall time via current offset
            mono_elapsed = time.monotonic() - self._last_sync_ts
            return datetime.now(tz=timezone.utc) - timedelta(seconds=mono_elapsed)

    @property
    def sync_server(self) -> Optional[str]:
        """Hostname of the NTP server used for the last successful sync."""
        with self._lock:
            return self._sync_server

    @property
    def servers(self) -> list[str]:
        """Configured NTP server list (copy)."""
        return list(self._servers)

    # ── Public methods ──────────────────────────────────────────────────────

    def sync(self) -> bool:
        """
        Query NTP servers in order and store the first successful offset.

        Tries each configured server once.  Stops at the first success.
        If every server fails (network error, timeout, bad response) the
        existing offset is preserved (if any) and the method returns False.

        Returns
        -------
        bool — True on success, False if all servers failed.
        """
        if not _NTPLIB_AVAILABLE:
            logger.warning(
                "ntplib is not installed — NTP sync unavailable.  "
                "Install it with: pip install ntplib"
            )
            return False

        client = ntplib.NTPClient()
        for server in self._servers:
            try:
                response = client.request(
                    server,
                    version=self._ntp_version,
                    timeout=self._timeout_s,
                )
                offset = float(response.offset)
                with self._lock:
                    self._offset_s     = offset
                    self._last_sync_ts = time.monotonic()
                    self._sync_server  = server
                logger.info(
                    "NTP sync OK via %s — offset %+.3f s  delay %.3f s",
                    server, offset, response.delay,
                )
                return True

            except ntplib.NTPException as exc:
                logger.warning("NTP error from %s: %s", server, exc)
            except OSError as exc:
                # Covers socket.gaierror (DNS failure), connection refused, etc.
                logger.warning("NTP socket error for %s: %s", server, exc)
            except Exception as exc:              # noqa: BLE001  broad-catch intentional
                logger.warning("Unexpected NTP error for %s: %s", server, exc)

        logger.warning(
            "NTP sync failed — all %d servers unreachable.  "
            "Slot scheduling will use the local system clock.",
            len(self._servers),
        )
        return False

    def sync_async(self, callback: Optional[Callable[[bool], None]] = None) -> threading.Thread:
        """
        Run sync() in a background thread and return the Thread object.

        This keeps the GUI responsive during the initial NTP query on startup.

        Parameters
        ----------
        callback : optional callable(success: bool) called on the background
                   thread when sync completes.  Use root.after() in tkinter to
                   marshal the result back to the main thread if needed.
        """
        def _worker() -> None:
            ok = self.sync()
            if callback is not None:
                try:
                    callback(ok)
                except Exception as exc:        # noqa: BLE001
                    logger.warning("NTP callback raised: %s", exc)

        t = threading.Thread(target=_worker, name="NtpSync", daemon=True)
        t.start()
        return t

    def utc_now(self) -> datetime:
        """
        Return the current UTC time corrected for the measured NTP offset.

        If no sync has been performed (or all servers failed), the local
        system clock is returned unchanged — identical to
        datetime.now(timezone.utc).
        """
        with self._lock:
            offset = self._offset_s
        now = datetime.now(tz=timezone.utc)
        if offset is not None:
            now += timedelta(seconds=offset)
        return now


# ═══════════════════════════════════════════════════════════════════════════════
# § 2  Ft8SlotTimer
# ═══════════════════════════════════════════════════════════════════════════════

class Ft8SlotTimer:
    """
    FT8 15-second UTC slot scheduler backed by NTP-corrected time.

    Provides the slot-boundary math required by Ft8QsoManager and the
    transmit pipeline, using an NtpTimeSync for precision timing.

    Parameters
    ----------
    ntp_sync : NtpTimeSync instance to use for corrected time.  If None,
               a default NtpTimeSync() is created (but not yet synced;
               call sync() separately to trigger the initial NTP query).
    slot_duration_s : FT8 slot length in seconds (default 15.0).

    Examples
    --------
    timer = Ft8SlotTimer()
    timer.sync()                        # one-shot NTP sync
    print(timer.seconds_to_next_slot()) # e.g. 7.342
    print(timer.current_slot_parity())  # 0 = even (Station A), 1 = odd (Station B)
    """

    def __init__(
        self,
        ntp_sync: Optional[NtpTimeSync] = None,
        slot_duration_s: float = FT8_SLOT_DURATION_S,
    ) -> None:
        self._ntp:  NtpTimeSync = ntp_sync if ntp_sync is not None else NtpTimeSync()
        self._slot: float       = float(slot_duration_s)

    # ── Sync passthrough ───────────────────────────────────────────────────

    @property
    def ntp_sync(self) -> NtpTimeSync:
        """The underlying NtpTimeSync instance."""
        return self._ntp

    def sync(self) -> bool:
        """Synchronous NTP query — see NtpTimeSync.sync()."""
        return self._ntp.sync()

    def sync_async(self, callback: Optional[Callable[[bool], None]] = None) -> threading.Thread:
        """Asynchronous NTP query — see NtpTimeSync.sync_async()."""
        return self._ntp.sync_async(callback=callback)

    # ── Time source ────────────────────────────────────────────────────────

    def utc_now(self) -> datetime:
        """Return NTP-corrected current UTC time."""
        return self._ntp.utc_now()

    # ── Slot math ─────────────────────────────────────────────────────────

    def seconds_to_next_slot(self) -> float:
        """
        Seconds until the start of the next FT8 UTC slot boundary.

        FT8 slots begin at :00, :15, :30, and :45 of each UTC minute.
        The returned value is in the range (0, slot_duration_s].

        Uses NTP-corrected time when a sync has been performed, otherwise
        falls back to the local system clock.
        """
        now     = self.utc_now()
        elapsed = (now.second % self._slot) + now.microsecond / 1_000_000
        wait    = self._slot - elapsed
        # Clamp to (0, slot_duration_s] — avoid returning 0 exactly, which
        # would cause a spin-wait if called right at a boundary.
        return wait if wait > 0.001 else self._slot

    def current_slot_index(self) -> int:
        """
        Index of the current FT8 slot within the UTC minute (0–3 for 15 s slots).

        Slot 0: :00–:14   Slot 1: :15–:29
        Slot 2: :30–:44   Slot 3: :45–:59
        """
        now = self.utc_now()
        return int(now.second // self._slot)

    def current_slot_parity(self) -> int:
        """
        Return 0 for even slots (:00, :30) or 1 for odd slots (:15, :45).

        By WSJT-X convention Station A (the CQ caller) transmits in even
        slots and Station B (the answering station) transmits in odd slots.
        """
        return self.current_slot_index() % 2

    def next_slot_utc(self) -> datetime:
        """
        UTC datetime of the start of the next FT8 slot boundary.

        Useful for displaying the scheduled TX time to the operator or for
        passing to threading.Timer / asyncio.call_at.
        """
        return self.utc_now() + timedelta(seconds=self.seconds_to_next_slot())


# ═══════════════════════════════════════════════════════════════════════════════
# § 3  Module-level default instance
# ═══════════════════════════════════════════════════════════════════════════════
# A shared Ft8SlotTimer so callers can do:
#
#     from ft8_ntp import default_slot_timer
#     default_slot_timer.sync()
#     wait = default_slot_timer.seconds_to_next_slot()
#
# The instance is created at import time but NOT synced automatically, giving
# the application full control over when the first NTP query happens.

default_slot_timer: Ft8SlotTimer = Ft8SlotTimer()
