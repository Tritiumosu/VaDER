"""
adif_log.py — ADIF (Amateur Data Interchange Format) contact logging for VaDER.

Provides a lightweight writer for the ADIF 3.1.4 file format used by all
major amateur-radio logging applications (WSJT-X, JTDX, HRD, etc.).

Public API
----------
AdifContact         — dataclass representing a single QSO for file storage
append_adif_contact — thread-safe append of one contact to an .adi file
qso_record_to_adif_contact — convert a ft8_qso.QsoRecord to AdifContact
"""
from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

ADIF_VERSION = "3.1.4"
PROGRAM_ID   = "VaDER"

# One global write lock so multiple threads can safely append to the same file.
_write_lock: threading.Lock = threading.Lock()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _adif_field(name: str, value: str) -> str:
    """
    Format a single ADIF field tag.

    Returns ``<NAME:LEN>VALUE`` or an empty string when *value* is empty.
    """
    v = str(value)
    if not v:
        return ""
    return f"<{name.upper()}:{len(v)}>{v}"


def _ensure_header(path: str) -> None:
    """
    Write an ADIF file header to *path* when the file is absent or empty.

    The header is written only once; subsequent calls on the same non-empty
    file are a no-op.
    """
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(
            f"{_adif_field('ADIF_VER', ADIF_VERSION)} "
            f"{_adif_field('PROGRAMID', PROGRAM_ID)}"
            " <EOH>\n"
        )


# ---------------------------------------------------------------------------
# Public dataclass
# ---------------------------------------------------------------------------

@dataclass
class AdifContact:
    """
    Represents a single amateur radio contact for ADIF logging.

    All fields correspond to standard ADIF 3.1.4 field names.  Only *call*,
    *qso_date*, and *time_on* are required; all others default to empty
    strings / zero and are omitted from the ADIF record when empty.

    Attributes
    ----------
    call              : str   — DX station callsign (required)
    qso_date          : str   — UTC contact date, ``YYYYMMDD`` (required)
    time_on           : str   — UTC contact start time, ``HHMMSS`` (required)
    freq_mhz          : float — Operating frequency in MHz (0 → omitted)
    band              : str   — Band label, e.g. ``'20m'``
    mode              : str   — Operating mode, e.g. ``'FT8'``, ``'SSB'``
    submode           : str   — Sub-mode (e.g. ``'USB'``)
    rst_sent          : str   — Signal report sent, e.g. ``'59'`` or ``'-05'``
    rst_rcvd          : str   — Signal report received
    station_callsign  : str   — Our callsign (STATION_CALLSIGN)
    my_gridsquare     : str   — Our Maidenhead grid (MY_GRIDSQUARE)
    gridsquare        : str   — Their Maidenhead grid (GRIDSQUARE)
    tx_pwr            : str   — TX power in watts, e.g. ``'10'``
    comment           : str   — Free-form comment
    name              : str   — Their operator name
    """

    call:             str
    qso_date:         str
    time_on:          str
    freq_mhz:         float = 0.0
    band:             str = ""
    mode:             str = ""
    submode:          str = ""
    rst_sent:         str = ""
    rst_rcvd:         str = ""
    station_callsign: str = ""
    my_gridsquare:    str = ""
    gridsquare:       str = ""
    tx_pwr:           str = ""
    comment:          str = ""
    name:             str = ""

    # ── Formatting ──────────────────────────────────────────────────────────

    def to_adif_record(self) -> str:
        """
        Return a complete ADIF record string ending with ``<EOR>``.

        Fields with empty / zero values are omitted.  The record is
        suitable for direct appending to an ``.adi`` log file.
        """
        parts: list[str] = []

        def add(adif_name: str, val: str) -> None:
            f = _adif_field(adif_name, val)
            if f:
                parts.append(f)

        add("CALL",             self.call)
        add("QSO_DATE",         self.qso_date)
        add("TIME_ON",          self.time_on)

        if self.freq_mhz > 0.0:
            # Format to 6 decimal places, strip trailing zeros
            freq_str = f"{self.freq_mhz:.6f}".rstrip("0").rstrip(".")
            add("FREQ", freq_str)

        add("BAND",             self.band)
        add("MODE",             self.mode)
        add("SUBMODE",          self.submode)
        add("RST_SENT",         self.rst_sent)
        add("RST_RCVD",         self.rst_rcvd)
        add("STATION_CALLSIGN", self.station_callsign)
        add("MY_GRIDSQUARE",    self.my_gridsquare)
        add("GRIDSQUARE",       self.gridsquare)
        add("TX_PWR",           self.tx_pwr)
        add("COMMENT",          self.comment)
        add("NAME",             self.name)

        parts.append("<EOR>")
        return " ".join(parts)


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

def append_adif_contact(path: str, contact: AdifContact) -> None:
    """
    Append *contact* to the ADIF log file at *path*.

    If the file does not exist or is empty a standard ADIF header is written
    first.  All writes are serialised with a module-level lock so this
    function is safe to call from multiple threads.

    Parameters
    ----------
    path    : str         — Absolute or relative path to the ``.adi`` file.
    contact : AdifContact — Contact to append.

    Raises
    ------
    OSError  If the file cannot be opened for writing (e.g. permissions).
    """
    with _write_lock:
        _ensure_header(path)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(contact.to_adif_record() + "\n")


def qso_record_to_adif_contact(
    record,             # ft8_qso.QsoRecord
    *,
    my_grid: str = "",
    dx_grid: str = "",
    tx_pwr: str = "",
    comment: str = "",
    operator_name: str = "",
) -> AdifContact:
    """
    Convert a :class:`~ft8_qso.QsoRecord` to an :class:`AdifContact`.

    This helper bridges the FT8 QSO state machine (which produces
    ``QsoRecord`` objects) and the ADIF file writer.

    Grid squares default to the values stored on the *record* itself
    (``record.my_grid`` / ``record.dx_grid``); the *my_grid* and *dx_grid*
    kwargs only need to be supplied when the caller wants to override those.

    Parameters
    ----------
    record        : QsoRecord  — Completed QSO record from ``Ft8QsoManager``.
    my_grid       : str        — Our Maidenhead grid (MY_GRIDSQUARE).
                                 Defaults to ``record.my_grid`` when blank.
    dx_grid       : str        — Their grid, if decoded from their CQ message.
                                 Defaults to ``record.dx_grid`` when blank.
    tx_pwr        : str        — TX power in watts (string, e.g. ``'10'``).
    comment       : str        — Optional free-form comment.
    operator_name : str        — Their operator name, if known.
    """
    resolved_my_grid = my_grid or getattr(record, "my_grid", "")
    resolved_dx_grid = dx_grid or getattr(record, "dx_grid", "")
    return AdifContact(
        call=record.dx_call,
        qso_date=record.adif_date(),
        time_on=record.adif_time(),
        freq_mhz=record.freq_mhz,
        band=record.band,
        mode=record.mode,
        rst_sent=record.rst_sent,
        rst_rcvd=record.rst_rcvd,
        station_callsign=record.our_call,
        my_gridsquare=resolved_my_grid,
        gridsquare=resolved_dx_grid,
        tx_pwr=tx_pwr,
        comment=comment,
        name=operator_name,
    )
