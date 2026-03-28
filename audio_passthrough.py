"""
audio_passthrough.py

Real-time audio bridge for voice operating mode.

Two primary classes:
- AudioPassthrough : Routes received radio audio (soundcard input → headphones/speakers).
                     Used for RX monitoring — you hear the radio through your computer.
- AudioTxCapture   : Routes computer microphone audio to the radio's audio input device.
                     Used during PTT — your voice goes to the radio for transmission.

Both classes follow the same lifecycle:
    obj.start()   # open streams and begin routing
    obj.stop()    # close streams and stop routing

An optional `rms_callback(float)` parameter allows the GUI to display a live
level meter without any extra threads.

Requires the optional dependency:
    pip install sounddevice
"""

from __future__ import annotations

import queue
import threading
from typing import Callable, Optional

try:
    import numpy as np
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "audio_passthrough requires numpy. Install it with: pip install numpy"
    ) from e


# ---------------------------------------------------------------------------
# Internal helper – audio output stream
# ---------------------------------------------------------------------------

class SoundCardAudioOutput:
    """
    Plays float32 mono audio through a soundcard output device.

    Call `play(samples)` from any thread to enqueue audio for playback.
    The stream's output callback will drain the queue; silence is produced
    when the queue is empty (underrun).
    """

    def __init__(
        self,
        *,
        fs: int = 48_000,
        block_size: int = 4_800,   # 100 ms at 48 kHz
        device: Optional[int | str] = None,
        max_queue_blocks: int = 20,
    ) -> None:
        self.fs = int(fs)
        self.block_size = int(block_size)
        self.device = device

        self._q: queue.Queue[np.ndarray] = queue.Queue(maxsize=max_queue_blocks)
        self._stream = None
        self._running = False

    # -- Public API --------------------------------------------------------

    def start(self) -> None:
        """Open the output stream and begin playback."""
        if self._running:
            return

        try:
            import sounddevice as sd  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "Live soundcard output requires 'sounddevice'.\n"
                "Install it with: pip install sounddevice"
            ) from exc

        self._running = True
        self._stream = sd.OutputStream(
            samplerate=self.fs,
            blocksize=self.block_size,
            device=self.device,
            channels=1,
            dtype="float32",
            callback=self._audio_callback,
        )
        self._stream.start()

    def stop(self) -> None:
        """Close the output stream."""
        self._running = False
        if self._stream is not None:
            try:
                self._stream.stop()
            finally:
                try:
                    self._stream.close()
                except Exception:
                    pass
            self._stream = None

    def play(self, samples: "np.ndarray") -> None:
        """
        Enqueue mono float32 samples for playback.

        Non-blocking: if the queue is full the newest block is silently dropped.
        """
        try:
            self._q.put_nowait(np.asarray(samples, dtype=np.float32))
        except queue.Full:
            pass

    # -- Internal ----------------------------------------------------------

    def _audio_callback(self, outdata, frames, time_info, status) -> None:
        """sounddevice output callback – runs in the audio thread."""
        try:
            data = self._q.get_nowait()
        except queue.Empty:
            outdata[:] = 0.0
            return

        n = min(len(data), frames)
        outdata[:n, 0] = data[:n]
        if n < frames:
            outdata[n:] = 0.0


# ---------------------------------------------------------------------------
# RX Monitoring – radio audio → headphones
# ---------------------------------------------------------------------------

class AudioPassthrough:
    """
    Routes audio from a soundcard **input** device (radio receive audio) to a
    soundcard **output** device (headphones / computer speakers).

    Typical use in a ham radio setup:
        input_device  = USB audio interface connected to radio audio output (e.g. headphone jack)
        output_device = computer headphones or speakers

    Usage::

        pt = AudioPassthrough(
            input_device=3,
            output_device=5,
            rms_callback=lambda rms: print(f"RX level: {rms:.4f}"),
        )
        pt.start()
        # ... (the radio audio now comes through the computer speakers)
        pt.stop()
    """

    def __init__(
        self,
        input_device: Optional[int | str],
        output_device: Optional[int | str],
        *,
        fs: int = 48_000,
        block_size: int = 4_800,
        rms_callback: Optional[Callable[[float], None]] = None,
    ) -> None:
        self._input_device  = input_device
        self._output_device = output_device
        self._fs            = int(fs)
        self._block_size    = int(block_size)
        self._rms_callback  = rms_callback

        self._out     = SoundCardAudioOutput(fs=fs, block_size=block_size, device=output_device)
        self._src     = None   # sounddevice.InputStream; created in start()
        self._thread  = None
        self._running = False
        self._stop_ev = threading.Event()

        # Internal queue fed by the sounddevice input callback
        self._in_q: queue.Queue[np.ndarray] = queue.Queue(maxsize=20)

    # -- Public API --------------------------------------------------------

    def start(self) -> None:
        """Open both audio streams and begin routing audio."""
        if self._running:
            return

        try:
            import sounddevice as sd  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "AudioPassthrough requires 'sounddevice'.\n"
                "Install it with: pip install sounddevice"
            ) from exc

        self._stop_ev.clear()
        self._running = True

        self._out.start()

        self._src = sd.InputStream(
            samplerate=self._fs,
            blocksize=self._block_size,
            device=self._input_device,
            channels=1,
            dtype="float32",
            callback=self._in_callback,
        )
        self._src.start()

        self._thread = threading.Thread(target=self._worker, daemon=True, name="AudioPassthrough")
        self._thread.start()

    def stop(self) -> None:
        """Stop audio routing and close both streams."""
        self._running = False
        self._stop_ev.set()

        if self._src is not None:
            try:
                self._src.stop()
            finally:
                try:
                    self._src.close()
                except Exception:
                    pass
            self._src = None

        self._out.stop()

        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    # -- Internal ----------------------------------------------------------

    def _in_callback(self, indata, frames, time_info, status) -> None:
        """sounddevice input callback – copy data to the internal queue."""
        try:
            self._in_q.put_nowait(indata[:, 0].copy())
        except queue.Full:
            pass

    def _worker(self) -> None:
        """Route audio from input queue to output stream and invoke RMS callback."""
        while self._running and not self._stop_ev.is_set():
            try:
                samples = self._in_q.get(timeout=0.5)
            except queue.Empty:
                continue

            self._out.play(samples)

            if self._rms_callback is not None:
                try:
                    rms = float(np.sqrt(np.mean(samples * samples))) if samples.size else 0.0
                    self._rms_callback(rms)
                except Exception:
                    pass


# ---------------------------------------------------------------------------
# TX Audio – microphone → radio audio input
# ---------------------------------------------------------------------------

class AudioTxCapture:
    """
    Captures audio from a **microphone** (or any input device) and routes it
    to a soundcard **output** device connected to the radio's audio input.

    Typical use in a ham radio setup:
        mic_device        = computer headset microphone
        radio_out_device  = USB audio interface connected to radio microphone / audio input

    This should be started when the PTT is pressed and stopped on release::

        tx = AudioTxCapture(mic_device=1, radio_out_device=4)
        tx.start()   # PTT pressed  → mic → radio
        tx.stop()    # PTT released → audio stops
    """

    def __init__(
        self,
        mic_device: Optional[int | str],
        radio_out_device: Optional[int | str],
        *,
        fs: int = 48_000,
        block_size: int = 4_800,
        rms_callback: Optional[Callable[[float], None]] = None,
    ) -> None:
        self._mic_device       = mic_device
        self._radio_out_device = radio_out_device
        self._fs               = int(fs)
        self._block_size       = int(block_size)
        self._rms_callback     = rms_callback

        self._out     = SoundCardAudioOutput(fs=fs, block_size=block_size, device=radio_out_device)
        self._src     = None
        self._thread  = None
        self._running = False
        self._stop_ev = threading.Event()

        self._in_q: queue.Queue[np.ndarray] = queue.Queue(maxsize=20)

    # -- Public API --------------------------------------------------------

    def start(self) -> None:
        """Open mic input and radio output streams; begin routing audio."""
        if self._running:
            return

        try:
            import sounddevice as sd  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "AudioTxCapture requires 'sounddevice'.\n"
                "Install it with: pip install sounddevice"
            ) from exc

        self._stop_ev.clear()
        self._running = True

        self._out.start()

        self._src = sd.InputStream(
            samplerate=self._fs,
            blocksize=self._block_size,
            device=self._mic_device,
            channels=1,
            dtype="float32",
            callback=self._in_callback,
        )
        self._src.start()

        self._thread = threading.Thread(target=self._worker, daemon=True, name="AudioTxCapture")
        self._thread.start()

    def stop(self) -> None:
        """Stop audio routing and close both streams."""
        self._running = False
        self._stop_ev.set()

        if self._src is not None:
            try:
                self._src.stop()
            finally:
                try:
                    self._src.close()
                except Exception:
                    pass
            self._src = None

        self._out.stop()

        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    # -- Internal ----------------------------------------------------------

    def _in_callback(self, indata, frames, time_info, status) -> None:
        """sounddevice input callback – copy data to the internal queue."""
        try:
            self._in_q.put_nowait(indata[:, 0].copy())
        except queue.Full:
            pass

    def _worker(self) -> None:
        """Route mic audio to radio output and invoke RMS callback."""
        while self._running and not self._stop_ev.is_set():
            try:
                samples = self._in_q.get(timeout=0.5)
            except queue.Empty:
                continue

            self._out.play(samples)

            if self._rms_callback is not None:
                try:
                    rms = float(np.sqrt(np.mean(samples * samples))) if samples.size else 0.0
                    self._rms_callback(rms)
                except Exception:
                    pass
