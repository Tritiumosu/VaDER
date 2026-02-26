"""
digi_input.py

Audio capture front-end (sound card) that implements the "just give the decoder audio" contract.

Contract (v1):
- Output: mono float32 audio chunks in [-1.0, +1.0]
- Known sample rate (device sample rate)
- Chunked streaming with timestamps (monotonic time in seconds)

Notes:
- This module intentionally does NOT demodulate. It only captures/conditions audio.
- Live soundcard capture requires an external backend (recommended: `sounddevice`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generator, Optional

import queue
import time

try:
    import numpy as np
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "This module requires numpy. Please add numpy to your environment."
    ) from e


@dataclass(frozen=True)
class AudioChunk:
    """A chunk of audio samples with timing metadata."""
    t0: float  # monotonic start time (seconds)
    fs: int  # sample rate (Hz)
    samples: "np.ndarray"  # shape: (n,), dtype float32, mono, range ~[-1, 1]


def _to_mono_float32(indata: "np.ndarray") -> "np.ndarray":
    """
    Convert an input block to mono float32.

    Expected input shapes from typical audio callbacks:
    - (frames,) or (frames, channels)
    """
    x = indata
    if x.ndim == 2:
        # Downmix channels -> mono
        x = x.mean(axis=1)
    x = np.asarray(x, dtype=np.float32)

    # If the backend gives int16-like data (some APIs do), normalize it.
    # sounddevice typically already provides float32 if requested, but this is defensive.
    if x.size and (x.max(initial=0.0) > 1.5 or x.min(initial=0.0) < -1.5):
        # Assume int16 range
        x = x / 32768.0

    # Remove any DC offset per chunk (simple, effective baseline)
    if x.size:
        x = x - float(x.mean())

    # Clip to avoid nasty surprises downstream
    np.clip(x, -1.0, 1.0, out=x)
    return x


class SoundCardAudioSource:
    """
    Captures audio from a sound card and yields AudioChunk objects.

    Requires the optional dependency:
        pip install sounddevice

    You can pass the chunks directly into a decoder that expects audio blocks.
    """

    def __init__(
            self,
            *,
            fs: int = 48_000,
            block_size: int = 4800,  # 100 ms at 48 kHz
            device: Optional[int | str] = None,
            channels: int = 1,
            dtype: str = "float32",
            max_queue_blocks: int = 200,
    ) -> None:
        self.fs = int(fs)
        self.block_size = int(block_size)
        self.device = device
        self.channels = int(channels)
        self.dtype = str(dtype)

        self._q: "queue.Queue[tuple[float, np.ndarray]]" = queue.Queue(maxsize=max_queue_blocks)
        self._stream = None
        self._running = False

    @staticmethod
    def list_devices() -> str:
        """Return a human-readable list of available devices (if sounddevice is installed)."""
        try:
            import sounddevice as sd  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "sounddevice is not installed. Install it to list audio devices."
            ) from e
        return str(sd.query_devices())

    def _audio_callback(self, indata, frames, time_info, status) -> None:
        # Called in (or near) the audio thread; keep it fast and non-blocking.
        if status:
            # Dropped buffers / overflow / etc. We don't print by default to avoid spamming.
            pass

        t0 = time.monotonic()
        try:
            x = _to_mono_float32(indata)
            self._q.put_nowait((t0, x))
        except queue.Full:
            # If the consumer can't keep up, drop the newest chunk.
            # For DSP/decoding, it's usually better to drop than to block the audio callback.
            pass

    def start(self) -> None:
        if self._running:
            return

        try:
            import sounddevice as sd  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "Live soundcard capture requires the optional dependency 'sounddevice'.\n"
                "Install it (in your active venv) and try again."
            ) from e

        self._running = True
        self._stream = sd.InputStream(
            samplerate=self.fs,
            blocksize=self.block_size,
            device=self.device,
            channels=self.channels,
            dtype=self.dtype,
            callback=self._audio_callback,
        )
        self._stream.start()

    def stop(self) -> None:
        self._running = False
        if self._stream is not None:
            try:
                self._stream.stop()
            finally:
                self._stream.close()
            self._stream = None

    def chunks(self, *, timeout_s: float = 1.0) -> Generator[AudioChunk, None, None]:
        """
        Yield AudioChunk objects as they arrive.

        This is your "contract" output:
        - mono float32 samples
        - consistent fs
        - monotonic timestamps
        """
        if not self._running:
            raise RuntimeError("SoundCardAudioSource is not started. Call start() first.")

        while self._running:
            try:
                t0, x = self._q.get(timeout=timeout_s)
            except queue.Empty:
                continue
            yield AudioChunk(t0=t0, fs=self.fs, samples=x)


def run_audio_to_decoder(
        decoder_feed: Callable[[AudioChunk], None],
        *,
        fs: int = 48_000,
        block_size: int = 4800,
        device: Optional[int | str] = None,
) -> None:
    """
    Convenience runner: capture audio and feed it into a decoder callback.

    The decoder_feed function is expected to accept AudioChunk objects.
    """
    src = SoundCardAudioSource(fs=fs, block_size=block_size, device=device)
    src.start()
    try:
        for chunk in src.chunks():
            decoder_feed(chunk)
    finally:
        src.stop()


# Example usage (kept minimal; you can remove or replace with your application wiring)
if __name__ == "__main__":
    def _debug_decoder_feed(chunk: AudioChunk) -> None:
        # Replace this with something like: ft8_decoder.feed(chunk.samples, chunk.fs, chunk.t0)
        rms = float(np.sqrt(np.mean(chunk.samples * chunk.samples))) if chunk.samples.size else 0.0
        print(f"t0={chunk.t0:.3f}s fs={chunk.fs} n={chunk.samples.size} rms={rms:.4f}")

    # Tip: If you don't know your device index/name, call:
    # print(SoundCardAudioSource.list_devices())
    run_audio_to_decoder(_debug_decoder_feed)