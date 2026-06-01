from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy import signal
from scipy.io import wavfile

from bass_extractor.kick_cleaner import KickCleanOptions, clean_kick_leakage, detect_kick_profile


class KickCleanerTests(unittest.TestCase):
    def test_detects_fixed_pitch_kick_pattern(self) -> None:
        sample_rate, reference, _ = _synthetic_kick_and_bass()
        profile = detect_kick_profile(
            reference,
            sample_rate,
            KickCleanOptions(strength=0.7, min_frequency=45.0, max_frequency=90.0),
        )
        self.assertGreaterEqual(profile.event_times.size, 3)
        self.assertIsNotNone(profile.kick_frequency_hz)
        self.assertAlmostEqual(profile.kick_frequency_hz or 0.0, 60.0, delta=7.0)

    def test_cleaner_reduces_kick_band_during_hits(self) -> None:
        sample_rate, reference, leaked_bass = _synthetic_kick_and_bass()
        with tempfile.TemporaryDirectory() as tmp:
            reference_path = Path(tmp) / "mix.wav"
            bass_path = Path(tmp) / "bass.wav"
            wavfile.write(reference_path, sample_rate, reference.astype(np.float32))
            wavfile.write(bass_path, sample_rate, leaked_bass.astype(np.float32))

            before = _kick_band_event_rms(leaked_bass, sample_rate)
            result = clean_kick_leakage(
                reference_path,
                bass_path,
                options=KickCleanOptions(strength=0.8, min_frequency=45.0, max_frequency=90.0),
            )
            _, cleaned = wavfile.read(result.cleaned_path)
            after = _kick_band_event_rms(cleaned.astype(np.float32), sample_rate)

        self.assertGreaterEqual(result.events_detected, 3)
        self.assertLess(after, before * 0.72)


def _synthetic_kick_and_bass() -> tuple[int, np.ndarray, np.ndarray]:
    sample_rate = 44100
    duration = 2.2
    t = np.linspace(0.0, duration, int(sample_rate * duration), endpoint=False)
    bass = 0.13 * np.sin(2.0 * np.pi * 110.0 * t)
    kick = np.zeros_like(t)
    for start_time in (0.20, 0.70, 1.20, 1.70):
        start = int(start_time * sample_rate)
        length = int(0.130 * sample_rate)
        kt = np.arange(length) / sample_rate
        envelope = np.exp(-kt * 28.0)
        kick[start : start + length] += 0.45 * np.sin(2.0 * np.pi * 60.0 * kt) * envelope

    reference = bass + kick
    leaked_bass = bass + kick * 0.55
    return sample_rate, reference.astype(np.float32), leaked_bass.astype(np.float32)


def _kick_band_event_rms(audio: np.ndarray, sample_rate: int) -> float:
    sos = signal.butter(4, [50.0 / (sample_rate / 2.0), 72.0 / (sample_rate / 2.0)], btype="bandpass", output="sos")
    filtered = signal.sosfiltfilt(sos, audio)
    values = []
    for start_time in (0.20, 0.70, 1.20, 1.70):
        start = int(start_time * sample_rate)
        end = start + int(0.120 * sample_rate)
        values.append(float(np.sqrt(np.mean(np.square(filtered[start:end])))))
    return float(np.mean(values))
