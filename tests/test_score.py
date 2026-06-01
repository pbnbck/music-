from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from bass_extractor.score import ScoreOptions, create_bass_score, estimate_bpm, estimate_key


class ScoreTests(unittest.TestCase):
    def test_create_musicxml_contains_score_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bass_path = Path(tmp) / "bass.wav"
            score_path = Path(tmp) / "bass.musicxml"
            sample_rate, audio = _bass_line()
            wavfile.write(bass_path, sample_rate, audio)

            result = create_bass_score(
                bass_path,
                score_path,
                ScoreOptions(tempo_override=120.0, key_override="C major", title="Bass Test"),
            )
            xml = score_path.read_text(encoding="utf-8")

        self.assertEqual(result.bpm, 120.0)
        self.assertEqual(result.key, "C major")
        self.assertGreater(result.note_count, 0)
        self.assertGreater(len(result.chords), 0)
        self.assertIn("<work-title>Bass Test</work-title>", xml)
        self.assertIn("<per-minute>120</per-minute>", xml)
        self.assertIn("<fifths>0</fifths>", xml)
        self.assertIn("<sign>F</sign>", xml)
        self.assertIn("<harmony>", xml)

    def test_estimators_return_music_values(self) -> None:
        sample_rate, audio = _bass_line()
        bpm = estimate_bpm(audio, sample_rate)
        key, fifths, mode = estimate_key(audio, sample_rate)

        self.assertGreaterEqual(bpm, 40.0)
        self.assertLessEqual(bpm, 220.0)
        self.assertIsInstance(key, str)
        self.assertIsInstance(fifths, int)
        self.assertIn(mode, {"major", "minor"})


def _bass_line() -> tuple[int, np.ndarray]:
    sample_rate = 44100
    bpm = 120.0
    quarter = 60.0 / bpm
    notes = [36, 40, 43, 48, 43, 40, 36, 31]
    chunks = []
    for midi in notes:
        frequency = 440.0 * 2.0 ** ((midi - 69) / 12.0)
        t = np.linspace(0.0, quarter, int(sample_rate * quarter), endpoint=False)
        envelope = np.minimum(1.0, np.linspace(0.0, 6.0, t.size)) * np.exp(-t * 0.35)
        chunks.append((0.4 * np.sin(2.0 * np.pi * frequency * t) * envelope).astype(np.float32))
    return sample_rate, np.concatenate(chunks)
