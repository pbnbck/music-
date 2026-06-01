from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from bass_extractor.core import PROFILES, SeparationOptions, build_demucs_command, resolve_output_path
from bass_extractor.quality import analyze_wav, build_quality_report


class CoreTests(unittest.TestCase):
    def test_profiles_have_studio_default(self) -> None:
        self.assertIn("studio", PROFILES)
        resolved = SeparationOptions().resolved()
        self.assertEqual(resolved.model, "htdemucs_ft")
        self.assertEqual(resolved.output_format, "wav")

    def test_output_path_defaults_to_bass_folder(self) -> None:
        source = Path("song.wav")
        output = resolve_output_path(source, None, "wav")
        self.assertEqual(output.as_posix(), "bass-extractor-output/song_bass.wav")

    def test_command_uses_two_stem_bass(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            options = SeparationOptions(profile="fast", output_format="wav").resolved()
            command = build_demucs_command(Path("mix.wav"), Path(tmp), options, "python")
        self.assertIn("--two-stems", command)
        self.assertIn("bass", command)
        self.assertIn("--float32", command)


class QualityTests(unittest.TestCase):
    def test_bass_sine_scores_as_bass_dominant(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bass.wav"
            sr = 44100
            t = np.linspace(0, 1.0, sr, endpoint=False)
            audio = (0.5 * np.sin(2 * np.pi * 80 * t)).astype(np.float32)
            wavfile.write(path, sr, audio)

            metrics = analyze_wav(path)
            self.assertGreater(metrics.low_band_energy_ratio or 0.0, 0.95)
            self.assertLess(metrics.high_leak_energy_ratio or 1.0, 0.01)

    def test_quality_report_detects_duration_match(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            input_path = Path(tmp) / "input.wav"
            bass_path = Path(tmp) / "bass.wav"
            sr = 8000
            samples = np.zeros(sr, dtype=np.float32)
            wavfile.write(input_path, sr, samples)
            wavfile.write(bass_path, sr, samples)

            report = build_quality_report(input_path, bass_path)
            self.assertEqual(report["duration_delta_seconds"], 0.0)


if __name__ == "__main__":
    unittest.main()
