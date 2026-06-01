from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import wavfile

try:
    from scipy.io.wavfile import WavFileWarning
except ImportError:  # pragma: no cover - compatibility with older scipy
    WavFileWarning = Warning


@dataclass(frozen=True)
class AudioMetrics:
    path: str
    sample_rate: int
    channels: int
    samples: int
    duration_seconds: float
    peak_dbfs: float | None
    rms_dbfs: float | None
    clipped_sample_ratio: float
    low_band_energy_ratio: float | None
    high_leak_energy_ratio: float | None
    analyzed_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "samples": self.samples,
            "duration_seconds": round(self.duration_seconds, 6),
            "peak_dbfs": _round_or_none(self.peak_dbfs),
            "rms_dbfs": _round_or_none(self.rms_dbfs),
            "clipped_sample_ratio": round(self.clipped_sample_ratio, 8),
            "low_band_energy_ratio": _round_or_none(self.low_band_energy_ratio, 6),
            "high_leak_energy_ratio": _round_or_none(self.high_leak_energy_ratio, 6),
            "analyzed_seconds": round(self.analyzed_seconds, 6),
        }


def build_quality_report(input_path: Path, bass_path: Path) -> dict[str, Any]:
    report: dict[str, Any] = {
        "input": {"path": str(input_path), "readable_wav": False},
        "bass": {"path": str(bass_path), "readable_wav": False},
        "checks": [],
        "warnings": [],
    }

    input_metrics = _try_metrics(input_path)
    bass_metrics = _try_metrics(bass_path)

    if input_metrics is not None:
        report["input"] = input_metrics.to_dict() | {"readable_wav": True}
    else:
        report["warnings"].append(
            "Input is not a readable WAV file for local QC. Demucs can still process it if ffmpeg is available."
        )

    if bass_metrics is not None:
        report["bass"] = bass_metrics.to_dict() | {"readable_wav": True}
        report["checks"].extend(_bass_checks(bass_metrics))
    else:
        report["warnings"].append("Bass output could not be read as WAV for QC metrics.")

    if input_metrics is not None and bass_metrics is not None:
        duration_delta = abs(input_metrics.duration_seconds - bass_metrics.duration_seconds)
        report["duration_delta_seconds"] = round(duration_delta, 6)
        if duration_delta > 0.05:
            report["warnings"].append(
                f"Input/output duration mismatch is {duration_delta:.3f}s; review the stem manually."
            )

    report["pass"] = not report["warnings"]
    return report


def _try_metrics(path: Path) -> AudioMetrics | None:
    try:
        return analyze_wav(path)
    except Exception:
        return None


def analyze_wav(path: Path, max_analysis_seconds: float = 180.0) -> AudioMetrics:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", WavFileWarning)
        sample_rate, raw = wavfile.read(path)
    data = _to_float_audio(np.asarray(raw))
    if data.ndim == 1:
        channels = 1
        mono = data
    else:
        channels = int(data.shape[1])
        mono = data.mean(axis=1)

    samples = int(mono.shape[0])
    duration = samples / float(sample_rate) if sample_rate else 0.0
    peak = float(np.max(np.abs(data))) if data.size else 0.0
    rms = float(np.sqrt(np.mean(np.square(data)))) if data.size else 0.0
    clipped = float(np.mean(np.abs(data) >= 0.999)) if data.size else 0.0

    analysis_samples = min(samples, int(sample_rate * max_analysis_seconds))
    if analysis_samples > 0:
        low_ratio, high_ratio = _band_ratios(mono[:analysis_samples], sample_rate)
    else:
        low_ratio, high_ratio = None, None

    return AudioMetrics(
        path=str(path),
        sample_rate=int(sample_rate),
        channels=channels,
        samples=samples,
        duration_seconds=duration,
        peak_dbfs=_dbfs(peak),
        rms_dbfs=_dbfs(rms),
        clipped_sample_ratio=clipped,
        low_band_energy_ratio=low_ratio,
        high_leak_energy_ratio=high_ratio,
        analyzed_seconds=analysis_samples / float(sample_rate) if sample_rate else 0.0,
    )


def _to_float_audio(data: np.ndarray) -> np.ndarray:
    if np.issubdtype(data.dtype, np.floating):
        return data.astype(np.float32, copy=False)
    if np.issubdtype(data.dtype, np.signedinteger):
        scale = float(max(abs(np.iinfo(data.dtype).min), np.iinfo(data.dtype).max))
        return data.astype(np.float32) / scale
    if np.issubdtype(data.dtype, np.unsignedinteger):
        info = np.iinfo(data.dtype)
        midpoint = (info.max + info.min) / 2.0
        return (data.astype(np.float32) - midpoint) / midpoint
    return data.astype(np.float32)


def _band_ratios(mono: np.ndarray, sample_rate: int) -> tuple[float | None, float | None]:
    if mono.size < 16 or sample_rate <= 0:
        return None, None

    mono = mono - float(np.mean(mono))
    window = np.hanning(mono.size)
    spectrum = np.abs(np.fft.rfft(mono * window)) ** 2
    freqs = np.fft.rfftfreq(mono.size, 1.0 / sample_rate)
    total = float(np.sum(spectrum))
    if total <= 0.0:
        return None, None

    low_band = (freqs >= 20.0) & (freqs <= 250.0)
    high_leak = freqs >= 800.0
    return float(np.sum(spectrum[low_band]) / total), float(np.sum(spectrum[high_leak]) / total)


def _bass_checks(metrics: AudioMetrics) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    checks.append(
        {
            "name": "no_clipping",
            "pass": metrics.clipped_sample_ratio < 0.000001,
            "value": round(metrics.clipped_sample_ratio, 8),
        }
    )
    checks.append(
        {
            "name": "bass_dominant_energy",
            "pass": metrics.low_band_energy_ratio is None or metrics.low_band_energy_ratio >= 0.45,
            "value": _round_or_none(metrics.low_band_energy_ratio, 6),
        }
    )
    checks.append(
        {
            "name": "limited_high_frequency_leak",
            "pass": metrics.high_leak_energy_ratio is None or metrics.high_leak_energy_ratio <= 0.25,
            "value": _round_or_none(metrics.high_leak_energy_ratio, 6),
        }
    )
    return checks


def _dbfs(value: float) -> float | None:
    if value <= 0.0 or math.isnan(value):
        return None
    return 20.0 * math.log10(value)


def _round_or_none(value: float | None, digits: int = 3) -> float | None:
    if value is None:
        return None
    return round(value, digits)
