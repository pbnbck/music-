from __future__ import annotations

import math
import shutil
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import signal
from scipy.io import wavfile

try:
    from scipy.io.wavfile import WavFileWarning
except ImportError:  # pragma: no cover - compatibility with older scipy
    WavFileWarning = Warning


class KickCleanupError(RuntimeError):
    """Raised when kick-aware cleanup cannot be applied."""


@dataclass(frozen=True)
class KickCleanOptions:
    strength: float = 0.65
    min_frequency: float = 35.0
    max_frequency: float = 135.0
    bandwidth_hz: float = 18.0
    pre_ms: float = 12.0
    post_ms: float = 150.0
    min_interval_ms: float = 160.0
    harmonic_count: int = 2
    onset_percentile: float = 84.0
    pitch_tolerance_ratio: float = 0.35


@dataclass(frozen=True)
class KickCleanResult:
    cleaned_path: Path
    events_detected: int
    kick_frequency_hz: float | None
    regularity_score: float | None
    strength: float
    attenuation_windows: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "cleaned_path": str(self.cleaned_path),
            "events_detected": self.events_detected,
            "kick_frequency_hz": _round_or_none(self.kick_frequency_hz),
            "regularity_score": _round_or_none(self.regularity_score, 4),
            "strength": round(self.strength, 3),
            "attenuation_windows": self.attenuation_windows,
        }


@dataclass(frozen=True)
class KickProfile:
    event_times: np.ndarray
    kick_frequency_hz: float | None
    regularity_score: float | None


def clean_kick_leakage(
    reference_path: Path,
    bass_path: Path,
    output_path: Path | None = None,
    options: KickCleanOptions | None = None,
) -> KickCleanResult:
    options = options or KickCleanOptions()
    _validate_options(options)

    reference_rate, reference_audio = _read_audio(reference_path)
    bass_rate, bass_audio = _read_audio(bass_path)
    if reference_rate != bass_rate:
        raise KickCleanupError(
            f"Reference and bass sample rates differ ({reference_rate} vs {bass_rate})."
        )

    reference_mono = _to_mono(reference_audio)
    profile = detect_kick_profile(reference_mono, reference_rate, options)
    if profile.kick_frequency_hz is None or profile.event_times.size == 0:
        raise KickCleanupError("No stable kick pattern was detected.")

    cleaned = attenuate_kick_windows(bass_audio, bass_rate, profile, options)
    destination = output_path or bass_path

    if destination == bass_path:
        with tempfile.NamedTemporaryFile(
            suffix=bass_path.suffix,
            prefix=bass_path.stem + "_kickclean_",
            dir=str(bass_path.parent),
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
        _write_audio(temp_path, bass_rate, cleaned)
        shutil.move(str(temp_path), str(bass_path))
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        _write_audio(destination, bass_rate, cleaned)

    return KickCleanResult(
        cleaned_path=destination,
        events_detected=int(profile.event_times.size),
        kick_frequency_hz=profile.kick_frequency_hz,
        regularity_score=profile.regularity_score,
        strength=options.strength,
        attenuation_windows=int(profile.event_times.size),
    )


def detect_kick_profile(
    mono_audio: np.ndarray,
    sample_rate: int,
    options: KickCleanOptions | None = None,
) -> KickProfile:
    options = options or KickCleanOptions()
    if mono_audio.size == 0:
        return KickProfile(np.array([], dtype=np.float32), None, None)

    filtered = _bandpass(mono_audio, sample_rate, options.min_frequency, options.max_frequency)
    hop = max(1, int(sample_rate * 0.010))
    frame = max(hop * 2, int(sample_rate * 0.040))
    envelope = _frame_rms(filtered, frame, hop)
    if envelope.size < 3:
        return KickProfile(np.array([], dtype=np.float32), None, None)

    onset = np.maximum(np.diff(np.log1p(envelope * 1000.0), prepend=0.0), 0.0)
    if not np.any(onset > 0.0):
        return KickProfile(np.array([], dtype=np.float32), None, None)

    distance = max(1, int(options.min_interval_ms / 1000.0 * sample_rate / hop))
    threshold = max(
        float(np.percentile(onset, options.onset_percentile)),
        float(np.mean(onset) + np.std(onset) * 0.75),
    )
    peaks, _ = signal.find_peaks(onset, height=threshold, distance=distance)
    if peaks.size == 0:
        return KickProfile(np.array([], dtype=np.float32), None, None)

    event_times = peaks.astype(np.float32) * hop / float(sample_rate)
    event_freqs = _event_frequencies(filtered, sample_rate, event_times, options)
    kick_frequency = _stable_frequency(event_freqs)
    if kick_frequency is not None and event_freqs.size:
        consistent = _consistent_pitch_mask(event_freqs, kick_frequency, options.pitch_tolerance_ratio)
        if np.count_nonzero(consistent) >= max(2, min(4, event_times.size)):
            event_times = event_times[consistent]

    regularity = _regularity_score(event_times)
    return KickProfile(event_times=event_times, kick_frequency_hz=kick_frequency, regularity_score=regularity)


def attenuate_kick_windows(
    bass_audio: np.ndarray,
    sample_rate: int,
    profile: KickProfile,
    options: KickCleanOptions | None = None,
) -> np.ndarray:
    options = options or KickCleanOptions()
    if profile.kick_frequency_hz is None or profile.event_times.size == 0:
        return bass_audio.copy()

    nperseg = min(4096, _largest_power_of_two(max(512, bass_audio.shape[0])))
    noverlap = int(nperseg * 0.75)
    cleaned_channels: list[np.ndarray] = []

    for channel_index in range(bass_audio.shape[1]):
        frequencies, times, spectrum = signal.stft(
            bass_audio[:, channel_index],
            fs=sample_rate,
            nperseg=nperseg,
            noverlap=noverlap,
            boundary="zeros",
        )
        mask = _kick_mask(frequencies, times, profile.event_times, profile.kick_frequency_hz, options)
        _, cleaned = signal.istft(
            spectrum * mask,
            fs=sample_rate,
            nperseg=nperseg,
            noverlap=noverlap,
            input_onesided=True,
            boundary=True,
        )
        cleaned_channels.append(_fit_length(cleaned, bass_audio.shape[0]))

    cleaned_audio = np.stack(cleaned_channels, axis=1).astype(np.float32)
    return np.clip(cleaned_audio, -1.0, 1.0)


def _kick_mask(
    frequencies: np.ndarray,
    times: np.ndarray,
    event_times: np.ndarray,
    kick_frequency_hz: float,
    options: KickCleanOptions,
) -> np.ndarray:
    frequency_weight = np.zeros_like(frequencies, dtype=np.float32)
    for harmonic in range(1, options.harmonic_count + 1):
        center = kick_frequency_hz * harmonic
        if center > 320.0:
            continue
        width = options.bandwidth_hz * (1.0 + 0.45 * (harmonic - 1))
        band = np.exp(-0.5 * np.square((frequencies - center) / width))
        frequency_weight = np.maximum(frequency_weight, band.astype(np.float32))

    mask = np.ones((frequencies.size, times.size), dtype=np.float32)
    pre_seconds = options.pre_ms / 1000.0
    post_seconds = options.post_ms / 1000.0
    duration = pre_seconds + post_seconds
    if duration <= 0.0:
        return mask

    for event_time in event_times:
        start = event_time - pre_seconds
        end = event_time + post_seconds
        active = (times >= start) & (times <= end)
        if not np.any(active):
            continue
        phase = (times[active] - start) / duration
        time_weight = np.sin(np.pi * np.clip(phase, 0.0, 1.0)).astype(np.float32)
        attenuation = options.strength * np.outer(frequency_weight, time_weight)
        mask[:, active] = np.minimum(mask[:, active], 1.0 - attenuation)

    return np.clip(mask, 1.0 - options.strength, 1.0)


def _read_audio(path: Path) -> tuple[int, np.ndarray]:
    suffix = path.suffix.lower()
    if suffix == ".wav":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", WavFileWarning)
            sample_rate, raw = wavfile.read(path)
        audio = _to_float_audio(np.asarray(raw))
    else:
        try:
            import soundfile as sf
        except Exception as exc:  # pragma: no cover - optional path
            raise KickCleanupError(f"{path.suffix} input requires soundfile support.") from exc
        try:
            audio, sample_rate = sf.read(path, always_2d=False, dtype="float32")
        except Exception as exc:  # pragma: no cover - depends on local codecs
            raise KickCleanupError(f"Could not read {path} for kick cleanup.") from exc

    if audio.ndim == 1:
        audio = audio[:, None]
    return int(sample_rate), audio.astype(np.float32, copy=False)


def _write_audio(path: Path, sample_rate: int, audio: np.ndarray) -> None:
    suffix = path.suffix.lower()
    if suffix == ".wav":
        wavfile.write(path, sample_rate, audio.astype(np.float32, copy=False))
        return
    try:
        import soundfile as sf
    except Exception as exc:  # pragma: no cover - optional path
        raise KickCleanupError(f"{path.suffix} output requires soundfile support.") from exc
    sf.write(path, audio.astype(np.float32, copy=False), sample_rate)


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


def _to_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio.astype(np.float32, copy=False)
    return audio.mean(axis=1).astype(np.float32, copy=False)


def _bandpass(audio: np.ndarray, sample_rate: int, low: float, high: float) -> np.ndarray:
    nyquist = sample_rate / 2.0
    high = min(high, nyquist * 0.95)
    low = max(1.0, min(low, high * 0.8))
    sos = signal.butter(4, [low / nyquist, high / nyquist], btype="bandpass", output="sos")
    return signal.sosfiltfilt(sos, audio).astype(np.float32)


def _frame_rms(audio: np.ndarray, frame: int, hop: int) -> np.ndarray:
    if audio.size < frame:
        padded = np.pad(audio, (0, frame - audio.size))
    else:
        pad = (hop - ((audio.size - frame) % hop)) % hop
        padded = np.pad(audio, (0, pad))

    frame_count = 1 + max(0, (padded.size - frame) // hop)
    values = np.empty(frame_count, dtype=np.float32)
    for index in range(frame_count):
        start = index * hop
        chunk = padded[start : start + frame]
        values[index] = math.sqrt(float(np.mean(np.square(chunk))) + 1e-12)
    return values


def _event_frequencies(
    filtered_audio: np.ndarray,
    sample_rate: int,
    event_times: np.ndarray,
    options: KickCleanOptions,
) -> np.ndarray:
    values: list[float] = []
    window = max(256, int(sample_rate * 0.110))
    n_fft = max(4096, _next_power_of_two(window * 4))
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sample_rate)
    band = (freqs >= options.min_frequency) & (freqs <= options.max_frequency)

    for event_time in event_times:
        center = int(event_time * sample_rate)
        start = max(0, center - int(sample_rate * 0.010))
        end = min(filtered_audio.size, start + window)
        segment = filtered_audio[start:end]
        if segment.size < 32:
            continue
        padded = np.zeros(n_fft, dtype=np.float32)
        padded[: segment.size] = segment * np.hanning(segment.size)
        spectrum = np.abs(np.fft.rfft(padded))
        if not np.any(spectrum[band] > 0.0):
            continue
        values.append(float(freqs[band][np.argmax(spectrum[band])]))

    return np.asarray(values, dtype=np.float32)


def _stable_frequency(event_freqs: np.ndarray) -> float | None:
    if event_freqs.size == 0:
        return None
    return float(np.median(event_freqs))


def _consistent_pitch_mask(event_freqs: np.ndarray, kick_frequency: float, tolerance_ratio: float) -> np.ndarray:
    low = kick_frequency * (1.0 - tolerance_ratio)
    high = kick_frequency * (1.0 + tolerance_ratio)
    direct = (event_freqs >= low) & (event_freqs <= high)
    half = (event_freqs / 2.0 >= low) & (event_freqs / 2.0 <= high)
    double = (event_freqs * 2.0 >= low) & (event_freqs * 2.0 <= high)
    return direct | half | double


def _regularity_score(event_times: np.ndarray) -> float | None:
    if event_times.size < 3:
        return None
    intervals = np.diff(event_times)
    median = float(np.median(intervals))
    if median <= 0.0:
        return None
    coefficient = float(np.std(intervals) / median)
    return 1.0 / (1.0 + coefficient)


def _fit_length(audio: np.ndarray, length: int) -> np.ndarray:
    if audio.size == length:
        return audio.astype(np.float32, copy=False)
    if audio.size > length:
        return audio[:length].astype(np.float32, copy=False)
    return np.pad(audio, (0, length - audio.size)).astype(np.float32)


def _largest_power_of_two(value: int) -> int:
    if value < 1:
        return 1
    return 1 << (value.bit_length() - 1)


def _next_power_of_two(value: int) -> int:
    if value <= 1:
        return 1
    return 1 << (value - 1).bit_length()


def _validate_options(options: KickCleanOptions) -> None:
    if not 0.0 <= options.strength <= 0.95:
        raise ValueError("kick strength must be between 0.0 and 0.95")
    if options.min_frequency <= 0.0 or options.max_frequency <= options.min_frequency:
        raise ValueError("kick frequency range is invalid")
    if options.harmonic_count < 1:
        raise ValueError("harmonic_count must be at least 1")


def _round_or_none(value: float | None, digits: int = 3) -> float | None:
    if value is None:
        return None
    return round(value, digits)
