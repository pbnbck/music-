from __future__ import annotations

import importlib.util
import json
import re
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

ProgressCallback = Callable[[str], None]
_PROGRESS_PATTERN = re.compile(r"(?P<percent>\d{1,3})%\|")


class BassExtractorError(RuntimeError):
    """Base error for extraction failures."""


class DependencyError(BassExtractorError):
    """Raised when a required runtime dependency is missing."""


class SeparationError(BassExtractorError):
    """Raised when Demucs fails to produce the requested stem."""


@dataclass(frozen=True)
class SeparationProfile:
    name: str
    model: str
    shifts: int
    overlap: float
    jobs: int
    description: str


PROFILES: dict[str, SeparationProfile] = {
    "studio": SeparationProfile(
        name="studio",
        model="htdemucs_ft",
        shifts=8,
        overlap=0.50,
        jobs=0,
        description="Highest quality. Slower, recommended for release work.",
    ),
    "balanced": SeparationProfile(
        name="balanced",
        model="htdemucs",
        shifts=4,
        overlap=0.35,
        jobs=0,
        description="Good quality/speed tradeoff.",
    ),
    "fast": SeparationProfile(
        name="fast",
        model="htdemucs",
        shifts=1,
        overlap=0.25,
        jobs=0,
        description="Fast preview, not recommended as final master.",
    ),
}


@dataclass(frozen=True)
class SeparationOptions:
    profile: str = "studio"
    model: str | None = None
    shifts: int | None = None
    overlap: float | None = None
    jobs: int | None = None
    device: str = "auto"
    output_format: str = "wav"
    mp3_bitrate: int = 320
    clip_mode: str = "rescale"
    keep_work_dir: bool = False
    work_dir: Path | None = None
    export_no_bass: bool = False
    kick_clean: bool = False
    kick_strength: float = 0.65
    kick_min_frequency: float = 35.0
    kick_max_frequency: float = 135.0
    kick_window_ms: float = 150.0
    make_score: bool = False
    score_path: Path | None = None
    score_tempo: float | None = None
    score_key: str | None = None
    score_title: str | None = None

    def resolved(self) -> "ResolvedOptions":
        profile = PROFILES.get(self.profile)
        if profile is None:
            valid = ", ".join(sorted(PROFILES))
            raise ValueError(f"Unknown profile '{self.profile}'. Valid profiles: {valid}")
        output_format = self.output_format.lower()
        if output_format not in {"wav", "flac", "mp3"}:
            raise ValueError("output_format must be wav, flac, or mp3")
        if self.kick_clean and output_format == "mp3":
            raise ValueError("kick_clean requires wav or flac output; use wav for final delivery")
        if self.make_score and output_format == "mp3":
            raise ValueError("score generation requires wav or flac output; use wav for final delivery")
        if not 0.0 <= self.kick_strength <= 0.95:
            raise ValueError("kick_strength must be between 0.0 and 0.95")
        return ResolvedOptions(
            profile=profile.name,
            model=self.model or profile.model,
            shifts=profile.shifts if self.shifts is None else self.shifts,
            overlap=profile.overlap if self.overlap is None else self.overlap,
            jobs=profile.jobs if self.jobs is None else self.jobs,
            device=self.device,
            output_format=output_format,
            mp3_bitrate=self.mp3_bitrate,
            clip_mode=self.clip_mode,
            keep_work_dir=self.keep_work_dir,
            work_dir=self.work_dir,
            export_no_bass=self.export_no_bass,
            kick_clean=self.kick_clean,
            kick_strength=self.kick_strength,
            kick_min_frequency=self.kick_min_frequency,
            kick_max_frequency=self.kick_max_frequency,
            kick_window_ms=self.kick_window_ms,
            make_score=self.make_score,
            score_path=self.score_path,
            score_tempo=self.score_tempo,
            score_key=self.score_key,
            score_title=self.score_title,
        )


@dataclass(frozen=True)
class ResolvedOptions:
    profile: str
    model: str
    shifts: int
    overlap: float
    jobs: int
    device: str
    output_format: str
    mp3_bitrate: int
    clip_mode: str
    keep_work_dir: bool
    work_dir: Path | None
    export_no_bass: bool
    kick_clean: bool
    kick_strength: float
    kick_min_frequency: float
    kick_max_frequency: float
    kick_window_ms: float
    make_score: bool
    score_path: Path | None
    score_tempo: float | None
    score_key: str | None
    score_title: str | None


@dataclass(frozen=True)
class SeparationResult:
    input_path: Path
    bass_path: Path
    report_path: Path
    work_dir: Path
    model: str
    command: list[str]
    no_bass_path: Path | None = None
    kick_cleanup: dict[str, Any] | None = None
    score_path: Path | None = None
    score: dict[str, Any] | None = None


def diagnose_environment(python_executable: str = sys.executable) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    checks.append(
        {
            "name": "python",
            "pass": True,
            "detail": f"{python_executable} ({sys.version.split()[0]})",
        }
    )
    checks.append(_module_check("demucs", "Demucs source separation engine"))
    checks.append(_module_check("numpy", "Numerical analysis"))
    checks.append(_module_check("scipy", "WAV quality analysis"))
    checks.append(_torch_check())
    ffmpeg_path = shutil.which("ffmpeg")
    checks.append(
        {
            "name": "ffmpeg",
            "pass": ffmpeg_path is not None,
            "detail": ffmpeg_path or "Missing. Required for MP3/FLAC/M4A and other compressed formats.",
        }
    )
    return checks


def separate_bass(
    input_path: Path,
    output_path: Path | None = None,
    options: SeparationOptions | None = None,
    progress: ProgressCallback | None = None,
    python_executable: str = sys.executable,
) -> SeparationResult:
    options = options or SeparationOptions()
    resolved = options.resolved()
    input_path = input_path.expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    _require_modules(("demucs", "numpy", "scipy"))

    output_path = resolve_output_path(input_path, output_path, resolved.output_format)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    work_dir = _prepare_work_dir(output_path, resolved)

    command = build_demucs_command(input_path, work_dir, resolved, python_executable)
    _emit(progress, "Starting Demucs separation...")
    _emit(progress, "Command: " + " ".join(_quote_for_log(part) for part in command))
    _run_command(command, progress, _demucs_environment())

    bass_stem = _locate_stem(work_dir, resolved.model, input_path.stem, "bass", resolved.output_format)
    shutil.copy2(bass_stem, output_path)

    no_bass_output: Path | None = None
    if resolved.export_no_bass:
        no_bass_stem = _locate_stem(
            work_dir,
            resolved.model,
            input_path.stem,
            "no_bass",
            resolved.output_format,
            required=False,
        )
        if no_bass_stem is not None:
            no_bass_output = output_path.with_name(output_path.stem + "_no_bass" + output_path.suffix)
            shutil.copy2(no_bass_stem, no_bass_output)

    kick_cleanup: dict[str, Any] | None = None
    if resolved.kick_clean:
        from .kick_cleaner import KickCleanOptions, clean_kick_leakage

        _emit(progress, "Running kick-aware bass cleanup...")
        cleanup_result = clean_kick_leakage(
            reference_path=input_path,
            bass_path=output_path,
            options=KickCleanOptions(
                strength=resolved.kick_strength,
                min_frequency=resolved.kick_min_frequency,
                max_frequency=resolved.kick_max_frequency,
                post_ms=resolved.kick_window_ms,
            ),
        )
        kick_cleanup = cleanup_result.to_dict()
        frequency = cleanup_result.kick_frequency_hz
        if frequency is None:
            _emit(progress, f"Kick cleanup: {cleanup_result.events_detected} events")
        else:
            _emit(
                progress,
                f"Kick cleanup: {cleanup_result.events_detected} events at about {frequency:.1f} Hz",
            )

    score: dict[str, Any] | None = None
    score_path: Path | None = None
    if resolved.make_score:
        from .score import ScoreOptions, create_bass_score

        _emit(progress, "Generating bass MusicXML score...")
        score_path = resolved.score_path or output_path.with_suffix(".musicxml")
        score_result = create_bass_score(
            bass_path=output_path,
            output_path=score_path,
            options=ScoreOptions(
                tempo_override=resolved.score_tempo,
                key_override=resolved.score_key,
                title=resolved.score_title or output_path.stem,
            ),
        )
        score = score_result.to_dict()
        score_path = score_result.score_path
        _emit(
            progress,
            f"Bass score written: {score_result.score_path} "
            f"({score_result.bpm:.1f} BPM, {score_result.key}, {len(score_result.chords)} chords)",
        )

    from .quality import build_quality_report

    report = build_quality_report(input_path, output_path)
    report["engine"] = {
        "name": "demucs",
        "model": resolved.model,
        "profile": resolved.profile,
        "shifts": resolved.shifts,
        "overlap": resolved.overlap,
        "device": resolved.device,
        "output_format": resolved.output_format,
    }
    if kick_cleanup is not None:
        report["kick_cleanup"] = kick_cleanup
    if score is not None:
        report["score"] = score
    report_path = output_path.with_suffix(output_path.suffix + ".quality.json")
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    if not resolved.keep_work_dir and resolved.work_dir is None:
        shutil.rmtree(work_dir, ignore_errors=True)

    _emit(progress, f"Bass stem written: {output_path}")
    _emit(progress, f"Quality report written: {report_path}")
    return SeparationResult(
        input_path=input_path,
        bass_path=output_path,
        report_path=report_path,
        work_dir=work_dir,
        model=resolved.model,
        command=command,
        no_bass_path=no_bass_output,
        kick_cleanup=kick_cleanup,
        score_path=score_path,
        score=score,
    )


def resolve_output_path(input_path: Path, output_path: Path | None, output_format: str) -> Path:
    suffix = "." + output_format.lower().lstrip(".")
    if output_path is None:
        return input_path.parent / "bass-extractor-output" / f"{input_path.stem}_bass{suffix}"
    output_path = output_path.expanduser()
    if output_path.exists() and output_path.is_dir():
        return output_path / f"{input_path.stem}_bass{suffix}"
    if output_path.suffix:
        return output_path
    return output_path.with_suffix(suffix)


def build_demucs_command(
    input_path: Path,
    work_dir: Path,
    options: ResolvedOptions,
    python_executable: str = sys.executable,
) -> list[str]:
    command = [
        python_executable,
        "-m",
        "demucs",
        "-n",
        options.model,
        "--two-stems",
        "bass",
        "-o",
        str(work_dir),
        "--filename",
        "{track}/{stem}.{ext}",
        "--shifts",
        str(options.shifts),
        "--overlap",
        str(options.overlap),
        "-j",
        str(options.jobs),
        "--clip-mode",
        options.clip_mode,
    ]
    if options.device != "auto":
        command.extend(["-d", options.device])
    if options.output_format == "flac":
        command.append("--flac")
    elif options.output_format == "mp3":
        command.extend(["--mp3", "--mp3-bitrate", str(options.mp3_bitrate)])
    else:
        command.append("--float32")
    command.append(str(input_path))
    return command


def _run_command(command: list[str], progress: ProgressCallback | None, env: dict[str, str]) -> None:
    last_progress: str | None = None
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        for message in _clean_process_output(line):
            if message.startswith("Demucs progress:"):
                if message == last_progress:
                    continue
                last_progress = message
            _emit(progress, message)
    return_code = process.wait()
    if return_code != 0:
        raise SeparationError(f"Demucs failed with exit code {return_code}.")


def _prepare_work_dir(output_path: Path, options: ResolvedOptions) -> Path:
    if options.work_dir is not None:
        work_dir = options.work_dir.expanduser().resolve()
        work_dir.mkdir(parents=True, exist_ok=True)
        return work_dir
    if options.keep_work_dir:
        work_dir = output_path.parent / "_demucs_work"
        work_dir.mkdir(parents=True, exist_ok=True)
        return work_dir
    return Path(tempfile.mkdtemp(prefix="bass_extractor_"))


def _demucs_environment() -> dict[str, str]:
    env = os.environ.copy()
    cache_root = Path.cwd() / ".model-cache"
    env.setdefault("XDG_CACHE_HOME", str(cache_root))
    env.setdefault("TORCH_HOME", str(cache_root / "torch"))
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["TQDM_ASCII"] = "1"
    Path(env["TORCH_HOME"]).mkdir(parents=True, exist_ok=True)
    return env


def _locate_stem(
    work_dir: Path,
    model: str,
    track_name: str,
    stem: str,
    output_format: str,
    required: bool = True,
) -> Path | None:
    expected = work_dir / model / track_name / f"{stem}.{output_format}"
    if expected.exists():
        return expected

    candidates = sorted(
        work_dir.glob(f"**/{stem}.{output_format}"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return candidates[0]
    if required:
        raise SeparationError(f"Demucs completed, but {stem}.{output_format} was not found in {work_dir}.")
    return None


def _module_check(module_name: str, purpose: str) -> dict[str, Any]:
    spec = importlib.util.find_spec(module_name)
    return {
        "name": module_name,
        "pass": spec is not None,
        "detail": purpose if spec is not None else f"Missing. Required for {purpose}.",
    }


def _require_modules(module_names: tuple[str, ...]) -> None:
    missing = [module_name for module_name in module_names if importlib.util.find_spec(module_name) is None]
    if missing:
        joined = ", ".join(missing)
        raise DependencyError(f"Missing Python dependencies: {joined}. Run install.ps1 first.")


def _torch_check() -> dict[str, Any]:
    spec = importlib.util.find_spec("torch")
    if spec is None:
        return {"name": "torch", "pass": False, "detail": "Missing. Required by Demucs."}
    try:
        import torch

        cuda = "cuda available" if torch.cuda.is_available() else "cpu only"
        return {"name": "torch", "pass": True, "detail": f"{torch.__version__}, {cuda}"}
    except Exception as exc:
        return {"name": "torch", "pass": False, "detail": str(exc)}


def _emit(progress: ProgressCallback | None, message: str) -> None:
    if progress is not None:
        progress(message)


def _clean_process_output(raw_line: str) -> list[str]:
    messages: list[str] = []
    for part in raw_line.replace("\r", "\n").splitlines():
        text = part.strip()
        if not text:
            continue

        progress_match = _PROGRESS_PATTERN.search(text)
        if progress_match:
            percent = min(100, int(progress_match.group("percent")))
            messages.append(f"Demucs progress: {percent}%")
            continue

        if _is_progress_noise(text):
            continue

        messages.append(text.replace("\ufffd", "?"))
    return messages


def _is_progress_noise(text: str) -> bool:
    if "?" not in text:
        return False
    question_ratio = text.count("?") / max(len(text), 1)
    return question_ratio > 0.25 and ("|" in text or "[" in text or "seconds" in text)


def _quote_for_log(value: str) -> str:
    if " " in value:
        return f'"{value}"'
    return value


def options_from_profile(profile: str) -> SeparationOptions:
    if profile not in PROFILES:
        valid = ", ".join(sorted(PROFILES))
        raise ValueError(f"Unknown profile '{profile}'. Valid profiles: {valid}")
    return replace(SeparationOptions(), profile=profile)
