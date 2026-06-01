from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .core import PROFILES, SeparationOptions, diagnose_environment, separate_bass


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.doctor:
        return _doctor()

    if args.gui:
        from .gui import main as gui_main

        gui_main()
        return 0

    if args.input is None:
        parser.error("input is required unless --doctor or --gui is used")

    options = SeparationOptions(
        profile=args.profile,
        model=args.model,
        shifts=args.shifts,
        overlap=args.overlap,
        jobs=args.jobs,
        device=args.device,
        output_format=args.format,
        mp3_bitrate=args.mp3_bitrate,
        clip_mode=args.clip_mode,
        keep_work_dir=args.keep_work_dir,
        work_dir=Path(args.work_dir) if args.work_dir else None,
        export_no_bass=args.export_no_bass,
        kick_clean=args.kick_clean,
        kick_strength=args.kick_strength,
        kick_min_frequency=args.kick_min_frequency,
        kick_max_frequency=args.kick_max_frequency,
        kick_window_ms=args.kick_window_ms,
        make_score=args.score,
        score_path=Path(args.score_path) if args.score_path else None,
        score_tempo=args.score_tempo,
        score_key=args.score_key,
        score_title=args.score_title,
    )

    try:
        result = separate_bass(
            input_path=Path(args.input),
            output_path=Path(args.output) if args.output else None,
            options=options,
            progress=_safe_print,
        )
    except Exception as exc:
        _safe_print(f"ERROR: {exc}", stream=sys.stderr)
        return 1

    payload = {"bass": str(result.bass_path), "report": str(result.report_path)}
    if result.kick_cleanup is not None:
        payload["kick_cleanup"] = result.kick_cleanup
    if result.score_path is not None:
        payload["score"] = str(result.score_path)
    if result.score is not None:
        payload["score_analysis"] = result.score
    _safe_print(json.dumps(payload, ensure_ascii=False))
    return 0


def _doctor() -> int:
    checks = diagnose_environment()
    ok = True
    for check in checks:
        mark = "OK" if check["pass"] else "MISSING"
        _safe_print(f"{mark:8} {check['name']}: {check['detail']}")
        ok = ok and bool(check["pass"])
    return 0 if ok else 2


def _safe_print(message: str, stream=None) -> None:
    stream = stream or sys.stdout
    encoding = stream.encoding or "utf-8"
    safe = message.encode(encoding, errors="replace").decode(encoding, errors="replace")
    stream.write(safe + "\n")
    stream.flush()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bass-extractor",
        description="Extract a bass stem from a mixed song using Demucs.",
    )
    parser.add_argument("input", nargs="?", help="Input song path.")
    parser.add_argument("-o", "--output", help="Output bass stem path or directory.")
    parser.add_argument("--gui", action="store_true", help="Open the desktop GUI.")
    parser.add_argument("--doctor", action="store_true", help="Check runtime dependencies.")
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILES),
        default="studio",
        help="Quality profile. studio is slowest and best for final work.",
    )
    parser.add_argument("--model", help="Override Demucs model name, e.g. htdemucs_ft.")
    parser.add_argument("--shifts", type=int, help="Override random shift averaging count.")
    parser.add_argument("--overlap", type=float, help="Override split overlap.")
    parser.add_argument("--jobs", type=int, help="Parallel jobs passed to Demucs.")
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Run on cpu, cuda, or let Demucs choose.",
    )
    parser.add_argument("--format", choices=["wav", "flac", "mp3"], default="wav")
    parser.add_argument("--mp3-bitrate", type=int, default=320)
    parser.add_argument("--clip-mode", choices=["rescale", "clamp", "none"], default="rescale")
    parser.add_argument("--keep-work-dir", action="store_true", help="Keep raw Demucs output folder.")
    parser.add_argument("--work-dir", help="Use a specific Demucs work/output folder.")
    parser.add_argument("--export-no-bass", action="store_true", help="Also export the no_bass stem.")
    parser.add_argument(
        "--kick-clean",
        action="store_true",
        help="Detect fixed-pitch kick hits and attenuate kick leakage in the bass stem.",
    )
    parser.add_argument("--kick-strength", type=float, default=0.65, help="Kick attenuation strength, 0.0-0.95.")
    parser.add_argument("--kick-min-frequency", type=float, default=35.0, help="Lowest expected kick pitch in Hz.")
    parser.add_argument("--kick-max-frequency", type=float, default=135.0, help="Highest expected kick pitch in Hz.")
    parser.add_argument("--kick-window-ms", type=float, default=150.0, help="Milliseconds to attenuate after each kick.")
    parser.add_argument(
        "--score",
        action="store_true",
        help="Create a bass staff MusicXML score after extraction.",
    )
    parser.add_argument("--score-path", help="MusicXML output path. Defaults to the bass output name with .musicxml.")
    parser.add_argument("--score-tempo", type=float, help="Override detected BPM for the generated score.")
    parser.add_argument("--score-key", help="Override detected key, e.g. 'C major' or 'A minor'.")
    parser.add_argument("--score-title", help="Title written into the MusicXML score.")
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
