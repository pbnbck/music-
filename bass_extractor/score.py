from __future__ import annotations

import math
import warnings
import xml.etree.ElementTree as ET
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


NOTE_NAMES_SHARP = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
MAJOR_PROFILE = np.asarray([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.asarray([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
KEY_FIFTHS = {
    "C": 0,
    "G": 1,
    "D": 2,
    "A": 3,
    "E": 4,
    "B": 5,
    "F#": 6,
    "C#": 7,
    "F": -1,
    "Bb": -2,
    "Eb": -3,
    "Ab": -4,
    "Db": -5,
    "Gb": -6,
    "Cb": -7,
}
PC_TO_FLAT_NAME = ("C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B")
MAJOR_QUALITIES = {0: "major", 2: "minor", 4: "minor", 5: "major", 7: "major", 9: "minor", 11: "diminished"}
MINOR_QUALITIES = {0: "minor", 2: "diminished", 3: "major", 5: "minor", 7: "minor", 8: "major", 10: "major"}


@dataclass(frozen=True)
class ScoreOptions:
    divisions_per_quarter: int = 4
    beats_per_measure: int = 4
    beat_unit: int = 4
    min_frequency: float = 35.0
    max_frequency: float = 320.0
    tempo_override: float | None = None
    key_override: str | None = None
    title: str | None = None
    minimum_note_ms: float = 80.0
    make_pdf: bool = True
    pdf_path: Path | None = None


@dataclass(frozen=True)
class NoteEvent:
    midi: int | None
    start_division: int
    duration_divisions: int


@dataclass(frozen=True)
class ChordEvent:
    measure: int
    root: str
    quality: str
    bass_midi: int

    @property
    def symbol(self) -> str:
        if self.quality == "major":
            return self.root
        if self.quality == "minor":
            return f"{self.root}m"
        if self.quality == "diminished":
            return f"{self.root}dim"
        return self.root

    def to_dict(self) -> dict[str, Any]:
        return {
            "measure": self.measure,
            "root": self.root,
            "quality": self.quality,
            "symbol": self.symbol,
            "bass_midi": self.bass_midi,
        }


@dataclass(frozen=True)
class ScoreResult:
    score_path: Path
    pdf_path: Path | None
    bpm: float
    key: str
    key_fifths: int
    mode: str
    note_count: int
    measure_count: int
    chords: list[ChordEvent]
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "score_path": str(self.score_path),
            "pdf_path": str(self.pdf_path) if self.pdf_path is not None else None,
            "bpm": round(self.bpm, 3),
            "key": self.key,
            "key_fifths": self.key_fifths,
            "mode": self.mode,
            "note_count": self.note_count,
            "measure_count": self.measure_count,
            "chords": [chord.to_dict() for chord in self.chords],
            "warnings": self.warnings,
        }


def create_bass_score(
    bass_path: Path,
    output_path: Path | None = None,
    options: ScoreOptions | None = None,
) -> ScoreResult:
    options = options or ScoreOptions()
    _validate_options(options)

    sample_rate, audio = _read_audio(bass_path)
    mono = _to_mono(audio)
    bpm = float(options.tempo_override or estimate_bpm(mono, sample_rate))
    key_name, key_fifths, mode = estimate_key(mono, sample_rate, options)
    if options.key_override:
        key_name, key_fifths, mode = _parse_key_override(options.key_override)

    frame_pitches, frame_times, frame_energies = track_bass_pitch(mono, sample_rate, options)
    note_events = frame_pitches_to_events(frame_pitches, frame_times, frame_energies, bpm, options)
    chords = infer_chords(note_events, key_name, mode, options)

    destination = output_path or bass_path.with_suffix(".musicxml")
    destination.parent.mkdir(parents=True, exist_ok=True)
    warnings_list = [
        "Chords are inferred from bass notes and key context, not guaranteed full harmonic transcription."
    ]
    write_musicxml(destination, note_events, chords, bpm, key_name, key_fifths, mode, options)
    pdf_path = options.pdf_path or destination.with_suffix(".pdf")
    if options.make_pdf:
        write_pdf_score(pdf_path, note_events, chords, bpm, key_name, key_fifths, mode, options)

    measure_count = _measure_count(note_events, options)
    return ScoreResult(
        score_path=destination,
        pdf_path=pdf_path if options.make_pdf else None,
        bpm=bpm,
        key=key_name,
        key_fifths=key_fifths,
        mode=mode,
        note_count=sum(1 for event in note_events if event.midi is not None),
        measure_count=measure_count,
        chords=chords,
        warnings=warnings_list,
    )


def estimate_bpm(mono_audio: np.ndarray, sample_rate: int) -> float:
    if mono_audio.size < sample_rate:
        return 120.0

    hop = max(1, int(sample_rate * 0.020))
    frame = max(hop * 2, int(sample_rate * 0.060))
    envelope = _frame_rms(_band_limit(mono_audio, sample_rate, 35.0, 420.0), frame, hop)
    onset = np.maximum(np.diff(np.log1p(envelope * 1000.0), prepend=0.0), 0.0)
    onset = onset - float(np.mean(onset))
    if not np.any(onset > 0):
        return 120.0

    autocorr = signal.correlate(onset, onset, mode="full")[onset.size - 1 :]
    autocorr[:2] = 0.0
    hop_seconds = hop / float(sample_rate)
    min_bpm, max_bpm = 60.0, 200.0
    min_lag = max(1, int((60.0 / max_bpm) / hop_seconds))
    max_lag = min(autocorr.size - 1, int((60.0 / min_bpm) / hop_seconds))
    if max_lag <= min_lag:
        return 120.0

    lag = min_lag + int(np.argmax(autocorr[min_lag : max_lag + 1]))
    bpm = 60.0 / (lag * hop_seconds)
    if bpm < 80.0:
        bpm *= 2.0
    elif bpm > 180.0:
        bpm /= 2.0
    return float(round(np.clip(bpm, 40.0, 220.0), 3))


def track_bass_pitch(
    mono_audio: np.ndarray,
    sample_rate: int,
    options: ScoreOptions | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    options = options or ScoreOptions()
    filtered = _band_limit(mono_audio, sample_rate, options.min_frequency, options.max_frequency * 1.35)
    frame = max(512, int(sample_rate * 0.090))
    hop = max(128, int(sample_rate * 0.025))
    if filtered.size < frame:
        filtered = np.pad(filtered, (0, frame - filtered.size))

    frame_count = 1 + max(0, (filtered.size - frame) // hop)
    midi_values = np.full(frame_count, -1, dtype=np.int16)
    times = np.empty(frame_count, dtype=np.float32)
    energies = np.empty(frame_count, dtype=np.float32)

    rms_values = []
    for index in range(frame_count):
        start = index * hop
        chunk = filtered[start : start + frame]
        rms_values.append(math.sqrt(float(np.mean(np.square(chunk))) + 1e-12))
    rms_array = np.asarray(rms_values, dtype=np.float32)
    energy_threshold = max(float(np.percentile(rms_array, 25)) * 0.45, float(np.max(rms_array)) * 0.02)

    for index in range(frame_count):
        start = index * hop
        chunk = filtered[start : start + frame].astype(np.float32, copy=True)
        times[index] = (start + frame / 2.0) / float(sample_rate)
        energies[index] = rms_array[index]
        if rms_array[index] < energy_threshold:
            continue
        frequency, confidence = _estimate_frame_frequency(chunk, sample_rate, options.min_frequency, options.max_frequency)
        if frequency is None or confidence < 0.22:
            continue
        midi_values[index] = int(round(_frequency_to_midi(frequency)))

    midi_values = _smooth_midi_track(midi_values)
    return midi_values, times, energies


def frame_pitches_to_events(
    frame_pitches: np.ndarray,
    frame_times: np.ndarray,
    frame_energies: np.ndarray,
    bpm: float,
    options: ScoreOptions | None = None,
) -> list[NoteEvent]:
    options = options or ScoreOptions()
    if frame_pitches.size == 0:
        return [NoteEvent(None, 0, options.beats_per_measure * options.divisions_per_quarter)]

    quarter_seconds = 60.0 / bpm
    divisions_per_second = options.divisions_per_quarter / quarter_seconds
    frame_step = float(np.median(np.diff(frame_times))) if frame_times.size > 1 else 0.025
    start_offset = max(0.0, frame_times[0] - frame_step / 2.0)
    min_duration = max(1, int(round(options.minimum_note_ms / 1000.0 * divisions_per_second)))

    raw_events: list[NoteEvent] = []
    current = int(frame_pitches[0])
    start_time = start_offset
    for index in range(1, frame_pitches.size):
        pitch = int(frame_pitches[index])
        if pitch == current:
            continue
        end_time = max(start_time + frame_step, frame_times[index] - frame_step / 2.0)
        raw_events.append(_event_from_times(current, start_time, end_time, divisions_per_second))
        current = pitch
        start_time = end_time
    final_end = frame_times[-1] + frame_step / 2.0
    raw_events.append(_event_from_times(current, start_time, final_end, divisions_per_second))

    merged = _merge_short_events(raw_events, min_duration)
    if not merged:
        return [NoteEvent(None, 0, options.beats_per_measure * options.divisions_per_quarter)]
    return _normalize_event_timeline(merged)


def estimate_key(
    mono_audio: np.ndarray,
    sample_rate: int,
    options: ScoreOptions | None = None,
) -> tuple[str, int, str]:
    options = options or ScoreOptions()
    frame_pitches, _, frame_energies = track_bass_pitch(mono_audio, sample_rate, options)
    histogram = np.zeros(12, dtype=np.float32)
    for midi, energy in zip(frame_pitches, frame_energies):
        if midi >= 0:
            histogram[int(midi) % 12] += max(float(energy), 1e-6)
    if not np.any(histogram > 0):
        return "C major", 0, "major"

    histogram = histogram / np.sum(histogram)
    best_score = -float("inf")
    best_tonic = 0
    best_mode = "major"
    for tonic in range(12):
        major_score = float(np.dot(histogram, np.roll(MAJOR_PROFILE, tonic)))
        minor_score = float(np.dot(histogram, np.roll(MINOR_PROFILE, tonic)))
        if major_score > best_score:
            best_score = major_score
            best_tonic = tonic
            best_mode = "major"
        if minor_score > best_score:
            best_score = minor_score
            best_tonic = tonic
            best_mode = "minor"

    tonic_name = _key_name_for_pc(best_tonic)
    key_label = f"{tonic_name} {best_mode}"
    return key_label, KEY_FIFTHS.get(tonic_name, 0), best_mode


def infer_chords(
    note_events: list[NoteEvent],
    key_name: str,
    mode: str,
    options: ScoreOptions | None = None,
) -> list[ChordEvent]:
    options = options or ScoreOptions()
    measure_length = options.beats_per_measure * options.divisions_per_quarter
    measure_count = _measure_count(note_events, options)
    tonic_pc = _key_label_to_pc(key_name)
    qualities = MINOR_QUALITIES if mode == "minor" else MAJOR_QUALITIES
    chords: list[ChordEvent] = []

    for measure_index in range(measure_count):
        start = measure_index * measure_length
        end = start + measure_length
        pitch_weights: dict[int, int] = {}
        for event in note_events:
            if event.midi is None:
                continue
            overlap = _overlap(start, end, event.start_division, event.start_division + event.duration_divisions)
            if overlap > 0:
                pitch_weights[event.midi] = pitch_weights.get(event.midi, 0) + overlap
        if not pitch_weights:
            continue
        bass_midi = max(pitch_weights.items(), key=lambda item: (item[1], -item[0]))[0]
        root_pc = bass_midi % 12
        degree = (root_pc - tonic_pc) % 12
        quality = qualities.get(degree, "major")
        chords.append(
            ChordEvent(
                measure=measure_index + 1,
                root=NOTE_NAMES_SHARP[root_pc],
                quality=quality,
                bass_midi=bass_midi,
            )
        )
    return chords


def write_musicxml(
    output_path: Path,
    note_events: list[NoteEvent],
    chords: list[ChordEvent],
    bpm: float,
    key_name: str,
    key_fifths: int,
    mode: str,
    options: ScoreOptions,
) -> None:
    score = ET.Element("score-partwise", version="3.1")
    work = ET.SubElement(score, "work")
    ET.SubElement(work, "work-title").text = options.title or output_path.stem

    part_list = ET.SubElement(score, "part-list")
    score_part = ET.SubElement(part_list, "score-part", id="P1")
    ET.SubElement(score_part, "part-name").text = "Bass"

    part = ET.SubElement(score, "part", id="P1")
    measure_length = options.beats_per_measure * options.divisions_per_quarter
    measure_count = _measure_count(note_events, options)
    chord_by_measure = {chord.measure: chord for chord in chords}
    pieces = _events_to_measure_pieces(note_events, options)

    for measure_number in range(1, measure_count + 1):
        measure = ET.SubElement(part, "measure", number=str(measure_number))
        if measure_number == 1:
            attributes = ET.SubElement(measure, "attributes")
            ET.SubElement(attributes, "divisions").text = str(options.divisions_per_quarter)
            key = ET.SubElement(attributes, "key")
            ET.SubElement(key, "fifths").text = str(key_fifths)
            ET.SubElement(key, "mode").text = mode
            time = ET.SubElement(attributes, "time")
            ET.SubElement(time, "beats").text = str(options.beats_per_measure)
            ET.SubElement(time, "beat-type").text = str(options.beat_unit)
            clef = ET.SubElement(attributes, "clef")
            ET.SubElement(clef, "sign").text = "F"
            ET.SubElement(clef, "line").text = "4"

            direction = ET.SubElement(measure, "direction", placement="above")
            direction_type = ET.SubElement(direction, "direction-type")
            metronome = ET.SubElement(direction_type, "metronome")
            ET.SubElement(metronome, "beat-unit").text = "quarter"
            ET.SubElement(metronome, "per-minute").text = str(int(round(bpm)))
            ET.SubElement(direction, "sound", tempo=str(round(bpm, 3)))

        chord = chord_by_measure.get(measure_number)
        if chord is not None:
            _add_harmony(measure, chord)

        for piece in pieces.get(measure_number, []):
            _add_note(measure, piece)

        if measure_number not in pieces:
            _add_note(measure, NotePiece(None, measure_length, False, False))

    ET.indent(score, space="  ")
    ET.ElementTree(score).write(output_path, encoding="utf-8", xml_declaration=True)


def write_pdf_score(
    output_path: Path,
    note_events: list[NoteEvent],
    chords: list[ChordEvent],
    bpm: float,
    key_name: str,
    key_fifths: int,
    mode: str,
    options: ScoreOptions,
) -> None:
    try:
        _write_matplotlib_pdf(output_path, note_events, chords, bpm, key_name, key_fifths, mode, options)
        return
    except Exception:
        pass

    del key_fifths, mode
    output_path.parent.mkdir(parents=True, exist_ok=True)
    page_width = 595.0
    page_height = 842.0
    margin = 50.0
    staff_left = 82.0
    staff_right = page_width - margin
    staff_width = staff_right - staff_left
    measures_per_system = 4
    systems_per_page = 5
    system_gap = 126.0
    line_gap = 8.0
    measure_length = options.beats_per_measure * options.divisions_per_quarter
    measure_count = _measure_count(note_events, options)
    system_count = max(1, math.ceil(measure_count / measures_per_system))
    page_count = max(1, math.ceil(system_count / systems_per_page))
    pieces = _events_to_measure_pieces(note_events, options)
    chord_by_measure = {chord.measure: chord for chord in chords}

    pages: list[list[str]] = [[] for _ in range(page_count)]
    for page_index, commands in enumerate(pages):
        _pdf_text(commands, margin, page_height - 45.0, options.title or output_path.stem, 18, "F2")
        subtitle = f"Bass score | BPM {bpm:.1f} | Key {key_name} | Chords inferred from bass"
        _pdf_text(commands, margin, page_height - 68.0, subtitle, 10, "F1")
        if page_count > 1:
            _pdf_text(commands, page_width - 95.0, 28.0, f"Page {page_index + 1}/{page_count}", 9, "F1")

    for system_index in range(system_count):
        page_index = system_index // systems_per_page
        system_in_page = system_index % systems_per_page
        commands = pages[page_index]
        staff_bottom = page_height - 150.0 - system_in_page * system_gap
        measure_width = staff_width / measures_per_system
        first_measure = system_index * measures_per_system + 1
        last_measure = min(measure_count, first_measure + measures_per_system - 1)
        active_width = measure_width * (last_measure - first_measure + 1)

        _draw_staff(commands, staff_left, staff_left + active_width, staff_bottom, line_gap)
        _pdf_text(commands, staff_left - 50.0, staff_bottom + line_gap * 2.0 - 7.0, "F", 28, "F2")
        _pdf_circle(commands, staff_left - 16.0, staff_bottom + line_gap * 2.5, 1.7, fill=True)
        _pdf_circle(commands, staff_left - 16.0, staff_bottom + line_gap * 1.5, 1.7, fill=True)
        _pdf_text(commands, staff_left - 50.0, staff_bottom - 18.0, "bass clef", 7, "F1")

        for measure_number in range(first_measure, last_measure + 1):
            slot = measure_number - first_measure
            measure_left = staff_left + slot * measure_width
            measure_right = measure_left + measure_width
            _pdf_line(commands, measure_left, staff_bottom, measure_left, staff_bottom + line_gap * 4.0, 0.8)
            _pdf_line(commands, measure_right, staff_bottom, measure_right, staff_bottom + line_gap * 4.0, 0.8)
            _pdf_text(commands, measure_left + 3.0, staff_bottom - 15.0, str(measure_number), 7, "F1")

            chord = chord_by_measure.get(measure_number)
            if chord is not None:
                _pdf_text(commands, measure_left + 10.0, staff_bottom + line_gap * 5.8, chord.symbol, 11, "F2")

            measure_start = (measure_number - 1) * measure_length
            for event in _events_overlapping_measure(note_events, measure_start, measure_length):
                if event.midi is None:
                    rest_x = measure_left + measure_width * 0.5
                    _pdf_text(commands, rest_x - 4.0, staff_bottom + line_gap * 2.0 - 3.0, "rest", 7, "F1")
                    continue
                event_offset = max(0, event.start_division - measure_start)
                x = measure_left + 16.0 + (event_offset / measure_length) * max(20.0, measure_width - 32.0)
                y = _midi_to_bass_staff_y(event.midi, staff_bottom, line_gap)
                _draw_ledger_lines(commands, x, y, staff_bottom, line_gap)
                _pdf_ellipse(commands, x, y, 4.8, 3.4, fill=True)
                _pdf_line(commands, x + 4.5, y, x + 4.5, y + 23.0, 0.8)

    _write_pdf(output_path, pages, page_width, page_height)


def _write_matplotlib_pdf(
    output_path: Path,
    note_events: list[NoteEvent],
    chords: list[ChordEvent],
    bpm: float,
    key_name: str,
    key_fifths: int,
    mode: str,
    options: ScoreOptions,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.patches import Ellipse

    del mode
    output_path.parent.mkdir(parents=True, exist_ok=True)
    page_width = 595.0
    page_height = 842.0
    margin = 46.0
    staff_left = 86.0
    staff_right = page_width - margin
    staff_width = staff_right - staff_left
    measures_per_system = 4
    systems_per_page = 5
    system_gap = 126.0
    line_gap = 8.5
    measure_length = options.beats_per_measure * options.divisions_per_quarter
    measure_count = _measure_count(note_events, options)
    system_count = max(1, math.ceil(measure_count / measures_per_system))
    page_count = max(1, math.ceil(system_count / systems_per_page))
    chord_by_measure = {chord.measure: chord for chord in chords}

    with PdfPages(output_path) as pdf:
        for page_index in range(page_count):
            figure, axis = plt.subplots(figsize=(8.27, 11.69))
            axis.set_xlim(0, page_width)
            axis.set_ylim(0, page_height)
            axis.axis("off")

            axis.text(margin, page_height - 43, options.title or output_path.stem, fontsize=18, fontweight="bold")
            axis.text(
                margin,
                page_height - 64,
                f"Bass score | BPM {bpm:.1f} | Key {key_name} | Chords inferred from bass",
                fontsize=9,
            )
            if page_count > 1:
                axis.text(page_width - 92, 26, f"Page {page_index + 1}/{page_count}", fontsize=8)

            for system_in_page in range(systems_per_page):
                system_index = page_index * systems_per_page + system_in_page
                if system_index >= system_count:
                    break
                staff_bottom = page_height - 150.0 - system_in_page * system_gap
                measure_width = staff_width / measures_per_system
                first_measure = system_index * measures_per_system + 1
                last_measure = min(measure_count, first_measure + measures_per_system - 1)
                active_width = measure_width * (last_measure - first_measure + 1)

                _mpl_draw_staff(axis, staff_left, staff_left + active_width, staff_bottom, line_gap)
                _mpl_draw_bass_clef(axis, staff_left - 44, staff_bottom, line_gap)
                signature_x = staff_left - 10
                signature_x += _mpl_draw_key_signature(axis, signature_x, staff_bottom, line_gap, key_fifths)
                _mpl_draw_time_signature(axis, signature_x + 8, staff_bottom, line_gap, options)

                for measure_number in range(first_measure, last_measure + 1):
                    slot = measure_number - first_measure
                    measure_left = staff_left + slot * measure_width
                    measure_right = measure_left + measure_width
                    axis.plot([measure_left, measure_left], [staff_bottom, staff_bottom + line_gap * 4], color="black", lw=0.9)
                    axis.plot([measure_right, measure_right], [staff_bottom, staff_bottom + line_gap * 4], color="black", lw=0.9)
                    axis.text(measure_left + 2, staff_bottom - 15, str(measure_number), fontsize=7, color="#555555")

                    chord = chord_by_measure.get(measure_number)
                    if chord is not None:
                        axis.text(measure_left + 12, staff_bottom + line_gap * 5.4, chord.symbol, fontsize=11, fontweight="bold")

                    measure_start = (measure_number - 1) * measure_length
                    for event in _events_overlapping_measure(note_events, measure_start, measure_length):
                        _mpl_draw_event(
                            axis,
                            event,
                            measure_left,
                            measure_width,
                            measure_start,
                            measure_length,
                            staff_bottom,
                            line_gap,
                            Ellipse,
                        )

            figure.tight_layout(pad=0)
            pdf.savefig(figure)
            plt.close(figure)


def _mpl_draw_staff(axis: Any, left: float, right: float, bottom: float, line_gap: float) -> None:
    for index in range(5):
        y = bottom + index * line_gap
        axis.plot([left, right], [y, y], color="black", lw=0.75)


def _mpl_draw_bass_clef(axis: Any, x: float, staff_bottom: float, line_gap: float) -> None:
    center_x = x + 22.0
    center_y = staff_bottom + line_gap * 3.0
    theta = np.linspace(-0.25 * np.pi, 1.25 * np.pi, 90)
    radius_x = np.linspace(18.0, 4.0, theta.size)
    radius_y = np.linspace(24.0, 6.0, theta.size)
    xs = center_x + radius_x * np.cos(theta)
    ys = center_y - radius_y * np.sin(theta)
    axis.plot(xs, ys, color="black", lw=2.1, solid_capstyle="round")
    axis.add_patch(Ellipse((center_x - 2.0, center_y + 0.5), width=7.6, height=7.6, facecolor="black", edgecolor="black"))
    axis.add_patch(Ellipse((x + 47.0, staff_bottom + line_gap * 2.55), width=3.4, height=3.4, facecolor="black", edgecolor="black"))
    axis.add_patch(Ellipse((x + 47.0, staff_bottom + line_gap * 1.55), width=3.4, height=3.4, facecolor="black", edgecolor="black"))
    axis.plot([x + 58, x + 58], [staff_bottom, staff_bottom + line_gap * 4], color="black", lw=0.7)


def _mpl_draw_time_signature(axis: Any, x: float, staff_bottom: float, line_gap: float, options: ScoreOptions) -> None:
    axis.text(x, staff_bottom + line_gap * 2.35, str(options.beats_per_measure), fontsize=15, fontweight="bold")
    axis.text(x, staff_bottom + line_gap * 0.65, str(options.beat_unit), fontsize=15, fontweight="bold")


def _mpl_draw_key_signature(axis: Any, x: float, staff_bottom: float, line_gap: float, key_fifths: int) -> float:
    if key_fifths == 0:
        return 0.0
    sharp_midis = [54, 49, 56, 51, 46, 53, 48]
    flat_midis = [47, 52, 45, 50, 43, 48, 41]
    symbol = "#" if key_fifths > 0 else "b"
    midis = sharp_midis if key_fifths > 0 else flat_midis
    count = min(7, abs(key_fifths))
    for index in range(count):
        y = _midi_to_bass_staff_y(midis[index], staff_bottom, line_gap) - 5
        axis.text(x + index * 8.5, y, symbol, fontsize=13, fontweight="bold")
    return count * 8.5 + 4.0


def _mpl_draw_event(
    axis: Any,
    event: NoteEvent,
    measure_left: float,
    measure_width: float,
    measure_start: int,
    measure_length: int,
    staff_bottom: float,
    line_gap: float,
    ellipse_class: Any,
) -> None:
    if event.midi is None:
        return
    event_offset = max(0, event.start_division - measure_start)
    x = measure_left + 17.0 + (event_offset / measure_length) * max(20.0, measure_width - 34.0)
    y = _midi_to_bass_staff_y(event.midi, staff_bottom, line_gap)
    _mpl_draw_ledger_lines(axis, x, y, staff_bottom, line_gap)
    duration = max(1, event.duration_divisions)
    filled = duration < 8
    note = ellipse_class((x, y), width=10.5, height=7.1, angle=-18, facecolor="black" if filled else "white", edgecolor="black", lw=1.0)
    axis.add_patch(note)
    if duration < 16:
        axis.plot([x + 4.7, x + 4.7], [y, y + 27], color="black", lw=0.9)
        if duration <= 2:
            axis.plot([x + 4.7, x + 14.0], [y + 27, y + 22], color="black", lw=0.9)


def _mpl_draw_ledger_lines(axis: Any, x: float, y: float, staff_bottom: float, line_gap: float) -> None:
    staff_top = staff_bottom + line_gap * 4.0
    if y < staff_bottom:
        ledger = staff_bottom - line_gap
        while ledger >= y - 1.0:
            axis.plot([x - 9, x + 9], [ledger, ledger], color="black", lw=0.65)
            ledger -= line_gap
    elif y > staff_top:
        ledger = staff_top + line_gap
        while ledger <= y + 1.0:
            axis.plot([x - 9, x + 9], [ledger, ledger], color="black", lw=0.65)
            ledger += line_gap


@dataclass(frozen=True)
class NotePiece:
    midi: int | None
    duration: int
    tie_start: bool
    tie_stop: bool


def _events_to_measure_pieces(note_events: list[NoteEvent], options: ScoreOptions) -> dict[int, list[NotePiece]]:
    measure_length = options.beats_per_measure * options.divisions_per_quarter
    by_measure: dict[int, list[NotePiece]] = {}
    for event in note_events:
        remaining = event.duration_divisions
        cursor = event.start_division
        components: list[tuple[int, int]] = []
        while remaining > 0:
            measure_number = cursor // measure_length + 1
            measure_remaining = measure_length - (cursor % measure_length)
            chunk = min(remaining, measure_remaining)
            for duration in _duration_components(chunk):
                components.append((measure_number, duration))
                cursor += duration
                remaining -= duration

        for index, (measure_number, duration) in enumerate(components):
            piece = NotePiece(
                midi=event.midi,
                duration=duration,
                tie_start=event.midi is not None and index < len(components) - 1,
                tie_stop=event.midi is not None and index > 0,
            )
            by_measure.setdefault(measure_number, []).append(piece)
    return by_measure


def _add_harmony(measure: ET.Element, chord: ChordEvent) -> None:
    harmony = ET.SubElement(measure, "harmony")
    root = ET.SubElement(harmony, "root")
    step, alter = _split_note_name(chord.root)
    ET.SubElement(root, "root-step").text = step
    if alter:
        ET.SubElement(root, "root-alter").text = str(alter)
    ET.SubElement(harmony, "kind", text=chord.symbol).text = chord.quality


def _add_note(measure: ET.Element, piece: NotePiece) -> None:
    note = ET.SubElement(measure, "note")
    if piece.midi is None:
        ET.SubElement(note, "rest")
    else:
        pitch = ET.SubElement(note, "pitch")
        step, alter, octave = _midi_to_pitch(piece.midi)
        ET.SubElement(pitch, "step").text = step
        if alter:
            ET.SubElement(pitch, "alter").text = str(alter)
        ET.SubElement(pitch, "octave").text = str(octave)
        if piece.tie_start:
            ET.SubElement(note, "tie", type="start")
        if piece.tie_stop:
            ET.SubElement(note, "tie", type="stop")
    ET.SubElement(note, "duration").text = str(piece.duration)
    ET.SubElement(note, "voice").text = "1"
    note_type, dots = _duration_type(piece.duration)
    ET.SubElement(note, "type").text = note_type
    for _ in range(dots):
        ET.SubElement(note, "dot")
    if piece.midi is not None and (piece.tie_start or piece.tie_stop):
        notations = ET.SubElement(note, "notations")
        if piece.tie_stop:
            ET.SubElement(notations, "tied", type="stop")
        if piece.tie_start:
            ET.SubElement(notations, "tied", type="start")


def _events_overlapping_measure(
    note_events: list[NoteEvent],
    measure_start: int,
    measure_length: int,
) -> list[NoteEvent]:
    measure_end = measure_start + measure_length
    values: list[NoteEvent] = []
    for event in note_events:
        event_end = event.start_division + event.duration_divisions
        if _overlap(measure_start, measure_end, event.start_division, event_end) > 0:
            values.append(event)
    return values


def _draw_staff(commands: list[str], left: float, right: float, bottom: float, line_gap: float) -> None:
    for index in range(5):
        y = bottom + index * line_gap
        _pdf_line(commands, left, y, right, y, 0.65)


def _draw_ledger_lines(commands: list[str], x: float, y: float, staff_bottom: float, line_gap: float) -> None:
    staff_top = staff_bottom + line_gap * 4.0
    ledger_gap = line_gap
    if y < staff_bottom:
        ledger = staff_bottom - ledger_gap
        while ledger >= y - 1.0:
            _pdf_line(commands, x - 8.0, ledger, x + 8.0, ledger, 0.55)
            ledger -= ledger_gap
    elif y > staff_top:
        ledger = staff_top + ledger_gap
        while ledger <= y + 1.0:
            _pdf_line(commands, x - 8.0, ledger, x + 8.0, ledger, 0.55)
            ledger += ledger_gap


def _midi_to_bass_staff_y(midi: int, staff_bottom: float, line_gap: float) -> float:
    step, _, octave = _midi_to_pitch(midi)
    letter_index = {"C": 0, "D": 1, "E": 2, "F": 3, "G": 4, "A": 5, "B": 6}[step]
    diatonic = octave * 7 + letter_index
    bass_bottom_line_g2 = 2 * 7 + 4
    return staff_bottom + (diatonic - bass_bottom_line_g2) * (line_gap / 2.0)


def _midi_label(midi: int) -> str:
    step, alter, octave = _midi_to_pitch(midi)
    accidental = "#" if alter == 1 else "b" if alter == -1 else ""
    return f"{step}{accidental}{octave}"


def _pdf_line(commands: list[str], x1: float, y1: float, x2: float, y2: float, width: float = 1.0) -> None:
    commands.append(f"{width:.3f} w {x1:.3f} {y1:.3f} m {x2:.3f} {y2:.3f} l S")


def _pdf_text(commands: list[str], x: float, y: float, text: str, size: int, font: str = "F1") -> None:
    commands.append(f"BT /{font} {size} Tf {x:.3f} {y:.3f} Td ({_pdf_escape(text)}) Tj ET")


def _pdf_circle(commands: list[str], x: float, y: float, radius: float, fill: bool = False) -> None:
    _pdf_ellipse(commands, x, y, radius, radius, fill=fill)


def _pdf_ellipse(commands: list[str], x: float, y: float, rx: float, ry: float, fill: bool = False) -> None:
    kappa = 0.5522847498
    op = "f" if fill else "S"
    commands.append(
        " ".join(
            [
                f"{x + rx:.3f} {y:.3f} m",
                f"{x + rx:.3f} {y + ry * kappa:.3f} {x + rx * kappa:.3f} {y + ry:.3f} {x:.3f} {y + ry:.3f} c",
                f"{x - rx * kappa:.3f} {y + ry:.3f} {x - rx:.3f} {y + ry * kappa:.3f} {x - rx:.3f} {y:.3f} c",
                f"{x - rx:.3f} {y - ry * kappa:.3f} {x - rx * kappa:.3f} {y - ry:.3f} {x:.3f} {y - ry:.3f} c",
                f"{x + rx * kappa:.3f} {y - ry:.3f} {x + rx:.3f} {y - ry * kappa:.3f} {x + rx:.3f} {y:.3f} c",
                op,
            ]
        )
    )


def _pdf_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _write_pdf(output_path: Path, pages: list[list[str]], page_width: float, page_height: float) -> None:
    objects: list[bytes] = []
    page_count = len(pages)
    catalog_id = 1
    pages_id = 2
    font_regular_id = 3
    font_bold_id = 4
    first_page_id = 5
    content_ids: list[int] = []
    page_ids: list[int] = []

    objects.append(f"<< /Type /Catalog /Pages {pages_id} 0 R >>".encode("ascii"))
    objects.append(b"")
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>")

    next_id = first_page_id
    for page_commands in pages:
        page_id = next_id
        content_id = next_id + 1
        next_id += 2
        page_ids.append(page_id)
        content_ids.append(content_id)
        objects.append(
            (
                f"<< /Type /Page /Parent {pages_id} 0 R /MediaBox [0 0 {page_width:.0f} {page_height:.0f}] "
                f"/Resources << /Font << /F1 {font_regular_id} 0 R /F2 {font_bold_id} 0 R >> >> "
                f"/Contents {content_id} 0 R >>"
            ).encode("ascii")
        )
        stream = "\n".join(page_commands).encode("latin-1", errors="replace")
        objects.append(b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n" + stream + b"\nendstream")

    kids = " ".join(f"{page_id} 0 R" for page_id in page_ids)
    objects[pages_id - 1] = f"<< /Type /Pages /Kids [{kids}] /Count {page_count} >>".encode("ascii")

    output = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for object_id, payload in enumerate(objects, start=1):
        offsets.append(len(output))
        output.extend(f"{object_id} 0 obj\n".encode("ascii"))
        output.extend(payload)
        output.extend(b"\nendobj\n")
    xref_offset = len(output)
    output.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    output.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        output.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    output.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root {catalog_id} 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF\n"
        ).encode("ascii")
    )
    output_path.write_bytes(bytes(output))


def _read_audio(path: Path) -> tuple[int, np.ndarray]:
    if path.suffix.lower() == ".wav":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", WavFileWarning)
            sample_rate, raw = wavfile.read(path)
        audio = _to_float_audio(np.asarray(raw))
    else:
        try:
            import soundfile as sf
        except Exception as exc:  # pragma: no cover - optional path
            raise RuntimeError(f"{path.suffix} score input requires soundfile support.") from exc
        audio, sample_rate = sf.read(path, always_2d=False, dtype="float32")
    if audio.ndim == 1:
        audio = audio[:, None]
    return int(sample_rate), audio.astype(np.float32, copy=False)


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


def _band_limit(audio: np.ndarray, sample_rate: int, low: float, high: float) -> np.ndarray:
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
        chunk = padded[index * hop : index * hop + frame]
        values[index] = math.sqrt(float(np.mean(np.square(chunk))) + 1e-12)
    return values


def _estimate_frame_frequency(
    chunk: np.ndarray,
    sample_rate: int,
    min_frequency: float,
    max_frequency: float,
) -> tuple[float | None, float]:
    chunk = chunk - float(np.mean(chunk))
    if not np.any(np.abs(chunk) > 1e-7):
        return None, 0.0
    chunk = chunk * np.hanning(chunk.size)
    autocorr = signal.correlate(chunk, chunk, mode="full")[chunk.size - 1 :]
    autocorr[0] = max(float(autocorr[0]), 1e-12)
    min_lag = max(1, int(sample_rate / max_frequency))
    max_lag = min(autocorr.size - 1, int(sample_rate / min_frequency))
    if max_lag <= min_lag:
        return None, 0.0
    search = autocorr[min_lag : max_lag + 1]
    lag = min_lag + int(np.argmax(search))
    confidence = float(autocorr[lag] / autocorr[0])
    return sample_rate / float(lag), confidence


def _smooth_midi_track(midi_values: np.ndarray) -> np.ndarray:
    smoothed = midi_values.copy()
    for index in range(1, midi_values.size - 1):
        neighborhood = midi_values[index - 1 : index + 2]
        valid = neighborhood[neighborhood >= 0]
        if valid.size >= 2:
            smoothed[index] = int(np.median(valid))
    return smoothed


def _frequency_to_midi(frequency: float) -> float:
    return 69.0 + 12.0 * math.log2(frequency / 440.0)


def _event_from_times(
    midi: int,
    start_time: float,
    end_time: float,
    divisions_per_second: float,
) -> NoteEvent:
    start_division = max(0, int(round(start_time * divisions_per_second)))
    end_division = max(start_division + 1, int(round(end_time * divisions_per_second)))
    return NoteEvent(None if midi < 0 else midi, start_division, end_division - start_division)


def _merge_short_events(events: list[NoteEvent], min_duration: int) -> list[NoteEvent]:
    merged: list[NoteEvent] = []
    for event in events:
        if event.duration_divisions >= min_duration or not merged:
            merged.append(event)
            continue
        previous = merged[-1]
        merged[-1] = NoteEvent(previous.midi, previous.start_division, previous.duration_divisions + event.duration_divisions)
    return merged


def _normalize_event_timeline(events: list[NoteEvent]) -> list[NoteEvent]:
    normalized: list[NoteEvent] = []
    cursor = 0
    for event in events:
        if event.start_division > cursor:
            normalized.append(NoteEvent(None, cursor, event.start_division - cursor))
            cursor = event.start_division
        normalized.append(NoteEvent(event.midi, cursor, max(1, event.duration_divisions)))
        cursor += max(1, event.duration_divisions)
    return normalized


def _measure_count(note_events: list[NoteEvent], options: ScoreOptions) -> int:
    measure_length = options.beats_per_measure * options.divisions_per_quarter
    if not note_events:
        return 1
    end = max(event.start_division + event.duration_divisions for event in note_events)
    return max(1, int(math.ceil(end / measure_length)))


def _duration_components(duration: int) -> list[int]:
    allowed = (16, 12, 8, 6, 4, 3, 2, 1)
    parts: list[int] = []
    remaining = duration
    while remaining > 0:
        for value in allowed:
            if value <= remaining:
                parts.append(value)
                remaining -= value
                break
    return parts


def _duration_type(duration: int) -> tuple[str, int]:
    mapping = {
        16: ("whole", 0),
        12: ("half", 1),
        8: ("half", 0),
        6: ("quarter", 1),
        4: ("quarter", 0),
        3: ("eighth", 1),
        2: ("eighth", 0),
        1: ("16th", 0),
    }
    return mapping.get(duration, ("quarter", 0))


def _overlap(start_a: int, end_a: int, start_b: int, end_b: int) -> int:
    return max(0, min(end_a, end_b) - max(start_a, start_b))


def _midi_to_pitch(midi: int) -> tuple[str, int, int]:
    pitch_class = midi % 12
    octave = midi // 12 - 1
    name = NOTE_NAMES_SHARP[pitch_class]
    step, alter = _split_note_name(name)
    return step, alter, octave


def _split_note_name(name: str) -> tuple[str, int]:
    if len(name) == 1:
        return name, 0
    if name[1] == "#":
        return name[0], 1
    if name[1] == "b":
        return name[0], -1
    return name[0], 0


def _key_name_for_pc(pc: int) -> str:
    sharp_name = NOTE_NAMES_SHARP[pc]
    if sharp_name in KEY_FIFTHS and abs(KEY_FIFTHS[sharp_name]) <= 4:
        return sharp_name
    return PC_TO_FLAT_NAME[pc]


def _key_label_to_pc(key_name: str) -> int:
    tonic = key_name.split()[0]
    lookup = {name: index for index, name in enumerate(NOTE_NAMES_SHARP)}
    lookup.update({name: index for index, name in enumerate(PC_TO_FLAT_NAME)})
    return lookup.get(tonic, 0)


def _parse_key_override(value: str) -> tuple[str, int, str]:
    parts = value.strip().replace("-", " ").split()
    if not parts:
        return "C major", 0, "major"
    tonic = parts[0]
    mode = parts[1].lower() if len(parts) > 1 else "major"
    if mode not in {"major", "minor"}:
        mode = "major"
    return f"{tonic} {mode}", KEY_FIFTHS.get(tonic, 0), mode


def _validate_options(options: ScoreOptions) -> None:
    if options.divisions_per_quarter < 1:
        raise ValueError("divisions_per_quarter must be positive")
    if options.beats_per_measure < 1:
        raise ValueError("beats_per_measure must be positive")
    if options.min_frequency <= 0 or options.max_frequency <= options.min_frequency:
        raise ValueError("score frequency range is invalid")
