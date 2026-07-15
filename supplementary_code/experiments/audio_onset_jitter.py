#!/usr/bin/env python3
"""
Audio onset alignment analysis for the Polyphony section (40-note chords).

Detects peaks in the audio and compares to theoretical onset timestamps to compute
jitter (std of actual - theoretical). Use to show that hardware correction
reduces vertical alignment error (empirical evidence).

Requires: librosa, numpy, scipy. Optional: path to reference MIDI for theoretical onsets.
"""

import argparse
import numpy as np


def load_audio(path: str, sr: int = 22050):
    try:
        import librosa
        y, fs = librosa.load(path, sr=sr, mono=True)
        return y, fs
    except Exception as e:
        raise RuntimeError(f"Load audio failed: {e}") from e


def detect_onsets(y, sr: int, hop_length: int = 512, backtrack: bool = True):
    """Return onset times in seconds."""
    import librosa
    try:
        onset_times = librosa.onset.onset_detect(
            y=y, sr=sr, hop_length=hop_length, backtrack=backtrack, units="time"
        )
        return np.asarray(onset_times)
    except TypeError:
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr, hop_length=hop_length, backtrack=backtrack)
        return librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length)


def theoretical_onsets_from_midi(midi_path: str, track_index: int = 0):
    """Return sorted list of onset times (seconds) from MIDI file."""
    try:
        import mido
        mid = mido.MidiFile(midi_path)
        ticks_per_beat = mid.ticks_per_beat
        tempo = 500000  # default 120 BPM in microseconds per beat
        t_ticks = 0
        onsets_ticks = []
        for msg in mid.tracks[track_index]:
            t_ticks += msg.time
            if hasattr(msg, "note") and getattr(msg, "type", None) == "note_on" and getattr(msg, "velocity", 0) > 0:
                onsets_ticks.append(t_ticks)
            if msg.type == "set_tempo":
                tempo = msg.tempo
        # Convert ticks to seconds: seconds = ticks * (tempo/1e6) / ticks_per_beat
        sec_per_tick = (tempo / 1e6) / ticks_per_beat
        return sorted(set(o * sec_per_tick for o in onsets_ticks))
    except Exception as e:
        raise RuntimeError(f"MIDI read failed: {e}") from e


def theoretical_onsets_polyphony_section(
    section_start_sec: float,
    section_end_sec: float,
    chord_interval_sec: float = 0.5,
    notes_per_chord: int = 40,
):
    """
    Generate theoretical onsets for the Polyphony section: 40-note chords every 500 ms.
    Returns one time per chord (leading edge); for full vertical alignment you may
    expand to 40 identical times per chord.
    """
    times = []
    t = section_start_sec
    while t < section_end_sec:
        for _ in range(notes_per_chord):
            times.append(t)
        t += chord_interval_sec
    return times


def align_onsets_to_theoretical(actual_onsets, theoretical_onsets, max_diff_sec: float = 0.1):
    """
    Pair each theoretical onset to nearest actual within max_diff_sec.
    Returns (paired_actual, paired_theoretical) and unpaired counts.
    """
    actual = np.asarray(actual_onsets)
    theory = np.asarray(theoretical_onsets)
    paired_a, paired_t = [], []
    used = set()
    for th in theory:
        d = np.abs(actual - th)
        idx = np.argmin(d)
        if d[idx] <= max_diff_sec and idx not in used:
            used.add(idx)
            paired_a.append(actual[idx])
            paired_t.append(th)
    return np.array(paired_a), np.array(paired_t), len(theory) - len(paired_t), len(actual) - len(paired_a)


def jitter_report(actual_onsets, theoretical_onsets, max_pair_sec: float = 0.1):
    """
    Compute jitter (std of actual - theoretical) and summary stats.
    Returns dict: mean_error, std_error (jitter), n_paired, n_theory, n_actual.
    """
    pa, pt, unpaired_t, unpaired_a = align_onsets_to_theoretical(
        actual_onsets, theoretical_onsets, max_diff_sec=max_pair_sec
    )
    if len(pa) == 0:
        return {
            "mean_error_sec": np.nan,
            "std_error_sec": np.nan,
            "jitter_ms": np.nan,
            "n_paired": 0,
            "n_theory": len(theoretical_onsets),
            "n_actual": len(actual_onsets),
            "unpaired_theory": unpaired_t,
            "unpaired_actual": unpaired_a,
        }
    errors_sec = pa - pt
    return {
        "mean_error_sec": float(np.mean(errors_sec)),
        "std_error_sec": float(np.std(errors_sec)),
        "jitter_ms": float(np.std(errors_sec)) * 1000,
        "n_paired": len(pa),
        "n_theory": len(theoretical_onsets),
        "n_actual": len(actual_onsets),
        "unpaired_theory": unpaired_t,
        "unpaired_actual": unpaired_a,
    }


def main():
    parser = argparse.ArgumentParser(description="Polyphony section: onset jitter (actual vs theoretical)")
    parser.add_argument("audio", help="Path to WAV (e.g. 90s demo; Polyphony section)")
    parser.add_argument("--polyphony-start", type=float, default=0, help="Polyphony section start (sec)")
    parser.add_argument("--polyphony-end", type=float, default=10, help="Polyphony section end (sec)")
    parser.add_argument("--chord-interval", type=float, default=0.5, help="Chord interval (sec)")
    parser.add_argument("--notes-per-chord", type=int, default=40)
    parser.add_argument("--midi", type=str, default=None, help="Optional: MIDI for theoretical onsets")
    parser.add_argument("--max-pair-sec", type=float, default=0.1, help="Max sec to pair onset")
    parser.add_argument("--sr", type=int, default=22050)
    args = parser.parse_args()

    y, sr = load_audio(args.audio, sr=args.sr)
    onset_times = detect_onsets(y, sr)
    # Restrict to Polyphony section
    start, end = args.polyphony_start, args.polyphony_end
    onset_times = onset_times[(onset_times >= start) & (onset_times <= end)]

    if args.midi:
        theory_all = theoretical_onsets_from_midi(args.midi)
        theory_sec = [t for t in theory_all if start <= t <= end]
    else:
        theory_sec = theoretical_onsets_polyphony_section(
            start, end, args.chord_interval, args.notes_per_chord
        )

    report = jitter_report(onset_times, theory_sec, max_pair_sec=args.max_pair_sec)
    print("Onset alignment (Polyphony section)")
    print("  Theoretical onsets (in segment):", report["n_theory"])
    print("  Detected onsets (in segment):   ", report["n_actual"])
    print("  Paired:                         ", report["n_paired"])
    print("  Mean error (sec):               ", report["mean_error_sec"])
    print("  Std error — Jitter (sec):       ", report["std_error_sec"])
    print("  Jitter (ms):                    ", report["jitter_ms"])
    return 0


if __name__ == "__main__":
    main()
