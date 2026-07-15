#!/usr/bin/env python3
"""
Hardware compensation (Layer 4) validation: Onset Jitter analysis

Compare pre-compensation and Amanous (post-compensation) MIDI performance recordings (.wav)
against Ground Truth (MIDI timestamps) to visualize jitter reduction after compensation
and proximity to scanning resolution (~1 ms).

Requires: librosa (onset detection)
"""

import sys
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
    """Return actual sound onset times (seconds)."""
    import librosa
    try:
        onset_times = librosa.onset.onset_detect(
            y=y, sr=sr, hop_length=hop_length, backtrack=backtrack, units="time"
        )
        return np.asarray(onset_times)
    except TypeError:
        onset_frames = librosa.onset.onset_detect(
            y=y, sr=sr, hop_length=hop_length, backtrack=backtrack
        )
        return librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length)


def theoretical_onsets_from_midi(midi_path: str, track_index: int = 0):
    """Return intended onset times (seconds) from MIDI file."""
    try:
        import mido
        mid = mido.MidiFile(midi_path)
        ticks_per_beat = mid.ticks_per_beat
        tempo = 500000
        t_ticks = 0
        onsets_ticks = []
        for msg in mid.tracks[track_index]:
            t_ticks += msg.time
            if getattr(msg, "type", None) == "note_on" and getattr(msg, "velocity", 0) > 0:
                onsets_ticks.append(t_ticks)
            if msg.type == "set_tempo":
                tempo = msg.tempo
        sec_per_tick = (tempo / 1e6) / ticks_per_beat
        return sorted(set(o * sec_per_tick for o in onsets_ticks))
    except Exception as e:
        raise RuntimeError(f"MIDI read failed: {e}") from e


def align_onsets_to_theoretical(actual_onsets, theoretical_onsets, max_diff_sec: float = 0.05):
    """
    Match each intended onset to nearest actual onset within max_diff_sec.
    Returns (paired_actual, paired_theoretical); error = actual - theoretical.
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
    return np.array(paired_a), np.array(paired_t)


def jitter_stats(actual_onsets, theoretical_onsets, max_pair_sec: float = 0.05):
    """
    Error = |actual onset time - intended MIDI time|.
    Returns mean error, std (jitter).
    """
    pa, pt = align_onsets_to_theoretical(
        actual_onsets, theoretical_onsets, max_diff_sec=max_pair_sec
    )
    if len(pa) == 0:
        return {
            "mean_error_sec": np.nan,
            "std_error_sec": np.nan,
            "mean_abs_error_sec": np.nan,
            "std_abs_error_sec": np.nan,
            "jitter_ms": np.nan,
            "n_paired": 0,
            "errors_sec": np.array([]),
            "abs_errors_ms": np.array([]),
        }
    errors_sec = pa - pt  # signed error (actual - intended)
    abs_errors = np.abs(errors_sec)  # Error = |actual onset - intended MIDI|
    return {
        "mean_error_sec": float(np.mean(errors_sec)),
        "std_error_sec": float(np.std(errors_sec)),
        "mean_abs_error_sec": float(np.mean(abs_errors)),
        "std_abs_error_sec": float(np.std(abs_errors)),
        "jitter_ms": float(np.std(errors_sec)) * 1000,
        "n_paired": len(pa),
        "errors_sec": errors_sec,
        "errors_ms": errors_sec * 1000,
        "abs_errors_ms": abs_errors * 1000,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Pre- vs post-hardware compensation onset jitter (WAV + MIDI Ground Truth)"
    )
    parser.add_argument("wav_before", help="Pre-compensation MIDI performance WAV")
    parser.add_argument("wav_after", help="Amanous post-compensation MIDI performance WAV")
    parser.add_argument("midi", help="Ground Truth MIDI file")
    parser.add_argument("--track", type=int, default=0, help="MIDI track index (0-based) for note onsets")
    parser.add_argument("--max-pair-sec", type=float, default=0.05,
                        help="Max onset matching distance (seconds)")
    parser.add_argument("--sr", type=int, default=22050)
    parser.add_argument("--hop-length", type=int, default=512)
    parser.add_argument("-o", "--output-plot", type=str, default="onset_jitter_comparison.png",
                        help="Histogram output path")
    parser.add_argument("--resolution-ms", type=float, default=1.0,
                        help="Hardware scanning resolution (ms) reference")
    args = parser.parse_args()

    # Ground Truth
    theory_onsets = theoretical_onsets_from_midi(args.midi, track_index=args.track)
    if not theory_onsets:
        print("Warning: No onsets found in MIDI.")
        return 1

    # Before compensation
    y_before, sr = load_audio(args.wav_before, sr=args.sr)
    onsets_before = detect_onsets(y_before, sr, hop_length=args.hop_length)
    stats_before = jitter_stats(onsets_before, theory_onsets, max_pair_sec=args.max_pair_sec)

    # After compensation
    y_after, sr = load_audio(args.wav_after, sr=args.sr)
    onsets_after = detect_onsets(y_after, sr, hop_length=args.hop_length)
    stats_after = jitter_stats(onsets_after, theory_onsets, max_pair_sec=args.max_pair_sec)

    # Report
    print("=" * 60)
    print("Onset Jitter: Hardware compensation (Layer 4) effect")
    print("=" * 60)
    print(f"Ground Truth (MIDI) onsets: {len(theory_onsets)}")
    print()
    print("Before compensation:")
    print(f"  Matched onsets:        {stats_before['n_paired']}")
    print(f"  Mean error (sec):     {stats_before['mean_error_sec']:.6f}")
    print(f"  Mean |Error| (sec):   {stats_before['mean_abs_error_sec']:.6f}  (Error = |actual - intended|)")
    print(f"  Std(Error) sec:       {stats_before.get('std_abs_error_sec', np.nan):.6f}  (std of |Error|)")
    print(f"  Jitter (signed std): {stats_before['std_error_sec']:.6f} sec = {stats_before['jitter_ms']:.2f} ms")
    print()
    print("After compensation (Amanous):")
    print(f"  Matched onsets:        {stats_after['n_paired']}")
    print(f"  Mean error (sec):     {stats_after['mean_error_sec']:.6f}")
    print(f"  Mean |Error| (sec):   {stats_after['mean_abs_error_sec']:.6f}")
    print(f"  Std(Error) sec:       {stats_after.get('std_abs_error_sec', np.nan):.6f}")
    print(f"  Jitter (signed std): {stats_after['std_error_sec']:.6f} sec = {stats_after['jitter_ms']:.2f} ms")
    print()

    # Jitter reduction
    if stats_before["jitter_ms"] > 0 and not np.isnan(stats_after["jitter_ms"]):
        reduction_pct = (1 - stats_after["jitter_ms"] / stats_before["jitter_ms"]) * 100
        print(f"Jitter reduction: {reduction_pct:.1f}% (after: {stats_after['jitter_ms']:.2f} ms)")
    print()
    print(f"Hardware scanning resolution reference: ~{args.resolution_ms} ms")
    print(f"Post-compensation jitter near resolution: {stats_after['jitter_ms']:.2f} ms")
    print()

    # Histogram
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not found; skipping histogram")
        return 0

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    err_before_ms = stats_before["errors_ms"]
    err_after_ms = stats_after["errors_ms"]

    if len(err_before_ms) > 0:
        axes[0].hist(err_before_ms, bins=min(50, max(10, len(err_before_ms)//5)),
                     color="tab:orange", alpha=0.8, edgecolor="black")
        axes[0].axvline(0, color="black", linestyle="-", linewidth=0.8)
        axes[0].axvline(args.resolution_ms, color="green", linestyle="--", label=f"~{args.resolution_ms} ms resolution")
        axes[0].axvline(-args.resolution_ms, color="green", linestyle="--")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Before correction: Onset error (actual - intended, ms)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    if len(err_after_ms) > 0:
        axes[1].hist(err_after_ms, bins=min(50, max(10, len(err_after_ms)//5)),
                     color="tab:blue", alpha=0.8, edgecolor="black")
        axes[1].axvline(0, color="black", linestyle="-", linewidth=0.8)
        axes[1].axvline(args.resolution_ms, color="green", linestyle="--", label=f"~{args.resolution_ms} ms resolution")
        axes[1].axvline(-args.resolution_ms, color="green", linestyle="--")
    axes[1].set_xlabel("Error (ms)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("After correction (Amanous): Onset error distribution")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(args.output_plot, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {args.output_plot}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
