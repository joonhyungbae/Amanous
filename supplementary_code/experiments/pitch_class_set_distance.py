#!/usr/bin/env python3
"""
Pitch Class Set (PCS) Distance: harmonic separation of two voices independent of dynamics.

Addresses VSS bias (velocity ~97%): measures pure pitch-class separation in time windows.
Output: PCS distance (higher = more separated harmonic territory) and optional
cosine similarity / Euclidean distance between 12-D PCS vectors per window.
"""

import sys
import os
import argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))


def events_to_pitch_lists(events):
    """events: list of dict with 'onset_time', 'pitch'. Returns (times, pitches)."""
    sorted_events = sorted(events, key=lambda e: e["onset_time"])
    times = [e["onset_time"] for e in sorted_events]
    pitches = [e["pitch"] for e in sorted_events]
    return times, pitches


def pcs_vector_for_window(times, pitches, t_start: float, t_end: float):
    """
    ​Pitch Class Set (0--11) count vector for notes in [t_start, t_end).
    Returns length-12 vector (normalized to sum 1).
    """
    vec = np.zeros(12)
    for t, p in zip(times, pitches):
        if t_start <= t < t_end:
            vec[p % 12] += 1
    s = vec.sum()
    if s > 0:
        vec = vec / s
    return vec


def pcs_distance_cosine(vec1: np.ndarray, vec2: np.ndarray):
    """1 - cosine similarity so that higher = more separated."""
    n1 = np.linalg.norm(vec1)
    n2 = np.linalg.norm(vec2)
    if n1 == 0 or n2 == 0:
        return 1.0
    sim = np.dot(vec1, vec2) / (n1 * n2)
    return 1 - float(np.clip(sim, -1, 1))


def pcs_distance_euclidean(vec1: np.ndarray, vec2: np.ndarray):
    """Euclidean distance between 12-D PCS vectors (normalized)."""
    return float(np.linalg.norm(vec1 - vec2))


def pitch_class_set_distance(
    voice1_events,
    voice2_events,
    window_sec: float = 1.0,
    hop_sec: float = None,
    metric: str = "cosine",
):
    """
    Compute mean PCS distance between two MIDI tracks over time windows.

    Parameters
    ----------
    voice1_events, voice2_events : list of dict
        Each dict: 'onset_time', 'pitch' (and optionally 'velocity')
    window_sec : float
        Window length in seconds
    hop_sec : float, optional
        Hop between windows (default: window_sec)
    metric : str
        'cosine' -> 1 - cosine similarity; 'euclidean' -> L2 distance

    Returns
    -------
    dict with keys: mean_distance, std_distance, n_windows, distances_per_window
    """
    if hop_sec is None:
        hop_sec = window_sec

    t1, p1 = events_to_pitch_lists(voice1_events)
    t2, p2 = events_to_pitch_lists(voice2_events)

    t_min = min(t1[0] if t1 else 0, t2[0] if t2 else 0)
    t_max = max(t1[-1] if t1 else 0, t2[-1] if t2 else 0)
    duration = max(0, t_max - t_min)
    if duration == 0:
        return {"mean_distance": np.nan, "std_distance": np.nan, "n_windows": 0, "distances_per_window": []}

    distances = []
    t_start = t_min
    while t_start + window_sec <= t_max:
        t_end = t_start + window_sec
        v1 = pcs_vector_for_window(t1, p1, t_start, t_end)
        v2 = pcs_vector_for_window(t2, p2, t_start, t_end)
        if metric == "cosine":
            d = pcs_distance_cosine(v1, v2)
        else:
            d = pcs_distance_euclidean(v1, v2)
        distances.append(d)
        t_start += hop_sec

    distances = np.array(distances)
    return {
        "mean_distance": float(np.mean(distances)),
        "std_distance": float(np.std(distances)) if len(distances) > 1 else 0,
        "n_windows": len(distances),
        "distances_per_window": distances.tolist(),
    }


def load_events_from_csv(path: str):
    """Load events from CSV with columns onset_time (or time), pitch, optionally velocity."""
    import csv
    events = []
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            t = float(row.get("onset_time", row.get("time", 0)))
            p = int(row.get("pitch", 60))
            v = int(row.get("velocity", 512))
            events.append({"onset_time": t, "pitch": p, "velocity": v})
    return events


def main():
    parser = argparse.ArgumentParser(description="Pitch Class Set distance between two voices")
    parser.add_argument("voice1_csv", nargs="?", help="CSV: onset_time, pitch [, velocity]")
    parser.add_argument("voice2_csv", nargs="?", help="CSV: onset_time, pitch [, velocity]")
    parser.add_argument("--window", type=float, default=1.0, help="Window length (seconds)")
    parser.add_argument("--hop", type=float, default=None, help="Hop (default: window)")
    parser.add_argument("--metric", choices=["cosine", "euclidean"], default="cosine")
    args = parser.parse_args()

    if args.voice1_csv and args.voice2_csv:
        v1 = load_events_from_csv(args.voice1_csv)
        v2 = load_events_from_csv(args.voice2_csv)
    else:
        # Demo with synthetic two-voice data
        rng = np.random.default_rng(42)
        v1 = [{"onset_time": t, "pitch": 60 + (i % 7), "velocity": 500} for i, t in enumerate(np.cumsum(rng.exponential(0.1, 200)))]
        v2 = [{"onset_time": t, "pitch": 72 + (i % 5), "velocity": 400} for i, t in enumerate(np.cumsum(rng.exponential(0.12, 200)))]
        print("No CSV provided; using synthetic two-voice demo.")

    result = pitch_class_set_distance(v1, v2, window_sec=args.window, hop_sec=args.hop, metric=args.metric)
    print("PCS distance ({}): mean = {:.4f}, std = {:.4f}, n_windows = {}".format(
        args.metric, result["mean_distance"], result["std_distance"], result["n_windows"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
