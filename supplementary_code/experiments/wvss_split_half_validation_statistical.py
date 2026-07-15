#!/usr/bin/env python3
"""
Split-half Cross-validation for wVSS (Weighted Voice Separation Score) — Statistical Validation.

Splits data 50:50 to validate weight stability and outputs tables (Pandas DataFrame) for the paper's
'Statistical Validation' section. Mitigates circular logic: checks that weights derived from one half
are consistent when applied to the other half.

- Weight definition: each half's domain ratio of Pitch, Velocity, Temporal Wasserstein distance sum.
- Output: Half A vs Half B weight comparison, Pearson correlation, VSS change when applying Half A weights to Half B.
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))
from coherence_metrics import (
    vss_component_distances,
    voice_separation_score,
)


def events_to_voices(events):
    """
    Convert MIDI event list into per-voice lists.
    events: list of dict with keys 'voice_id', 'pitch', 'velocity', 'onset_time' (or trigger_time)
    """
    voices = sorted(set(e.get('voice_id', 0) for e in events))
    return [
        [e for e in events if e.get('voice_id') == v]
        for v in voices
    ]


def pairwise_component_means(voices_list):
    """
    Mean of Pitch, Velocity, Temporal Wasserstein distances per voice pair.
    voices_list: list of N lists of events (each list = one voice).
    Returns: {'pitch': mean_W_pitch, 'velocity': mean_W_vel, 'temporal': mean_W_temporal}
    """
    n = len(voices_list)
    p, v, t = [], [], []
    for i in range(n):
        for j in range(i + 1, n):
            d = vss_component_distances(voices_list[i], voices_list[j])
            p.append(d['pitch'])
            v.append(d['velocity'])
            t.append(d['temporal'])
    return {
        'pitch': float(np.mean(p)) if p else 0.0,
        'velocity': float(np.mean(v)) if v else 0.0,
        'temporal': float(np.mean(t)) if t else 0.0,
    }


def weights_from_total_distance(component_means):
    """
    Define weights as each domain's ratio of total distance (sum = 100%).
    component_means: {'pitch': W_pitch, 'velocity': W_vel, 'temporal': W_temporal}
    """
    total = sum(component_means.values())
    if total == 0:
        return {'pitch': 0.0, 'velocity': 0.0, 'temporal': 0.0}
    return {k: (v / total) * 100 for k, v in component_means.items()}


def split_half_even_odd(events_per_voice):
    """
    50:50 index-based split (even/odd). Sort by onset_time per voice; even indices → Half A, odd → Half B.
    Returns: (half_a_voices, half_b_voices), each a list of voice event lists.
    """
    half_a = []
    half_b = []
    for evs in events_per_voice:
        evs_sorted = sorted(evs, key=lambda e: e.get('onset_time', e.get('trigger_time', 0)))
        half_a.append([evs_sorted[i] for i in range(0, len(evs_sorted), 2)])
        half_b.append([evs_sorted[i] for i in range(1, len(evs_sorted), 2)])
    return half_a, half_b


def mean_wvss_over_voices(voices_list, weights_dict):
    """
    Mean wVSS over voice pairs with given weights.
    weights_dict: {'pitch': pct, 'velocity': pct, 'temporal': pct} (sum 100, 0–100 scale)
    """
    w = {k: v / 100.0 for k, v in weights_dict.items()}  # scale to 0–1
    n = len(voices_list)
    scores = []
    for i in range(n):
        for j in range(i + 1, n):
            s = voice_separation_score(voices_list[i], voices_list[j], weighted=True, weights=w)
            scores.append(s)
    return float(np.mean(scores)) if scores else 0.0


def dominant_domain(weights_dict):
    """Domain name with the largest weight."""
    return max(weights_dict.keys(), key=lambda k: weights_dict[k])


def run_split_half_validation(events=None, events_per_voice=None):
    """
    Run split-half cross-validation.

    Input: events (full event list) or events_per_voice (per-voice event lists).
    If neither is provided, uses default synthetic data.
    """
    if events_per_voice is None:
        if events is not None:
            events_per_voice = events_to_voices(events)
        else:
            events_per_voice = _default_synthetic_voices()

    half_a_voices, half_b_voices = split_half_even_odd(events_per_voice)

    # Half A: component means → weights
    means_a = pairwise_component_means(half_a_voices)
    weights_a = weights_from_total_distance(means_a)

    # Half B: component means → weights
    means_b = pairwise_component_means(half_b_voices)
    weights_b = weights_from_total_distance(means_b)

    # Dominant domain consistency
    dom_a = dominant_domain(weights_a)
    dom_b = dominant_domain(weights_b)
    dominant_consistent = dom_a == dom_b

    # Pearson correlation between weight vectors
    vec_a = np.array([weights_a['pitch'], weights_a['velocity'], weights_a['temporal']])
    vec_b = np.array([weights_b['pitch'], weights_b['velocity'], weights_b['temporal']])
    pearson_r = float(stats.pearsonr(vec_a, vec_b)[0]) if (np.std(vec_a) > 0 and np.std(vec_b) > 0) else np.nan

    # VSS change when applying Half A weights to Half B
    vss_b_own = mean_wvss_over_voices(half_b_voices, weights_b)
    vss_b_with_a_weights = mean_wvss_over_voices(half_b_voices, weights_a)
    if vss_b_own != 0:
        vss_change_pct = (vss_b_with_a_weights - vss_b_own) / vss_b_own * 100
    else:
        vss_change_pct = 0.0 if vss_b_with_a_weights == 0 else np.nan

    # --- Tables for paper: Pandas DataFrame ---

    # Table 1: Half A vs Half B weight comparison
    df_weights = pd.DataFrame({
        'Domain': ['Pitch', 'Velocity', 'Temporal'],
        'Half A (%)': [weights_a['pitch'], weights_a['velocity'], weights_a['temporal']],
        'Half B (%)': [weights_b['pitch'], weights_b['velocity'], weights_b['temporal']],
        'Abs. Diff. (pp)': [
            abs(weights_a['pitch'] - weights_b['pitch']),
            abs(weights_a['velocity'] - weights_b['velocity']),
            abs(weights_a['temporal'] - weights_b['temporal']),
        ],
    })
    df_weights = df_weights.set_index('Domain')

    # Table 2: Statistical Validation summary (single row)
    df_validation = pd.DataFrame([{
        'Dominant (Half A)': dom_a,
        'Dominant (Half B)': dom_b,
        'Dominant consistent': dominant_consistent,
        'Pearson r (weight vectors)': pearson_r,
        'VSS change (%): Half A weights on Half B': vss_change_pct,
    }])

    # Table 3: Combined table for paper (Statistical Validation section)
    df_paper = pd.DataFrame({
        'Metric': [
            'w_pitch (%)',
            'w_velocity (%)',
            'w_temporal (%)',
            'Dominant domain (Half A)',
            'Dominant domain (Half B)',
            'Pearson r (Half A vs Half B weights)',
            'VSS change (%) when applying Half A weights to Half B',
        ],
        'Half A': [
            f"{weights_a['pitch']:.2f}",
            f"{weights_a['velocity']:.2f}",
            f"{weights_a['temporal']:.2f}",
            dom_a,
            '—',
            '—',
            '—',
        ],
        'Half B': [
            f"{weights_b['pitch']:.2f}",
            f"{weights_b['velocity']:.2f}",
            f"{weights_b['temporal']:.2f}",
            '—',
            dom_b,
            '—',
            '—',
        ],
        'Summary': [
            f"Δ = {abs(weights_a['pitch'] - weights_b['pitch']):.2f} pp",
            f"Δ = {abs(weights_a['velocity'] - weights_b['velocity']):.2f} pp",
            f"Δ = {abs(weights_a['temporal'] - weights_b['temporal']):.2f} pp",
            'Consistent' if dominant_consistent else 'Different',
            '',
            f"{pearson_r:.4f}" if not np.isnan(pearson_r) else '—',
            f"{vss_change_pct:.2f}%",
        ],
    })

    return {
        'weights_a': weights_a,
        'weights_b': weights_b,
        'pearson_r': pearson_r,
        'vss_change_pct': vss_change_pct,
        'dominant_consistent': dominant_consistent,
        'df_weights': df_weights,
        'df_validation': df_validation,
        'df_paper': df_paper,
    }


def _default_synthetic_voices():
    """Default synthetic data: 4 voices, 125 events per voice (500 total), high-density condition."""
    N_VOICES = 4
    EVENTS_PER_VOICE = 125
    SEED = 42
    R_PITCH = 7
    duration = (N_VOICES * EVENTS_PER_VOICE) / 120.0
    rng = np.random.default_rng(SEED)
    bands = [(200, 350), (400, 550), (550, 700), (700, 900)]
    events_by_voice = []
    for v in range(N_VOICES):
        low, high = bands[v]
        base_pitch = 48 + v * R_PITCH
        dt = duration / max(1, EVENTS_PER_VOICE - 1)
        times = np.arange(EVENTS_PER_VOICE) * dt
        events = []
        for t in times:
            events.append({
                'onset_time': float(t),
                'pitch': int(np.clip(rng.normal(base_pitch, 2), 21, 108)),
                'velocity': int(rng.uniform(low, high)),
                'voice_id': v,
            })
        events_by_voice.append(events)
    return events_by_voice


def print_tables(result):
    """Print tables for paper Statistical Validation section."""
    print("\n" + "=" * 70)
    print("wVSS Split-half Cross-validation — Statistical Validation (Table output)")
    print("=" * 70)

    print("\n--- Table 1: Half A vs Half B weight comparison ---")
    print(result['df_weights'].to_string())

    print("\n--- Table 2: Validation summary ---")
    print(result['df_validation'].to_string(index=False))

    print("\n--- Table 3: For paper (Statistical Validation section) ---")
    print(result['df_paper'].to_string(index=False))

    print("\n--- Interpretation ---")
    print(f"  Dominant domain consistent (Half A == Half B): {result['dominant_consistent']}")
    print(f"  Pearson r (weight vectors): {result['pearson_r']:.4f}")
    print(f"  VSS change when applying Half A weights to Half B: {result['vss_change_pct']:.2f}%")
    print("=" * 70)


if __name__ == "__main__":
    result = run_split_half_validation()
    print_tables(result)

    # Optional: save CSV
    out_dir = os.path.join(os.path.dirname(__file__))
    result['df_weights'].to_csv(os.path.join(out_dir, "wvss_split_half_weights_table.csv"))
    result['df_validation'].to_csv(os.path.join(out_dir, "wvss_split_half_validation_summary.csv"), index=False)
    result['df_paper'].to_csv(os.path.join(out_dir, "wvss_split_half_paper_table.csv"), index=False)
    print("\nTables saved: wvss_split_half_weights_table.csv, wvss_split_half_validation_summary.csv, wvss_split_half_paper_table.csv")
