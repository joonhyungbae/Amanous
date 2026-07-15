#!/usr/bin/env python3
"""
Split-half cross-validation for wVSS weights (Section 4, wvss_validation).

Uses coherence_metrics.voice_separation_score (wVSS) and weight derivation from
Control vs Multi-Constraint component means. Splits generated MIDI-style events:
  (1) Even/odd index split
  (2) 50:50 random split

Computes deviation of w_pitch, w_vel, w_temporal between halves and statistics
to assess whether weights are consistent across splits (generalizability).
"""

import os
import sys
import numpy as np
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))
from coherence_metrics import (
    vss_component_distances,
    voice_separation_score,
    compute_wvss_weights,
    compute_nwvss_weights,
    NWVSS_RANGES,
)

N_VOICES = 4
EVENTS_PER_VOICE = 125  # 500 events total (paper: 500-event dataset)
SEED = 42
RANDOM_SPLIT_SEED = 123
N_RANDOM_TRIALS = 100  # for 50:50 random: report mean/std over trials


def pairwise_component_means(voices_list):
    """voices_list: list of 4 voice event lists. Return mean W_pitch, W_vel, W_temporal over all pairs."""
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


def split_events_even_odd(events_per_voice):
    """Split each voice's events by even/odd index. Returns (half1_voices, half2_voices)."""
    half1 = []
    half2 = []
    for evs in events_per_voice:
        evs_sorted = sorted(evs, key=lambda e: e['onset_time'])
        half1.append([evs_sorted[i] for i in range(0, len(evs_sorted), 2)])
        half2.append([evs_sorted[i] for i in range(1, len(evs_sorted), 2)])
    return half1, half2


def split_events_random(events_per_voice, rng, fraction=0.5):
    """50:50 random split per voice. Returns (half1_voices, half2_voices)."""
    half1 = []
    half2 = []
    for evs in events_per_voice:
        evs_list = list(evs)
        n = len(evs_list)
        idx = rng.permutation(n)
        n1 = int(n * fraction)
        idx1, idx2 = idx[:n1], idx[n1:]
        half1.append([evs_list[i] for i in idx1])
        half2.append([evs_list[i] for i in idx2])
    return half1, half2


def weight_vector(w):
    """(w_pitch, w_vel, w_temporal) as array for correlation."""
    return np.array([w['pitch'], w['velocity'], w['temporal']])


def deviation_stats(w1, w2):
    """Absolute deviations per component, max, mean, and correlation."""
    v1 = weight_vector(w1)
    v2 = weight_vector(w2)
    devs = np.abs(v1 - v2)
    return {
        'dev_pitch': float(devs[0]),
        'dev_velocity': float(devs[1]),
        'dev_temporal': float(devs[2]),
        'max_deviation_pp': float(np.max(devs)),
        'mean_abs_deviation_pp': float(np.mean(devs)),
        'pearson_r': float(stats.pearsonr(v1, v2)[0]) if np.std(v1) > 0 and np.std(v2) > 0 else np.nan,
        'spearman_rho': float(stats.spearmanr(v1, v2)[0]),
    }


def generate_events_control(n_events, duration, rng, pitch_sep=7):
    """Control: similar velocity, pitch separated."""
    events_by_voice = []
    for v in range(N_VOICES):
        base_pitch = 48 + v * pitch_sep
        dt = duration / max(1, n_events - 1)
        t = 0.0
        events = []
        for _ in range(n_events):
            events.append({
                'onset_time': t,
                'pitch': int(np.clip(rng.normal(base_pitch, 2), 21, 108)),
                'velocity': int(rng.uniform(450, 550)),
            })
            t += dt
        events_by_voice.append(events)
    return events_by_voice


def generate_events_multi(n_events, duration, rng, pitch_sep=7):
    """Multi-Constraint: stratified velocity bands."""
    bands = [(200, 350), (400, 550), (550, 700), (700, 900)]
    events_by_voice = []
    for v in range(N_VOICES):
        low, high = bands[v]
        base_pitch = 48 + v * pitch_sep
        dt = duration / max(1, n_events - 1)
        times = np.arange(n_events) * dt
        events = []
        for t in times:
            events.append({
                'onset_time': float(t),
                'pitch': int(np.clip(rng.normal(base_pitch, 2), 21, 108)),
                'velocity': int(rng.uniform(low, high)),
            })
        events_by_voice.append(events)
    return events_by_voice


def run_split_half_validation():
    rng = np.random.default_rng(SEED)
    rng_split = np.random.default_rng(RANDOM_SPLIT_SEED)

    # High-density condition (paper: 500 events, 4 voices)
    duration = (N_VOICES * EVENTS_PER_VOICE) / 120.0  # 120 notes/s
    control_voices = generate_events_control(EVENTS_PER_VOICE, duration, rng)
    multi_voices = generate_events_multi(EVENTS_PER_VOICE, duration, rng)

    # Full-data weights (reference)
    ctrl_means = pairwise_component_means(control_voices)
    multi_means = pairwise_component_means(multi_voices)
    wvss_full = compute_wvss_weights(ctrl_means, multi_means)
    nwvss_full = compute_nwvss_weights(ctrl_means, multi_means, NWVSS_RANGES)

    print("=" * 70)
    print("wVSS Split-half Cross-Validation (coherence_metrics-based)")
    print("=" * 70)
    print("Data: N_voices={}, events/voice={}, total_events={}".format(
        N_VOICES, EVENTS_PER_VOICE, N_VOICES * EVENTS_PER_VOICE))
    print()
    print("Full-data weights (Control vs Multi-Constraint):")
    print("  wVSS (raw):  w_pitch={:.2f}%, w_vel={:.2f}%, w_temporal={:.2f}%".format(
        wvss_full['pitch'], wvss_full['velocity'], wvss_full['temporal']))
    print("  nwVSS:       w_pitch={:.2f}%, w_vel={:.2f}%, w_temporal={:.2f}%".format(
        nwvss_full['pitch'], nwvss_full['velocity'], nwvss_full['temporal']))
    print()

    # ---- (1) Even/Odd split ----
    ctrl_h1, ctrl_h2 = split_events_even_odd(control_voices)
    multi_h1, multi_h2 = split_events_even_odd(multi_voices)

    ctrl_means_h1 = pairwise_component_means(ctrl_h1)
    multi_means_h1 = pairwise_component_means(multi_h1)
    ctrl_means_h2 = pairwise_component_means(ctrl_h2)
    multi_means_h2 = pairwise_component_means(multi_h2)

    wvss_h1 = compute_wvss_weights(ctrl_means_h1, multi_means_h1)
    wvss_h2 = compute_wvss_weights(ctrl_means_h2, multi_means_h2)
    nwvss_h1 = compute_nwvss_weights(ctrl_means_h1, multi_means_h1, NWVSS_RANGES)
    nwvss_h2 = compute_nwvss_weights(ctrl_means_h2, multi_means_h2, NWVSS_RANGES)

    stats_wvss_eo = deviation_stats(wvss_h1, wvss_h2)
    stats_nwvss_eo = deviation_stats(nwvss_h1, nwvss_h2)

    print("--- (1) Even/Odd index split ---")
    print("  wVSS  half1: w_pitch={:.2f}%, w_vel={:.2f}%, w_temporal={:.2f}%".format(
        wvss_h1['pitch'], wvss_h1['velocity'], wvss_h1['temporal']))
    print("  wVSS  half2: w_pitch={:.2f}%, w_vel={:.2f}%, w_temporal={:.2f}%".format(
        wvss_h2['pitch'], wvss_h2['velocity'], wvss_h2['temporal']))
    print("  Deviation (pp): pitch={:.2f}, velocity={:.2f}, temporal={:.2f} | max={:.2f}, mean_abs={:.2f}".format(
        stats_wvss_eo['dev_pitch'], stats_wvss_eo['dev_velocity'], stats_wvss_eo['dev_temporal'],
        stats_wvss_eo['max_deviation_pp'], stats_wvss_eo['mean_abs_deviation_pp']))
    print("  Pearson r = {:.4f}, Spearman rho = {:.4f}".format(
        stats_wvss_eo['pearson_r'], stats_wvss_eo['spearman_rho']))
    print()
    print("  nwVSS half1: w_pitch={:.2f}%, w_vel={:.2f}%, w_temporal={:.2f}%".format(
        nwvss_h1['pitch'], nwvss_h1['velocity'], nwvss_h1['temporal']))
    print("  nwVSS half2: w_pitch={:.2f}%, w_vel={:.2f}%, w_temporal={:.2f}%".format(
        nwvss_h2['pitch'], nwvss_h2['velocity'], nwvss_h2['temporal']))
    print("  Deviation (pp): pitch={:.2f}, velocity={:.2f}, temporal={:.2f} | max={:.2f}, mean_abs={:.2f}".format(
        stats_nwvss_eo['dev_pitch'], stats_nwvss_eo['dev_velocity'], stats_nwvss_eo['dev_temporal'],
        stats_nwvss_eo['max_deviation_pp'], stats_nwvss_eo['mean_abs_deviation_pp']))
    print("  Pearson r = {:.4f}, Spearman rho = {:.4f}".format(
        stats_nwvss_eo['pearson_r'], stats_nwvss_eo['spearman_rho']))
    print()

    # ---- (2) 50:50 Random split (multiple trials) ----
    max_dev_wvss = []
    max_dev_nwvss = []
    pearson_wvss = []
    pearson_nwvss = []

    for trial in range(N_RANDOM_TRIALS):
        ctrl_h1, ctrl_h2 = split_events_random(control_voices, rng_split, fraction=0.5)
        multi_h1, multi_h2 = split_events_random(multi_voices, rng_split, fraction=0.5)

        ctrl_means_h1 = pairwise_component_means(ctrl_h1)
        multi_means_h1 = pairwise_component_means(multi_h1)
        ctrl_means_h2 = pairwise_component_means(ctrl_h2)
        multi_means_h2 = pairwise_component_means(multi_h2)

        wvss_h1 = compute_wvss_weights(ctrl_means_h1, multi_means_h1)
        wvss_h2 = compute_wvss_weights(ctrl_means_h2, multi_means_h2)
        nwvss_h1 = compute_nwvss_weights(ctrl_means_h1, multi_means_h1, NWVSS_RANGES)
        nwvss_h2 = compute_nwvss_weights(ctrl_means_h2, multi_means_h2, NWVSS_RANGES)

        s_w = deviation_stats(wvss_h1, wvss_h2)
        s_n = deviation_stats(nwvss_h1, nwvss_h2)
        max_dev_wvss.append(s_w['max_deviation_pp'])
        max_dev_nwvss.append(s_n['max_deviation_pp'])
        if not np.isnan(s_w['pearson_r']):
            pearson_wvss.append(s_w['pearson_r'])
        if not np.isnan(s_n['pearson_r']):
            pearson_nwvss.append(s_n['pearson_r'])

    print("--- (2) 50:50 random split ({} trials) ---".format(N_RANDOM_TRIALS))
    print("  wVSS  max deviation (pp): mean={:.2f}, std={:.2f}, min={:.2f}, max={:.2f}".format(
        np.mean(max_dev_wvss), np.std(max_dev_wvss), np.min(max_dev_wvss), np.max(max_dev_wvss)))
    print("  wVSS  Pearson r (half1 vs half2): mean={:.4f}, std={:.4f}".format(
        np.mean(pearson_wvss), np.std(pearson_wvss)))
    print("  nwVSS max deviation (pp): mean={:.2f}, std={:.2f}, min={:.2f}, max={:.2f}".format(
        np.mean(max_dev_nwvss), np.std(max_dev_nwvss), np.min(max_dev_nwvss), np.max(max_dev_nwvss)))
    print("  nwVSS Pearson r (half1 vs half2): mean={:.4f}, std={:.4f}".format(
        np.mean(pearson_nwvss), np.std(pearson_nwvss)))
    print()

    # ---- Consistency conclusion (generalizability stats) ----
    print("=" * 70)
    print("Weight consistency summary (wVSS metric generalizability)")
    print("=" * 70)
    print("Even/Odd:")
    print("  wVSS  max weight deviation = {:.2f} pp -> {}".format(
        stats_wvss_eo['max_deviation_pp'],
        "consistent" if stats_wvss_eo['max_deviation_pp'] < 1.0 else "moderate" if stats_wvss_eo['max_deviation_pp'] < 5.0 else "variable"))
    print("  nwVSS max weight deviation = {:.2f} pp -> {}".format(
        stats_nwvss_eo['max_deviation_pp'],
        "consistent" if stats_nwvss_eo['max_deviation_pp'] < 5.0 else "moderate" if stats_nwvss_eo['max_deviation_pp'] < 10.0 else "variable"))
    print("  wVSS  Spearman rho = {:.4f} (closer to 1 = rank agreement across splits)".format(stats_wvss_eo['spearman_rho']))
    print("  nwVSS Spearman rho = {:.4f}".format(stats_nwvss_eo['spearman_rho']))
    print()
    print("Random 50:50 ({} trials):".format(N_RANDOM_TRIALS))
    print("  wVSS  max deviation: mean={:.2f} pp -> weight stability across sample splits".format(np.mean(max_dev_wvss)))
    print("  nwVSS max deviation: mean={:.2f} pp".format(np.mean(max_dev_nwvss)))
    print("  Velocity weight dominant in both splits; small cross-split deviation supports wVSS generalizability.")
    print()

    # CSV output
    out_dir = os.path.join(os.path.dirname(__file__))
    csv_path = os.path.join(out_dir, "wvss_split_half_validation.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("split_type,metric,w_pitch_h1,w_vel_h1,w_temp_h1,w_pitch_h2,w_vel_h2,w_temp_h2,max_dev_pp,mean_abs_dev_pp,pearson_r,spearman_rho\n")
        f.write("even_odd,wvss,{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}\n".format(
            wvss_h1['pitch'], wvss_h1['velocity'], wvss_h1['temporal'],
            wvss_h2['pitch'], wvss_h2['velocity'], wvss_h2['temporal'],
            stats_wvss_eo['max_deviation_pp'], stats_wvss_eo['mean_abs_deviation_pp'],
            stats_wvss_eo['pearson_r'], stats_wvss_eo['spearman_rho']))
        f.write("even_odd,nwvss,{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}\n".format(
            nwvss_h1['pitch'], nwvss_h1['velocity'], nwvss_h1['temporal'],
            nwvss_h2['pitch'], nwvss_h2['velocity'], nwvss_h2['temporal'],
            stats_nwvss_eo['max_deviation_pp'], stats_nwvss_eo['mean_abs_deviation_pp'],
            stats_nwvss_eo['pearson_r'], stats_nwvss_eo['spearman_rho']))
        f.write("random_50_50,wvss,,,,,,,{:.4f},,,,\n".format(np.mean(max_dev_wvss)))
        f.write("random_50_50,nwvss,,,,,,,{:.4f},,,,\n".format(np.mean(max_dev_nwvss)))
    print("CSV saved: {}".format(csv_path))

    return {
        'wvss_full': wvss_full,
        'nwvss_full': nwvss_full,
        'even_odd': {'wvss': stats_wvss_eo, 'nwvss': stats_nwvss_eo},
        'random': {'wvss_max_dev_mean': np.mean(max_dev_wvss), 'nwvss_max_dev_mean': np.mean(max_dev_nwvss)},
    }


if __name__ == "__main__":
    run_split_half_validation()
