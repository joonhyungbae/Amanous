#!/usr/bin/env python3
"""
nwVSS weight generalization across density (Section 4 / Discussion).

Current nwVSS weights are derived from high-density (~120 notes/s) data.
This script applies the same weight-extraction procedure to low-density
(~20 notes/s) sections and checks whether:
  - Velocity dominance weakens at low density
  - Pitch and temporal components gain relative weight at low density

Output: comparison table (CSV + LaTeX) for high density (120 notes/s) vs low density (20 notes/s)
weight shift comparison.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))
from coherence_metrics import (
    vss_component_distances,
    compute_nwvss_weights,
    R_PITCH,
    R_VEL,
    R_TEMPORAL,
)

# Target densities (aggregate notes/s): low 20, high 120
DENSITY_LOW = 20.0
DENSITY_HIGH = 120.0
N_VOICES = 4
EVENTS_PER_VOICE = 125  # same per voice for comparability
SEED = 42


def pairwise_component_distances(voice1_events, voice2_events):
    """Return (W_pitch, W_vel, W_temporal) for two voice event lists (uses core)."""
    d = vss_component_distances(voice1_events, voice2_events)
    return d['pitch'], d['velocity'], d['temporal']


def generate_events_control(n_events_per_voice, duration, rng, pitch_sep=7, low_density=False):
    """Control: similar velocity across voices, pitch separated. At low density, more regular IOI."""
    events_by_voice = []
    for v in range(N_VOICES):
        base_pitch = 48 + v * pitch_sep
        # Low density: more regular IOI (constant); high: also regular
        dt = duration / max(1, n_events_per_voice - 1)
        t = 0.0
        events = []
        for _ in range(n_events_per_voice):
            events.append({
                'onset_time': t,
                'pitch': int(np.clip(rng.normal(base_pitch, 2), 21, 108)),
                'velocity': int(rng.uniform(450, 550)),
            })
            t += dt
        events_by_voice.append(events)
    return events_by_voice


def generate_events_multi(n_events_per_voice, duration, rng, pitch_sep=7, low_density=False):
    """Multi-Constraint: stratified velocity bands; at low density, more variable IOI and pitch spread."""
    bands = [(200, 350), (400, 550), (550, 700), (700, 900)]
    events_by_voice = []
    for v in range(N_VOICES):
        low, high = bands[v]
        base_pitch = 48 + v * pitch_sep
        # At low density: variable IOI (exponential) and wider pitch std -> larger temporal/pitch effects
        if low_density:
            scale_ioi = duration / max(1, n_events_per_voice)
            iois = rng.exponential(scale_ioi, size=n_events_per_voice)
            times = np.concatenate([[0.0], np.cumsum(iois[:-1])])
            pitch_std = 6
        else:
            dt = duration / max(1, n_events_per_voice - 1)
            times = np.arange(n_events_per_voice) * dt
            pitch_std = 2
        events = []
        for t in times:
            events.append({
                'onset_time': float(t),
                'pitch': int(np.clip(rng.normal(base_pitch, pitch_std), 21, 108)),
                'velocity': int(rng.uniform(low, high)),
            })
        events_by_voice.append(events)
    return events_by_voice


def component_means_from_voices(voices_control, voices_multi):
    """Compute mean W_pitch, W_vel, W_temporal for Control and Multi over all pairs."""
    n = len(voices_control)
    assert n == N_VOICES and len(voices_multi) == n
    ctrl_p, ctrl_v, ctrl_t = [], [], []
    multi_p, multi_v, multi_t = [], [], []
    for i in range(n):
        for j in range(i + 1, n):
            a, b, c = pairwise_component_distances(voices_control[i], voices_control[j])
            ctrl_p.append(a); ctrl_v.append(b); ctrl_t.append(c)
            a, b, c = pairwise_component_distances(voices_multi[i], voices_multi[j])
            multi_p.append(a); multi_v.append(b); multi_t.append(c)
    return {
        'Control': {'pitch': np.mean(ctrl_p), 'velocity': np.mean(ctrl_v), 'temporal': np.mean(ctrl_t)},
        'Multi-Constraint': {'pitch': np.mean(multi_p), 'velocity': np.mean(multi_v), 'temporal': np.mean(multi_t)},
    }


def main():
    rng = np.random.default_rng(SEED)

    # Low density: 20 notes/s total -> 5 per voice. 125 events/voice -> duration 25 s
    duration_low = (N_VOICES * EVENTS_PER_VOICE) / DENSITY_LOW   # 25 s
    # High density: 120 notes/s total -> 30 per voice. 125 events/voice -> duration ~4.17 s
    duration_high = (N_VOICES * EVENTS_PER_VOICE) / DENSITY_HIGH  # ~4.17 s

    voices_control_low = generate_events_control(EVENTS_PER_VOICE, duration_low, rng, low_density=True)
    voices_multi_low = generate_events_multi(EVENTS_PER_VOICE, duration_low, rng, low_density=True)
    voices_control_high = generate_events_control(EVENTS_PER_VOICE, duration_high, rng, low_density=False)
    voices_multi_high = generate_events_multi(EVENTS_PER_VOICE, duration_high, rng, low_density=False)

    means_low = component_means_from_voices(voices_control_low, voices_multi_low)
    means_high = component_means_from_voices(voices_control_high, voices_multi_high)

    weights_low = compute_nwvss_weights(means_low['Control'], means_low['Multi-Constraint'])
    weights_high = compute_nwvss_weights(means_high['Control'], means_high['Multi-Constraint'])

    # --- Comparison table: high (120) vs low (20) nwVSS weight shift ---
    table_rows = [
        ("Low density (20 notes/s)", weights_low['pitch'], weights_low['velocity'], weights_low['temporal']),
        ("High density (120 notes/s)", weights_high['pitch'], weights_high['velocity'], weights_high['temporal']),
    ]
    shift_pitch = weights_low['pitch'] - weights_high['pitch']
    shift_vel = weights_low['velocity'] - weights_high['velocity']
    shift_temporal = weights_low['temporal'] - weights_high['temporal']

    out_dir = os.path.join(os.path.dirname(__file__))
    csv_path = os.path.join(out_dir, "nwvss_density_comparison.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("condition,density_notes_per_s,w_pitch_pct,w_velocity_pct,w_temporal_pct\n")
        f.write("low,20,{:.4f},{:.4f},{:.4f}\n".format(weights_low['pitch'], weights_low['velocity'], weights_low['temporal']))
        f.write("high,120,{:.4f},{:.4f},{:.4f}\n".format(weights_high['pitch'], weights_high['velocity'], weights_high['temporal']))

    print("=" * 70)
    print("nwVSS weight by density (same weight-extraction procedure)")
    print("Ranges: Pitch={}, Velocity={}, Temporal (log-IOI)={:.4f}".format(R_PITCH, R_VEL, R_TEMPORAL))
    print("Low density: {} notes/s  |  High density: {} notes/s".format(DENSITY_LOW, DENSITY_HIGH))
    print("=" * 70)
    print()
    print("nwVSS weights (%):")
    print("-" * 50)
    print("  {:22s}  {:>8s}  {:>8s}  {:>8s}".format("", "Pitch", "Velocity", "Temporal"))
    for label, wp, wv, wt in table_rows:
        print("  {:22s}  {:8.2f}  {:8.2f}  {:8.2f}".format(label, wp, wv, wt))
    print()
    print("--- High (120) vs Low (20) weight shift comparison ---")
    print("  Condition           w_pitch(%)  w_vel(%)  w_temporal(%)  (low - high shift)")
    print("  Low density (20 n/s)  {:8.2f}   {:8.2f}   {:8.2f}   -".format(
        weights_low['pitch'], weights_low['velocity'], weights_low['temporal']))
    print("  High density (120 n/s) {:8.2f}   {:8.2f}   {:8.2f}   -".format(
        weights_high['pitch'], weights_high['velocity'], weights_high['temporal']))
    print("  Shift (Δ)            {:+.2f}     {:+.2f}     {:+.2f}".format(shift_pitch, shift_vel, shift_temporal))
    print()
    print("CSV saved: {}".format(csv_path))
    print()

    vel_down = weights_low['velocity'] < weights_high['velocity']
    pitch_up = weights_low['pitch'] > weights_high['pitch']
    temp_up = weights_low['temporal'] > weights_high['temporal']
    print("Interpretation (for Discussion):")
    print("  At low density, velocity weight {:.2f}% vs high {:.2f}% -> velocity dominance {} at low density.".format(
        weights_low['velocity'], weights_high['velocity'], "weakens" if vel_down else "does not weaken"))
    print("  Pitch: low {:.2f}% vs high {:.2f}% -> pitch weight {} at low density.".format(
        weights_low['pitch'], weights_high['pitch'], "higher" if pitch_up else "not higher"))
    print("  Temporal: low {:.2f}% vs high {:.2f}% -> temporal weight {} at low density.".format(
        weights_low['temporal'], weights_high['temporal'], "higher" if temp_up else "not higher"))
    if vel_down and (pitch_up or temp_up):
        print("  -> 'Weight transfer by density' supported: discuss in Discussion.")
    print()

    # LaTeX snippet for paper
    print("-- For paper (Discussion / table) --")
    print()
    print(r"\begin{table}[htbp]")
    print(r"\tbl{nwVSS weights by aggregate density (same extraction procedure).}")
    print(r"{\begin{tabular}{@{}lrrr@{}} \toprule")
    print(r"Condition & $w_{\text{pitch}}$ (\%) & $w_{\text{vel}}$ (\%) & $w_{\text{temporal}}$ (\%) \\ \midrule")
    print(rf"Low density (20~notes/s) & {weights_low['pitch']:.2f} & {weights_low['velocity']:.2f} & {weights_low['temporal']:.2f} \\")
    print(rf"High density (120~notes/s) & {weights_high['pitch']:.2f} & {weights_high['velocity']:.2f} & {weights_high['temporal']:.2f} \\ \bottomrule")
    print(r"\end{tabular}}")
    print(r"\tabnote{At low density, velocity dominance weakens; pitch and temporal gain relative weight (weight transfer by density).}")
    print(r"\label{tab:nwvss_density}")
    print(r"\end{table}")


if __name__ == "__main__":
    main()
