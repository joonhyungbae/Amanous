#!/usr/bin/env python3
"""
Layer 4 HAL validation: Latency Mismatch simulation.

Defines a 'virtual real piano' model: Power-law (c=0.5) plus ±10% random noise and parameter drift.

Compares onset timing std (Jitter) under three conditions:
  (A) No correction (Raw MIDI)
  (B) Amanous HAL correction (c=0.5 Power-law)
  (C) Ideal correction (perfect match to virtual real model)

Tests whether (B) is statistically significantly more accurate than (A) even when the model
is imperfect, demonstrating that following a general physical law (Power-law) yields
sufficient benefit without perfect measured data.

Reference: paper.tex Section 4 (Layer 4), Eq. (5) L(v) power-law.
"""

import csv
import os
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Power-law latency model (paper Eq. 5): L = L_max - (L_max - L_min) * (v/v_max)^c
# -----------------------------------------------------------------------------
VELOCITY_MAX = 1023.0
LATENCY_MIN = 10.0   # ms at max velocity
LATENCY_MAX = 30.0   # ms at min velocity
C_EXPONENT = 0.5     # Amanous HAL assumed value (paper)


def L_powerlaw_ms(velocity: float, c: float, v_max: float = VELOCITY_MAX) -> float:
    """L(v) = L_max - (L_max - L_min) * (v/v_max)^c [ms]."""
    v_norm = np.clip(velocity / v_max, 0.0, 1.0)
    return LATENCY_MAX - (LATENCY_MAX - LATENCY_MIN) * (v_norm ** c)


def virtual_real_piano_latencies(
    velocities: np.ndarray,
    noise_pct: float = 0.10,
    drift_scale: float = 0.03,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """
    Virtual real piano: Power-law(c=0.5) + ±noise_pct noise + parameter drift.

    - Base: L_base(v) = L(v; c=0.5).
    - Noise: per note L *= (1 + U(-noise_pct, +noise_pct)).
    - Drift: c random walk by note index. c_i in [0.5 - drift_scale, 0.5 + drift_scale].
      L_base computed with c_i so curve varies slightly with velocity.

    Returns
    -------
    L_actual : (n,) array, ms. Actual latency per note.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    n = len(velocities)

    # Parameter drift: c random walk around 0.5 (clip to ~±10%)
    c_walk = np.cumsum(rng.standard_normal(n) * 0.015)
    c_walk = np.clip(c_walk, -drift_scale, drift_scale)
    c_vals = C_EXPONENT + c_walk  # c in [0.47, 0.53]

    L_base = np.array([
        L_powerlaw_ms(velocities[i], c_vals[i]) for i in range(n)
    ])
    # ±noise_pct scale noise (multiplicative)
    noise = rng.uniform(-noise_pct, noise_pct, size=n)
    L_actual = L_base * (1.0 + noise)
    return L_actual.astype(float)


def jitter_std_ms(latency_errors: np.ndarray) -> float:
    """Std of onset timing error (Jitter, ms)."""
    return float(np.std(latency_errors))


def condition_a_raw(velocities: np.ndarray, L_actual: np.ndarray) -> float:
    """(A) No correction: timing error = actual latency. Jitter = std(L_actual)."""
    return jitter_std_ms(L_actual)


def condition_b_hal(velocities: np.ndarray, L_actual: np.ndarray) -> float:
    """(B) Amanous HAL: correction L_hal(v)=L(v;c=0.5). Residual = L_actual - L_hal."""
    L_hal = np.array([L_powerlaw_ms(v, C_EXPONENT) for v in velocities])
    residual = L_actual - L_hal
    return jitter_std_ms(residual)


def condition_c_ideal(velocities: np.ndarray, L_actual: np.ndarray) -> float:
    """(C) Ideal: correction perfectly matching virtual real model. Residual = 0 (theoretical)."""
    # Perfect correction -> residual = 0; numerically 0
    return 0.0


def load_velocities_from_csv(path: str, max_notes: int = 526) -> np.ndarray:
    """Load velocity column from CSV. If 0--127, scale to Disklavier (×8)."""
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            v = float(row.get("velocity", row.get("velocity_midi", 64)))
            if v <= 127:
                v = v * 8
            rows.append(min(max(v, 0), VELOCITY_MAX))
            if len(rows) >= max_notes:
                break
    return np.array(rows) if rows else np.array([])


def generate_synthetic_velocities(n: int = 526, seed: int = 42) -> np.ndarray:
    """Velocity samples for ~30s density (Disklavier 0--1023)."""
    rng = np.random.default_rng(seed)
    v = rng.integers(50, 1024, size=n)
    return np.clip(v, 0, VELOCITY_MAX).astype(float)


def run_simulation(
    velocities: np.ndarray,
    n_trials: int = 200,
    seed: int = 42,
    noise_pct: float = 0.10,
    drift_scale: float = 0.03,
):
    """
    Run n_trials: each trial resamples virtual piano and computes (A)(B)(C) Jitter.

    Returns
    -------
    jitter_a, jitter_b, jitter_c : each (n_trials,) array.
    """
    rng = np.random.default_rng(seed)
    jitter_a = np.zeros(n_trials)
    jitter_b = np.zeros(n_trials)
    jitter_c = np.zeros(n_trials)

    for t in range(n_trials):
        trial_rng = np.random.default_rng(seed + t)
        L_actual = virtual_real_piano_latencies(
            velocities, noise_pct=noise_pct, drift_scale=drift_scale, rng=trial_rng
        )
        jitter_a[t] = condition_a_raw(velocities, L_actual)
        jitter_b[t] = condition_b_hal(velocities, L_actual)
        jitter_c[t] = condition_c_ideal(velocities, L_actual)

    return jitter_a, jitter_b, jitter_c


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.normpath(os.path.join(script_dir, "..", ".."))
    csv_path = os.path.join(
        repo_root, "supplementary_code", "data", "csv", "note_by_note_error_analysis.csv"
    )

    if os.path.isfile(csv_path):
        velocities = load_velocities_from_csv(csv_path, max_notes=526)
        print(f"[Data] Loaded {len(velocities)} velocities from CSV")
    else:
        velocities = generate_synthetic_velocities(526)
        print(f"[Data] Generated {len(velocities)} synthetic velocities")

    if len(velocities) == 0:
        velocities = generate_synthetic_velocities(526)

    n_trials = 200
    noise_pct = 0.10   # ±10%
    drift_scale = 0.03 # c in [0.47, 0.53]

    print()
    print("Running Latency Mismatch Simulation (virtual real piano: Power-law c=0.5 + ±10% noise + drift)...")
    jitter_a, jitter_b, jitter_c = run_simulation(
        velocities, n_trials=n_trials, seed=42, noise_pct=noise_pct, drift_scale=drift_scale
    )

    # Summary statistics
    mean_a, std_a = np.mean(jitter_a), np.std(jitter_a)
    mean_b, std_b = np.mean(jitter_b), np.std(jitter_b)
    mean_c, std_c = np.mean(jitter_c), np.std(jitter_c)

    # (A) vs (B) test: B has smaller Jitter than A (paired, one-sided: A > B)
    # Paired t-test: H0: mean(A - B) <= 0, H1: mean(A - B) > 0
    t_stat, p_value_ttest = stats.ttest_rel(jitter_a, jitter_b, alternative="greater")
    # Wilcoxon signed-rank (nonparametric)
    w_stat, p_value_wilcoxon = stats.wilcoxon(jitter_a, jitter_b, alternative="greater")
    # Effect size: mean difference (ms)
    mean_diff_ab = float(np.mean(jitter_a - jitter_b))
    cohen_d = mean_diff_ab / (np.std(jitter_a - jitter_b) + 1e-12)

    print()
    print("=" * 72)
    print("Latency Mismatch Simulation: HAL validation")
    print("  Virtual real piano: Power-law(c=0.5) + ±10% noise + parameter drift")
    print("=" * 72)
    print(f"  (A) No correction (Raw MIDI) Jitter = {mean_a:.4f} ± {std_a:.4f} ms  (mean ± std over {n_trials} trials)")
    print(f"  (B) Amanous HAL correction   Jitter = {mean_b:.4f} ± {std_b:.4f} ms")
    print(f"  (C) Ideal correction        Jitter = {mean_c:.4f} ± {std_c:.4f} ms")
    print("-" * 72)
    print("  Statistical test (A vs B): Is HAL (B) Jitter smaller than uncorrected (A)?")
    print(f"    Paired t-test (one-sided, A > B):  t = {t_stat:.4f},  p = {p_value_ttest:.6f}")
    print(f"    Wilcoxon signed-rank (one-sided):  W = {w_stat},  p = {p_value_wilcoxon:.6f}")
    print(f"    Mean difference (A - B): {mean_diff_ab:.4f} ms  (positive -> B more accurate)")
    print(f"    Cohen's d (effect size): {cohen_d:.4f}")
    if p_value_ttest < 0.05:
        print("  -> Conclusion: HAL (B) is statistically significantly more accurate than uncorrected (A) (p < 0.05).")
    else:
        print("  -> Conclusion: No significant difference at alpha=0.05 (check trial count or effect size).")
    print("=" * 72)
    print()

    # Save CSV
    path_csv = os.path.join(script_dir, "latency_mismatch_simulation_results.csv")
    with open(path_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "condition", "jitter_mean_ms", "jitter_std_ms", "n_trials", "n_notes",
            "noise_pct", "drift_scale", "p_value_ttest_A_vs_B", "p_value_wilcoxon_A_vs_B",
            "mean_diff_A_minus_B_ms", "cohen_d"
        ])
        w.writerow(["A_raw", round(mean_a, 6), round(std_a, 6), n_trials, len(velocities), noise_pct, drift_scale, "", "", "", ""])
        w.writerow(["B_hal", round(mean_b, 6), round(std_b, 6), n_trials, len(velocities), noise_pct, drift_scale, "", "", "", ""])
        w.writerow(["C_ideal", round(mean_c, 6), round(std_c, 6), n_trials, len(velocities), noise_pct, drift_scale, "", "", "", ""])
        w.writerow(["summary", "", "", n_trials, len(velocities), noise_pct, drift_scale,
                    round(p_value_ttest, 6), round(p_value_wilcoxon, 6), round(mean_diff_ab, 6), round(cohen_d, 6)])
    print(f"[Output] {path_csv}")

    # Save per-trial A, B (for plots/reanalysis)
    path_trials = os.path.join(script_dir, "latency_mismatch_simulation_trials.csv")
    with open(path_trials, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["trial", "jitter_A_raw_ms", "jitter_B_hal_ms", "jitter_C_ideal_ms"])
        for t in range(n_trials):
            w.writerow([t, round(jitter_a[t], 6), round(jitter_b[t], 6), round(jitter_c[t], 6)])
    print(f"[Output] {path_trials}")

    # Plot: (A)(B)(C) Jitter distribution (boxplot + mean)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    data = [jitter_a, jitter_b, jitter_c]
    labels = ["(A) Raw\n(no correction)", "(B) Amanous HAL\n(c=0.5 Power-law)", "(C) Ideal\n(perfect match)"]
    colors = ["C1", "C0", "C2"]
    bp = ax.boxplot(data, labels=labels, patch_artist=True, showmeans=True)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_ylabel("Onset timing jitter (std, ms)", fontsize=11)
    ax.set_title("Latency Mismatch Simulation: Virtual Real Piano\n"
                 "Power-law(c=0.5) + ±10% noise + parameter drift")
    ax.grid(True, axis="y", alpha=0.3)
    y_data_max = max(np.max(jitter_a), np.max(jitter_b))
    ax.set_ylim(0, y_data_max * 1.35)
    # A vs B significance bracket (above boxes 1 and 2)
    y_brace = y_data_max * 1.12
    ax.plot([1, 2], [y_brace, y_brace], "k-", linewidth=1)
    ax.plot([1, 1], [y_brace - 0.15, y_brace], "k-", linewidth=1)
    ax.plot([2, 2], [y_brace - 0.15, y_brace], "k-", linewidth=1)
    p_str = f"p = {p_value_ttest:.4f}" if p_value_ttest >= 0.0001 else "p < 0.0001"
    ax.text(1.5, y_brace + 0.15, p_str, ha="center", fontsize=9)
    plt.tight_layout()
    path_fig = os.path.join(script_dir, "latency_mismatch_simulation.png")
    plt.savefig(path_fig, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Output] {path_fig}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
