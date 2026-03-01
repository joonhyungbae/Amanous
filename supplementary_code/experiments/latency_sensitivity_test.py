#!/usr/bin/env python3
"""
Layer 4 Latency Compensation validation: Latency Model Mismatch Sensitivity Test.

Simulates real piano latency deviating from assumed model L(v) by ±10%, ±20%.

- Jitter (ms) without correction
- Jitter (ms) with correction applied even when model is slightly wrong

Compares these to produce a Sensitivity Curve showing how robust the HAL is to parameter error.

Reference: paper.tex Section 4 (Layer 4), Eq. (5) L(v) power-law.
"""

import csv
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# L(v) model: paper Eq. (5) Power-law, c=0.5, 10--30 ms
# -----------------------------------------------------------------------------
VELOCITY_MAX = 1023.0
LATENCY_MIN = 10.0   # ms at max velocity (loudest)
LATENCY_MAX = 30.0   # ms at min velocity (softest)
C_EXPONENT = 0.5     # calibrated (paper)


def L_model_ms(velocity: float, c: float = C_EXPONENT) -> float:
    """Assumed latency model L(v) = L_max - (L_max - L_min) * (v/v_max)^c [ms]."""
    v_norm = np.clip(velocity / VELOCITY_MAX, 0.0, 1.0)
    return LATENCY_MAX - (LATENCY_MAX - LATENCY_MIN) * (v_norm ** c)


def L_actual_ms(velocity: float, mismatch_pct: float, c: float = C_EXPONENT) -> float:
    """
    Real piano latency: model deviates by ±mismatch_pct.
    L_actual(v) = (1 + mismatch_pct) * L_model(v).
    mismatch_pct: -0.2 = -20%, +0.1 = +10%, etc.
    """
    return (1.0 + mismatch_pct) * L_model_ms(velocity, c)


def jitter_uncorrected_std_ms(velocities: np.ndarray, mismatch_pct: float) -> float:
    """Uncorrected: timing error = actual latency, so Jitter = std(L_actual)."""
    L_act = np.array([L_actual_ms(v, mismatch_pct) for v in velocities])
    return float(np.std(L_act))


def jitter_corrected_std_ms(velocities: np.ndarray, mismatch_pct: float) -> float:
    """
    Corrected: send trigger at (intended - L_model), so
    actual arrival = intended - L_model + L_actual = intended + (L_actual - L_model).
    Residual = L_actual - L_model = mismatch_pct * L_model -> Jitter = std(residual).
    """
    L_mod = np.array([L_model_ms(v) for v in velocities])
    L_act = np.array([L_actual_ms(v, mismatch_pct) for v in velocities])
    residual_ms = L_act - L_mod
    return float(np.std(residual_ms))


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

    # Mismatch: ±10%, ±20% and in between (scale error)
    mismatch_pcts = [-0.20, -0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15, 0.20]
    rows = []
    for pct in mismatch_pcts:
        j_no = jitter_uncorrected_std_ms(velocities, pct)
        j_hal = jitter_corrected_std_ms(velocities, pct)
        rows.append((pct, j_no, j_hal))

    # Console output
    print()
    print("=" * 70)
    print("Layer 4 Latency Sensitivity: L(v) Model Mismatch ±10%, ±20%")
    print("  (actual L_actual = (1 + mismatch) × L_model)")
    print("=" * 70)
    print(f"  {'Mismatch':>10}  {'Jitter (no correction)':>22}  {'Jitter (HAL applied)':>22}  HAL better")
    print("-" * 70)
    for pct, j_no, j_hal in rows:
        better = "✓" if j_hal < j_no else "—"
        print(f"  {pct:>+9.0%}  {j_no:>20.4f} ms  {j_hal:>20.4f} ms  {better}")
    print("=" * 70)
    print()

    # Sensitivity curve plot
    pcts = [r[0] for r in rows]
    jitter_no = [r[1] for r in rows]
    jitter_hal = [r[2] for r in rows]
    pct_labels = [f"{p*100:+.0f}%" for p in pcts]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(np.array(pcts) * 100, jitter_no, "o-", color="C1", label="Uncorrected (no HAL)",
            linewidth=2, markersize=8)
    ax.plot(np.array(pcts) * 100, jitter_hal, "s-", color="C0", label="HAL applied (L(v) compensation)",
            linewidth=2, markersize=8)
    ax.axvline(0, color="gray", linestyle="--", alpha=0.7, label="Perfect match (0%)")
    ax.set_xlabel("Latency model mismatch (actual $L_{\\mathrm{actual}} = (1 + \\delta) \\cdot L(v)$)", fontsize=11)
    ax.set_ylabel("Jitter (timing error std, ms)", fontsize=11)
    ax.set_title("Layer 4 Sensitivity: HAL Robustness to Parameter Error\n"
                 "($\\pm$10%, $\\pm$20% scale mismatch)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)
    ax.set_xticks([p * 100 for p in pcts])
    ax.set_xticklabels(pct_labels)
    plt.tight_layout()
    path_fig = os.path.join(script_dir, "latency_sensitivity_curve.png")
    plt.savefig(path_fig, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Output] {path_fig}")

    # Save CSV
    path_csv = os.path.join(script_dir, "latency_sensitivity_mismatch.csv")
    with open(path_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["mismatch_pct", "jitter_std_ms_no_correction", "jitter_std_ms_hal_applied", "n_notes"])
        for pct, j_no, j_hal in rows:
            w.writerow([round(pct, 4), round(j_no, 6), round(j_hal, 6), len(velocities)])
    print(f"[Output] {path_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
