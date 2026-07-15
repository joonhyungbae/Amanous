#!/usr/bin/env python3
"""
Run all three ablation experiments and print the LaTeX table for Section 4.1.
Order: (a) No L-system, (b) No tempo canon, (c) No hardware compensation.
"""

import os
import sys
import json
import subprocess

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(CODE_DIR)


def run_script(name: str, extra_args: list = None) -> dict:
    """Run a script and load its JSON output."""
    out_name = name.replace(".py", "")  # e.g. ablation_a_no_lsystem
    json_name = out_name + ".json"
    cmd = [sys.executable, name] + (extra_args or [])
    subprocess.check_call(cmd, cwd=CODE_DIR)
    path = os.path.join(CODE_DIR, json_name)
    with open(path) as f:
        return json.load(f)


def format_val(x, fmt=".4f"):
    if x is None or (isinstance(x, float) and (x != x or x == float("inf"))):
        return "---"
    if isinstance(x, (int, float)):
        return ("{:" + fmt + "}").format(x)
    return str(x)


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Run all ablations and print LaTeX table")
    ap.add_argument("--n-runs", type=int, default=100, help="Number of runs for ablation (a)")
    ap.add_argument("--seed", type=int, default=42)
    args_main = ap.parse_args()
    common = ["--seed", str(args_main.seed)]

    print("Running ablation (a): No L-system...")
    da = run_script("ablation_a_no_lsystem.py", ["--n-runs", str(args_main.n_runs)] + common)
    print("\nRunning ablation (b): No tempo canon...")
    db = run_script("ablation_b_no_canon.py", common)
    print("\nRunning ablation (c): No hardware compensation...")
    dc = run_script("ablation_c_no_hwcomp.py", common)

    full_a = da["full_pipeline"]
    abl_a = da["ablated"]
    comp_a = da["comparison"]
    full_b = db["full_pipeline"]
    abl_b = db["ablated"]
    comp_b = db["comparison"]
    full_c = dc["full_pipeline"]
    abl_c = dc["ablated"]
    comp_c = dc["comparison"]

    # Table row strings
    seq_mean = abl_a.get("sequential_self_sim_mc_mean")
    seq_std = abl_a.get("sequential_self_sim_mc_std")
    seq_z = comp_a.get("z_sequential_self_sim")
    seq_p = comp_a.get("p_sequential_self_sim")
    row_a1_full = format_val(full_a["sequential_self_sim_mc"])
    row_a1_abl = f"{seq_mean:.3f}$\\pm${seq_std:.3f}" if seq_mean is not None and seq_std is not None else "---"
    row_a1_delta = f"z={seq_z:.2f}, p={seq_p:.2e}" if seq_z is not None and seq_p is not None else "---"

    mc_mean = abl_a.get("same_symbol_mc_mean")
    mc_std = abl_a.get("same_symbol_mc_std")
    mc_z = comp_a.get("z_same_symbol_mc")
    mc_p = comp_a.get("p_same_symbol_mc")
    row_a2_full = format_val(full_a["same_symbol_mc"])
    row_a2_abl = f"{mc_mean:.3f}$\\pm${mc_std:.3f}" if mc_mean is not None and mc_std is not None else "---"
    row_a2_delta = f"z={mc_z:.2f}, p={mc_p:.2e}" if mc_z is not None and mc_p is not None else "---"

    sl = comp_b.get("section_level") or {}
    vss_t_full = full_b["vss_temporal"]
    vss_t_abl = abl_b["vss_temporal"]
    pct_t = comp_b.get("pct_change_vss_temporal", 0)
    row_b1_full = format_val(vss_t_full, ".6f")
    row_b1_abl = format_val(vss_t_abl, ".6f")
    u_vss, p_vss, r_vss = sl.get("vss_temporal_U"), sl.get("vss_temporal_p"), sl.get("vss_temporal_r")
    row_b1_delta = f"{pct_t:+.1f}\\%"
    if u_vss is not None and p_vss is not None and r_vss is not None:
        row_b1_delta += f", U={u_vss:.0f}, p={p_vss:.2e}, r={r_vss:.2f}"

    rc_full_b = full_b["same_symbol_rc"]
    rc_abl_b = abl_b["same_symbol_rc"]
    pct_rc = comp_b.get("pct_change_rc", 0)
    row_b2_full = format_val(rc_full_b)
    row_b2_abl = format_val(rc_abl_b)
    u_rc, p_rc, r_rc = sl.get("rc_U"), sl.get("rc_p"), sl.get("rc_r")
    row_b2_delta = f"{pct_rc:+.1f}\\%"
    if u_rc is not None and p_rc is not None and r_rc is not None:
        row_b2_delta += f", U={u_rc:.0f}, p={p_rc:.2e}, r={r_rc:.2f}"

    # (c) ablated now has linear and powerlaw_c05
    abl_c_lin = abl_c.get("linear") or abl_c
    align_full = full_c["onset_alignment_sd_ms"]
    align_abl = abl_c_lin.get("onset_alignment_sd_ms")
    pct_align = comp_c.get("pct_change_align_sd", 0)
    row_c1_full = format_val(align_full)
    row_c1_abl = format_val(align_abl)
    row_c1_delta = f"{pct_align:+.1f}\\%"

    r_full = full_c["velocity_timing_r"]
    r_abl = abl_c_lin.get("velocity_timing_r")
    p_abl = abl_c_lin.get("velocity_timing_p")
    row_c2_full = format_val(r_full)
    row_c2_abl = format_val(r_abl)
    row_c2_delta = f"r={r_abl:.2f}, p={p_abl:.2e}" if p_abl is not None else format_val(r_abl)

    latex = r"""
\begin{table}[htbp]
\tbl{Pipeline component ablation: effect of removing each layer on key metrics.}
{\begin{tabular}{@{}llccl@{}} \toprule
\textbf{Ablation} & \textbf{Metric} & \textbf{Full Pipeline} & \textbf{Ablated} & \textbf{$\Delta$ / Test} \\ \midrule
(a) No L-system & Sequential self-sim.\ (MC) & """ + row_a1_full + r""" & """ + row_a1_abl + r""" & """ + row_a1_delta + r""" \\
(a) No L-system & Same-symbol MC & """ + row_a2_full + r""" & """ + row_a2_abl + r""" & """ + row_a2_delta + r""" \\
(b) No tempo canon & VSS (temporal component) & """ + row_b1_full + r""" & """ + row_b1_abl + r""" & """ + row_b1_delta + r""" \\
(b) No tempo canon & RC (same-symbol) & """ + row_b2_full + r""" & """ + row_b2_abl + r""" & """ + row_b2_delta + r""" \\
(c) No hw.\ comp. & Onset alignment SD (ms) & """ + row_c1_full + r""" & """ + row_c1_abl + r""" & """ + row_c1_delta + r""" \\
(c) No hw.\ comp. & Vel.--timing correlation & """ + row_c2_full + r""" & """ + row_c2_abl + r""" & """ + row_c2_delta + r""" \\ \bottomrule
\end{tabular}}
\label{tab:ablation}
\end{table}
"""
    print("\n" + "=" * 70)
    print("LaTeX table (Section 4.1)")
    print("=" * 70)
    print(latex)
    return 0


if __name__ == "__main__":
    sys.exit(main())
