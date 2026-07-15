#!/usr/bin/env python3
"""
Epsilon sensitivity test for Convergence Point (CP) Calculus (Section 4.4).

Varies epsilon in {10, 20, 50, 100} ms and computes the number of
convergence events (switch opportunities) in a 60 s window for:
  (1) Rational 3:4 canon (IOI 1.0 s and 0.75 s)
  (2) Irrational e:pi canon (tempo ratio e : pi)

Output: table suitable for Appendix (consistent trend: not tied to a single epsilon).
Reference: paper.tex Section 4.4, Definition (Convergence Point).
"""

import numpy as np

# -----------------------------------------------------------------------------
# Parameters
# -----------------------------------------------------------------------------
DURATION = 60.0  # seconds
EPSILON_MS = [10, 20, 50, 100]


def rational_34_convergence_events(epsilon_sec: float) -> int:
    """
    Two voices: T1(n) = n * 1.0, T2(m) = m * 0.75.
    CP when |T1 - T2| < epsilon. Count distinct convergence events in [0, DURATION].
    For 3:4, exact convergences at t = 3k (k=1,2,...,20) in 60 s.
    """
    events = []
    # Voice 1: n * 1.0  (n such that n*1.0 <= 60)
    # Voice 2: m * 0.75 (m such that m*0.75 <= 60)
    n_max = int(DURATION / 1.0) + 1
    m_max = int(DURATION / 0.75) + 1
    for n in range(n_max):
        t1 = n * 1.0
        if t1 > DURATION:
            break
        for m in range(m_max):
            t2 = m * 0.75
            if t2 > DURATION:
                break
            if abs(t1 - t2) < epsilon_sec:
                # event time = midpoint (or min)
                t_event = (t1 + t2) / 2.0
                if 0 <= t_event <= DURATION:
                    events.append(t_event)
    if not events:
        return 0
    # Merge events within epsilon (same "switch" opportunity)
    events = np.sort(np.unique(np.round(events, 6)))
    merged = [events[0]]
    for t in events[1:]:
        if t - merged[-1] > epsilon_sec:
            merged.append(t)
    return len(merged)


def irrational_epi_convergence_events(epsilon_sec: float) -> int:
    """
    Two voices: T1(n) = n / e, T2(m) = m / pi (so ratio of periods = e : pi).
    CP when |T1 - T2| < epsilon. Count distinct convergence events in [0, DURATION].
    """
    e_val = np.e
    pi_val = np.pi
    # Voice 1: n / e  ->  n_max such that n/e <= 60  -> n <= 60*e
    # Voice 2: m / pi ->  m_max such that m/pi <= 60 -> m <= 60*pi
    n_max = int(DURATION * e_val) + 2
    m_max = int(DURATION * pi_val) + 2
    events = []
    for n in range(n_max):
        t1 = n / e_val
        if t1 > DURATION:
            break
        for m in range(m_max):
            t2 = m / pi_val
            if t2 > DURATION:
                break
            if abs(t1 - t2) < epsilon_sec:
                t_event = (t1 + t2) / 2.0
                if 0 <= t_event <= DURATION:
                    events.append(t_event)
    if not events:
        return 0
    events = np.sort(np.unique(np.round(events, 6)))
    merged = [events[0]]
    for t in events[1:]:
        if t - merged[-1] > epsilon_sec:
            merged.append(t)
    return len(merged)


def main():
    print("=" * 70)
    print("Epsilon sensitivity: CP Calculus (Section 4.4)")
    print("Convergence events in 60 s for |T_i(n) - T_j(m)| < epsilon")
    print("=" * 70)

    rows = []
    for eps_ms in EPSILON_MS:
        eps_sec = eps_ms / 1000.0
        n_rational = rational_34_convergence_events(eps_sec)
        n_irrational = irrational_epi_convergence_events(eps_sec)
        freq_r = n_rational / DURATION
        freq_i = n_irrational / DURATION
        rows.append((eps_ms, n_rational, freq_r, n_irrational, freq_i))

    print()
    print("Rational 3:4 canon (IOI 1.0 s, 0.75 s):")
    print("-" * 50)
    for eps_ms, n_r, f_r, n_i, f_i in rows:
        print(f"  epsilon = {eps_ms:3d} ms  ->  {n_r} events in 60 s  ({f_r:.4f} /s)")
    print()
    print("Irrational e:pi canon:")
    print("-" * 50)
    for eps_ms, n_r, f_r, n_i, f_i in rows:
        print(f"  epsilon = {eps_ms:3d} ms  ->  {n_i} events in 60 s  ({f_i:.4f} /s)")
    print()

    # LaTeX table for Appendix
    print("-- For paper.tex Appendix (epsilon sensitivity table) --")
    print()
    print(r"\begin{table}[htbp]")
    print(r"\tbl{Convergence-point count and switching frequency for varying $\epsilon$ ($N = 60$~s).}")
    print(r"{\begin{tabular}{@{}lrrrr@{}} \toprule")
    print(r"$\epsilon$ (ms) & \textbf{3:4 count} & \textbf{3:4 (/s)} & \textbf{$e:\pi$ count} & \textbf{$e:\pi$ (/s)} \\ \midrule")
    for eps_ms, n_r, f_r, n_i, f_i in rows:
        print(rf"  {eps_ms} & {n_r} & {f_r:.4f} & {n_i} & {f_i:.4f} \\")
    print(r"  \bottomrule")
    print(r"\end{tabular}}")
    print(r"\tabnote{Rational 3:4: exact convergences every 3~s; count stable across $\epsilon$. Irrational $e:\pi$: count increases with $\epsilon$, showing consistent trend.}")
    print(r"\label{tab:epsilon_sensitivity}")
    print(r"\end{table}")
    print()

    # Consistency check: rational should be stable (21 for 60s including t=0)
    assert all(r[1] == 21 for r in rows), "Rational 3:4 should yield 21 events in 60 s"
    # Irrational: larger epsilon -> more events (or at least non-decreasing)
    irr_counts = [r[3] for r in rows]
    assert irr_counts == sorted(irr_counts), "Irrational count should be non-decreasing with epsilon"
    print("(Consistency checks passed.)")


if __name__ == "__main__":
    main()
