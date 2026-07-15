#!/usr/bin/env python3
"""
Window-sensitivity of the perceptual saturation break point
===========================================================

The decisive test of whether the retention transition is a real consequence of
temporal integration or an artifact of the metric. If integration is the mechanism,
the break point must move with the integration window w, tracking roughly 1/w. If the
break point sits at the same density regardless of w, the transition is a property of
the density grid or the metric, not of integration, and the mechanistic claim fails.

Streams do not depend on w, so each (density, trial) stream is generated once and all
windows are applied to it. This shares the expensive generation across windows and
makes the comparison exact rather than seed-dependent.

Outputs perceptual_saturation_wsweep.json.
"""

import argparse
import json
import sys

import numpy as np

from config import CODE_DIR, SUPPLEMENTARY_DIR

sys.path.insert(0, str(SUPPLEMENTARY_DIR / 'experiments'))
sys.path.insert(0, str(SUPPLEMENTARY_DIR / 'core'))

from density_sweep_null_model_comparison import (  # noqa: E402
    generate_amanous_stream_at_density, generate_random_midi, HAS_AMANOUS,
)
from perceptual_saturation import (  # noqa: E402
    DENSITIES, retention, estimate_breakpoint, piecewise_r2,
)


def main(windows_ms, seed, trials, events, n_boot):
    if not HAS_AMANOUS:
        raise SystemExit("Real composer not importable; refusing to run on the fallback generator.")
    rng = np.random.default_rng(seed)

    # Generate every stream once, keyed by (density, trial). Windows are applied later.
    streams = {}
    for d in DENSITIES:
        for t in range(trials):
            np.random.seed(seed + 1000 * t + int(d))
            ta, pa = generate_amanous_stream_at_density(d, events, rng)
            tn, pn = generate_random_midi(d, events, rng)
            streams[(d, t)] = (ta, pa, tn, pn)

    out = {'windows': [], 'seed': seed, 'n_trials': trials, 'n_events': events}
    x = np.array(DENSITIES, float)

    print(f"{'w(ms)':>6}  {'1/w':>6}  {'breakpoint':>10}  {'95% CI':>16}  {'R2 pw':>6}  {'R2 lin':>6}")
    for w_ms in windows_ms:
        w = w_ms / 1000.0
        per_trial = {d: [] for d in DENSITIES}
        for d in DENSITIES:
            for t in range(trials):
                ta, pa, tn, pn = streams[(d, t)]
                r = retention(ta, pa, w)
                if not np.isnan(r):
                    per_trial[d].append(r)
        y = np.array([np.mean(per_trial[d]) for d in DENSITIES])
        bp = estimate_breakpoint(x, y)
        fit = piecewise_r2(x, y, bp)

        # bootstrap CI over trials
        brng = np.random.default_rng(seed + int(w_ms))
        bps = []
        for _ in range(n_boot):
            yb = np.array([np.mean(brng.choice(per_trial[d], size=len(per_trial[d]), replace=True))
                           for d in DENSITIES])
            bps.append(estimate_breakpoint(x, yb))
        lo, hi = float(np.percentile(bps, 2.5)), float(np.percentile(bps, 97.5))

        rec = {'window_ms': w_ms, 'inv_w_nps': 1000.0 / w_ms, 'breakpoint_nps': bp,
               'ci95': [lo, hi], **fit,
               'retention_curve': {int(d): float(np.mean(per_trial[d])) for d in DENSITIES}}
        out['windows'].append(rec)
        print(f"{w_ms:6.0f}  {1000.0/w_ms:6.0f}  {bp:10.1f}  [{lo:5.1f},{hi:5.1f}]  "
              f"{fit['r2_piecewise']:6.3f}  {fit['r2_linear']:6.3f}")

    outp = CODE_DIR / 'perceptual_saturation_wsweep.json'
    with open(outp, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {outp}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--windows', type=str, default='20,25,33,40,50')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--trials', type=int, default=20)
    ap.add_argument('--events', type=int, default=200)
    ap.add_argument('--bootstrap', type=int, default=2000)
    a = ap.parse_args()
    windows = [float(x) for x in a.windows.split(',')]
    main(windows, a.seed, a.trials, a.events, a.bootstrap)
