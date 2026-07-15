#!/usr/bin/env python3
"""
DEPRECATED. Do not use.

This script fitted a piecewise regression to fourteen (density, coherence) pairs
that were typed in by hand and were never produced by any generator. The break point
it reported (28.4 notes/s) was therefore an artifact of hand-authored input, not a
measurement of the system.

Measure the density behaviour instead:

  * density_sweep_breakpoint.py     -- Single-Voice Coherence swept over density with
                                       the real generator. Shows the SVC curve is flat,
                                       i.e. a pitch-interval metric on a fixed event
                                       count cannot vary with density.
  * perceptual_saturation.py        -- melodic-transition retention under a temporal
                                       integration window, which does depend on density.
  * perceptual_saturation_wsweep.py -- window-sensitivity check showing the retention
                                       knee does not track 1/w (no sharp threshold).
  * discriminability_analysis.py    -- structured-versus-null separability by density,
                                       with the peak near 25 notes/s and its collapse
                                       band. This is what the paper reports.

The hand-authored input (density_sweep_results.csv) has been removed from the
repository along with the fabricated curve that lived here.
"""

import sys

if __name__ == "__main__":
    sys.exit(
        "breakpoint_bootstrap.py is deprecated: it fit a hand-authored curve.\n"
        "Run discriminability_analysis.py (and density_sweep_breakpoint.py) instead."
    )
