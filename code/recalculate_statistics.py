#!/usr/bin/env python3
"""
DEPRECATED. Do not use.

This script read an extracted-outputs directory that does not exist in the repository
and hard-coded several statistics (including the continuous convergence-point tracking
values r = 0.907 / 0.888 / 0.933). None of its numbers is a live source for the paper.

The reproducible sources are:

  * cp_continuous_tracking.py  -- continuous CP tracking, measured from the specified
                                  inhomogeneous Poisson process (seed 42: r = 0.928).
  * discriminability_analysis.py, analyze_canonical.py -- the main results.
  * supplementary_code/data/csv/*.csv -- the hardware-compensation and constraint
                                  statistics (e.g. latency_filter_effectiveness_comparison.csv
                                  for the filter SD, correction_pipeline_validation_results.csv
                                  for the onset-error figures).

Kept only as a tombstone so that the retired numbers are not mistaken for a source.
"""

import sys

if __name__ == "__main__":
    sys.exit(
        "recalculate_statistics.py is deprecated: it read a nonexistent directory and\n"
        "hard-coded retired values. Use cp_continuous_tracking.py, discriminability_analysis.py,\n"
        "analyze_canonical.py, and the saved CSVs under supplementary_code/data/csv/."
    )
