#!/usr/bin/env python3
"""
Custom Amanous Composition (paper Excerpt 2 only)
==================================================

Generates Phase Music — Minimalist Study (Excerpt 2) as in the paper.
Other paper pieces (Canonical, Beyond-human, Convergence Point) are generated via amanous_composer.py presets.
"""

from amanous_composer import (
    CompositionConfig, SymbolConfig, DistributionConfig,
    compose, events_to_midi
)


def create_minimalist_composition():
    """
    Phase Music — Minimalist Study (Excerpt 2).
    Reich-inspired phase-shifting effect.
    """
    pentatonic = [0, 2, 4, 7, 9]  # C, D, E, G, A

    symbol_configs = {
        'S': SymbolConfig(
            tempo_ratios=(1.0, 1.0),
            duration=8.0,
            ioi_dist=DistributionConfig('constant', {'value': 0.125}),
            pitch_dist=DistributionConfig('gaussian', {'mean': 60, 'std': 4}),
            velocity_dist=DistributionConfig('constant', {'value': 550}),
            pitch_set=pentatonic,
            mode='melodic'
        ),
        'P': SymbolConfig(
            tempo_ratios=(1.0, 1.01),  # 1% tempo difference for phase shift
            duration=16.0,
            ioi_dist=DistributionConfig('constant', {'value': 0.125}),
            pitch_dist=DistributionConfig('gaussian', {'mean': 60, 'std': 4}),
            velocity_dist=DistributionConfig('gaussian', {'mean': 520, 'std': 30}),
            pitch_set=pentatonic,
            mode='melodic'
        ),
        'D': SymbolConfig(
            tempo_ratios=(1.0, 1.0),
            duration=8.0,
            ioi_dist=DistributionConfig('constant', {'value': 0.0625}),
            pitch_dist=DistributionConfig('gaussian', {'mean': 60, 'std': 6}),
            velocity_dist=DistributionConfig('gaussian', {'mean': 600, 'std': 50}),
            pitch_set=pentatonic,
            mode='melodic'
        ),
    }

    return CompositionConfig(
        axiom='S',
        production_rules={'S': 'SP', 'P': 'PD', 'D': 'S'},
        iterations=3,
        symbol_configs=symbol_configs,
        voices=2,
        seed=1234,
        title="Phase Music — Minimalist Study"
    )


# =============================================================================
# Main: generate paper Excerpt 2 only
# =============================================================================

if __name__ == "__main__":
    import os

    output_dir = os.path.join(os.path.dirname(__file__), "..", "compositions")
    os.makedirs(output_dir, exist_ok=True)

    name = "minimalist_phase"
    config = create_minimalist_composition()

    print("=" * 70)
    print("AMANOUS — Phase Music (Excerpt 2)")
    print("=" * 70)
    print(f"\n>>> Generating: {config.title}")
    events, sequence, summary = compose(config)
    print(summary)

    midi_file = os.path.join(output_dir, f"{name}.mid")
    events_to_midi(events, midi_file, tempo=100, title=config.title)
    print(f"✓ MIDI saved: {midi_file}")

    import pandas as pd
    csv_file = os.path.join(output_dir, f"{name}_events.csv")
    pd.DataFrame(events).to_csv(csv_file, index=False)
    print(f"✓ CSV saved: {csv_file}")
    print("=" * 70)
