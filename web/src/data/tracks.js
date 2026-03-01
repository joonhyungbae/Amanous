/**
 * Composition metadata (aligned with code/play_audio.py DESCRIPTIONS).
 * Order follows paper section appearance: 4.1.5 Canonical → Beyond-human → Phase Music; 4.4 Convergence Point.
 * Audio URL: {BASE_URL}audio/{id}.wav
 */
export const TRACK_ORDER = [
  'canonical_abaababa',   // Excerpt 3 (Section 4.1.5)
  'beyond_human_demo',    // Excerpt 1 (Section 4.1.5)
  'minimalist_phase',    // Excerpt 2 (Section 4.1.5)
  'convergence_point',   // Excerpt 4 (Section 4.4)
];

// Titles aligned with paper Appendix F (Supplementary audio Excerpts)
export const TRACKS = {
  canonical_abaababa: {
    title: 'Canonical ABAABABA Validation Composition',
    style: 'L-system & 3:4 tempo canon',
    description: 'L-system macro-form with deterministic (A) and textural (B) sections; 3:4 tempo canon (paper Excerpt 3).',
    duration: '74s',
    highlight: '3:4 tempo canon, C major vs chromatic',
  },
  beyond_human_demo: {
    title: 'Beyond-Human-Density',
    style: 'Superhuman piano textures',
    description: 'Rendered on Disklavier. Polyphony (40-note chords), 30 Hz multi-key trill, 6-octave arpeggio (paper Excerpt 1).',
    duration: '34s',
    highlight: 'Peak density 138 notes/s, 4 voices',
  },
  minimalist_phase: {
    title: 'Phase Music — Minimalist Study',
    style: 'Phase-shift · pentatonic',
    description: 'Reich-inspired phase-shift; pentatonic set, 1:1.01 tempo drift between voices (paper Excerpt 2).',
    duration: '80s',
    highlight: 'High tonal stability (TS=0.37), regular rhythm',
  },
  convergence_point: {
    title: 'Convergence Point (3:4 Canon)',
    style: '3:4 tempo canon · texture switch',
    description: 'Pre-CP sparse/melodic and post-CP dense/textural switch at t = 15 s (paper Excerpt 4).',
    duration: '30s',
    highlight: 'Density and tonality shift at convergence',
  },
};
