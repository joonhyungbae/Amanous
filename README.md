# Amanous

**Distribution-switching for superhuman piano density on Yamaha Disklavier.**

Amanous is a hardware-aware algorithmic composition system that unifies **L-systems**, **tempo canons**, and **stochastic distributions** in a single pipeline. It generates piano music at note densities, polyphony, and register spans beyond human performance limits, while respecting the Disklavier’s actuation constraints.

- **Live excerpts:** [amanous.xyz](https://www.amanous.xyz)
- **Paper:** *Amanous: Distribution-Switching for Superhuman Piano Density on Disklavier* (JCMS)

---

## Features

- **Hierarchical distribution-switching** — L-system symbols select distinct distributional regimes (not just parameter tweaks), producing statistically separable sections with large effect sizes.
- **Hardware abstraction layer (HAL)** — Velocity-dependent latency and key-reset limits are formalized and compensated so that superhuman textures stay within the instrument’s actuable envelope.
- **Density saturation threshold** — A computational transition around 24–30 notes/s is identified (bootstrap 95% CI: 23.3–50.0), beyond which single-domain melodic metrics lose discriminative power.
- **Convergence point calculus** — Tempo-canon convergence events drive distribution switches, linking macro temporal structure to micro-level texture.

All results are computational (MIDI/statistical); psychoacoustic validation is proposed for future work. The pipeline has been run on a physical Disklavier with sub-millisecond software precision.

---

## Project structure

| Path | Description |
|------|-------------|
| `code/` | Main composition pipeline: `amanous_composer.py`, ablations, MIDI→audio, analysis |
| `supplementary_code/` | Experiments, RQ validations, coherence metrics, HAL and latency robustness |
| `compositions/` | Example composition data (event CSVs; MIDI/WAV when generated) |
| `audio_hq/` | High-quality WAV renders of selected compositions |
| `web/` | React + Vite site for [amanous.xyz](https://www.amanous.xyz) — play excerpts, metadata aligned with paper |
| `paper.tex` | LaTeX manuscript (JCMS); `reference.bib` for bibliography |

---

## Getting started

### Python (composition and analysis)

Core pipeline and supplementary code use Python 3.

```bash
# From repo root
pip install -r supplementary_code/requirements.txt
# Or for code/ only: numpy scipy pandas midiutil
```

Run the main composer (example):

```bash
cd code
python amanous_composer.py   # or use custom_composition.py for custom configs
```

Ablations and experiments live under `supplementary_code/` (see `experiments/`, `rq1_distribution_switching/`, etc.). Check individual scripts for usage.

### SoundFonts (MIDI → WAV)

SoundFont files (`.sf2`) are **not** in the repo (large binaries; see `.gitignore`). To convert MIDI to WAV you need either a system GM soundfont or a piano soundfont in `soundfonts/`.

**Option A — Quick start (Linux):** Install FluidSynth and the GM soundfont; the pipeline will use it automatically.

```bash
sudo apt install -y fluidsynth fluid-soundfont-gm
```

**Option B — Better piano quality:** Use the Salamander Grand Piano (Yamaha C5, Disklavier-like). Run the setup script, then download and place the `.sf2` as instructed:

```bash
cd code
python download_soundfont.py
# Follow the printed links; put the .sf2 in repo root's soundfonts/ as SalamanderGrandPiano.sf2 or SalamanderC5-Lite.sf2
```

Optional: `python download_soundfont.py --download-salamander` attempts an automatic download (Google Drive may require manual confirmation). Check available soundfonts: `python midi_to_audio.py --list-sf`.

### Web (listen to excerpts)

```bash
./dev.sh
# Opens Vite dev server; open http://localhost:5173
```

Place WAV files in `web/public/audio/` (e.g. `canonical_abaababa.wav`, `convergence_point.wav`) so the player can load them. Track list and descriptions are in `web/src/data/tracks.js`.


## License

This project is licensed under **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)**. You may share and adapt the material for non-commercial use with attribution. See [LICENSE](LICENSE) for the full text.
