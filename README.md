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
| `compositions/` | Example MIDI outputs (e.g. `minimalist_phase.mid`, `convergence_point.mid`) |
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

### Web (listen to excerpts)

```bash
./dev.sh
# Opens Vite dev server; open http://localhost:5173
```

Place WAV files in `web/public/audio/` (e.g. `canonical_abaababa.wav`, `convergence_point.wav`) so the player can load them. Track list and descriptions are in `web/src/data/tracks.js`.

### Deploy web to GitHub Pages

```bash
./deploy.sh
# Cleans node_modules, installs, builds, and runs gh-pages deploy
```

---

## Citation

If you use Amanous or build on its methodology, please cite the paper:

- *Amanous: Distribution-Switching for Superhuman Piano Density on Disklavier*, Journal of Creative Music Systems. (See `paper.tex` and `reference.bib` for full reference.)

---

## License

This project is licensed under **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)**. You may share and adapt the material for non-commercial use with attribution. See [LICENSE](LICENSE) for the full text.
