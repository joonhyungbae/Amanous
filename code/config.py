#!/usr/bin/env python3
"""
Amanous path configuration
==========================

Single place where every filesystem location used by the pipeline is resolved.

Paths default to locations relative to the repository root, so a fresh clone
works with no edits. Each path can be overridden with an environment variable
when the layout differs, for example when audio renders live on a separate
volume:

    AMANOUS_ROOT           repository root (default: parent of this file's dir)
    AMANOUS_AUDIO_DIR      high-quality WAV renders      (default: $ROOT/audio_hq)
    AMANOUS_SOUNDFONT_DIR  SoundFont (.sf2) location     (default: $ROOT/soundfonts)
    AMANOUS_SOUNDFONT      explicit .sf2 file, wins over the search list
    AMANOUS_OUTPUT_DIR     rendered audio for the web app (default: $ROOT/web/public/audio)
    AMANOUS_CODE_EXTRACTED extracted experiment outputs   (default: $ROOT/code_extracted)
"""

import os
from pathlib import Path


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser().resolve() if value else default


REPO_ROOT = _env_path("AMANOUS_ROOT", Path(__file__).resolve().parent.parent)

CODE_DIR = REPO_ROOT / "code"
COMPOSITIONS_DIR = REPO_ROOT / "compositions"
SUPPLEMENTARY_DIR = REPO_ROOT / "supplementary_code"

AUDIO_DIR = _env_path("AMANOUS_AUDIO_DIR", REPO_ROOT / "audio_hq")
SOUNDFONT_DIR = _env_path("AMANOUS_SOUNDFONT_DIR", REPO_ROOT / "soundfonts")
OUTPUT_DIR = _env_path("AMANOUS_OUTPUT_DIR", REPO_ROOT / "web" / "public" / "audio")
CODE_EXTRACTED = _env_path("AMANOUS_CODE_EXTRACTED", REPO_ROOT / "code_extracted")

# SoundFont search order. An explicit AMANOUS_SOUNDFONT takes precedence, then the
# repo-local files, then common system installs.
_SYSTEM_SOUNDFONTS = [
    Path("/usr/share/sounds/sf2/FluidR3_GM.sf2"),
    Path("/usr/share/soundfonts/FluidR3_GM.sf2"),
    Path("/usr/share/sounds/sf2/default-GM.sf2"),
]


def soundfont_candidates() -> list[Path]:
    """Return the SoundFont files to try, in priority order."""
    candidates: list[Path] = []
    explicit = os.environ.get("AMANOUS_SOUNDFONT")
    if explicit:
        candidates.append(Path(explicit).expanduser())
    candidates += [
        SOUNDFONT_DIR / "SalamanderGrandPiano.sf2",
        SOUNDFONT_DIR / "SalamanderC5-Lite.sf2",
    ]
    candidates += _SYSTEM_SOUNDFONTS
    return candidates


def find_soundfont() -> Path | None:
    """First SoundFont that exists on disk, or None if the user has not installed one."""
    for candidate in soundfont_candidates():
        if candidate.is_file():
            return candidate
    return None
