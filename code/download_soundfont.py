#!/usr/bin/env python3
"""
SoundFont setup for Amanous MIDI → WAV conversion
=================================================

SoundFont files are not stored in the repo (large binaries). This script helps you:
  1. Install system soundfont (FluidR3 GM) on Linux/macOS
  2. Download Salamander Grand Piano (recommended for piano quality) into soundfonts/

Run: python download_soundfont.py
See: README.md "SoundFonts (MIDI → WAV)"
"""

import os
import subprocess
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SOUNDFONTS_DIR = REPO_ROOT / "soundfonts"

# Salamander C5 Light (CC BY-NC; good for piano, Disklavier-like)
# HED Sounds: https://sites.google.com/view/hed-sounds/salamander-c5-light
# Zip contains .sf2; extract to soundfonts/ as SalamanderC5-Lite.sf2 or SalamanderGrandPiano.sf2
SALAMANDER_ZIP_URL = "https://drive.google.com/uc?export=download&id=0B5gPxvwx-I4KWjZ2SHZOLU42dHM"
SALAMANDER_PAGE = "https://sites.google.com/view/hed-sounds/salamander-c5-light"


def ensure_soundfonts_dir():
    SOUNDFONTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Directory: {SOUNDFONTS_DIR}")


def check_system_soundfont():
    """Check for FluidR3 / system GM soundfont."""
    candidates = [
        "/usr/share/sounds/sf2/FluidR3_GM.sf2",
        "/usr/share/soundfonts/FluidR3_GM.sf2",
        "/usr/share/sounds/sf2/default-GM.sf2",
    ]
    for p in candidates:
        if Path(p).exists():
            print(f"  ✓ System soundfont found: {p}")
            return True
    print("  ✗ No system soundfont found.")
    return False


def suggest_system_install():
    """Print install commands for FluidSynth + GM soundfont."""
    print("\n--- Install system soundfont (quick start) ---")
    try:
        with open("/etc/os-release") as f:
            if "debian" in f.read().lower() or "ubuntu" in f.read().lower():
                print("  Debian/Ubuntu:")
                print("    sudo apt update")
                print("    sudo apt install -y fluidsynth fluid-soundfont-gm")
                print("  Then FluidR3_GM.sf2 will be used automatically.")
                return
    except FileNotFoundError:
        pass
    print("  Linux (Debian/Ubuntu):  sudo apt install -y fluidsynth fluid-soundfont-gm")
    print("  macOS:                  brew install fluid-synth")
    print("  (On macOS, download a .sf2 and pass it with midi_to_audio.py --soundfont <path>)")


def suggest_salamander():
    """Print instructions to download Salamander Grand Piano."""
    print("\n--- Salamander Grand Piano (recommended for piano quality) ---")
    print("  1. Create directory (if needed):  mkdir -p soundfonts")
    print("  2. Download from:")
    print(f"     {SALAMANDER_PAGE}")
    print("     (Salamander C5 Light, ~24 MB; CC BY-NC)")
    print("  3. Extract the .sf2 from the zip into the repo's soundfonts/ folder.")
    print("  4. Rename to one of: SalamanderGrandPiano.sf2  or  SalamanderC5-Lite.sf2")
    print(f"     So that the file is: {SOUNDFONTS_DIR / 'SalamanderGrandPiano.sf2'}")


def try_download_salamander():
    """Attempt to download Salamander C5 Light zip (Google Drive may require manual confirm)."""
    ensure_soundfonts_dir()
    zip_path = SOUNDFONTS_DIR / "SalamanderC5-Lite.zip"
    if zip_path.exists():
        print(f"  Found existing zip: {zip_path}")
        return extract_salamander_zip(zip_path)

    print("  Attempting download (Google Drive may show a confirmation page)...")
    try:
        import urllib.request
        urllib.request.urlretrieve(SALAMANDER_ZIP_URL, zip_path)
    except Exception as e:
        print(f"  Download failed: {e}")
        print("  Please download manually from:", SALAMANDER_PAGE)
        return False
    return extract_salamander_zip(zip_path)


def extract_salamander_zip(zip_path: Path) -> bool:
    """Extract .sf2 from zip into soundfonts/ with a known name."""
    if not zip_path.exists():
        return False
    try:
        with zipfile.ZipFile(zip_path, "r") as z:
            names = [n for n in z.namelist() if n.lower().endswith(".sf2")]
            if not names:
                print("  No .sf2 file found in zip.")
                return False
            # Prefer a name we already look for
            target = SOUNDFONTS_DIR / "SalamanderC5-Lite.sf2"
            with z.open(names[0]) as src:
                target.write_bytes(src.read())
            print(f"  Extracted: {target}")
            return True
    except Exception as e:
        print(f"  Extract failed: {e}")
        return False


def list_available():
    """List which soundfonts are available (same logic as midi_to_audio.py --list-sf)."""
    from midi_to_audio import DEFAULT_SOUNDFONTS
    print("Soundfont paths (checked by midi_to_audio.py):")
    for sf in DEFAULT_SOUNDFONTS:
        exists = "✓" if os.path.exists(sf) else "✗"
        print(f"  {exists} {sf}")
    any_ok = any(os.path.exists(sf) for sf in DEFAULT_SOUNDFONTS)
    if not any_ok:
        print("\nNo soundfont found. Run: python download_soundfont.py")
    return 0 if any_ok else 1


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="SoundFont setup for Amanous (MIDI → WAV)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available soundfont paths and exit",
    )
    parser.add_argument(
        "--download-salamander",
        action="store_true",
        help="Try to download Salamander C5 Light into soundfonts/",
    )
    args = parser.parse_args()

    if args.list:
        return list_available()

    print("SoundFont setup for Amanous")
    print("=" * 50)
    ensure_soundfonts_dir()

    if args.download_salamander:
        try_download_salamander()
        list_available()
        return 0

    has_system = check_system_soundfont()
    if has_system:
        print("\nYou can run midi_to_audio.py without extra steps.")
    suggest_system_install()
    suggest_salamander()
    print("\nVerify:  python midi_to_audio.py --list-sf")
    return 0


if __name__ == "__main__":
    sys.exit(main())
