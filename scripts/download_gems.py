#!/usr/bin/env python3
"""
scripts/download_gems.py

Download the large GeMS_A10.hdf5 (~14 GB) from the HF Hub into:
  <PROJECT_ROOT>/data/spectra/GeMS_A10.hdf5

No login needed (public repo). Works on Linux/macOS/Windows.
"""

import os
import sys
import shutil
from pathlib import Path

# Figure out project root as the repo root (parent of scripts/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEST_DIR = PROJECT_ROOT / "data" / "spectra"
DEST_DIR.mkdir(parents=True, exist_ok=True)
DEST_PATH = DEST_DIR / "GeMS_A10.hdf5"

REPO_ID = "roman-bushuiev/MassSpecGym"
REPO_FILE = "data/spectra/GeMS_A10.hdf5"  # path inside the HF repo

# Prefer huggingface_hub (resumable, cached). Fallback to requests streaming.
try:
    from huggingface_hub import hf_hub_download
    use_hf = True
except ImportError:
    use_hf = False

def link_or_copy(src: Path, dst: Path):
    """Try to hardlink (fast, no extra space). Fallback to copy."""
    try:
        if dst.exists():
            return
        os.link(src, dst)
    except Exception:
        shutil.copy2(src, dst)

def main():
    if DEST_PATH.exists():
        print(f"[SKIP] {DEST_PATH} already exists")
        return

    print(f"[INFO] Downloading {REPO_ID}/{REPO_FILE}")
    print(f"[INFO] Destination: {DEST_PATH}")

    if use_hf:
        try:
            cached = hf_hub_download(
                repo_id=REPO_ID,
                repo_type="dataset",
                filename=REPO_FILE,
                revision="main",
                resume_download=True,   # good for large files
            )
        except Exception as e:
            print(f"[ERROR] huggingface_hub download failed: {e}")
            sys.exit(1)

        # Copy (or hardlink) from cache into our project data folder
        link_or_copy(Path(cached), DEST_PATH)
        print("[OK] Download complete (via huggingface_hub).")
        return

    # Fallback: raw HTTPS with streaming
    try:
        import requests
    except ImportError:
        print("[ERROR] Install either huggingface_hub or requests to download.")
        sys.exit(1)

    url = f"https://huggingface.co/{REPO_ID}/resolve/main/{REPO_FILE}"
    tmp = DEST_PATH.with_suffix(".part")

    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        downloaded = 0
        chunk = 1024 * 1024  # 1MB

        with open(tmp, "wb") as f:
            for block in r.iter_content(chunk_size=chunk):
                if block:
                    f.write(block)
                    downloaded += len(block)
                    if total:
                        pct = downloaded * 100 // total
                        print(f"\r[DOWN] {downloaded/1e9:.2f} / {total/1e9:.2f} GB ({pct}%)", end="")
        print()

    tmp.rename(DEST_PATH)
    print("[OK] Download complete (via requests streaming).")

if __name__ == "__main__":
    main()