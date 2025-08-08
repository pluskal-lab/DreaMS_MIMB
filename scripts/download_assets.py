#!/usr/bin/env python3
"""
scripts/download_assets.py

Download MassSpecGym data and DreaMS checkpoints from HF Hub
into your PROJECT_ROOT/data/... folders.
"""

import sys
import shutil
from pathlib import Path

# 1) locate project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 2) prepare local target dirs
DATA_DIR = PROJECT_ROOT / "data"
MASS_DIR = DATA_DIR / "massspecgym"
CKPT_DIR = DATA_DIR / "model_checkpoints"
for d in (MASS_DIR, CKPT_DIR):
    d.mkdir(parents=True, exist_ok=True)

# 3) define what to fetch:
TO_DOWNLOAD = [
    ("roman-bushuiev/MassSpecGym",
     "data/auxiliary/MassSpecGym.mgf",
     MASS_DIR / "MassSpecGym.mgf"),
    ("roman-bushuiev/DreaMS",
     "ssl_model.ckpt",
     CKPT_DIR / "ssl_model.ckpt"),
    ("roman-bushuiev/DreaMS",
     "embedding_model.ckpt",
     CKPT_DIR / "embedding_model.ckpt"),
]

# 4) try huggingface_hub first
try:
    from huggingface_hub import hf_hub_download
    _use_hf = True
except ImportError:
    _use_hf = False

# 5) download loop
for repo_id, in_path, out_path in TO_DOWNLOAD:
    if out_path.exists():
        print(f"[SKIP] {out_path} already exists")
        continue

    print(f"[DOWN] {repo_id}/{in_path} → {out_path}")
    out_tmp: Path
    if _use_hf:
        # repo_type: dataset for MassSpecGym, model for DreaMS
        repo_type = "dataset" if repo_id.endswith("MassSpecGym") else "model"
        try:
            hf_path = hf_hub_download(
                repo_id=repo_id,
                filename=in_path,
                repo_type=repo_type,
                revision="main"
            )
        except Exception as e:
            print(f"  ✗ huggingface_hub failed: {e}")
            sys.exit(1)
        out_tmp = Path(hf_path)
    else:
        # fallback to HTTP GET
        import requests
        url = f"https://huggingface.co/{repo_id}/resolve/main/{in_path}"
        r = requests.get(url, stream=True)
        r.raise_for_status()
        out_tmp = out_path.with_name(out_path.name + ".part")
        with open(out_tmp, "wb") as f:
            for chunk in r.iter_content(1024 * 1024):
                f.write(chunk)

    # copy from hf cache or .part → final
    shutil.copy(out_tmp, out_path)
    if not _use_hf:
        out_tmp.unlink()
    print("  ✓ done")

print("All files are in place!")