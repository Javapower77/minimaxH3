#!/usr/bin/env python3
"""Install the isolated ComfyUI worker and pruned MiniMax-H3 FL2VA weights."""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMFY = ROOT / ".runtime" / "ComfyUI"
COMFY_REVISION = "1d48d9cf7bcecb6022a87b3cb13e0fb435bf9b8a"
REPO = "Comfy-Org/MiniMax-H3"
FILES = (
    "diffusion_models/minimax_h3_fl2va_pruned_bf16.safetensors",
    "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
    "vae/minimax_h3_video_vae_fp16.safetensors",
    "vae/minimax_h3_audio_vae_fp32.safetensors",
)
PRUNED_SHA256 = "a32572fb90b5508b201ec7c2eddcc184b13ddfd3c6f6d2cf06a0b46535d541b4"


def run(*args: str) -> None:
    subprocess.run(args, check=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-install", action="store_true", help="Only download/check model files")
    args = parser.parse_args()

    if not args.skip_install:
        COMFY.parent.mkdir(parents=True, exist_ok=True)
        if not (COMFY / ".git").is_dir():
            run("git", "clone", "--depth", "1", "https://github.com/Comfy-Org/ComfyUI.git", str(COMFY))
        run("git", "-C", str(COMFY), "fetch", "--depth", "1", "origin", COMFY_REVISION)
        run("git", "-C", str(COMFY), "checkout", "--detach", COMFY_REVISION)
        venv = COMFY / ".venv"
        if not (venv / "bin/python").is_file():
            run("python3.11", "-m", "venv", str(venv))
        python = str(venv / "bin/python")
        run(python, "-m", "pip", "install", "--upgrade", "pip")
        run(
            python, "-m", "pip", "install", "torch==2.11.0", "torchvision", "torchaudio",
            "--index-url", "https://download.pytorch.org/whl/cu128",
        )
        run(python, "-m", "pip", "install", "-r", str(COMFY / "requirements.txt"))

    sys.path.insert(0, str(ROOT / "src"))
    from minimax_h3_fl2v.offline import allow_online_for_download

    allow_online_for_download()
    os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
    from huggingface_hub import hf_hub_download

    for filename in FILES:
        print(f"Downloading {REPO}/{filename}", flush=True)
        hf_hub_download(repo_id=REPO, filename=filename, local_dir=str(COMFY / "models"))

    model = COMFY / "models/diffusion_models/minimax_h3_fl2va_pruned_bf16.safetensors"
    actual = sha256(model)
    if actual != PRUNED_SHA256:
        raise RuntimeError(f"Pruned base SHA-256 mismatch: expected {PRUNED_SHA256}, got {actual}")

    lora_target = COMFY / "models/loras"
    lora_target.mkdir(parents=True, exist_ok=True)
    for source in (ROOT / "models/loras").glob("*.safetensors"):
        destination = lora_target / source.name
        if destination.exists() or destination.is_symlink():
            destination.unlink()
        destination.symlink_to(source.resolve())

    print("Pruned FL2VA backend is ready.")


if __name__ == "__main__":
    main()
