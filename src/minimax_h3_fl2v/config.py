"""Typed configuration for the MiniMax-H3 FL2VA studio."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = ROOT / "configs" / "default.yaml"
DEFAULT_LORA_CATALOG_PATH = ROOT / "configs" / "loras.yaml"


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _local_attention(value: Any) -> Optional[str]:
    from .offline import local_attention_backend

    if value is None or str(value).strip() == "":
        return "_flash_3"
    return local_attention_backend(str(value))


@dataclass
class LoRASpec:
    id: str
    name: str
    repo: Optional[str] = None
    filename: Optional[str] = None
    nfe: int = 8
    video_shift: float = 12.0
    audio_shift: float = 3.0
    lora_alpha: int = 8
    lora_scale: float = 1.0
    megapixels: float = 1.0
    notes: str = ""
    recommended: bool = False
    local_path: Optional[Path] = None
    backend: str = "diffusers"

    @property
    def is_base(self) -> bool:
        return not self.filename and not self.local_path

    def resolved_path(self, lora_dir: Path) -> Optional[Path]:
        if self.local_path is not None:
            return Path(self.local_path)
        if self.filename:
            return lora_dir / self.filename
        return None


@dataclass
class GenerationRequest:
    prompt: str
    first_image: Optional[Path] = None
    last_image: Optional[Path] = None
    duration_seconds: float = 5.0
    megapixels: float = 1.0
    aspect_ratio: str = "auto"
    nfe: int = 8
    seed: int = 42
    lora_id: str = "fl2va_turbo_8step_768p"
    extra_loras: list[tuple[Path, float]] = field(default_factory=list)
    # Deprecated single-extra compatibility used by the CLI.
    extra_lora_path: Optional[Path] = None
    extra_lora_scale: float = 1.0
    lora_scale: float = 1.0
    video_shift: Optional[float] = None
    audio_shift: Optional[float] = None
    lora_alpha: Optional[int] = None
    structured_prompt: bool = True

    @property
    def mode(self) -> str:
        if self.first_image and self.last_image:
            return "fl2va"
        if self.first_image or self.last_image:
            return "i2va"
        return "t2va"


@dataclass
class AppConfig:
    model_id: str = "MiniMaxAI/MiniMax-H3"
    local_dir: Path = ROOT / "models" / "MiniMax-H3"
    workflow: str = "fl2va"
    device: str = "cuda"
    dtype: str = "bfloat16"
    cpu_offload: bool = True
    memory_reserve_margin: str = "20GB"
    attention_backend: Optional[str] = "_flash_3"
    output_dir: Path = ROOT / "outputs"
    lora_dir: Path = ROOT / "models" / "loras"
    upload_dir: Path = ROOT / "uploads"
    default_lora_id: str = "fl2va_turbo_8step_768p"
    nfe: int = 8
    video_shift: float = 6.0
    audio_shift: float = 3.0
    lora_alpha: int = 128
    lora_scale: float = 1.0
    fuse_lora: bool = False
    megapixels: float = 1.0
    aspect_ratio: str = "auto"
    duration_seconds: float = 5.0
    seed: int = 42
    fps: int = 24
    hf_token: Optional[str] = None
    server_name: str = "0.0.0.0"
    server_port: int = 7860
    share: bool = False
    max_queue: int = 4
    title: str = "MiniMax-H3 FL2VA Studio"
    catalog: list[LoRASpec] = field(default_factory=list)

    @property
    def pretrained_path(self) -> str:
        from .offline import require_local_snapshot

        return str(require_local_snapshot(self.local_dir))

    def lora_by_id(self, lora_id: str) -> LoRASpec:
        for spec in self.catalog:
            if spec.id == lora_id:
                return spec
        raise KeyError(f"Unknown LoRA id: {lora_id}")


def _resolve_path(value: Any, default: Path) -> Path:
    if not value:
        return default
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path


def load_lora_catalog(path: Path = DEFAULT_LORA_CATALOG_PATH) -> list[LoRASpec]:
    document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    catalog = []
    for item in document.get("catalog", []):
        catalog.append(
            LoRASpec(
                id=item["id"],
                name=item["name"],
                repo=item.get("repo"),
                filename=item.get("filename"),
                nfe=int(item.get("nfe", 8)),
                video_shift=float(item.get("video_shift", 12.0)),
                audio_shift=float(item.get("audio_shift", 3.0)),
                lora_alpha=int(item.get("lora_alpha", 8)),
                lora_scale=float(item.get("lora_scale", 1.0)),
                megapixels=float(item.get("megapixels", 1.0)),
                notes=item.get("notes", ""),
                recommended=bool(item.get("recommended", False)),
                backend=str(item.get("backend", "diffusers")),
            )
        )
    return catalog


def load_config(
    config_path: Path = DEFAULT_CONFIG_PATH,
    catalog_path: Path = DEFAULT_LORA_CATALOG_PATH,
) -> AppConfig:
    load_dotenv(ROOT / ".env")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    gradio = raw.get("gradio", {})

    cfg = AppConfig(
        model_id=os.getenv("MINIMAX_H3_MODEL_ID", raw.get("model_id", "MiniMaxAI/MiniMax-H3")),
        local_dir=_resolve_path(
            os.getenv("MINIMAX_H3_LOCAL_DIR", raw.get("local_dir")),
            ROOT / "models" / "MiniMax-H3",
        ),
        workflow=raw.get("workflow", "fl2va"),
        device=os.getenv("MINIMAX_H3_DEVICE", raw.get("device", "cuda")),
        dtype=os.getenv("MINIMAX_H3_DTYPE", raw.get("dtype", "bfloat16")),
        cpu_offload=_as_bool(os.getenv("MINIMAX_H3_CPU_OFFLOAD"), _as_bool(raw.get("cpu_offload"), True)),
        memory_reserve_margin=os.getenv(
            "MINIMAX_H3_MEMORY_RESERVE_MARGIN",
            raw.get("memory_reserve_margin", "20GB"),
        ),
        attention_backend=_local_attention(
            os.getenv("MINIMAX_H3_ATTENTION_BACKEND", raw.get("attention_backend"))
        ),
        output_dir=_resolve_path(raw.get("output_dir"), ROOT / "outputs"),
        lora_dir=_resolve_path(
            os.getenv("MINIMAX_H3_LORA_DIR", raw.get("lora_dir")),
            ROOT / "models" / "loras",
        ),
        upload_dir=_resolve_path(raw.get("upload_dir"), ROOT / "uploads"),
        default_lora_id=os.getenv(
            "MINIMAX_H3_DEFAULT_LORA",
            raw.get("default_lora_id", "fl2va_turbo_8step_768p"),
        ),
        nfe=int(raw.get("nfe", 8)),
        video_shift=float(raw.get("video_shift", 6.0)),
        audio_shift=float(raw.get("audio_shift", 3.0)),
        lora_alpha=int(raw.get("lora_alpha", 128)),
        lora_scale=float(raw.get("lora_scale", 1.0)),
        fuse_lora=_as_bool(raw.get("fuse_lora"), False),
        megapixels=float(raw.get("megapixels", 1.0)),
        aspect_ratio=str(raw.get("aspect_ratio", "auto")),
        duration_seconds=float(raw.get("duration_seconds", 5.0)),
        seed=int(raw.get("seed", 42)),
        fps=int(raw.get("fps", 24)),
        hf_token=os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN"),
        server_name="0.0.0.0",
        server_port=int(os.getenv("GRADIO_SERVER_PORT", gradio.get("server_port", 7860))),
        share=False,
        max_queue=int(gradio.get("max_queue", 4)),
        title=gradio.get("title", "MiniMax-H3 FL2VA Studio"),
        catalog=load_lora_catalog(catalog_path),
    )
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    cfg.lora_dir.mkdir(parents=True, exist_ok=True)
    cfg.upload_dir.mkdir(parents=True, exist_ok=True)
    cfg.local_dir.mkdir(parents=True, exist_ok=True)
    return cfg
