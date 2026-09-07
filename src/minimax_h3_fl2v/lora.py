"""Inspect PEFT LoRA SafeTensors for MiniMax-H3.

Runtime loading uses Diffusers' ``MiniMaxH3LoraLoaderMixin.load_lora_weights``,
which converts DiffSynth-Studio layouts, reads ``lora_alpha`` from metadata,
and injects mixed-rank adapters (64 on attention/FFN, 16 on AdaLN).

This module still validates files for the catalog/tests:

* keys contain ``lora_A`` / ``lora_B``
* ComfyUI-named files (``lora_unet_``, ``lora_up`` / ``lora_down``) are rejected
"""

from __future__ import annotations

import gc
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from peft import LoraConfig
from safetensors.torch import load_file as load_safetensors_file

logger = logging.getLogger(__name__)

LORA_TARGET_MODULES = (
    "to_q",
    "to_k",
    "to_v",
    "to_out.0",
    "ff.net.0.proj",
    "ff.net.2",
)
LORA_A_SUFFIX = ".lora_A.default.weight"
LORA_B_SUFFIX = ".lora_B.default.weight"


@dataclass(frozen=True)
class LoadedLoRA:
    adapter_name: str
    path: Path
    rank: int
    alpha: int
    scale: float
    fused: bool

    @property
    def effective_scale(self) -> float:
        return self.scale * self.alpha / self.rank


def _load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"LoRA checkpoint does not exist: {path}")
    if path.suffix.lower() != ".safetensors":
        raise ValueError(
            f"{path.name} is not a SafeTensors file. MiniMax-H3 LoRAs must be .safetensors."
        )
    checkpoint = load_safetensors_file(str(path), device="cpu")
    if isinstance(checkpoint, Mapping) and isinstance(checkpoint.get("state_dict"), Mapping):
        checkpoint = checkpoint["state_dict"]
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"Expected a state-dict mapping in {path}")
    return {str(key): value for key, value in checkpoint.items()}


def _normalize_keys(state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Rewrite common PEFT / Diffusers prefixes into ModelTC's expected names."""
    normalized: dict[str, torch.Tensor] = {}
    for key, tensor in state_dict.items():
        name = key
        for prefix in (
            "base_model.model.",
            "base_model.",
            "transformer.",
            "diffusion_model.",
        ):
            if name.startswith(prefix):
                name = name[len(prefix) :]
        if name.endswith(".lora_A.weight"):
            name = name[: -len(".lora_A.weight")] + LORA_A_SUFFIX
        elif name.endswith(".lora_B.weight"):
            name = name[: -len(".lora_B.weight")] + LORA_B_SUFFIX
        normalized[name] = tensor
    return normalized


def inspect_lora(path: Path) -> tuple[int, int]:
    """Return ``(rank, tensor_count)`` after validating a PEFT LoRA file."""
    state_dict = _normalize_keys(_load_state_dict(path))
    rank = _validate_state_dict(state_dict, path)
    return rank, len(state_dict)


def _validate_state_dict(state_dict: Mapping[str, torch.Tensor], path: Path) -> int:
    lora_a: dict[str, torch.Tensor] = {}
    lora_b: dict[str, torch.Tensor] = {}
    unsupported: list[str] = []
    comfy_keys = [
        key
        for key in state_dict
        if key.startswith("lora_unet_") or ".lora_up." in key or ".lora_down." in key
    ]
    if comfy_keys:
        raise ValueError(
            f"{path.name} looks like a ComfyUI LoRA ({comfy_keys[0]!r}). "
            "Use the Diffusers PEFT files from lightx2v/Minimax-h3-Turbo."
        )
    for key, tensor in state_dict.items():
        if key.endswith(LORA_A_SUFFIX) or key.endswith(".lora_A.weight"):
            lora_a[key.rsplit(".lora_A", 1)[0]] = tensor
        elif key.endswith(LORA_B_SUFFIX) or key.endswith(".lora_B.weight"):
            lora_b[key.rsplit(".lora_B", 1)[0]] = tensor
        else:
            unsupported.append(key)

    if unsupported:
        preview = ", ".join(unsupported[:4])
        raise ValueError(
            f"{path.name} is not a Diffusers PEFT LoRA. "
            "Use the LightX2V Diffusers files (not ComfyUI). "
            f"Unsupported keys: {preview}"
        )
    if not lora_a:
        raise ValueError(f"No {LORA_A_SUFFIX} tensors found in {path}")

    missing_a = sorted(lora_b.keys() - lora_a.keys())
    missing_b = sorted(lora_a.keys() - lora_b.keys())
    if missing_a or missing_b:
        raise ValueError(
            f"Unpaired LoRA tensors in {path.name}: missing A={missing_a[:3]}, missing B={missing_b[:3]}"
        )

    ranks: set[int] = set()
    for module_name, a_tensor in lora_a.items():
        b_tensor = lora_b[module_name]
        if a_tensor.ndim != 2 or b_tensor.ndim != 2:
            raise ValueError(f"LoRA tensors for {module_name} must be matrices")
        if a_tensor.shape[0] != b_tensor.shape[1]:
            raise ValueError(
                f"LoRA rank mismatch for {module_name}: A{tuple(a_tensor.shape)} B{tuple(b_tensor.shape)}"
            )
        ranks.add(int(a_tensor.shape[0]))
    # Official H3 LoRAs are mixed-rank: 64 on attention/FFN, 16 on AdaLN.
    return max(ranks)


def unload_loras(transformer: torch.nn.Module) -> None:
    peft_config = getattr(transformer, "peft_config", None)
    if not peft_config:
        return
    names = list(peft_config.keys())
    if hasattr(transformer, "delete_adapters"):
        transformer.delete_adapters(names)
    elif hasattr(transformer, "unload_lora"):
        transformer.unload_lora()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_lora_adapter(
    transformer: torch.nn.Module,
    lora_path: Path,
    *,
    lora_alpha: int,
    lora_scale: float = 1.0,
    adapter_name: str = "default",
    fuse_lora: bool = False,
) -> LoadedLoRA:
    """Inject and load a PEFT LoRA checkpoint into the H3 transformer."""
    state_dict = _normalize_keys(_load_state_dict(lora_path))
    rank = _validate_state_dict(state_dict, lora_path)

    existing = getattr(transformer, "peft_config", None) or {}
    if adapter_name in existing:
        if hasattr(transformer, "delete_adapters"):
            transformer.delete_adapters([adapter_name])

    transformer.add_adapter(
        LoraConfig(
            r=rank,
            lora_alpha=lora_alpha,
            init_lora_weights=False,
            target_modules=list(LORA_TARGET_MODULES),
            use_rslora=False,
        ),
        adapter_name=adapter_name,
    )

    if adapter_name != "default":
        state_dict = {
            key.replace(".default.weight", f".{adapter_name}.weight"): tensor
            for key, tensor in state_dict.items()
        }

    adapter_parameters = {
        name: parameter
        for name, parameter in transformer.named_parameters()
        if ".lora_A." in name or ".lora_B." in name
    }
    # Only check tensors that belong to this adapter.
    expected = {k: v for k, v in adapter_parameters.items() if f".{adapter_name}." in k}
    missing = sorted(expected.keys() - state_dict.keys())
    unexpected = sorted(state_dict.keys() - expected.keys())
    shape_mismatches = [
        (name, tuple(state_dict[name].shape), tuple(parameter.shape))
        for name, parameter in expected.items()
        if name in state_dict and state_dict[name].shape != parameter.shape
    ]
    if missing or unexpected or shape_mismatches:
        raise ValueError(
            "LoRA checkpoint is incompatible with MiniMax-H3 transformer: "
            f"missing={missing[:3]}, unexpected={unexpected[:3]}, "
            f"shape_mismatches={shape_mismatches[:3]}"
        )

    incompatible = transformer.load_state_dict(state_dict, strict=False)
    missing_lora = [
        key
        for key in incompatible.missing_keys
        if f".lora_A.{adapter_name}." in key or f".lora_B.{adapter_name}." in key
    ]
    if incompatible.unexpected_keys or missing_lora:
        raise RuntimeError(
            "LoRA loading did not consume the expected adapter tensors: "
            f"missing={missing_lora[:3]}, unexpected={incompatible.unexpected_keys[:3]}"
        )

    transformer.set_adapters(adapter_name, weights=lora_scale)
    if fuse_lora:
        transformer.fuse_lora(lora_scale=1.0, safe_fusing=True, adapter_names=[adapter_name])
        transformer.unload_lora()

    transformer.requires_grad_(False)
    transformer.eval()
    tensor_count = len(state_dict)
    del state_dict
    gc.collect()
    logger.info(
        "Loaded LoRA path=%s tensors=%s rank=%s alpha=%s scale=%s fused=%s",
        lora_path,
        tensor_count,
        rank,
        lora_alpha,
        lora_scale,
        fuse_lora,
    )
    return LoadedLoRA(
        adapter_name=adapter_name,
        path=lora_path,
        rank=rank,
        alpha=lora_alpha,
        scale=lora_scale,
        fused=fuse_lora,
    )


def set_adapter_weights(
    transformer: torch.nn.Module,
    names: list[str],
    weights: list[float],
) -> None:
    if not names:
        if hasattr(transformer, "disable_adapters"):
            transformer.disable_adapters()
        return
    transformer.set_adapters(names, weights=weights)
