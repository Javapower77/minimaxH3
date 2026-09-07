"""MiniMax-H3 FL2VA engine: load once, generate many, swap LoRAs."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch

from .config import AppConfig, GenerationRequest, LoRASpec
from .frames import duration_to_frames, frames_to_duration
from .lora import LoadedLoRA
from .media import fit_image_without_crop, load_rgb_image, save_result_video
from .offline import (
    disable_safety_guards,
    enforce_offline_runtime,
    local_attention_backend,
    pin_component_specs_to_local,
    require_local_snapshot,
    strip_hub_kwargs,
)
from .prompts import expand_prompt
from .resolution import resolve_output_size, size_from_image

logger = logging.getLogger(__name__)

REQUIRED_COMPONENTS = (
    "text_encoder",
    "tokenizer",
    "processor",
    "vae",
    "scheduler",
    "audio_scheduler",
    "transformer",
    "audio_vae",
)
ATTENTION_BACKEND_FALLBACKS = ("_flash_3",)


@dataclass
class GenerationResult:
    path: Path
    mode: str
    width: int
    height: int
    frames: int
    duration: float
    seed: int
    nfe: int
    elapsed_seconds: float
    lora: str
    prompt: str


class MiniMaxH3Engine:
    """Thread-safe single-GPU FL2VA runner."""

    def __init__(self, config: AppConfig):
        self.config = config
        self._lock = threading.Lock()
        self.pipe = None
        self.transformer = None
        self.components_manager = None
        self.ready = False
        self.status = "not loaded"
        self.active_lora_id: Optional[str] = None
        self.active_lora: Optional[LoadedLoRA] = None
        self._loaded_adapters: dict[str, Path] = {}
        self._active_adapter_names: tuple[str, ...] = ()
        self._active_adapter_weights: tuple[float, ...] = ()

    def load(self) -> None:
        with self._lock:
            self._load_unlocked()

    def _torch_dtype(self) -> torch.dtype:
        mapping = {
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float16": torch.float16,
            "fp16": torch.float16,
            "float32": torch.float32,
        }
        return mapping.get(self.config.dtype.lower(), torch.bfloat16)

    def _load_unlocked(self) -> None:
        if self.ready:
            return
        if self.config.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but no CUDA device is available.")

        from diffusers import ComponentsManager, ModularPipeline

        enforce_offline_runtime()
        model_id = str(require_local_snapshot(self.config.local_dir))
        self.status = f"loading {model_id} (workflow={self.config.workflow}, offline)"
        logger.info(self.status)
        dtype = self._torch_dtype()
        load_kwargs: dict = strip_hub_kwargs(
            {
                "workflow": self.config.workflow,
                "local_files_only": True,
            }
        )

        manager = None
        if self.config.cpu_offload:
            manager = ComponentsManager()
            load_kwargs["components_manager"] = manager

        pipe = ModularPipeline.from_pretrained(model_id, **load_kwargs)
        pin_component_specs_to_local(pipe, model_id)
        pipe.load_components(
            dtype=dtype,
            local_files_only=True,
            pretrained_model_name_or_path=model_id,
        )
        disable_safety_guards(pipe)
        missing = [name for name in REQUIRED_COMPONENTS if getattr(pipe, name, None) is None]
        if missing:
            raise RuntimeError(
                f"Failed to load required components from {model_id!r}: {', '.join(missing)}"
            )

        transformer = pipe.transformer
        transformer.requires_grad_(False)
        transformer.eval()
        self._set_attention_backend(transformer)

        if self.config.cpu_offload and manager is not None:
            manager.enable_auto_cpu_offload(
                device=self.config.device,
                memory_reserve_margin=self.config.memory_reserve_margin,
            )
        else:
            pipe.to(self.config.device)

        pipe.scheduler.set_shift(self.config.video_shift)
        pipe.audio_scheduler.set_shift(self.config.audio_shift)

        self.pipe = pipe
        self.transformer = transformer
        self.components_manager = manager
        self.ready = True
        self.status = "ready"
        logger.info("MiniMax-H3 FL2VA pipeline ready on %s", self.config.device)

    def _set_attention_backend(self, transformer) -> None:
        requested: list[str] = []
        backend = local_attention_backend(self.config.attention_backend)
        if backend:
            requested.append(backend)
        for name in ATTENTION_BACKEND_FALLBACKS:
            if name not in requested:
                requested.append(name)
        last_error: Optional[Exception] = None
        for name in requested:
            try:
                transformer.set_attention_backend(name)
                logger.info("Attention backend: %s", name)
                return
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                logger.warning("Attention backend %s unavailable (%s)", name, exc)
        logger.warning("Falling back to default SDPA attention (%s)", last_error)

    def _restore_adapter_dtype(self) -> None:
        """DiffSynth-style H3 LoRAs inject fp32 factors; keep the DiT in bf16."""
        transformer = getattr(self.pipe, "transformer", None)
        if transformer is None:
            return
        dtype = self._torch_dtype()
        for name, parameter in transformer.named_parameters():
            if "lora_" in name and parameter.dtype != dtype:
                parameter.data = parameter.data.to(dtype)

    def _set_adapters_for_inference(self, names: list[str], weights: list[float]) -> None:
        """Activate PEFT adapters without ever making their tensors trainable.

        Diffusers' pipeline-level ``set_adapters`` calls PEFT's
        ``BaseTunerLayer.set_adapter(names)`` with ``inference_mode=False``.
        On a second generation that attempts ``requires_grad_(True)`` on tensors
        already touched by ``torch.inference_mode`` and raises. Calling the PEFT
        layer API with ``inference_mode=True`` keeps all adapters frozen.
        """
        from peft.tuners.tuners_utils import BaseTunerLayer

        for module in self.transformer.modules():
            if not isinstance(module, BaseTunerLayer):
                continue
            module.set_adapter(names, inference_mode=True)
            for adapter_name, weight in zip(names, weights):
                module.set_scale(adapter_name, weight)
        self.transformer.requires_grad_(False)
        self._active_adapter_names = tuple(names)
        self._active_adapter_weights = tuple(weights)

    def apply_lora(
        self,
        spec: LoRASpec,
        *,
        scale: Optional[float] = None,
        extra_path: Optional[Path] = None,
        extra_scale: float = 1.0,
    ) -> str:
        if self.pipe is None:
            raise RuntimeError("Pipeline is not loaded")

        turbo_scale = scale if scale is not None else spec.lora_scale
        desired: dict[str, Path] = {}
        weights: dict[str, float] = {}
        if not spec.is_base:
            path = spec.resolved_path(self.config.lora_dir)
            if path is None or not path.is_file():
                raise FileNotFoundError(
                    f"LoRA file missing: {path}. Run: python scripts/download_models.py --loras"
                )
            desired["turbo"] = path.expanduser().resolve()
            weights["turbo"] = float(turbo_scale)
        if extra_path:
            extra = Path(extra_path).expanduser().resolve()
            if not extra.is_file():
                raise FileNotFoundError(f"Extra LoRA file missing: {extra}")
            desired["style"] = extra
            weights["style"] = float(extra_scale)

        reloaded = desired != self._loaded_adapters
        if reloaded:
            if hasattr(self.pipe, "unload_lora_weights"):
                self.pipe.unload_lora_weights()
            self._loaded_adapters = {}
            for adapter_name, path in desired.items():
                logger.info("Loading LoRA adapter=%s path=%s", adapter_name, path)
                self.pipe.load_lora_weights(
                    str(path),
                    adapter_name=adapter_name,
                    local_files_only=True,
                )
                self._loaded_adapters[adapter_name] = path
            self._restore_adapter_dtype()
            self._active_adapter_names = ()
            self._active_adapter_weights = ()

        self.active_lora_id = spec.id
        self.active_lora = None
        if "turbo" in desired:
            self.active_lora = LoadedLoRA(
                adapter_name="turbo",
                path=desired["turbo"],
                rank=0,
                alpha=spec.lora_alpha,
                scale=turbo_scale,
                fused=False,
            )

        if desired:
            names = list(desired)
            adapter_weights = [weights[name] for name in names]
            signature = (tuple(names), tuple(adapter_weights))
            if signature != (self._active_adapter_names, self._active_adapter_weights):
                self._set_adapters_for_inference(names, adapter_weights)
            if self.config.fuse_lora and reloaded:
                self.pipe.fuse_lora(adapter_names=names, lora_scale=1.0)
            return " + ".join(names)
        if hasattr(self.pipe, "disable_lora"):
            try:
                self.pipe.disable_lora()
            except Exception:  # noqa: BLE001
                pass
        self._active_adapter_names = ()
        self._active_adapter_weights = ()
        return "base"

    def generate(self, request: GenerationRequest) -> GenerationResult:
        if not self.ready:
            self.load()
        with self._lock:
            return self._generate_unlocked(request)

    def _generate_unlocked(self, request: GenerationRequest) -> GenerationResult:
        spec = self.config.lora_by_id(request.lora_id)
        video_shift = request.video_shift if request.video_shift is not None else spec.video_shift
        audio_shift = request.audio_shift if request.audio_shift is not None else spec.audio_shift
        nfe = request.nfe or spec.nfe
        megapixels = request.megapixels or spec.megapixels

        self.pipe.scheduler.set_shift(video_shift)
        self.pipe.audio_scheduler.set_shift(audio_shift)
        self.apply_lora(
            spec,
            scale=request.lora_scale,
            extra_path=request.extra_lora_path,
            extra_scale=request.extra_lora_scale,
        )

        first = load_rgb_image(request.first_image) if request.first_image else None
        last = load_rgb_image(request.last_image) if request.last_image else None
        anchor = first or last

        if anchor is not None:
            # A reference image is authoritative: presets/manual ratios never
            # override it. H3 still requires dimensions on a 32-pixel grid.
            width, height = size_from_image(anchor.width, anchor.height, megapixels)
        else:
            aspect = request.aspect_ratio if request.aspect_ratio != "auto" else self.config.aspect_ratio
            if aspect == "auto":
                aspect = "16:9"
            width, height = resolve_output_size(megapixels, aspect)

        # Bypass H3's built-in stretch/cover-crop branches. Both keyframes are
        # exactly canvas-sized, while their complete source content is retained.
        if first is not None:
            first = fit_image_without_crop(first, width, height)
        if last is not None:
            last = fit_image_without_crop(last, width, height)

        frames = duration_to_frames(request.duration_seconds, fps=self.config.fps)
        prompt = expand_prompt(request.prompt, structured=request.structured_prompt)
        # MiniMaxH3Scheduler counts the terminal sigma point, so N NFE needs N+1.
        scheduler_grid_points = int(nfe) + 1
        generator = torch.Generator().manual_seed(int(request.seed))

        kwargs: dict = {}
        if first is not None:
            kwargs["image"] = first
        if last is not None:
            kwargs["last_image"] = last

        self.status = f"generating {request.mode} {width}x{height} {frames}f nfe={nfe}"
        started = time.perf_counter()
        # ``no_grad`` is sufficient for inference and, unlike inference_mode,
        # does not permanently mark PEFT/offload tensors as inference-only.
        with torch.no_grad():
            result = self.pipe(
                prompt=prompt,
                height=height,
                width=width,
                num_frames=frames,
                num_inference_steps=scheduler_grid_points,
                generator=generator,
                output_type="np",
                output=["videos", "audio", "sampling_rate"],
                **kwargs,
            )
        elapsed = time.perf_counter() - started

        stamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"{stamp}_{request.mode}_seed{request.seed}_{width}x{height}.mp4"
        output_path = self.config.output_dir / filename
        save_result_video(result, output_path, fps=self.config.fps)
        del result
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.status = "ready"
        return GenerationResult(
            path=output_path,
            mode=request.mode,
            width=width,
            height=height,
            frames=frames,
            duration=frames_to_duration(frames, self.config.fps),
            seed=int(request.seed),
            nfe=int(nfe),
            elapsed_seconds=elapsed,
            lora=spec.name,
            prompt=prompt,
        )


_ENGINE: Optional[MiniMaxH3Engine] = None
_ENGINE_LOCK = threading.Lock()


def get_engine(config: Optional[AppConfig] = None) -> MiniMaxH3Engine:
    global _ENGINE
    with _ENGINE_LOCK:
        if _ENGINE is None:
            if config is None:
                from .config import load_config

                config = load_config()
            _ENGINE = MiniMaxH3Engine(config)
        return _ENGINE
