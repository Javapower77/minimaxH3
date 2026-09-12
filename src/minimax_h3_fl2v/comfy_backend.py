"""Local ComfyUI backend for pruned MiniMax-H3 FL2VA checkpoints."""

from __future__ import annotations

import io
import json
import logging
import os
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Optional

import requests
import websocket
from PIL import Image

from .config import AppConfig, GenerationRequest, LoRASpec, ROOT
from .frames import duration_to_frames, frames_to_duration
from .media import fit_image_without_crop, load_rgb_image
from .lora import validate_pruned_fl2va_lora
from .prompts import expand_prompt
from .resolution import resolve_output_size, size_from_image

logger = logging.getLogger(__name__)

COMFY_HOST = "127.0.0.1"
COMFY_PORT = 8188
COMFY_URL = f"http://{COMFY_HOST}:{COMFY_PORT}"
COMFY_ROOT = ROOT / ".runtime" / "ComfyUI"
COMFY_MODEL = "minimax_h3_fl2va_pruned_bf16.safetensors"
COMFY_CLIP = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
COMFY_VIDEO_VAE = "minimax_h3_video_vae_fp16.safetensors"
COMFY_AUDIO_VAE = "minimax_h3_audio_vae_fp32.safetensors"
ProgressCallback = Callable[[float, str], None]

NODE_STAGES = {
    "1": (0.08, "🔷 Loading pruned BF16 transformer"),
    "2": (0.11, "🔤 Loading Qwen3-VL text encoder"),
    "3": (0.13, "🎞️ Loading video VAE"),
    "4": (0.14, "🔊 Loading audio VAE"),
    "7": (0.15, "🖼️ Loading first frame"),
    "8": (0.16, "🖼️ Loading last frame"),
    "9": (0.19, "🧠 Encoding prompt and keyframes"),
    "10": (0.20, "🎲 Preparing seeded noise"),
    "11": (0.21, "⚙️ Selecting Euler sampler"),
    "12": (0.22, "📈 Building simple sigma schedule"),
    "13": (0.23, "🧭 Preparing guidance"),
    "15": (0.25, "⚡ Sampling video and audio latents"),
    "16": (0.84, "🎞️ Decoding video frames"),
    "17": (0.88, "🔊 Decoding stereo audio"),
    "18": (0.92, "🎬 Creating synchronized video"),
    "19": (0.96, "💾 Encoding H.264 MP4"),
}


class PrunedComfyBackend:
    """Submit pruned FL2VA jobs to an isolated loopback-only ComfyUI worker."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.root = Path(os.getenv("MINIMAX_H3_COMFY_ROOT", COMFY_ROOT)).resolve()
        self.url = os.getenv("MINIMAX_H3_COMFY_URL", COMFY_URL).rstrip("/")
        self.process: Optional[subprocess.Popen] = None
        self.output_dir = (config.output_dir / "comfyui").resolve()
        self.input_dir = (config.upload_dir / "comfyui").resolve()

    def _required_paths(self) -> list[Path]:
        return [
            self.root / ".venv/bin/python",
            self.root / "main.py",
            self.root / "models/diffusion_models" / COMFY_MODEL,
            self.root / "models/text_encoders" / COMFY_CLIP,
            self.root / "models/vae" / COMFY_VIDEO_VAE,
            self.root / "models/vae" / COMFY_AUDIO_VAE,
        ]

    def validate_installation(self) -> None:
        missing = [str(path) for path in self._required_paths() if not path.exists()]
        if missing:
            raise RuntimeError(
                "Pruned FL2VA backend is incomplete. Run "
                "python scripts/setup_pruned_backend.py first. Missing: " + ", ".join(missing)
            )

    def _is_ready(self) -> bool:
        try:
            response = requests.get(f"{self.url}/system_stats", timeout=2)
            return response.ok
        except requests.RequestException:
            return False

    def ensure_worker(
        self,
        timeout: float = 90.0,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> None:
        report = progress_callback or (lambda fraction, message: None)
        if self._is_ready():
            report(0.04, "🟣 Pruned worker ready · validating local models…")
            return
        report(0.02, "🟣 Starting isolated pruned ComfyUI worker…")
        self.validate_installation()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.input_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.root / "comfyui-worker.log"
        log_handle = log_path.open("ab")
        command = [
            str(self.root / ".venv/bin/python"),
            str(self.root / "main.py"),
            "--listen", COMFY_HOST,
            "--port", str(COMFY_PORT),
            "--disable-auto-launch",
            "--disable-all-custom-nodes",
            "--reserve-vram", "20",
            "--output-directory", str(self.output_dir),
            "--input-directory", str(self.input_dir),
        ]
        env = os.environ.copy()
        env.update({
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "DIFFUSERS_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
        })
        self.process = subprocess.Popen(
            command,
            cwd=self.root,
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._is_ready():
                logger.info("Pruned ComfyUI worker ready at %s", self.url)
                report(0.04, "🟣 Pruned worker started · validating local models…")
                return
            if self.process.poll() is not None:
                raise RuntimeError(f"Pruned ComfyUI worker exited. See {log_path}")
            time.sleep(0.5)
        raise RuntimeError(f"Timed out starting pruned ComfyUI worker. See {log_path}")

    def _upload_image(self, image: Image.Image, name: str) -> str:
        payload = io.BytesIO()
        image.save(payload, format="PNG")
        payload.seek(0)
        response = requests.post(
            f"{self.url}/upload/image",
            files={"image": (name, payload, "image/png")},
            data={"type": "input", "overwrite": "true"},
            timeout=60,
        )
        response.raise_for_status()
        result = response.json()
        return str(Path(result.get("subfolder", "")) / result["name"])

    def _sync_loras(self, paths: list[Path]) -> None:
        """Expose selected app LoRAs to ComfyUI and verify live discovery."""
        target_dir = self.root / "models/loras"
        target_dir.mkdir(parents=True, exist_ok=True)
        expected: set[str] = set()
        for value in paths:
            source = Path(value).expanduser().resolve()
            if not source.is_file():
                raise FileNotFoundError(f"LoRA file missing: {source}")
            expected.add(source.name)
            target = target_dir / source.name
            if target.is_symlink() and target.resolve() == source:
                continue
            if target.exists() or target.is_symlink():
                target.unlink()
            temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
            temporary.symlink_to(source)
            temporary.replace(target)

        response = requests.get(f"{self.url}/models/loras", timeout=30)
        response.raise_for_status()
        available = set(response.json())
        missing = sorted(expected - available)
        if missing:
            raise RuntimeError(
                "Pruned worker did not discover selected LoRA files: "
                + ", ".join(missing)
                + ". Restart the application or rerun scripts/setup_pruned_backend.py."
            )

    @staticmethod
    def build_workflow(
        *,
        prompt: str,
        width: int,
        height: int,
        frames: int,
        seed: int,
        nfe: int,
        loras: list[tuple[str, float]],
        first_image: Optional[str] = None,
        last_image: Optional[str] = None,
        filename_prefix: str = "pruned/minimax_h3",
    ) -> dict[str, dict[str, Any]]:
        graph: dict[str, dict[str, Any]] = {
            "1": {"class_type": "UNETLoader", "inputs": {"unet_name": COMFY_MODEL, "weight_dtype": "default"}},
            "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": COMFY_CLIP, "type": "minimax", "device": "default"}},
            "3": {"class_type": "VAELoader", "inputs": {"vae_name": COMFY_VIDEO_VAE}},
            "4": {"class_type": "VAELoader", "inputs": {"vae_name": COMFY_AUDIO_VAE}},
            "10": {"class_type": "RandomNoise", "inputs": {"noise_seed": int(seed)}},
            "11": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
            "16": {"class_type": "VAEDecode", "inputs": {"samples": ["15", 0], "vae": ["3", 0]}},
            "17": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["15", 0], "vae": ["4", 0]}},
            "18": {"class_type": "CreateVideo", "inputs": {"images": ["16", 0], "audio": ["17", 0], "fps": 24.0, "bit_depth": 8, "color_space": "sRGB"}},
            "19": {"class_type": "SaveVideo", "inputs": {"video": ["18", 0], "filename_prefix": filename_prefix, "format": "mp4", "codec": "h264"}},
        }
        model_node = "1"
        for index, (filename, strength) in enumerate(loras, start=1):
            node = str(20 + index)
            graph[node] = {
                "class_type": "LoraLoaderModelOnly",
                "inputs": {"model": [model_node, 0], "lora_name": filename, "strength_model": float(strength)},
            }
            model_node = node
        graph["12"] = {"class_type": "BasicScheduler", "inputs": {"model": [model_node, 0], "scheduler": "simple", "steps": int(nfe), "denoise": 1.0}}
        graph["13"] = {"class_type": "BasicGuider", "inputs": {"model": [model_node, 0], "conditioning": ["9", 0]}}
        graph["15"] = {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["10", 0], "guider": ["13", 0], "sampler": ["11", 0], "sigmas": ["12", 0], "latent_image": ["9", 1]}}
        condition_inputs: dict[str, Any] = {
            "clip": ["2", 0], "vae": ["3", 0], "prompt": prompt,
            "width": int(width), "height": int(height), "length": int(frames),
        }
        if first_image:
            graph["7"] = {"class_type": "LoadImage", "inputs": {"image": first_image}}
            condition_inputs["first_frame"] = ["7", 0]
        if last_image:
            graph["8"] = {"class_type": "LoadImage", "inputs": {"image": last_image}}
            condition_inputs["last_frame"] = ["8", 0]
        graph["9"] = {"class_type": "MiniMaxH3ImageToVideo", "inputs": condition_inputs}
        return graph

    def _wait_for_output(
        self,
        prompt_id: str,
        *,
        socket,
        started: float,
        progress_callback: Optional[ProgressCallback] = None,
        timeout: float = 3600.0,
    ) -> tuple[Path, dict[str, float]]:
        report = progress_callback or (lambda fraction, message: None)
        deadline = time.monotonic() + timeout
        timings: dict[str, float] = {}
        active_node: Optional[str] = None
        active_started = time.perf_counter()
        last_fraction = 0.08
        last_label = "⏳ Waiting in worker queue"
        last_heartbeat = 0.0

        def emit(fraction: float, label: str) -> None:
            nonlocal last_fraction, last_label
            last_fraction = max(last_fraction, float(fraction))
            last_label = label
            report(last_fraction, label)

        while time.monotonic() < deadline:
            try:
                raw = socket.recv()
                if isinstance(raw, str):
                    event = json.loads(raw)
                    event_type = event.get("type")
                    data = event.get("data", {})
                    if data.get("prompt_id") in {None, prompt_id}:
                        if event_type == "executing":
                            node = data.get("node")
                            now = time.perf_counter()
                            if active_node is not None:
                                timings[NODE_STAGES.get(active_node, (0, f"node_{active_node}"))[1]] = (
                                    timings.get(NODE_STAGES.get(active_node, (0, f"node_{active_node}"))[1], 0.0)
                                    + now - active_started
                                )
                            active_node = str(node) if node is not None else None
                            active_started = now
                            if active_node is not None:
                                fraction, label = NODE_STAGES.get(
                                    active_node, (0.24, f"🔧 Processing node {active_node}")
                                )
                                emit(fraction, f"{label} · elapsed {now - started:.1f}s")
                        elif event_type == "progress":
                            value = int(data.get("value", 0))
                            total = max(1, int(data.get("max", 1)))
                            fraction = value / total
                            emit(
                                0.25 + 0.57 * fraction,
                                f"⚡ Sampling step {value}/{total} · {fraction * 100:.0f}% · "
                                f"elapsed {time.perf_counter() - started:.1f}s",
                            )
                        elif event_type == "execution_error":
                            raise RuntimeError(
                                f"Pruned ComfyUI generation failed: {data.get('exception_message', data)}"
                            )
            except websocket.WebSocketTimeoutException:
                now = time.perf_counter()
                if now - last_heartbeat >= 5.0:
                    stage = last_label.split(" · elapsed", 1)[0]
                    emit(last_fraction, f"{stage} · still working · elapsed {now - started:.1f}s")
                    last_heartbeat = now
            response = requests.get(f"{self.url}/history/{prompt_id}", timeout=30)
            response.raise_for_status()
            history = response.json().get(prompt_id)
            if history:
                status = history.get("status", {})
                if status.get("status_str") == "error":
                    messages = status.get("messages", [])
                    raise RuntimeError(f"Pruned ComfyUI generation failed: {messages[-1] if messages else status}")
                for output in history.get("outputs", {}).values():
                    for key in ("videos", "files", "gifs", "images"):
                        for item in output.get(key, []):
                            filename = item.get("filename")
                            if not filename:
                                continue
                            path = self.output_dir / item.get("subfolder", "") / filename
                            if path.is_file() and path.suffix.lower() in {".mp4", ".mkv", ".webm"}:
                                now = time.perf_counter()
                                if active_node is not None:
                                    label = NODE_STAGES.get(active_node, (0, f"node_{active_node}"))[1]
                                    timings[label] = timings.get(label, 0.0) + now - active_started
                                return path.resolve(), timings
        raise TimeoutError(f"Timed out waiting for pruned generation {prompt_id}")

    def generate(
        self,
        request: GenerationRequest,
        spec: LoRASpec,
        progress_callback: Optional[ProgressCallback] = None,
    ):
        from .pipeline import GenerationResult

        report = progress_callback or (lambda fraction, message: None)
        overall_started = time.perf_counter()
        timings: dict[str, float] = {}
        self.ensure_worker(progress_callback=report)
        report(0.05, "🖼️ Preparing and uploading reference frames…")
        prepare_started = time.perf_counter()
        first = load_rgb_image(request.first_image) if request.first_image else None
        last = load_rgb_image(request.last_image) if request.last_image else None
        anchor = first or last
        megapixels = request.megapixels or spec.megapixels
        if anchor is not None:
            width, height = size_from_image(anchor.width, anchor.height, megapixels)
        else:
            aspect = request.aspect_ratio if request.aspect_ratio != "auto" else self.config.aspect_ratio
            width, height = resolve_output_size(megapixels, "16:9" if aspect == "auto" else aspect)
        first_name = last_name = None
        token = uuid.uuid4().hex
        if first is not None:
            first_name = self._upload_image(fit_image_without_crop(first, width, height), f"{token}-first.png")
        if last is not None:
            last_name = self._upload_image(fit_image_without_crop(last, width, height), f"{token}-last.png")
        frames = duration_to_frames(request.duration_seconds, fps=self.config.fps)
        nfe = int(request.nfe or spec.nfe)
        selected_loras = [(spec.filename, float(request.lora_scale))]
        selected_paths = [spec.resolved_path(self.config.lora_dir)]
        for path, scale in request.extra_loras:
            validate_pruned_fl2va_lora(Path(path))
            selected_loras.append((Path(path).name, float(scale)))
            selected_paths.append(Path(path))
        report(0.07, f"🧬 Synchronizing {len(selected_loras)} pruned LoRA adapter(s)…")
        self._sync_loras([Path(path) for path in selected_paths if path is not None])
        timings["prepare"] = time.perf_counter() - prepare_started
        graph = self.build_workflow(
            prompt=expand_prompt(request.prompt, structured=request.structured_prompt),
            width=width, height=height, frames=frames, seed=int(request.seed), nfe=nfe,
            loras=selected_loras, first_image=first_name, last_image=last_name,
            filename_prefix=f"pruned/{time.strftime('%Y%m%d-%H%M%S')}_{request.mode}_seed{request.seed}",
        )
        started = time.perf_counter()
        report(0.08, "📤 Submitting pruned workflow to local worker…")
        socket = websocket.create_connection(
            self.url.replace("http://", "ws://").replace("https://", "wss://")
            + f"/ws?clientId={token}",
            timeout=2,
        )
        response = requests.post(f"{self.url}/prompt", json={"prompt": graph, "client_id": token}, timeout=60)
        if not response.ok:
            raise RuntimeError(f"Pruned workflow rejected: {response.text}")
        try:
            output, worker_timings = self._wait_for_output(
                response.json()["prompt_id"],
                socket=socket,
                started=started,
                progress_callback=report,
            )
            timings.update(worker_timings)
        finally:
            socket.close()
            try:
                requests.post(
                    f"{self.url}/free",
                    json={"unload_models": True, "free_memory": True},
                    timeout=30,
                )
            except requests.RequestException:
                logger.warning("Could not request ComfyUI model unload", exc_info=True)
        elapsed = time.perf_counter() - started
        final = self.config.output_dir / f"{time.strftime('%Y%m%d-%H%M%S')}_{request.mode}_seed{request.seed}_{width}x{height}_pruned.mp4"
        final.parent.mkdir(parents=True, exist_ok=True)
        if output != final.resolve():
            shutil.copy2(output, final)
        timings["worker"] = elapsed
        timings["total"] = time.perf_counter() - overall_started
        report(1.0, f"✅ Complete · total {timings['total']:.1f}s · saved {final.name}")
        return GenerationResult(
            path=final, mode=request.mode, width=width, height=height, frames=frames,
            duration=frames_to_duration(frames, self.config.fps), seed=int(request.seed),
            nfe=nfe, elapsed_seconds=timings["total"], lora=spec.name,
            prompt=expand_prompt(request.prompt, structured=request.structured_prompt),
            timings=timings,
        )
