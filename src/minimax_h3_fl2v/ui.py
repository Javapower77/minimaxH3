"""Gradio studio for MiniMax-H3 FL2VA + LoRA."""

from __future__ import annotations

import logging
import os
import shutil
import socket
import threading
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

from .offline import enforce_offline_runtime

enforce_offline_runtime()

import gradio as gr

from .config import AppConfig, GenerationRequest, load_config
from .pipeline import MiniMaxH3Engine, get_engine
from .prompts import describe_mode
from .resolution import PRESET_LABELS, SUPPORTED_ASPECT_RATIOS

logger = logging.getLogger(__name__)

LISTEN_HOST = "0.0.0.0"


def _nic_ipv4() -> Optional[str]:
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.connect(("1.1.1.1", 80))
        ip = sock.getsockname()[0]
        sock.close()
        return ip
    except OSError:
        return None


def _imds_get(url: str, timeout: float = 2.0) -> Optional[str]:
    req = urllib.request.Request(url, headers={"Metadata": "true"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read().decode("utf-8", errors="ignore").strip()
    except (urllib.error.URLError, TimeoutError, OSError):
        return None


def _azure_public_ipv4() -> Optional[str]:
    import json

    nic = _imds_get(
        "http://169.254.169.254/metadata/instance/network/interface/0/"
        "ipv4/ipAddress/0/publicIpAddress?api-version=2021-02-01&format=text"
    )
    if nic:
        return nic
    payload = _imds_get("http://169.254.169.254/metadata/loadbalancer?api-version=2020-10-01")
    if not payload:
        return None
    try:
        data = json.loads(payload)
        addrs = data.get("loadbalancer", {}).get("publicIpAddresses") or []
        for item in addrs:
            ip = str(item.get("frontendIpAddress") or "").strip()
            if ip:
                return ip
    except json.JSONDecodeError:
        return None
    return None


def _print_listen_urls(port: int) -> None:
    print(f"* Bound on {LISTEN_HOST}:{port} (all interfaces, no Gradio share tunnel)")
    nic = _nic_ipv4()
    if nic:
        print(f"* Private NIC:  http://{nic}:{port}")
    public = _azure_public_ipv4()
    if public:
        print(f"* Public URL:   http://{public}:{port}")
        print("  Open that URL from the client IP allowed in your Azure NSG.")
    else:
        print(f"* Public URL:   http://<vm-public-ip>:{port}")

CSS = """
:root { --body-bg: #0b0d12; }
.gradio-container { font-family: "IBM Plex Sans", "Segoe UI", sans-serif; }
#title-block h1 { letter-spacing: 0.04em; font-weight: 650; }
.hint { color: #9aa3b2; font-size: 0.92rem; }
"""


def _file_path(value) -> Optional[Path]:
    if value is None or value == "":
        return None
    if isinstance(value, (list, tuple)):
        return _file_path(value[0] if value else None)
    name = getattr(value, "name", None)
    if name:
        return Path(str(name))
    return Path(str(value))


def _lora_choices(config: AppConfig) -> list[tuple[str, str]]:
    choices = []
    for spec in config.catalog:
        mark = " ★" if spec.recommended else ""
        choices.append((f"{spec.name}{mark}", spec.id))
    return choices


def _stored_lora_choices(config: AppConfig) -> list[tuple[str, str | None]]:
    """List persistent local LoRAs, newest first, with an explicit None choice."""
    files = sorted(
        config.lora_dir.glob("*.safetensors"),
        key=lambda path: (path.stat().st_mtime, path.name.lower()),
        reverse=True,
    )
    return [("None (Turbo/catalog only)", None), *[(path.name, str(path.resolve())) for path in files]]


def _save_uploaded_lora(config: AppConfig, upload) -> tuple[gr.Dropdown, None]:
    source = _file_path(upload)
    if source is None:
        return gr.Dropdown(choices=_stored_lora_choices(config), value=None), None
    if source.suffix.lower() != ".safetensors":
        raise gr.Error("Only .safetensors LoRA files are accepted.")
    destination = (config.lora_dir / source.name).resolve()
    config.lora_dir.mkdir(parents=True, exist_ok=True)
    if source.resolve() != destination:
        shutil.copy2(source, destination)
    return (
        gr.Dropdown(choices=_stored_lora_choices(config), value=str(destination)),
        None,
    )


def build_app(config: Optional[AppConfig] = None) -> gr.Blocks:
    config = config or load_config()
    engine = get_engine(config)

    def load_model():
        try:
            engine.load()
            return f"Ready · {config.pretrained_path} · {config.device} · CPU offload={config.cpu_offload}"
        except Exception as exc:  # noqa: BLE001
            logger.exception("Model load failed")
            return f"Load failed: {exc}"

    def generate(
        prompt,
        first_image,
        last_image,
        lora_id,
        stored_lora,
        lora_upload,
        duration,
        preset,
        aspect_ratio,
        nfe,
        seed,
        lora_scale,
        extra_scale,
        structured,
        progress=gr.Progress(track_tqdm=False),
    ):
        if not prompt or not str(prompt).strip():
            raise gr.Error("Prompt is required.")
        if not engine.ready:
            progress(0.05, desc="Loading MiniMax-H3…")
            engine.load()

        extra_path = _file_path(stored_lora)
        uploaded = _file_path(lora_upload)
        if uploaded is not None:
            dest = config.lora_dir / uploaded.name
            if uploaded.resolve() != dest.resolve():
                shutil.copy2(uploaded, dest)
            extra_path = dest

        has_reference = bool(first_image or last_image)
        if preset and preset in PRESET_LABELS:
            megapixels, preset_aspect = PRESET_LABELS[preset]
            if aspect_ratio == "from preset" and not has_reference:
                aspect_ratio = preset_aspect
        else:
            megapixels = config.megapixels
        if has_reference:
            # Reference geometry always wins. The engine letterboxes the tiny
            # 32-grid rounding difference and never crops source pixels.
            aspect_ratio = "auto"
        elif aspect_ratio in {"from preset", "match reference (no crop)"}:
            aspect_ratio = "16:9"

        request = GenerationRequest(
            prompt=prompt,
            first_image=_file_path(first_image),
            last_image=_file_path(last_image),
            duration_seconds=float(duration),
            megapixels=float(megapixels),
            aspect_ratio=str(aspect_ratio),
            nfe=int(nfe),
            seed=int(seed),
            lora_id=lora_id,
            extra_lora_path=extra_path,
            extra_lora_scale=float(extra_scale),
            lora_scale=float(lora_scale),
            structured_prompt=bool(structured),
        )
        progress(0.15, desc=f"{describe_mode(first_image, last_image)} · sampling")
        result = engine.generate(request)
        summary = (
            f"{result.mode.upper()}  {result.width}×{result.height}  "
            f"{result.frames} frames ({result.duration:.2f}s)  "
            f"NFE={result.nfe}  seed={result.seed}  "
            f"{result.elapsed_seconds:.1f}s  LoRA={result.lora}\n"
            f"{result.path}"
        )
        return str(result.path), summary

    theme = gr.themes.Soft(
        primary_hue="amber",
        secondary_hue="slate",
        neutral_hue="slate",
    ).set(
        body_background_fill="#0b0d12",
        block_background_fill="#141821",
        border_color_primary="#2a3140",
    )

    with gr.Blocks(title=config.title) as demo:
        gr.Markdown(
            f"# {config.title}\n"
            "Local **MiniMax-H3 FL2VA** — first/last-frame to **video + stereo audio**. "
            "Offline, unfiltered. LoRAs are Diffusers PEFT **SafeTensors**. "
            "Tuned for a single **H100 80 GB** with CPU offload."
        )
        status = gr.Textbox(label="Engine", value="Click Load model, or generate (auto-loads).", interactive=False)
        with gr.Row():
            load_btn = gr.Button("Load model", variant="secondary")
            load_btn.click(load_model, outputs=status)

        with gr.Row():
            with gr.Column(scale=5):
                prompt = gr.Textbox(
                    label="Prompt",
                    lines=6,
                    placeholder=(
                        "A woman in a rust-red coat walks through neon rain toward a subway entrance. "
                        "Handheld camera, shallow depth of field, sodium and cyan highlights. "
                        "Rain hiss, distant traffic, heels on wet concrete, no score."
                    ),
                )
                structured = gr.Checkbox(value=True, label="Wrap as H3-Context-IR (vision + soundscape + music)")
                with gr.Row():
                    first_image = gr.Image(label="First frame (optional)", type="filepath")
                    last_image = gr.Image(label="Last frame (optional, FL2VA)", type="filepath")
                with gr.Row():
                    stored_lora = gr.Dropdown(
                        choices=_stored_lora_choices(config),
                        value=None,
                        label="Stored extra LoRA",
                        allow_custom_value=False,
                        scale=5,
                    )
                    refresh_loras = gr.Button("Refresh", scale=1)
                lora_upload = gr.File(
                    label="Upload new LoRA once (saved into models/loras)",
                    file_types=[".safetensors"],
                    type="filepath",
                )
            with gr.Column(scale=4):
                lora_id = gr.Dropdown(
                    choices=_lora_choices(config),
                    value=config.default_lora_id,
                    label="Turbo / catalog LoRA",
                )
                lora_notes = gr.Markdown(value=config.lora_by_id(config.default_lora_id).notes)
                preset = gr.Dropdown(
                    choices=list(PRESET_LABELS.keys()),
                    value="768p 16:9 (recommended)",
                    label="Canvas preset",
                )
                aspect_ratio = gr.Dropdown(
                    choices=["match reference (no crop)", "from preset", *SUPPORTED_ASPECT_RATIOS],
                    value="match reference (no crop)",
                    label="Aspect ratio",
                )
                duration = gr.Slider(5.0, 15.0, value=5.0, step=0.5, label="Duration (seconds, snapped to 17n+5 frames)")
                nfe = gr.Slider(4, 50, value=config.nfe, step=1, label="NFE (transformer steps)")
                lora_scale = gr.Slider(0.0, 1.5, value=1.0, step=0.05, label="Turbo LoRA strength")
                extra_scale = gr.Slider(0.0, 1.5, value=0.8, step=0.05, label="Extra LoRA strength")
                seed = gr.Number(value=42, precision=0, label="Seed")
                run_btn = gr.Button("Generate video + audio", variant="primary")

        video = gr.Video(label="Output MP4 (H.264 + AAC)", autoplay=True)
        summary = gr.Textbox(label="Run summary", lines=3)

        lora_id.change(
            lambda lora: (
                config.lora_by_id(lora).nfe,
                config.lora_by_id(lora).lora_scale,
                config.lora_by_id(lora).notes,
            ),
            inputs=lora_id,
            outputs=[nfe, lora_scale, lora_notes],
        )
        refresh_loras.click(
            lambda: gr.Dropdown(choices=_stored_lora_choices(config)),
            outputs=stored_lora,
        )
        lora_upload.upload(
            lambda upload: _save_uploaded_lora(config, upload),
            inputs=lora_upload,
            outputs=[stored_lora, lora_upload],
        )

        run_btn.click(
            generate,
            inputs=[
                prompt,
                first_image,
                last_image,
                lora_id,
                stored_lora,
                lora_upload,
                duration,
                preset,
                aspect_ratio,
                nfe,
                seed,
                lora_scale,
                extra_scale,
                structured,
            ],
            outputs=[video, summary],
        )

        gr.Markdown(
            "### Notes\n"
            "- **T2VA**: prompt only · **I2VA**: first frame · **FL2VA**: first + last frame.\n"
            "- A reference image controls the output ratio; only 32-pixel-grid rounding is applied. "
            "Frames are contained with padding, never cropped.\n"
            "- Turbo LoRAs are trained at **4–8 NFE**. Base model wants ~**50 NFE**.\n"
            "- Uploaded LoRAs persist in `models/loras`; select them later from **Stored extra LoRA**.\n"
            "- No Hub, no share tunnel, no safety checker. See `docs/AZURE_H100.md`."
        )

    # Gradio 6 moved these from Blocks(...) to launch(...).
    demo._minimax_theme = theme
    return demo


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    enforce_offline_runtime()
    config = load_config()
    demo = build_app(config)

    def _warmup():
        try:
            get_engine(config).load()
        except Exception:
            logger.exception("Background model load failed")

    threading.Thread(target=_warmup, name="h3-warmup", daemon=True).start()
    os.environ["GRADIO_SERVER_NAME"] = LISTEN_HOST
    port = int(config.server_port)
    _print_listen_urls(port)
    demo.queue(max_size=config.max_queue).launch(
        server_name=LISTEN_HOST,
        server_port=port,
        theme=getattr(demo, "_minimax_theme", None),
        css=CSS,
        share=False,
        ssr_mode=False,
        mcp_server=False,
        enable_monitoring=False,
        inbrowser=False,
        quiet=False,
        show_error=True,
        allowed_paths=[str(config.output_dir), str(config.lora_dir), str(config.upload_dir)],
    )


if __name__ == "__main__":
    main()
