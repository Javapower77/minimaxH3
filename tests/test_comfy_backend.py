import pytest
import json
from pathlib import Path

from minimax_h3_fl2v.comfy_backend import PrunedComfyBackend
from minimax_h3_fl2v.config import AppConfig, GenerationRequest, LoRASpec
from minimax_h3_fl2v.pipeline import MiniMaxH3Engine, _ProgressBridge


def test_pruned_workflow_contains_first_last_and_stacked_loras():
    graph = PrunedComfyBackend.build_workflow(
        prompt="test",
        width=1344,
        height=768,
        frames=124,
        seed=42,
        nfe=8,
        loras=[("rank144.safetensors", 1.0), ("style.safetensors", 0.4)],
        first_image="first.png",
        last_image="last.png",
    )
    assert graph["1"]["inputs"]["unet_name"] == "minimax_h3_fl2va_pruned_bf16.safetensors"
    assert graph["21"]["class_type"] == "LoraLoaderModelOnly"
    assert graph["22"]["inputs"]["model"] == ["21", 0]
    assert graph["9"]["inputs"]["first_frame"] == ["7", 0]
    assert graph["9"]["inputs"]["last_frame"] == ["8", 0]
    assert graph["12"]["inputs"]["steps"] == 8
    assert graph["11"]["inputs"]["sampler_name"] == "euler"


def test_pruned_catalog_routes_without_loading_diffusers(monkeypatch):
    spec = LoRASpec(
        id="pruned",
        name="Pruned",
        filename="pruned.safetensors",
        backend="comfy_pruned",
    )
    config = AppConfig(catalog=[spec])
    engine = MiniMaxH3Engine(config)
    expected = object()
    monkeypatch.setattr(engine, "_load_unlocked", lambda: (_ for _ in ()).throw(AssertionError("Diffusers loaded")))
    monkeypatch.setattr(
        "minimax_h3_fl2v.comfy_backend.PrunedComfyBackend.generate",
        lambda self, request, selected, progress_callback=None: expected,
    )
    assert engine.generate(GenerationRequest(prompt="test", lora_id="pruned")) is expected


def test_sync_loras_links_new_files_and_checks_live_discovery(tmp_path, monkeypatch):
    comfy = tmp_path / "ComfyUI"
    source = tmp_path / "new-style.safetensors"
    source.write_bytes(b"weights")
    backend = PrunedComfyBackend(AppConfig())
    backend.root = comfy

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return [source.name]

    monkeypatch.setattr("minimax_h3_fl2v.comfy_backend.requests.get", lambda *args, **kwargs: Response())
    backend._sync_loras([source])

    linked = comfy / "models/loras" / source.name
    assert linked.is_symlink()
    assert linked.resolve() == source.resolve()


def test_diffusers_progress_bridge_reports_sampling_steps(monkeypatch):
    events = []
    times = iter([10.0, 11.0, 12.0, 13.0])
    monkeypatch.setattr("minimax_h3_fl2v.pipeline.time.perf_counter", lambda: next(times))
    bridge = _ProgressBridge(2, lambda fraction, message: events.append((fraction, message)), 10.0)
    with bridge:
        bridge.update()
        bridge.update()
    assert "step 1/2" in events[0][1]
    assert "50%" in events[0][1]
    assert "step 2/2" in events[1][1]
    assert events[1][0] == pytest.approx(0.8)
    assert "Sampling complete" in events[2][1]


def test_pruned_progress_stays_monotonic_during_timeout_heartbeat(tmp_path, monkeypatch):
    backend = PrunedComfyBackend(AppConfig(output_dir=tmp_path))
    backend.output_dir = tmp_path
    output = tmp_path / "result.mp4"
    output.write_bytes(b"video")
    events = []

    class Socket:
        def __init__(self):
            self.calls = 0

        def recv(self):
            self.calls += 1
            if self.calls == 1:
                return json.dumps(
                    {"type": "executing", "data": {"prompt_id": "job", "node": "15"}}
                )
            raise __import__("websocket").WebSocketTimeoutException()

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "job": {
                    "status": {"status_str": "success"},
                    "outputs": {"19": {"videos": [{"filename": output.name}]}}
                }
            }

    monkeypatch.setattr("minimax_h3_fl2v.comfy_backend.requests.get", lambda *a, **k: Response())
    path, _ = backend._wait_for_output(
        "job",
        socket=Socket(),
        started=0.0,
        progress_callback=lambda fraction, message: events.append((fraction, message)),
        timeout=1,
    )
    assert path == output.resolve()
    assert all(right[0] >= left[0] for left, right in zip(events, events[1:]))
