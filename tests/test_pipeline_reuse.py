from pathlib import Path

from minimax_h3_fl2v.config import AppConfig, LoRASpec
from minimax_h3_fl2v.pipeline import MiniMaxH3Engine


class _Transformer:
    def named_parameters(self):
        return []

    def requires_grad_(self, value):
        return self


class _Pipe:
    def __init__(self):
        self.load_calls = []
        self.unload_calls = 0

    def unload_lora_weights(self):
        self.unload_calls += 1

    def load_lora_weights(self, path, **kwargs):
        self.load_calls.append((path, kwargs))


def test_second_generation_does_not_reactivate_unchanged_adapter(tmp_path):
    lora = tmp_path / "turbo.safetensors"
    lora.write_bytes(b"weights")
    spec = LoRASpec(id="turbo", name="Turbo", filename=lora.name)
    engine = MiniMaxH3Engine(AppConfig(lora_dir=tmp_path))
    engine.pipe = _Pipe()
    engine.transformer = _Transformer()
    activations = []
    engine._set_adapters_for_inference = lambda names, weights: (
        activations.append((tuple(names), tuple(weights))),
        setattr(engine, "_active_adapter_names", tuple(names)),
        setattr(engine, "_active_adapter_weights", tuple(weights)),
    )

    engine.apply_lora(spec, scale=1.0)
    engine.apply_lora(spec, scale=1.0)

    assert len(engine.pipe.load_calls) == 1
    assert activations == [(('turbo',), (1.0,))]


def test_scale_change_reactivates_without_reloading_weights(tmp_path):
    lora = tmp_path / "turbo.safetensors"
    lora.write_bytes(b"weights")
    spec = LoRASpec(id="turbo", name="Turbo", filename=lora.name)
    engine = MiniMaxH3Engine(AppConfig(lora_dir=tmp_path))
    engine.pipe = _Pipe()
    engine.transformer = _Transformer()
    activations = []

    def activate(names, weights):
        activations.append((tuple(names), tuple(weights)))
        engine._active_adapter_names = tuple(names)
        engine._active_adapter_weights = tuple(weights)

    engine._set_adapters_for_inference = activate
    engine.apply_lora(spec, scale=1.0)
    engine.apply_lora(spec, scale=0.75)

    assert len(engine.pipe.load_calls) == 1
    assert activations == [(('turbo',), (1.0,)), (('turbo',), (0.75,))]