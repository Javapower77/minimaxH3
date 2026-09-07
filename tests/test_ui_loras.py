from pathlib import Path

from minimax_h3_fl2v.config import AppConfig
from minimax_h3_fl2v.ui import _save_uploaded_lora, _stored_lora_choices


def test_stored_lora_choices_include_local_safetensors(tmp_path):
    cfg = AppConfig(lora_dir=tmp_path)
    (tmp_path / "style-a.safetensors").write_bytes(b"a")
    (tmp_path / "ignore.txt").write_text("x")
    choices = _stored_lora_choices(cfg)
    assert choices[0] == ("None (Turbo/catalog only)", None)
    assert ("style-a.safetensors", str((tmp_path / "style-a.safetensors").resolve())) in choices
    assert all("ignore.txt" not in label for label, _ in choices)


def test_uploaded_lora_is_persisted_and_selected(tmp_path):
    incoming = tmp_path / "incoming"
    incoming.mkdir()
    upload = incoming / "custom.safetensors"
    upload.write_bytes(b"weights")
    lora_dir = tmp_path / "loras"
    cfg = AppConfig(lora_dir=lora_dir)
    update, cleared = _save_uploaded_lora(cfg, str(upload))
    assert (lora_dir / "custom.safetensors").read_bytes() == b"weights"
    assert update.value == str((lora_dir / "custom.safetensors").resolve())
    assert cleared is None