from minimax_h3_fl2v.config import load_config


def test_default_h100_config():
    cfg = load_config()
    assert cfg.workflow == "fl2va"
    assert cfg.cpu_offload is True
    assert cfg.memory_reserve_margin == "20GB"
    assert cfg.dtype == "bfloat16"
    assert cfg.default_lora_id == "fl2va_turbo_8step_768p"
    assert cfg.server_name == "0.0.0.0"
    assert cfg.server_port == 7860
    assert cfg.share is False
    spec = cfg.lora_by_id(cfg.default_lora_id)
    assert spec.nfe == 8
    assert spec.lora_alpha == 128
    assert spec.filename.endswith(".safetensors")
