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


def test_civitai_multistep_ranks_are_catalogued():
    cfg = load_config()
    expected = {
        "dasiwa_multistep_r48_pruned": 48,
        "dasiwa_multistep_r96_pruned": 96,
        "dasiwa_multistep_r144_pruned": 144,
        "dasiwa_multistep_r512_pruned": 512,
    }
    for lora_id, rank in expected.items():
        spec = cfg.lora_by_id(lora_id)
        assert f"r{rank}_pruned" in spec.filename
        assert spec.nfe == 8
        assert spec.video_shift == 12.0
        assert spec.backend == "comfy_pruned"
        assert "pruned ComfyUI backend" in spec.notes
