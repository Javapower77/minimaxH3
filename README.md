# MiniMax-H3 FL2VA local studio

Local **first / last frame → video + stereo audio** inference for
**[MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3)** on a single
**Azure Linux H100 80 GB** VM.

This is **not a chat LLM**. MiniMax-H3 is an open video+audio generator
(T2VA / I2VA / FL2VA). Text is a **prompt**, not a conversation. The
Qwen3-VL-32B encoder is used only to condition the DiT.

| Item | Value |
| --- | --- |
| Base model | `MiniMaxAI/MiniMax-H3` (Apache-2.0) |
| Default workflow | `fl2va` (loads `transformer/`, **not** `transformer_ref/`) |
| Default turbo LoRA | LightX2V FL2VA 8-step v1.0 **768p** SafeTensors |
| UI | Gradio 6 (local, no share, no analytics) |
| Weights | SafeTensors + PEFT LoRA |
| Python | **3.11 venv** |
| GPU | 1× NVIDIA H100 80 GB, CUDA 12.8, bf16, CPU offload |

Official sources:

- Model card: [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3)
- Diffusers API: [MiniMax-H3 pipeline](https://huggingface.co/docs/diffusers/main/en/api/pipelines/minimax_h3)
- Turbo LoRAs: [lightx2v/Minimax-h3-Turbo](https://huggingface.co/lightx2v/Minimax-h3-Turbo)
- Community LoRA: [larryvrh/MiniMax-H3-Turbo-Lora](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora)

## What FL2VA does

| Mode | Inputs | Output |
| --- | --- | --- |
| **T2VA** | text prompt | ~5–15 s video + audio |
| **I2VA** | prompt + first frame | video starts from that image |
| **FL2VA** | prompt + first + last frame | video interpolates between the two stills |

Constraints from the official model card:

- 24 FPS, duration **5–15 s**
- frame count `17n + 5` (shortest clip = **124 frames ≈ 5.17 s**)
- native **768p**, 16:9 / 9:16
- native audio **48 kHz stereo**
- **no CFG**
- text encoder is **Qwen3-VL-32B**

## Why CPU offload on one H100

A single 80 GB H100 **cannot** hold full bf16 MiniMax-H3 + Qwen3-VL-32B.
The official single-GPU recipe is Diffusers `ComponentsManager` auto CPU
offload with a **12 GB** reserve. This project uses that recipe.

Optional faster path (not the default): load
`lightx2v/MiniMax-H3-int8c` (INT8 transformer, ~33 GB) and skip CPU
offload. See [docs/AZURE_H100.md](docs/AZURE_H100.md).

## Quick start (Azure Linux + H100)

```bash
cd /mnt/disk2TB/minimaxH3
bash scripts/azure_h100_setup.sh
bash scripts/setup_venv.sh
source .venv/bin/activate
cp .env.example .env          # HF_TOKEN is only for this download step
python scripts/download_models.py --all   # last online step
python scripts/smoke_test.py
python app.py                 # Gradio stays local → http://<vm-ip>:7860
```

CLI:

```bash
python generate.py \
  --prompt "$(cat examples/fl2va_prompt.txt)" \
  --first-image /path/to/first.png \
  --last-image /path/to/last.png \
  --lora-id fl2va_turbo_8step_768p \
  --duration 5 \
  --aspect-ratio 16:9 \
  --nfe 8 \
  --seed 42
```

MP4 files land in `outputs/`.

## Layout

```
minimaxH3/
├── app.py / generate.py
├── configs/default.yaml          # H100 runtime
├── configs/loras.yaml            # SafeTensors LoRA catalog
├── src/minimax_h3_fl2v/          # pipeline, LoRA, Gradio, CLI
├── scripts/setup_venv.sh         # Python 3.11 + torch cu128
├── scripts/download_models.py    # FL2VA partition + LoRAs
├── scripts/azure_h100_setup.sh
├── docs/                         # Azure, LoRA, prompting, troubleshooting
└── models/                       # created on download (gitignored)
```

## Recommended H100 settings

| Setting | Value |
| --- | --- |
| LoRA | FL2VA Turbo 8-step v1.0 768p |
| NFE | 8 (scheduler grid = 9; the extra point is terminal sigma) |
| `lora_alpha` | 128 |
| `lora_scale` | 1.0 |
| video shift | 6 |
| audio shift | 3 |
| canvas | 1.0 MP 16:9 → **1376×768** |
| duration | 5 s (124 frames) |
| attention | `_flash_3`, then local SDPA (no Hub kernels) |
| offload | on, `memory_reserve_margin=12GB` |

4-step 768p is faster and slightly softer. Base model (no LoRA) wants **~50 NFE**.

## LoRA rules

- Only **Diffusers PEFT** files (`*.lora_A.default.weight` / `*.lora_B.default.weight`).
- **Do not** load ComfyUI-named files from the same Hub repo.
- Targets: `to_q`, `to_k`, `to_v`, `to_out.0`, `ff.net.0.proj`, `ff.net.2`.
- Gradio keeps adapters **unfused** so you can swap LoRAs without reloading 60 GB.
- Optional second `.safetensors` (style) stacks on the turbo adapter.
- UI uploads are copied once to `models/loras/` and remain available in the
  **Stored extra LoRA** dropdown. Use **Refresh** after copying a file there
  outside the UI.

Details: [docs/LORA.md](docs/LORA.md).

## Reference geometry / no crop

When a first or last reference frame is present, its actual width:height ratio
always controls the output canvas. Named presets no longer override it. H3
requires a 32-pixel spatial grid, so width/height are rounded to that grid while
minimizing ratio error. Both keyframes are then **contained** on that canvas:
the full source remains visible, and any small mismatch is letterboxed or
pillarboxed instead of stretched or cropped.

## Prompting

H3-Context-IR likes a compact multimodal block. The UI checkbox wraps a
plain sentence into that format **without calling another model**.

See [docs/PROMPTING.md](docs/PROMPTING.md) and `examples/fl2va_prompt.txt`.

## Offline + unfiltered

Runtime (`app.py` / `generate.py`) is **air-gapped**:

- loads only `models/MiniMax-H3` and `models/loras`
- `local_files_only=True`, `HF_HUB_OFFLINE=1`
- no Gradio share tunnel, no Gradio analytics, no Google Fonts CDN
- no FlashAttention Hub kernel (`_flash_3_hub` is rejected)
- no safety checker, NSFW filter, watermark, or prompt blacklist

The **only** network path is `python scripts/download_models.py` while you
still have Hub access. After that, inference does not contact Hugging Face.

## Security / networking

Gradio binds **`0.0.0.0:7860`** (all NICs) with `share=False` — no Gradio
tunnel. From a browser on the IP allowed in NSG:

```text
http://<vm-public-ip>:7860
```

On this VM that is typically `http://20.70.201.100:7860`.

NSG must allow TCP **7860** from your client IP only (not `0.0.0.0/0`).
SSH tunnel still works if you prefer not to open the NSG:

```bash
ssh -L 7860:127.0.0.1:7860 azureuser@<vm-public-ip>
```

## License

This studio code is Apache-2.0. MiniMax-H3 weights are Apache-2.0.
Turbo LoRAs keep their upstream licenses (LightX2V / community).
