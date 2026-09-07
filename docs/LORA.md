# LoRA (SafeTensors / PEFT)

MiniMax-H3 turbo LoRAs are **PEFT adapters** stored as **SafeTensors**.
This studio loads them with Diffusers' official
`MiniMaxH3ModularPipeline.load_lora_weights` (same conversion path as
the Hugging Face MiniMax-H3 docs).

## File format

Accepted keys:

```
<module>.lora_A.default.weight
<module>.lora_B.default.weight
```

Also rewritten automatically:

| Incoming key | Normalized to |
| --- | --- |
| `base_model.model.*.lora_A.weight` | `*.lora_A.default.weight` |
| `transformer.*.lora_A.weight` | `*.lora_A.default.weight` |
| `diffusion_model.*.lora_A.weight` | `*.lora_A.default.weight` |

Rejected:

- ComfyUI-named files (`*_comfyui_*.safetensors`, `lora_unet_*`, `lora_up` / `lora_down`)
- non-`.safetensors` pickles
- unpaired A/B matrices

Official MiniMax-H3 LoRAs are **mixed-rank** (64 on attention/FFN, 16 on AdaLN). Diffusers' `load_lora_weights` injects those adapters and honors a `__metadata__` `alpha` when present. Alpha-less files load at `alpha == rank`.

Typical target modules:

```
to_q  to_k  to_v  to_out.0
ff.net.0.proj  ff.net.2
AdaLN / modulation projections (rank 16)
```

## Effective scale

```
effective_scale = lora_scale * lora_alpha / rank
```

LightX2V 8-step 768p files often record `alpha` in SafeTensors metadata
(Diffusers honors that). The 544p mixed-AR files typically load at
`alpha == rank`. Catalog `lora_alpha` values in `configs/loras.yaml` are
kept as documentation for the UI; runtime scaling is `lora_scale` on the
adapter plus whatever alpha the official loader applied.

`lora_scale` is the UI strength slider (default `1.0`).

## Catalog

| id | File | NFE | alpha | canvas |
| --- | --- | --- | --- | --- |
| `fl2va_turbo_8step_768p` | `minimax_h3_fl2v_turbo_8step_v1.0_768p_bf16.safetensors` | 8 | 128 | 1.0 MP |
| `fl2va_turbo_4step_768p` | `minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors` | 4 | 128 | 1.0 MP |
| `fl2va_turbo_8step` | `minimax_h3_fl2v_turbo_8step_v1.0_bf16.safetensors` | 8 | 8 | 0.5 MP |
| `fl2va_turbo_4step_v01` | `minimax_h3_fl2v_turbo_4step_v0.1.safetensors` | 4 | 8 | 0.5 MP |
| `larryvrh_turbo_v4` | `minimax_h3_turbo_v4_step600_ema.safetensors` | 6 | 8 | 1.0 MP |
| `none` | — | 50 | — | 1.0 MP |

Hub: [lightx2v/Minimax-h3-Turbo](https://huggingface.co/lightx2v/Minimax-h3-Turbo)
and [larryvrh/MiniMax-H3-Turbo-Lora](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora).

## Scheduler grid

MiniMax-H3's scheduler treats `num_inference_steps` as **sigma grid
points including terminal zero**. For **N** transformer evaluations the
code sends **N + 1**. The UI NFE slider is the human-facing N.

## Swapping and stacking

Gradio **does not fuse** LoRAs. Unfused adapters can be deleted and
replaced without reloading the 62 GB transformer.

You can stack:

1. Catalog turbo LoRA (`adapter_name=turbo`)
2. Extra uploaded `.safetensors` (`adapter_name=style`)

Do **not** stack a Ref2VA LoRA onto the FL2VA transformer. Ref2VA uses
`transformer_ref/` and is out of scope for this studio.

## Adding your own LoRA

1. Export a PEFT adapter as SafeTensors with the keys above.
2. Copy it to `models/loras/`.
3. Add an entry to `configs/loras.yaml`.
4. Restart Gradio.

Or upload it in the UI as **Extra style LoRA** (no catalog edit).

## Fuse (CLI only)

Fusing bakes A/B into the base weights and unloads the adapter. Faster
for a long batch of the **same** LoRA; you cannot swap afterwards
without reloading the pipeline. The Gradio path never fuses.
