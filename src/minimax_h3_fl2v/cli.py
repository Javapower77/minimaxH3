"""Command-line FL2VA generation."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .config import GenerationRequest, load_config
from .offline import enforce_offline_runtime
from .pipeline import MiniMaxH3Engine
from .resolution import SUPPORTED_ASPECT_RATIOS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate MiniMax-H3 FL2VA video+audio locally.",
    )
    parser.add_argument("--prompt", required=True, help="Scene + audio description")
    parser.add_argument("--first-image", type=Path, default=None)
    parser.add_argument("--last-image", type=Path, default=None)
    parser.add_argument("--lora-id", default=None)
    parser.add_argument("--extra-lora", type=Path, default=None, help="Optional extra .safetensors LoRA")
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--megapixels", type=float, default=None)
    parser.add_argument(
        "--aspect-ratio",
        choices=["auto", *SUPPORTED_ASPECT_RATIOS],
        default="auto",
        help="auto preserves the reference image ratio without cropping",
    )
    parser.add_argument("--nfe", type=int, default=None, help="Transformer evaluations (4–8 for turbo)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora-scale", type=float, default=1.0)
    parser.add_argument("--no-structured-prompt", action="store_true")
    parser.add_argument("--load-only", action="store_true", help="Load weights and exit")
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    enforce_offline_runtime()
    config = load_config()
    engine = MiniMaxH3Engine(config)
    engine.load()
    if args.load_only:
        print(engine.status)
        return

    lora_id = args.lora_id or config.default_lora_id
    spec = config.lora_by_id(lora_id)
    request = GenerationRequest(
        prompt=args.prompt,
        first_image=args.first_image,
        last_image=args.last_image,
        duration_seconds=args.duration,
        megapixels=args.megapixels if args.megapixels is not None else spec.megapixels,
        aspect_ratio=args.aspect_ratio,
        nfe=args.nfe if args.nfe is not None else spec.nfe,
        seed=args.seed,
        lora_id=lora_id,
        extra_lora_path=args.extra_lora,
        lora_scale=args.lora_scale,
        structured_prompt=not args.no_structured_prompt,
    )
    result = engine.generate(request)
    print(
        f"Saved {result.path}  {result.width}x{result.height}  "
        f"{result.frames}f/{result.duration:.2f}s  nfe={result.nfe}  "
        f"{result.elapsed_seconds:.1f}s  lora={result.lora}"
    )


if __name__ == "__main__":
    main()
