#!/usr/bin/env -S uv run
# /// script
# dependencies = ["tqdm"]
# ///
"""Fake video diffusion model for lite-runner demos and tests.

Mimics a real generative model's stdout: checkpoint loading messages,
tqdm denoise progress, per-step loss, decode timing, and attention
sparsity metric. Useful as a stand-in for asciicast recordings and
integration tests without GPU or model weights.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

from tqdm import tqdm


def main() -> None:
    """Run a fake video generation model for testing."""
    parser = argparse.ArgumentParser(description="Fake video diffusion model")
    parser.add_argument("--prompt", required=True, help="Text prompt")
    parser.add_argument("--output-path", required=True, help="Output video path")
    parser.add_argument("--debug-output", default=None, help="Debug artifact path")
    parser.add_argument(
        "--image", nargs="+", default=None, help="Input image path + frame + strength"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=-3.2)
    parser.add_argument("--mode", choices=["calib", "fast", "quality"], default="calib")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    rng = random.Random(args.seed)  # noqa: S311  # deterministic demo, not crypto

    print("Loading checkpoint: fake-video-diffusion-v2.safetensors", flush=True)
    time.sleep(0.2)
    print("Moved model to cuda:0  (13.4 GiB / 24.0 GiB allocated)", flush=True)
    time.sleep(0.1)
    print("Text encoder, VAE, transformer ready", flush=True)
    print(f"prompt: {args.prompt!r}", flush=True)
    print(
        f"seed={args.seed}  mode={args.mode}  threshold={args.threshold}",
        flush=True,
    )
    print(
        "UserWarning: torch.cuda.amp.autocast is deprecated, "
        "use torch.amp.autocast instead",
        file=sys.stderr,
        flush=True,
    )

    num_steps = 30
    loss = 0.0
    pbar = tqdm(range(num_steps), desc="denoise", unit="step", file=sys.stderr)
    for step in pbar:
        time.sleep(0.04)
        loss = 0.30 * (1.0 - step / num_steps) ** 2 + 0.005 * rng.random()
        pbar.set_postfix(loss=f"{loss:.4f}")
    pbar.close()

    print(f"denoise complete, final loss={loss:.4f}", flush=True)

    time.sleep(0.2)
    decode_ms = int(rng.uniform(280, 340))
    print(f"decoded 121 frames in {decode_ms}ms", flush=True)
    skipped = rng.uniform(28, 36)
    print(f"attention sparsity: skipped={skipped:.1f}% of tokens", flush=True)

    out = Path(args.output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(b"\x00" * 1024)
    print(f"saved video to {out}", flush=True)
    if args.debug_output:
        dbg = Path(args.debug_output)
        dbg.parent.mkdir(parents=True, exist_ok=True)
        dbg.write_bytes(b"\x00" * 512)
        print(f"saved debug tensor to {dbg}", flush=True)

    meta = {
        "prompt": args.prompt,
        "seed": args.seed,
        "threshold": args.threshold,
        "final_loss": loss,
    }
    Path("model_metadata.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
