#!/usr/bin/env -S uv run
# /// script
# dependencies = ["lite-runner"]
# ///
"""Example sweep: same prompt, varying seed."""

from lite_runner import Metric, Param, Runner

runner = Runner(
    command="./examples/fake_model.py",
    params=[
        Param("prompt", help="Text prompt for generation"),
        Param("seed", type="int", default=42, help="Random seed"),
        Param("output-path", value="$output/video.mp4", type="path-video"),
    ],
    metrics=[
        Metric("loss", pattern=r"final loss=([\d.]+)"),
        Metric("skipped_pct", pattern=r"skipped=([\d.]+)%"),
    ],
    tags=["sweep", "seed"],
    run_group="seed-sweep",  # groups all runs in W&B UI
)

if __name__ == "__main__":
    for s in [42, 123, 456, 789]:
        print(f"\n{'=' * 60}")
        print(f"SWEEP: seed={s}")
        print(f"{'=' * 60}\n")
        runner.override(seed=s).run(no_interactive=True)
