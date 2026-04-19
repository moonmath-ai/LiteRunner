# lite-runner cookbook

Recipes for the situations that aren't covered by the 80% patterns
in `SKILL.md`. Each one is self-contained and can be copy-pasted.

## PEP 723 single-file shebang script

Ship a `run.py` that `chmod +x`-and-run works, with dependencies
declared inline:

```python
#!/usr/bin/env -S uv run
# /// script
# dependencies = [
#     "lite-runner @ git+https://github.com/moonmath-ai/LiteRunner",
# ]
# ///
"""Run config for my model."""

from lite_runner import Param, Runner

runner = Runner(command="python my_model.py", params=[Param("prompt")])

if __name__ == "__main__":
    runner.run()
```

`uv` auto-installs the dependencies into an ephemeral venv.

## Defaulting a multi-value param with some values unset

Use `UNSET` inside a list literal. The user sees the other defaults
pre-filled in the TUI and is prompted for the UNSET slot:

```python
from lite_runner import UNSET, Param

Param(
    "image",
    type=["path-image", "int", "float"],
    labels=["path", "frame", "strength"],
    default=[UNSET, 0, 0.8],
)
```

If the user also leaves the path blank at the TUI, the entire param
becomes `UNSET` and is omitted from the subprocess command (the image
flag won't be passed at all).

## Env vars: set and unset

```python
Runner(
    command="python generate.py",
    env={
        "CUDA_VISIBLE_DEVICES": "0",   # set
        "TORCH_COMPILE": "1",          # set
        "NOISY_DEBUG_VAR": None,       # unset (even if inherited)
    },
)
```

`None` values are removed from the subprocess env, even if they were
present in the parent `os.environ`.

## Branching a base runner for multiple experiments

Pipeline methods return fresh copies; you can parse CLI once and then
fan out:

```python
base = runner.parse_cli()              # parse sys.argv[1:]
for seed in [42, 99, 123]:
    base.override(seed=seed).run(no_interactive=True)
```

`base` itself is untouched; each `.override(...)` returns a new Runner
with `seed` source-tagged `"override"`.

## Metadata per run in a sweep

`with_metadata()` updates project / group / tags:

```python
for variant, tags in [("baseline", ["v1"]), ("tuned", ["v2", "fast"])]:
    (runner
        .with_metadata(tags=tags)
        .override(variant=variant)
        .run(no_interactive=True))
```

Each call is independent; tags don't accumulate across iterations.

## Scraping a progress bar with timedelta

tqdm-style progress bars print the same line many times; the last line
shows the final elapsed time. Combined with `last match wins`, this
works out of the box:

```python
from lite_runner import Metric

Metric(
    "stage1_time",
    pattern=r"40/40 \[(\d\d:\d\d)<",   # e.g. "40/40 [03:12<00:00, ...]"
    type="timedelta",                  # "03:12" → 192.0 seconds
)
```

## `UNSET` for a path-typed required input

If your model has a required conditioning input that you sometimes
want to skip (e.g. pure text-to-video vs. image-to-video), declare a
`path-image` param with `default=UNSET` and the TUI skip sentinel `-`:

```python
Param("ref_image", type="path-image", default=UNSET, prompt=True)
```

When the user types `-` at the prompt (or on the CLI: `--ref-image=-`),
the flag is omitted from the command and the param's config value is
logged as `"<unset>"`. No file upload is attempted.

## Multi-value `UNSET` on the CLI

Every element must be `-` to unset a multi-value param from the CLI:

```bash
./run.py --image - - -      # unsets the whole image param
```

Partial unsetting (e.g. `--image photo.jpg - 0.8`) is not supported;
any `-` in the list makes the whole param `UNSET`.

## Skipping the interactive TUI for some params only

`prompt=False` disables interactive prompting for one param (it still
appears on the CLI and in logs):

```python
Param("threshold", type="float", default=-3.0, prompt=False)
```

The user can still pass `--threshold 2.5`. When not passed and not
overridden, it falls through to the default without asking.

## No W&B (local JSON only)

Either pass `--no-wandb` on the CLI, or set `no_wandb=True` in `run()`:

```python
runner.run(no_wandb=True)
```

`JsonBackend` still writes `<output_dir>/run_info.json` with config,
metrics, summary, and a list of logged files.

## `--dry-run`

Prints the command without executing it. Uses `DryRunBackend`, which
logs every intended action to stderr. Useful for validating a sweep
before committing to GPU hours:

```bash
./run.py --prompt "a cat" --dry-run
```

## Reading back results programmatically

`run()` returns a `RunResult`:

```python
result = runner.run()
print(result.output_dir)        # Path to this run's directory
print(result.exit_code)         # 0 on success
print(result.param_sources)     # {"prompt": "cli", "seed": "default", ...}
```

`param_sources` values: `"cli"`, `"override"`, `"default"`, `"fixed"`,
`"prompt"`.

## Testing runners without hitting W&B

The upstream test suite mocks W&B by patching `lite_runner.backends`
and `lite_runner.runner._collect_git_info`. Minimal pattern:

```python
from unittest.mock import patch

def test_my_runner(tmp_path):
    with (
        patch("lite_runner.runner._collect_git_info", return_value={
            "repo": "test-repo", "commit": "abc", "branch": "main", "dirty": False,
        }),
        patch("lite_runner.backends.create_repo_archive", return_value=None),
        patch("lite_runner.backends.create_repo_diff", return_value=None),
        patch("lite_runner.runner.RUNS_DIR", tmp_path / "lite_runs"),
    ):
        runner.override(prompt="a cat").run(no_interactive=True, no_wandb=True)
```

Pass `no_wandb=True` to avoid `wandb.init`.

## Wrap a distributed launcher (`torchrun`, `accelerate launch`, `mpirun`)

You wrap the **launcher**, not the inner training script. Pass
`command=` as a list so the launcher's own flags don't get
shlex-split:

```python
from lite_runner import Param, Runner

runner = Runner(
    command=[
        "torchrun",
        "--nproc_per_node=8",
        "--rdzv_backend=c10d",
        "train.py",
    ],
    params=[
        Param("config", type="path", help="Training config YAML"),
        Param("lr", type="float", default=1e-4),
        Param("batch-size", type="int", default=32),
        Param("epochs", type="int", default=10),
        # final checkpoint, written by train.py into $output
        Param("checkpoint", value="$output/model.pt", type="path-artifact"),
    ],
    env={
        "NCCL_DEBUG": "INFO",
        "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
        "TORCH_DISTRIBUTED_DEBUG": "DETAIL",
    },
    tags=["distributed", "8gpu"],
)
```

Notes:

- The list-form `command=` is essential — `shlex.split("torchrun --nproc_per_node=8 train.py")` would still work here, but for any launcher whose args contain `=`, spaces, or shell metachars, the list form is the only safe option.
- `env={...}` flows to **all** worker processes spawned by `torchrun`.
- The inner `train.py` still receives every `Param` as a CLI flag. Make sure your training script accepts `--lr`, `--batch-size`, `--checkpoint`, etc. (`argparse` is fine.)

## Benchmark / perf regression tracking

`lite-runner`'s git snapshot ties every metric to a commit, so you
get commit-to-perf traceability in the W&B UI for free:

```python
from lite_runner import Metric, Output, Param, Runner

runner = Runner(
    command=["hyperfine", "--export-json", "results.json", "--warmup", "3"],
    params=[
        Param("target", help="Binary or shell command to benchmark"),
    ],
    metrics=[
        # hyperfine prints lines like:
        #   Time (mean ± σ):     123.4 ms ±   5.6 ms    [User: ...]
        #   Range (min … max):   110.0 ms … 140.0 ms    10 runs
        Metric("mean_ms",  pattern=r"Time \(mean ± σ\):\s+([\d.]+) ms"),
        Metric("stddev_ms", pattern=r"Time \(mean ± σ\):\s+[\d.]+ ms ±\s+([\d.]+) ms"),
        Metric("min_ms",   pattern=r"Range \(min … max\):\s+([\d.]+) ms"),
    ],
    outputs=[
        Output("results.json", log_as="artifact",
               copy_to="$output/results.json"),
    ],
    run_group="perf-regression",
    tags=["benchmark"],
)
```

Run this in CI on every commit. In the W&B UI, group by `run_group`
and plot `mean_ms` against `git/commit` to spot regressions. Pair
with `--no-interactive` so CI doesn't hang on missing params.

## LLM evaluation harness (lm-eval / HELM / custom)

Eval harnesses typically print a metrics table at the end and write
a results JSON. Scrape the table; archive the JSON.

```python
from lite_runner import Metric, Output, Param, Runner

runner = Runner(
    command=["lm_eval"],
    params=[
        Param("model", default="hf"),
        Param("model_args", help="e.g. pretrained=meta-llama/Llama-3-8B"),
        Param("tasks", help="Comma-separated task list, e.g. mmlu,hellaswag"),
        Param("batch-size", default="auto"),
        Param("limit", type="int", default=None, prompt=False),
        # lm_eval writes results to --output_path; point it at $output
        Param("output_path", value="$output", type="path"),
    ],
    metrics=[
        # lm-eval prints a markdown-ish table:
        #   |hellaswag|acc       |↑  |0.7421|±  |0.0044|
        # Adjust the regex to match your harness's exact format and
        # add one Metric per task you care about.
        Metric("hellaswag_acc",
               pattern=r"\|hellaswag\s*\|acc\s*\|.*?\|\s*([\d.]+)\s*\|"),
        Metric("mmlu_acc",
               pattern=r"\|mmlu\s*\|acc\s*\|.*?\|\s*([\d.]+)\s*\|"),
    ],
    outputs=[
        # Whole results dir as one zip artifact
        Output("$output/results", log_as="zip", name="lm-eval-results"),
    ],
    tags=["llm-eval"],
)
```

Two caveats:

- `Metric` is flat — you get one named scalar per pattern. For a
  multi-task eval, declare one `Metric` per task you care about, or
  scrape the JSON post-run with your own glue.
- `last match wins` works in your favor: lm-eval prints intermediate
  progress, but the final summary table is last, so the scraped
  number is the final score.

## Common anti-patterns, expanded

### ❌ Passing both `value=` and `default=`

```python
Param("x", value="$output/x.mp4", default="fallback.mp4")   # don't
```

`value=` makes it fixed (never prompted, never parsed from CLI).
`default=` is only used when the param is promptable. Combining them
is ambiguous; `value=` wins.

### ❌ Regex without a capture group

```python
Metric("loss", pattern=r"loss=[\d.]+")                      # don't
Metric("loss", pattern=r"loss=([\d.]+)")                    # do
```

`re.findall` returns the full match for patterns without groups and
the capture for patterns with one group; your caster will then receive
`"loss=0.123"` instead of `"0.123"` and explode.

### ❌ Using a directory with `log_as="image"`

```python
Output("frames/", log_as="image")   # warns, uploads every file individually
Output("frames/", log_as="zip")     # zips the directory (usually what you want)
```

For a bare directory, the runner logs a warning and uploads each file
one-by-one; you usually want `"zip"`.

### ❌ Relying on first-match semantics for metrics

```python
# Model prints: loss=1.0 ... loss=0.5 ... loss=0.1
Metric("loss", pattern=r"loss=([\d.]+)")   # stored as 0.1 (last), NOT 1.0
```

If you want the first occurrence, scrape it yourself post-run rather
than relying on `Metric`.

### ❌ Mutating the base runner in a sweep loop

```python
base = runner.parse_cli()
for seed in seeds:
    base.param_values["seed"] = seed   # don't touch internals
    base.run()
```

Use `override()`:

```python
for seed in seeds:
    base.override(seed=seed).run(no_interactive=True)
```

### ❌ Forgetting to interpolate `$output` in `copy_to`

```python
Output("foo.json", log_as="artifact", copy_to="foo.json")          # writes to cwd
Output("foo.json", log_as="artifact", copy_to="$output/foo.json")  # writes to run dir
```

`copy_to` without `$output` writes to whatever the current working
directory happens to be at post-run time. Almost always a bug.
