# LiteRunner

[![Tests][tests-badge]][tests-link]
[![codecov][codecov-badge]][codecov-link]
[![PyPI version][pypi-version-badge]][pypi-link]
[![PyPI platforms][pypi-platforms-badge]][pypi-link]
[![Total downloads][pepy-badge]][pepy-link]
\
[![Made Using tsvikas/python-template][template-badge]][template-link]
[![GitHub Discussion][github-discussions-badge]][github-discussions-link]
[![PRs Welcome][prs-welcome-badge]][prs-welcome-link]

## Overview

Add MLOps-style experiment tracking to any CLI command, without modifying the model's code.

Got a command (say, a generative model) that runs from the terminal? Write a small
`run.py` declaring its inputs, outputs, and metrics; `lite-runner` then upgrades each
run in four ways without touching the source code:

1. **History of every run.** Inputs, outputs, logs, code, and metadata are captured
   locally to `~/lite_runs/`. Optionally uploaded to a backend of your choice (currently
   supporting Weights & Biases).
1. **Better ergonomics.** Set your own defaults per param. Fill in missing values via an
   interactive TUI instead of editing the command line. Keep one `run.py` per scenario
   so you don't re-remember the right flags.
1. **Bridge from terminal to Python.** Drive the CLI from Python, so a sweep is a plain
   `for` loop instead of a shell script.
1. **No lock-in.** Underneath, `lite-runner` builds and runs a normal shell command,
   which is logged with every run. You can copy-paste it and reproduce the run without
   `lite-runner` in the loop.

## Quick start

Two ways to get a `run.py` for your model:

**1. Let Claude Code write it.** Install the bundled
[Claude Code](https://claude.com/claude-code) skill (`lite-runner`), which knows the API
and writes idiomatic `run.py` scripts (and sweeps) for any command you point it at:

Inside Claude Code:

```text
/plugin marketplace add moonmath-ai/LiteRunner
/plugin install lite-runner@lite-runner-marketplace
```

**2. Write it by hand.** Create `run.py`:

```python
#!/usr/bin/env -S uv run
# /// script
# dependencies = ["lite-runner"]
# ///
from lite_runner import Runner, Param, Metric

runner = Runner(
    command="python generate.py",
    params=[
        Param("prompt", help="Text prompt"),
        Param("seed", type="int", default=42),
        Param("output-path", value="$output/video.mp4", type="path-video"),
    ],
    metrics=[
        Metric("loss", pattern=r"loss=([\d.]+)"),
    ],
)

if __name__ == "__main__":
    runner.run()
```

Then run it (requires [uv](https://docs.astral.sh/uv/getting-started/installation/)):

```bash
chmod +x run.py
./run.py --prompt "a cat walking"           # interactive TUI fills missing params
./run.py --prompt "a cat" --no-interactive  # non-interactive, fail if missing
./run.py --prompt "a cat" --dry-run         # print command, don't run
./run.py --seed=-                           # unset a param (omit from command)
./run.py --image - - -                      # unset a multi-value param
```

## What it does

Each `runner.run()` call records the run to
`~/lite_runs/<project>/<timestamp>_<run_name>/`, and mirrors it to Weights & Biases
when enabled:

1. Parses CLI args (all params are optional in argparse; missing ones trigger TUI prompts)
1. Creates the output directory
1. Saves a code snapshot under `code/` (git archive, plus `dirty.patch` if there are uncommitted changes)
1. Copies input files under `input/`
1. Builds and runs the subprocess, streaming stdout/stderr to terminal and to log files
1. Extracts metrics from stdout via regex
1. Records output files (videos, images, artifacts)
1. Writes `run_info.json` with params, git info, metadata, metrics, and exit status

## When to use this

`lite-runner` is built for the case where you're running a model someone else wrote
(HuggingFace, a GitHub clone, a vendor binary) that you can't edit, or don't want to.
It wraps the existing CLI, so the model code stays untouched.

Reach for it when:

- You want params, outputs, and metrics tracked without adding tracking code to the model.
- You want interactive prompts when you forget a param, not an argparse error.
- You want sweeps as a plain Python for-loop, not a YAML config.
- You want a code snapshot saved with every run, even when the code isn't yours.

Reach for something else when:

- You own the model code and are happy adding tracking calls directly: W&B's native SDK
  is more powerful (per-step metrics, custom charts, system metrics).
- You need hierarchical, composable configs: [Hydra](https://hydra.cc/) is built for that.
- You need a server-side sweep scheduler that hands out jobs across machines: use
  [`wandb sweep`](https://docs.wandb.ai/guides/sweeps/).

## Param

<!-- blacken-docs:off -->

```python
Param("name")                               # basic string param
Param("seed", type="int", default=42)       # typed with default
Param("mode", choices=["fast", "quality"])  # select from choices
Param("verbose", type="bool")               # --verbose flag (store_true, always defaults to False)
Param("image", type="path-image")           # file input, uploaded to W&B before run
Param(
    "output-path",
    value="$output/video.mp4",              # fixed value, $output interpolated
    type="path-video",
)                                           # after the run, uploaded to W&B as a video
Param(
    "input-image",
    type=["path-image", "float", "float"],  # multi-value flag
    labels=["img", "start", "strength"],
)                                           # each part prompted separately in TUI
```

<!-- blacken-docs:on -->

**Type** — controls parsing, casting, and file upload intent.
All param values are logged to `run.config`.
The `path-*` variants additionally upload the *file at that path* to W&B:

| Type              | Parsed as | File upload |
| ----------------- | --------- | ----------- |
| `"str"` (default) | `str`     | —           |
| `"int"`           | `int`     | —           |
| `"float"`         | `float`   | —           |
| `"bool"`          | flag      | —           |
| `"path"`          | `str`     | —           |
| `"path-image"`    | `str`     | as image    |
| `"path-video"`    | `str`     | as video    |
| `"path-artifact"` | `str`     | as artifact |
| `"path-text"`     | `str`     | as text     |

`"bool"` params are special: they generate a `--flag` (no value), always default
to `False` (any other `default=` is ignored), and cannot appear in multi-value type lists.

**Other fields:**

- `value=` makes a param fixed (never prompted, not in CLI). Can be a callable (called at resolve time).
- `default=` can be a callable (called at resolve time to compute the default)
- `flag=` overrides the CLI flag name (default: `--<name with hyphens>`)
- `prompt=False` skips interactive prompting (falls through to default). Requires a `default=`. The param still accepts CLI flags and is logged normally.
- `$output` in value is replaced with the run's output directory
- `log_when=` auto-inferred: `"before"` for inputs, `"after"` for `$output` paths
- `type=[...]` gives per-element types for multi-value flags (nargs inferred from length)
- Pass `-` on CLI to unset a param (omit it from the subprocess command).
  For single-value: `--seed=-`. For multi-value: `--image - - -` (one `-` per element).
  This mirrors typing `-` at the interactive TUI prompt.

## Output

For files the model writes to uncontrolled locations:

```python
Output("model_metadata.json", log_as="artifact", copy_to="$output/model_metadata.json")
```

Supports glob patterns and directory zipping:

<!-- blacken-docs:off -->

```python
Output("debug/**/*.png", log_as="image")                # upload each matched png
Output("debug/", log_as="image")                        # upload each file in directory
Output("debug/", log_as="zip")                          # zip entire directory, upload as artifact
Output("$output/frames/*.jpg", log_as="zip")            # zip glob matches into archive
Output("weights/", log_as="zip", name="model-weights")  # name= sets the W&B key (disambiguates zips)
```

<!-- blacken-docs:on -->

## Metric

Extract values from stdout:

```python
Metric("loss", pattern=r"loss=([\d.]+)")
Metric("status", pattern=r"status: (\w+)", type="str")
Metric("steps_per_sec", pattern=r"steps/s=([\d.]+)", type="int")
Metric(
    "elapsed", pattern=r"elapsed: ([\d:.]+)", type="timedelta"
)  # [[HH:]MM:]SS[.ddd] → seconds
```

Last match wins. Patterns are matched against both stdout and stderr. Stored in `wandb.run.summary`.

Supported types: `"float"` (default), `"int"`, `"str"`, `"timedelta"`.

## Sweeps

Loop with `override()`. Runs are grouped in W&B for easy comparison:

```python
runner = Runner(
    command="python gen.py",
    params=[...],
    run_group="lr-sweep",  # groups all runs together in W&B UI
)
for lr in [1e-3, 1e-4, 1e-5]:
    runner.override(learning_rate=lr).run(no_interactive=True)
```

Each call creates a separate W&B run, all grouped under the same `group`.

You can also update metadata per-run:

```python
runner.override(seed=42).with_metadata(tags=["baseline"]).run()
```

## Runner options

<!-- blacken-docs:off -->

```python
Runner(
    command="python gen.py",  # str or list[str] (list avoids shell splitting)
    params=[...],
    outputs=[...],
    metrics=[...],
    tags=["experiment-1"],    # W&B run tags
    env={
        "CUDA_VISIBLE_DEVICES": "0",
        "NOISY_VAR": None,
    },                        # set or unset env vars
    secret_env={
        "HF_TOKEN": "hf_xxx",
    },                        # like env, but redacted in logs / recorded config
    project="my-project",     # default: git repo name
    run_group="my-sweep",     # W&B run group for sweeps (None = no grouping)
)
```

<!-- blacken-docs:on -->

## Pipeline API

Each method returns a new `Runner` (immutable copies), so you can branch:

<!-- blacken-docs:off -->

```python
base = runner.parse_cli()    # parse sys.argv
r1 = base.override(seed=42)  # override params by name
r2 = base.override(seed=99)
r1.run()                     # auto-resolves defaults & prompts
r2.run()
```

<!-- blacken-docs:on -->

Methods:

| Method                                       | Description                                   |
| -------------------------------------------- | --------------------------------------------- |
| `parse_cli(argv)`                            | Parse CLI args (default: `sys.argv[1:]`)      |
| `override(**kwargs)`                         | Set param values by name                      |
| `with_metadata(project=, run_group=, tags=)` | Update W&B metadata                           |
| `resolve_defaults()`                         | Apply defaults and fixed values               |
| `ask_user(no_interactive=)`                  | Prompt for missing values                     |
| `run(...)`                                   | Auto-calls any unapplied steps, then executes |

`run()` accepts kwargs `dry_run`, `min_free_space_gib`, `no_interactive`, `no_wandb`, `project`, `run_name` as alternatives to CLI flags. It returns a `RunResult` with fields: `output_dir`, `exit_code`, `duration`, `run_name`, `project`, `config`, `param_values`, `param_sources`.

## Built-in CLI flags

| Flag                     | Description                                   |
| ------------------------ | --------------------------------------------- |
| `--dry-run`              | Print command and exit                        |
| `--min-free-space-gib N` | Minimum free disk space in GiB (default: 1.0) |
| `--no-interactive`       | Fail if required params missing               |
| `--no-wandb`             | Skip W&B logging (still logs to JSON)         |
| `--run-name NAME`        | Override W&B run name                         |
| `--project NAME`         | Override project name                         |

## What gets logged

Every run writes to `~/lite_runs/<project>/<timestamp>_<run_name>/`:
`run_info.json` (structured metadata), `run.log` / `stdout.log` / `stderr.log`,
the code snapshot under `code/` (`source.tar.gz` + `dirty.patch` if the repo has
uncommitted changes), copies of input files under `input/`,
and any output files the command produces.

When W&B is enabled (the default), the same data is also uploaded:

| Location                       | Content                                                                                  |
| ------------------------------ | ---------------------------------------------------------------------------------------- |
| `run.config["param/*"]`        | All param values                                                                         |
| `run.config["param_source/*"]` | Where each param value came from (cli, default, fixed, override, prompt)                 |
| `run.config["git/*"]`          | commit, branch, repo, dirty                                                              |
| `run.config["meta/*"]`         | hostname, user, cwd, datetime, command, full_command, output_dir, env (secrets as `***`) |
| `run.summary`                  | exit_code, duration_seconds, status, metrics                                             |
| Artifacts                      | Log files, code snapshot, artifact-type outputs                                          |
| Media                          | Videos and images from `path-*` type params/outputs                                      |

In `run_info.json` these are under top-level keys `metadata`, `config`, `metrics`,
`summary`, and `files_logged`.

## Contributing

Interested in contributing?
See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and guideline.

[codecov-badge]: https://codecov.io/gh/moonmath-ai/LiteRunner/graph/badge.svg
[codecov-link]: https://codecov.io/gh/moonmath-ai/LiteRunner
[github-discussions-badge]: https://img.shields.io/static/v1?label=Discussions&message=Ask&color=blue&logo=github
[github-discussions-link]: https://github.com/moonmath-ai/LiteRunner/discussions
[pepy-badge]: https://img.shields.io/pepy/dt/lite-runner
[pepy-link]: https://pepy.tech/project/lite-runner
[prs-welcome-badge]: https://img.shields.io/badge/PRs-welcome-brightgreen.svg
[prs-welcome-link]: https://opensource.guide/how-to-contribute/
[pypi-link]: https://pypi.org/project/lite-runner/
[pypi-platforms-badge]: https://img.shields.io/pypi/pyversions/lite-runner
[pypi-version-badge]: https://img.shields.io/pypi/v/lite-runner
[template-badge]: https://img.shields.io/badge/%F0%9F%9A%80_Made_Using-tsvikas%2Fpython--template-gold
[template-link]: https://github.com/tsvikas/python-template
[tests-badge]: https://github.com/moonmath-ai/LiteRunner/actions/workflows/ci.yml/badge.svg
[tests-link]: https://github.com/moonmath-ai/LiteRunner/actions/workflows/ci.yml
