# lite-runner API reference

Complete field-by-field reference for the public API exported from
`lite_runner`. Cross-check your code against this page when a detail
doesn't appear in `SKILL.md`.

## Public re-exports (`lite_runner.__all__`)

| Name           | Kind     | What it is                                                  |
| -------------- | -------- | ----------------------------------------------------------- |
| `Runner`       | class    | Main orchestrator.                                          |
| `Param`        | dataclass| Declaration of one CLI parameter / fixed value.             |
| `Output`       | dataclass| Declaration of an extra output file (uncontrolled path).    |
| `Metric`       | dataclass| Regex scraped from stdout+stderr into `wandb.run.summary`.  |
| `RunResult`    | dataclass| Frozen result returned by `Runner.run()`.                   |
| `ParamType`    | `Literal`| The string literal type for single `Param.type` values.     |
| `UNSET`        | sentinel | Marker meaning "param explicitly skipped — omit from cmd".  |
| `LogBackend`   | Protocol | Structural type for custom backends.                        |
| `WandbBackend` | class    | Default Weights & Biases backend.                           |
| `JsonBackend`  | class    | Always-on local `run_info.json` backend.                    |

## `Param`

```python
@dataclass
class Param:
    name: str
    type: ParamType | Sequence[ParamType] = "str"
    default: object = None
    choices: list[str] | None = None
    help: str = ""
    flag: str | None = None
    value: object = None
    labels: list[str] | None = None
    log_when: str | None = None
    prompt: bool = True
```

### Fields

- `name` — Used as the argparse `dest` (`-` → `_`) and the CLI flag
  (default: `--<name with hyphens>`). Names clashing with built-in
  flags (`dry_run`, `min_free_space_gib`, `no_interactive`, `no_wandb`,
  `project`, `run_name`) raise `ValueError` at `Runner(...)`.
- `type` — One of `ParamType` for single-value, or a sequence of
  `ParamType` for multi-value flags. See "ParamType values" below.
  `"bool"` cannot appear inside a multi-value list.
- `default` — Used when neither CLI nor `override()` provides a value.
  May be a zero-arg callable; called at `resolve_defaults()` time.
  Ignored (with a warning) for `type="bool"`.
- `choices` — List of allowed string values. Shown as a `questionary.select`
  in the TUI. Combines with any scalar `type=`.
- `help` — Help string for both `--help` and TUI prompt label.
- `flag` — Override the CLI flag (default `--<name.replace("_","-")>`).
- `value` — Fixed value. Param is **never** prompted and **not** added
  to argparse. `$output` in the string is interpolated at runtime.
  May be a list for multi-value params. May be a zero-arg callable.
- `labels` — Per-element labels for multi-value params. Used as argparse
  metavar (shown in `--help`) and as prompt labels in the TUI. Length
  must match `len(type)`.
- `log_when` — `"before"` (input, uploaded pre-run) or `"after"` (output,
  uploaded post-run). Auto-inferred when `type` encodes an upload intent:
  `"after"` if the value contains `$output`, else `"before"`.
- `prompt` — `False` disables interactive prompting for this param.
  Requires `default=` (otherwise raises). The param still appears in
  argparse and is logged normally.

### Derived properties

- `Param.dest` — `name.replace("-", "_")`
- `Param.nargs` — `len(type)` if `type` is a sequence else `None`
- `Param.type_list` — types as a list (single-value wrapped)
- `Param.is_fixed` — `value is not None`

### `ParamType` values

| Value             | Parsed as | File uploaded to W&B |
| ----------------- | --------- | -------------------- |
| `"str"` (default) | `str`     | —                    |
| `"int"`           | `int`     | —                    |
| `"float"`         | `float`   | —                    |
| `"bool"`          | flag      | —                    |
| `"path"`          | `str`     | —                    |
| `"path-image"`    | `str`     | as `wandb.Image`     |
| `"path-video"`    | `str`     | as `wandb.Video` (format inferred from extension: gif/mp4/webm/ogg) |
| `"path-artifact"` | `str`     | as W&B artifact      |
| `"path-text"`     | `str`     | as `wandb.Html("<pre>...</pre>")` of the file's text |

`"bool"` params generate `--flag` (no value), always default to `False`,
and are not added to the command when `False`.

### `UNSET`

Typing `-` at a TUI prompt or passing `-` on the CLI sets the param to
`UNSET`. For multi-value params, `-` for any element unsets the whole
param. `UNSET` params are **omitted** from the built command and logged
as `"<unset>"` in `wandb.run.config`.

## `Output`

```python
@dataclass
class Output:
    path: str
    log_as: str = "artifact"
    name: str | None = None
    copy_to: str | None = None
```

- `path` — Absolute path, `$output`-relative path, or glob (`*`, `?`,
  `[...]`). Trailing `/` or directory paths trigger directory mode.
- `log_as` — `"video"`, `"image"`, `"artifact"`, `"text"`, or `"zip"`.
  With globs or directories: `"zip"` collects matches into a single zip
  uploaded as an artifact; other values upload each matched file
  individually. For a bare directory with a non-zip `log_as`, a warning
  is logged.
- `name` — Overrides the W&B key. Required to disambiguate multiple
  zips (duplicate labels raise).
- `copy_to` — Copy the file to this path (with `$output` interpolated)
  before logging. Not supported with glob patterns.

## `Metric`

```python
@dataclass
class Metric:
    name: str
    pattern: str
    type: str = "float"
```

- `pattern` — Regex with **exactly one capture group**, matched with
  `re.findall` against the concatenation `stdout_text + "\n" + stderr_text`.
  **Last match wins.**
- `type` — `"float"` (default), `"int"`, `"str"`, or `"timedelta"`.
  `"timedelta"` parses `[[HH:]MM:]SS[.ddd]` into total seconds (float):

  ```python
  "1:02:03.5" → 3723.5
  "05:30"     → 330.0
  "42"        → 42.0
  ```

  If casting fails, the raw string is stored.

## `Runner`

```python
@dataclass
class Runner:
    command: str | list[str]
    params: list[Param] = []
    outputs: list[Output] = []
    metrics: list[Metric] = []
    env: dict[str, str | None] = {}
    project: str | None = None
    run_group: str | None = None
    tags: list[str] = []
```

### Fields

- `command` — `str` is split via `shlex.split`; `list[str]` is used as-is.
- `env` — Extra env vars for the subprocess. `value=None` means
  **unset** the var (even if inherited from parent env).
- `project` — W&B project. Default: git repo name at runtime.
- `run_group` — W&B group (for sweeps).
- `tags` — W&B run tags.

### Pipeline methods (all return a new Runner)

| Method                                                    | Behavior |
| --------------------------------------------------------- | -------- |
| `copy()`                                                  | `copy.deepcopy(self)`. |
| `parse_cli(argv=None)`                                    | Parses CLI args (default `sys.argv[1:]`). Sets `param_sources[name] = "cli"`, `cli_parsed = True`. Does not overwrite existing `"override"` sources. |
| `override(**kwargs)`                                      | Sets param values by name (hyphens OR underscores). Sets `param_sources = "override"`. Unknown names raise `ValueError`. |
| `with_metadata(project=, run_group=, tags=)`              | Replaces the corresponding fields if non-None. |
| `resolve_defaults()`                                      | Fills fixed (`value=`) and default (`default=`) values. Doesn't overwrite cli/override. Sets `defaults_resolved = True`. Bool → `False`. Callable `default`/`value` called here. |
| `ask_user(no_interactive=None)`                           | Auto-calls `resolve_defaults()` if needed. Prompts for every promptable param (not fixed, `prompt=True`, not set via cli/override). In non-interactive mode with missing params: raises `ValueError` (which `run()` turns into `SystemExit(2)`). Sets `filled = True`. |

### `run(*, dry_run=None, min_free_space_gib=None, no_interactive=None, no_wandb=None, project=None, run_name=None) -> RunResult`

Auto-calls any unapplied pipeline steps, then executes:

1. Parses CLI if `cli_parsed` is False.
2. Merges `run()` kwargs over CLI flags (warns on conflict).
3. Checks disk free-space if `min_free_space_gib` is set.
4. Resolves defaults, then prompts (or errors if `no_interactive`).
5. Resolves `project` via kwarg → CLI flag → `self.project` → git repo
   name. If still `None`: `ValueError`.
6. Collects `meta/*`, `git/*`, `param/*` into `config`.
7. Inits backends: `DryRunBackend` if `--dry-run`; else `WandbBackend`
   (unless `--no-wandb`) + `JsonBackend` (always).
8. Creates output dir `~/lite_runs/<project>/<date>_[<group>_]<name>/`.
9. Snapshots git: `code/source.tar.gz` (archive of HEAD) and
   `code/dirty.patch` (diff HEAD vs working tree). Submodules are
   archived into the same tar.
10. Interpolates `$output` in param values.
11. Logs input files (`log_when == "before"`) and copies them to
    `<output_dir>/input/` for local reproducibility.
12. Builds command via `build_command()`.
13. Runs subprocess, streaming stdout to `stdout.log`, stderr to
    `stderr.log`, combined to `run.log` (stderr lines prefixed
    `[stderr] `). `Ctrl-C` sends `SIGTERM`, then `SIGKILL` after 10s.
14. Collects metrics, post-run files (`log_when == "after"` params,
    `Output` declarations, run logs), logs them with per-step
    try/except so one failure doesn't skip others.
15. Sends to each backend with per-backend try/except, then `finish()`
    each backend individually.
16. Returns `RunResult`. `sys.exit(1)` if the subprocess failed or was
    aborted.

Every post-run step is individually try-excepted so W&B always finishes.

## `RunResult`

```python
@dataclass(frozen=True)
class RunResult:
    output_dir: Path
    exit_code: int
    duration: float
    run_name: str
    project: str
    config: dict[str, object]
    param_values: dict[str, object]
    param_sources: dict[str, str]  # values: "cli", "override", "default", "fixed", "prompt"
```

## Built-in CLI flags (always present)

| Flag                     | Type   | Meaning                                          |
| ------------------------ | ------ | ------------------------------------------------ |
| `--dry-run`              | flag   | Print command, use `DryRunBackend`, exit.        |
| `--min-free-space-gib N` | float  | Abort if runs dir has less than N GiB free.      |
| `--no-interactive`       | flag   | Fail on missing params instead of prompting.     |
| `--no-wandb`             | flag   | Skip W&B entirely; `JsonBackend` still runs.     |
| `--project NAME`         | str    | Override project name.                           |
| `--run-name NAME`        | str    | Override run name.                               |

## `LogBackend` protocol

Any class with this signature is a valid backend (duck-typed):

```python
class LogBackend(Protocol):
    def __init__(self, project, name, group, tags, config): ...
    @property
    def run_name(self) -> str: ...
    def update_config(self, updates: dict[str, object]) -> None: ...
    def log_file(self, path: Path, log_as: str, key: str) -> None: ...
    def set_metric(self, name: str, value: object) -> None: ...
    def set_summary(self, summary: dict[str, object]) -> None: ...
    def set_tags(self, tags: list[str]) -> None: ...
    def finish(self, exit_code: int) -> None: ...
```

Currently there's no public hook to register a custom backend without
editing `runner.py`; implement it by subclassing or monkeypatching
`Runner.run` if needed.

## What gets logged to W&B

| Location                 | Content                                             |
| ------------------------ | --------------------------------------------------- |
| `run.config["param/*"]`  | All param values (UNSET → `"<unset>"`).             |
| `run.config["git/*"]`    | `repo`, `commit`, `branch`, `dirty`.                |
| `run.config["meta/*"]`   | `hostname`, `datetime`, `command`, `output_dir`, `full_command`. |
| `run.config["wandb/url"]`| W&B run URL (or `"(no wandb)"`).                    |
| `run.summary`            | `exit_code`, `duration_seconds`, `status`, all Metric values. |
| Artifacts                | `code` (tar.gz of HEAD), `code-diff` (dirty patch, local-only), `path-artifact` params, `Output(log_as="artifact")`, `Output(log_as="zip")`, run logs as text. |
| Media                    | Videos (`path-video`, `Output(log_as="video")`) and images (`path-image`, `Output(log_as="image")`). |

## Layout of `~/lite_runs/<project>/<date>_<run_name>/`

```
input/            # copies of input files (path-* params with log_when="before")
code/
  source.tar.gz   # git archive of HEAD at run time
  dirty.patch     # diff HEAD vs working tree (only if dirty)
run.log           # combined stdout+stderr (stderr prefixed "[stderr] ")
stdout.log
stderr.log
run_info.json     # JsonBackend summary (config + metrics + summary + files_logged)
<your output files, e.g. video.mp4 from $output interpolation>
```
