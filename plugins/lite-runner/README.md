# lite-runner plugin for Claude Code

A [Claude Code](https://claude.com/claude-code) plugin that teaches
Claude how to use the [`lite-runner`](https://github.com/moonmath-ai/LiteRunner)
Python package — a runner for generative models with local and W&B
tracking.

The plugin ships a single skill (`lite-runner`) that auto-triggers
whenever Claude is asked to write or edit code that uses
`lite_runner`: declaring `Param` / `Output` / `Metric` / `Runner`
objects, building a `run.py` launcher, or driving a sweep via
`runner.override(...).run(...)`.

## Install

From the root of the LiteRunner repo (or the GitHub remote):

```text
/plugin marketplace add moonmath-ai/LiteRunner
/plugin install lite-runner@lite-runner-marketplace
```

Or from a local clone:

```text
/plugin marketplace add .
/plugin install lite-runner@lite-runner-marketplace
```

## What's inside

- `skills/lite-runner/SKILL.md` — mental model, setup, the 80%
  patterns, and common gotchas. Auto-loads when Claude detects
  `lite_runner` usage.
- `skills/lite-runner/references/api.md` — full field-by-field API
  reference (every `Param` / `Output` / `Metric` / `Runner` field,
  every `ParamType`, every built-in CLI flag).
- `skills/lite-runner/references/cookbook.md` — recipes and anti-patterns
  (PEP 723 shebangs, `UNSET` handling, env vars, sweeps, testing
  without W&B).

## Links

- Package on PyPI: <https://pypi.org/project/lite-runner/>
- Source: <https://github.com/moonmath-ai/LiteRunner>
- Issues: <https://github.com/moonmath-ai/LiteRunner/issues>
