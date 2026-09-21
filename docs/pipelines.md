# Pipelines and experiments

## Pipeline

One TOML file, a list of stages run in order per work item.

```toml
name = "bare-then-cleanup"

[[stage]]
name = "implement"
harness = "claude"
model = "claude-sonnet-5"
prompt = "implement.md"
instructions = "none"
harness_hooks = []
pre_commit_hooks = []
max_turns = 60
max_cost_usd = 3.0

[[stage]]
name = "cleanup"
harness = "claude"
model = "claude-haiku-4-5"
prompt = "cleanup.md"
prompt_args = { categories = "lint, types" }
harness_hooks = ["ruff", "pyright"]
pre_commit_hooks = ["ruff", "pyright"]
```

| Field | Meaning |
|---|---|
| `harness` | `claude` or `liuba` |
| `model` | Model id, per stage |
| `prompt` | Template in `prompts/`. The implement template receives the ticket. Cleanup templates receive `prompt_args`. Whether cleanup sees the ticket is a template choice |
| `instructions` | Instruction file written into the working copy (`CLAUDE.md` or liubai equivalent). `none` writes nothing |
| `harness_hooks` | Names from the hook catalogue, rendered into Claude Code settings or liubai environment variables |
| `pre_commit_hooks` | Names from the hook catalogue, rendered into a pre-commit configuration in the working copy |
| `max_turns`, `max_cost_usd` | Caps. Hitting one ends the stage with end reason `limit` |

Everything a stage resolves to is hashed into `stage.json`. Runs are comparable only if hashes match.

## Hook catalogue

One entry per hook name: event, command, action (`block` or `feedback`). Adapters render the same name for both harnesses. Scorers never depend on hooks.

## Experiment

```toml
name = "hooks-vs-cleanup-pilot"
corpus = { url = "git@...:corpus.git", commit = "abc123" }
pipelines = ["hooked", "bare-then-cleanup", "bare-then-cleanup-split"]
tasks = ["*"]
repeats = 3
concurrency = 4
```

Results go to `results/<name>/`.
