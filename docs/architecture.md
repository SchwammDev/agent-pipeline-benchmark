# Architecture

Vocabulary is defined in [vision.md](vision.md). Added here: an **experiment** is a named set of pipelines, tasks and repeat count against one corpus at one commit.

## Boundary

| Corpus (separate git repo) | Benchmark (this repo) |
|---|---|
| Starting repository | Base container image, per-task image build |
| Work items: ticket, hidden tests, reference solution | Harness adapters, hook catalogue, pre-commit rendering |
| Nothing else | Prompt templates, instruction files, pipelines, experiments |
| | Runner, scorers, analysis, inspector |

The corpus knows nothing about harnesses, prompts, hooks or containers. Protocol: [corpus-protocol.md](corpus-protocol.md).

## Stack

| Part | Choice |
|---|---|
| Language | Python 3.13, `uv`, `pytest` |
| Definitions | TOML, validated by Pydantic |
| Harness invocation | `subprocess`: `claude -p --output-format stream-json`, `pi --mode json`; one adapter per harness parses events into a common event model |
| Containers | `docker` CLI, podman-compatible; one base image with Python, node, Claude Code, pi, analyzers; per-task image adds the repository's dependencies via `uv sync` |
| Snapshots | git commit in the working copy after every stage, hooks bypassed; only the stage diff is kept |
| Run records | Files only: `record.json` per run, raw `events.jsonl` per stage. No database |
| Storage | `results/` mirrored to a remote with rclone; backend open |
| Analyzers | ruff, basedpyright, radon, vulture, jscpd; mutmut later |
| Analysis | polars, numpy, matplotlib |
| Inspector | FastAPI, server-rendered Jinja, read-only over the results directory |
| Concurrency | `asyncio` subprocesses, one container per run, concurrency limit |

## Run flow

1. Clone corpus at the pinned commit; build the task image if the digest is missing.
2. Per repeat: fresh container, fresh working copy.
3. Per work item, per stage: render prompt, instruction file, harness settings and pre-commit config into the working copy; invoke the harness; capture events; commit snapshot with hooks bypassed; copy session files out.
4. Per work item: copy hidden tests in, run, remove; run cheap scorers on the snapshots.
5. Write `record.json`. Slow scorers rescore later from the raw files.

## Results layout

```
results/<experiment>/<pipeline>/<task>/<run-id>/
  record.json                   derived from everything below
  work-items/<NN-name>/
    stages/<NN-name>/
      stage.json                resolved config, rendered prompt and settings, hashes
      events.jsonl              raw harness event stream
      session/                  harness session files, reopenable
      diff.patch                what this stage changed
    scoring/
      hidden-tests.json
      static.json
```

- `record.json` is derived. Deleting all records and rescoring reproduces them. Adding a scorer means rescoring old runs.
- A finished run is immutable. Rescoring rewrites only `record.json` and `scoring/`.
- No repository copy. Corpus commit plus stage diffs in order reproduce any snapshot; the inspector rebuilds on demand.
- `<run-id>` includes the machine name and a random suffix so runs from different machines never collide.
- `results/` is mirrored to a remote with rclone. Records are small enough to version in git later if a paper needs a fixed dataset.

## record.json

| Group | Fields |
|---|---|
| Identity | experiment, pipeline, task, run number, corpus URL and commit, benchmark commit, image digest, harness versions, model ids, start, end |
| Per work item | passed; fail-to-pass count; pass-to-pass count |
| Per stage | uncached input, cache read, cache write, output tokens; USD; wall-clock; turns; tool calls; hook events; end reason (finished, limit, error); static measures before and after |
| Totals | passed work items, tokens, USD, wall-clock |

Analysis reads all records with one glob into one polars table.
