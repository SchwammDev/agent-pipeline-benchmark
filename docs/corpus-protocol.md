# Corpus protocol

A corpus is a git repository. The runner is pointed at its URL and a commit. It contains one directory per task and nothing about harnesses, prompts, hooks, analyzers or containers.

## Layout

```
<task-name>/
  task.toml              name, language, notes
  repo/                  starting repository
  work-items/
    01-<name>/
      requirement.md     the ticket; the only thing the agent sees
      tests/             hidden tests
      reference.diff     reference solution, applied on the previous reference
    02-<name>/
      ...
```

## Rules

- **Repository.** Installs with `uv sync`, tests run with `pytest`. Its own lint configuration is ignored at scoring time.
- **Work item order** is the directory order. Tests of work item N must go from failing to passing. Tests of all earlier work items must keep passing.
- **Ticket** states the public interface precisely: command-line call, function signature, or endpoint, plus acceptance criteria. Hidden tests exercise only that boundary. Internals are the agent's choice.
- **Hidden tests** never enter the working copy during a run. They are copied in at scoring time and removed after.
- **Reference solutions chain.** The reference for work item N applies on top of the reference for N-1, never on agent code. They serve as control, as static baseline, and as the base for the flakiness check.
- **Later work items** shift requirements to stress the design of earlier ones.

## Borrowed tasks

Converted into this layout by the benchmark, one work item per task. Any special install steps stay in the converter.

## Fixtures

The benchmark repository carries two or three toy tasks in this layout as test fixtures and as the executable example of the protocol. They are never used for results.
