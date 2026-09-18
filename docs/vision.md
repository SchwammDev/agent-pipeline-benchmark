# Vision

A benchmark for comparing how coding agents are set up, not which model is best. It answers questions like: does hook nudging during coding beat a cleanup pass afterwards? What does each cost? Can a weaker model do the first pass? Does splitting cleanup by category help? Does Claude Code or pi do this better? Underneath all of them: which setup produces code that an agent can keep changing cheaply.

## End goal

The goal is extensibility: code that an agent, starting with no memory of writing it, can change correctly and cheaply. The benchmark measures this directly, not through proxies.

A task is a sequence of work items on one repository. The first builds something. The later ones change or extend it, with shifting requirements chosen to stress the design the way real projects do. Each work item has hidden tests. The pipeline under test does every work item, each on the code it produced for the one before. Every stage is a fresh invocation, so no agent remembers writing the earlier code.

The headline is the total over the whole task: did each work item pass its hidden tests, and what did the whole task cost. A pipeline that is fast on the first work item but produces code that makes the third expensive pays for it in the total.

Static measures such as complexity, lint findings, duplication and diff size are recorded, but they are hypotheses, not scores. Whether they predict the cost of later work items when the maintainer is an agent is an open question the benchmark tests. A proxy that does not predict is reported as such.

## Vocabulary

| Term | Meaning |
|---|---|
| Work item | One requirement description with hidden tests, written like a real ticket |
| Task | A repository at a fixed commit plus an ordered list of work items |
| Stage | One agent invocation: harness, model, hooks, instructions |
| Pipeline | How a work item is implemented: one or more stages in order, e.g. "no hooks, then one cleanup stage" |
| Run | One pipeline applied to one task, all work items, repeated because agents are random |
| Snapshot | The state of the working directory after a stage, kept as the stage's diff and rebuilt on demand |
| Run record | Everything measured about one run, stored as data |

## Measurement

In order of trust. All scoring reads snapshots, never the live run.

1. **Hidden tests, per work item.** A few tests fail before the work item and must pass after it, proving it was done. All others pass before and must still pass after, proving nothing broke. A work item is solved when both hold. Only solved work items get quality measured, or a tiny under-fix looks clean.
2. **Cost, per stage and per work item.** Uncached input tokens, cache reads, cache writes, output tokens, dollars, wall-clock, turns. Both harnesses expose these per call.
3. **Static measures, as candidate predictors.** Change before versus after, not repository averages. Suppressions such as `# noqa` count as findings. One fixed quality policy per language, independent of each repository's own lint configuration.
4. **Test strength.** Mutation testing injects small bugs and checks whether the tests catch them. Slow, so it runs later over stored snapshots.
5. **Trace facts** from the harness event logs. Tool calls, which hooks were present, when each fired and what it fed back, whether tests ran before the agent claimed done.
6. **Model-based judgment**, if ever, only as a pairwise tie-breaker with both orders tried. Never a headline number.

Running and scoring are separate. Cheap scorers run immediately. Slow or new scorers run later over stored snapshots, including old runs.

## Hooks

Hooks belong to the pipeline under test. A stage declares which harness hooks and which pre-commit hooks are on. Harness hooks give feedback at tool time, pre-commit hooks at commit time, a cleanup stage afterwards; the same feedback at three moments. In the pi harness a harness hook is an environment variable; for Claude Code the same knob writes a settings file. Scorers do not depend on hooks in any way. Recorded: hook presence per stage and every hook event in the trace.

## Reproducibility

- One container image per task pins Python, node, pi, Claude Code, analyzers and dependencies. Runs with docker or rootless podman. Claude Code's built-in sandbox is off inside the container.
- Every run record stores the version of everything and the exact model snapshot.
- Two controls per task. A do-nothing agent must score zero. The reference solution must pass the hidden tests and defines the baseline for static measures.
- Borrowed repositories bring test suites written by other people, and a few of those tests fail now and then regardless of the code, because of timing, the current date, the network, or test order. When a task is built, its suite runs ten times on the reference solution and any test that fails in some of those runs is excluded from the hidden set. For our own tasks we write the tests, so this is only a safety net.

## Statistics

The same pipeline on the same tasks gives a different pass rate every time, and the spread is about as large as the effect a hook is expected to have.

- Every run is repeated, at least five times per task and pipeline. A small first experiment with three repeats measures how much results vary and sizes the full experiment.
- Pipelines are compared task by task: for each task, subtract one pipeline's result from the other's. Task difficulty cancels.
- Every difference is reported as a range. Draw a thousand random subsets of the tasks, recompute the difference on each, sort the values, report the middle 95 percent. Tasks from the same repository share its quirks, so a subset is drawn by repository, taking all of a repository's tasks together.
- Two success numbers per pipeline: the average pass rate, and the share of tasks solved in every repeat. The second measures reliability.
- Cost is an outcome. Results are shown as quality against cost, and as "solved within a fixed budget".

Budget arithmetic: thirty tasks, five repeats, six pipelines is 900 runs per model, each covering all work items of its task.

## Run inspector

A human-readable frontend for qualitative judgment. Navigation: pipeline, task, run, work item, stage. At the bottom, the full transcript: thoughts, tool calls, hook events, and the diff of each stage. Both harnesses keep session files and can reopen a stored session interactively, so the inspector hands off to the original session for the deep look and owns only the cross-run navigation.

## Scope

| | First version | Later |
|---|---|---|
| Language | Python | TypeScript, then others |
| Harnesses | Claude Code, pi | Any with headless mode and hooks |
| Tasks | ~20 SWE-bench tasks for calibration, 5–10 own multi-work-item tasks | More own tasks |
| Scoring | Hidden tests, cost, static measures, trace facts | Mutation testing |
| Tooling | Runner, run inspector | |

A new language costs one analyzer bundle and one container base. A new harness costs one adapter for invocation, event-log parsing and hook knobs.

Non-goals: a public leaderboard, ranking models, any harness without headless mode.

## Open decisions

- How later work items are authored so they stress design rather than reward luck.
- Whether model-based judgment is included at all in the first version.
