# Decisions

Current decisions with their reason. Changed decisions replace the line, history lives in git.

| Decision | Why |
|---|---|
| Extensibility measured as total cost and success over a task's work items, each done by the pipeline under test on its own code | Real projects pay the total. Isolating "code quality" from "pipeline speed" answers a why-question we do not need |
| Static measures are hypotheses, not scores | Clean-code advice targets human maintainers; whether it predicts agent cost is what the benchmark tests |
| Harnesses invoked directly as subprocesses | No framework around the harness |
| Claude Code and pi only, Python only, first | Get it working before widening |
| Containers only, no bubblewrap | One path gives both pinned toolchain and isolation |
| Corpus is a separate git repo, pointed at by URL and commit | Private, so tasks stay out of training data; swappable; benchmark owns the protocol |
| Corpus holds only repository, tickets, hidden tests, reference solutions | Harness, prompts, hooks, containers are benchmark concerns |
| Tickets pin the public interface; hidden tests touch only that boundary | Tests need a contract; internals must stay free for the quality comparison |
| Hooks are part of the pipeline; scorers are independent of them | Sharing analyzers would let hooked pipelines win by construction |
| Pre-commit hooks are a stage setting alongside harness hooks | Same feedback at commit time instead of tool time; a third timing point |
| Whether cleanup sees the ticket is a prompt template choice | It is an experimental variable, not a benchmark default |
| Files, not a database | Append-only runs, each in its own directory; one glob read gives the whole table |
| `record.json` is derived from raw files | Rescoring must reproduce it; new scorers apply to old runs |
| Snapshot after every stage, runner commits bypass hooks | Per-stage diffs and costs come free; runner must not trigger the agent's feedback |
| Keep stage diffs, not a repository copy | Corpus commit plus diffs reproduce any snapshot; a repository copy per run is hundreds of megabytes for borrowed tasks |
| Finished runs are immutable; run ids carry machine name and random suffix | Mirroring needs no reconciliation; runs from several machines never collide |
| Results mirrored with rclone, backend open | Any NAS, bucket or cloud drive works without benchmark tooling |
| Running and scoring separate | Slow scorers such as mutation testing run later |
| Web inspector, server-rendered | Long transcripts and side-by-side diffs; hands off to the original harness session |
| At least five repeats, paired per task, ranges by resampling whole repositories | Harness effects are the size of run-to-run noise |
