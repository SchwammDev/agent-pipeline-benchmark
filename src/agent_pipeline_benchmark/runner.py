import json
import secrets
import shutil
import socket
import subprocess
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from agent_pipeline_benchmark.corpus import Task, WorkItem, load_task
from agent_pipeline_benchmark.definitions import ExperimentDefinition, PipelineDefinition, StageDefinition
from agent_pipeline_benchmark.development_environments import (
    DevelopmentEnvironment,
    new_uv_development_environment,
)
from agent_pipeline_benchmark.harnesses import (
    Harness,
    StageOutcome,
    harness_named,
    the_measures_of,
)
from agent_pipeline_benchmark.hidden_tests import TestVerdict, score_hidden_tests, test_movements
from agent_pipeline_benchmark.prompts import render_prompt
from agent_pipeline_benchmark.snapshots import commit_snapshot, initialise_snapshot, the_staged_change

HarnessResolver = Callable[[str], Harness]
Scorer = Callable[[WorkItem, DevelopmentEnvironment], list[TestVerdict]]
EnvironmentFactory = Callable[[Path], DevelopmentEnvironment]


@dataclass(frozen=True)
class StageRecord:
    name: str
    harness: str
    tokens: int
    usd: float
    duration_seconds: float
    turns: int | None = None
    tool_calls: int | None = None
    end_reason: str | None = None


@dataclass(frozen=True)
class WorkItemRecord:
    name: str
    solved: bool
    progressed: int
    preserved: int
    stages: tuple[StageRecord, ...]


@dataclass(frozen=True)
class RunRecord:
    experiment: str
    pipeline: str
    task: str
    run_id: str
    run_number: int
    corpus: Path
    start: datetime
    end: datetime
    harnesses: tuple[str, ...]
    models: tuple[str, ...]
    work_items: tuple[WorkItemRecord, ...]
    harness_versions: dict[str, str] = field(default_factory=dict)

    def totals(self) -> dict:
        stages = [stage for item in self.work_items for stage in item.stages]
        return {
            "solved": sum(1 for item in self.work_items if item.solved),
            "tokens": sum(stage.tokens for stage in stages),
            "usd": sum(stage.usd for stage in stages),
            "duration_seconds": sum(stage.duration_seconds for stage in stages),
        }

    def as_json(self) -> dict:
        return {
            "identity": self.identity(),
            "work_items": [work_item_as_json(item) for item in self.work_items],
            "totals": self.totals(),
        }

    def identity(self) -> dict:
        identity: dict = {
            "experiment": self.experiment,
            "pipeline": self.pipeline,
            "task": self.task,
            "run_id": self.run_id,
            "run_number": self.run_number,
            "corpus": str(self.corpus),
            "benchmark_commit": benchmark_commit(),
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "harnesses": list(self.harnesses),
        }
        if self.models:
            identity["models"] = list(self.models)
        if self.harness_versions:
            identity["harness_versions"] = dict(self.harness_versions)
        return identity


def work_item_as_json(item: WorkItemRecord) -> dict:
    return {
        "name": item.name,
        "solved": item.solved,
        "progressed": item.progressed,
        "preserved": item.preserved,
        "stages": [stage_as_json(stage) for stage in item.stages],
    }


def stage_as_json(stage: StageRecord) -> dict:
    record = {"name": stage.name, "harness": stage.harness, "tokens": stage.tokens, "usd": stage.usd, "duration_seconds": stage.duration_seconds}
    for field in ("turns", "tool_calls", "end_reason"):
        value = getattr(stage, field)
        if value is not None:
            record[field] = value
    return record


def run_experiment(
    experiment: ExperimentDefinition,
    results: Path,
    *,
    harness_named: HarnessResolver = harness_named,
    score: Scorer = score_hidden_tests,
    new_environment: EnvironmentFactory = new_uv_development_environment,
) -> list[Path]:
    written_records = []
    for pipeline in experiment.pipelines:
        for task_name in experiment.tasks:
            task = load_task(experiment.corpus, task_name)
            for run_number in range(1, experiment.repeats + 1):
                record = run_pipeline_on_task(
                    experiment.name,
                    pipeline,
                    task,
                    run_number,
                    corpus=experiment.corpus,
                    harness_named=harness_named,
                    score=score,
                    new_environment=new_environment,
                    results=results,
                )
                written_records.append(write_record(record, results))
    return written_records


def benchmark_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).parents[2],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def run_pipeline_on_task(
    experiment: str,
    pipeline: PipelineDefinition,
    task: Task,
    run_number: int,
    *,
    corpus: Path,
    harness_named: HarnessResolver = harness_named,
    score: Scorer = score_hidden_tests,
    new_environment: EnvironmentFactory = new_uv_development_environment,
    results: Path | None = None,
) -> RunRecord:
    start = datetime.now(timezone.utc)
    run_id = new_run_id()
    with tempfile.TemporaryDirectory() as working_copy_root:
        working_copy = Path(working_copy_root)
        shutil.copytree(
            task.repository,
            working_copy,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns(".venv", "__pycache__"),
        )
        environment = new_environment(working_copy)
        environment.prepare()
        initialise_snapshot(working_copy)
        run_directory = the_run_directory(results, experiment, pipeline.name, task.name, run_id)
        work_items = tuple(
            run_work_item(pipeline, work_item, working_copy, environment, harness_named, score, run_directory)
            for work_item in task.work_items
        )
        harnesses = tuple(dict.fromkeys(stage.harness for stage in pipeline.stages))
        harness_versions = {
            name: version
            for name in harnesses
            if (version := harness_named(name).version()) is not None
        }
    return RunRecord(
        experiment=experiment,
        pipeline=pipeline.name,
        task=task.name,
        run_id=run_id,
        run_number=run_number,
        corpus=corpus,
        start=start,
        end=datetime.now(timezone.utc),
        harnesses=harnesses,
        models=tuple(dict.fromkeys(stage.model for stage in pipeline.stages if stage.model is not None)),
        harness_versions=harness_versions,
        work_items=work_items,
    )


def the_run_directory(results: Path | None, experiment: str, pipeline: str, task: str, run_id: str) -> Path | None:
    if results is None:
        return None
    return results / experiment / pipeline / task / run_id


def run_work_item(
    pipeline: PipelineDefinition,
    work_item: WorkItem,
    working_copy: Path,
    environment: DevelopmentEnvironment,
    harness_named: HarnessResolver,
    score: Scorer,
    run_directory: Path | None,
) -> WorkItemRecord:
    before = frozenset(verdict.name for verdict in score(work_item, environment) if verdict.passed)
    stages = tuple(
        run_stage(stage, work_item, working_copy, harness_named, the_stage_directory(run_directory, work_item, stage, number))
        for number, stage in enumerate(pipeline.stages, start=1)
    )
    after_verdicts = score(work_item, environment)
    after = frozenset(verdict.name for verdict in after_verdicts if verdict.passed)
    movements = test_movements(before, after)
    scoring_directory = the_scoring_directory(run_directory, work_item)
    if scoring_directory is not None:
        write_hidden_tests(scoring_directory, after_verdicts)
    return WorkItemRecord(
        name=work_item.name,
        solved=movements.solved,
        progressed=movements.progressed,
        preserved=movements.preserved,
        stages=stages,
    )


def the_scoring_directory(run_directory: Path | None, work_item: WorkItem) -> Path | None:
    if run_directory is None:
        return None
    return run_directory / "work-items" / work_item.name / "scoring"


def write_hidden_tests(scoring_directory: Path, verdicts: list[TestVerdict]) -> None:
    scoring_directory.mkdir(parents=True, exist_ok=True)
    (scoring_directory / "hidden-tests.json").write_text(
        json.dumps({"tests": [{"name": verdict.name, "passed": verdict.passed} for verdict in verdicts]})
    )


def the_stage_directory(
    run_directory: Path | None, work_item: WorkItem, stage: StageDefinition, number: int
) -> Path | None:
    if run_directory is None:
        return None
    return run_directory / "work-items" / work_item.name / "stages" / f"{number:02d}-{stage.name}"


def run_stage(
    stage: StageDefinition,
    work_item: WorkItem,
    working_copy: Path,
    harness_named: HarnessResolver,
    stage_directory: Path | None,
) -> StageRecord:
    prompt = render_prompt(stage.prompt, work_item) if stage.prompt is not None else ""
    started = time.monotonic()
    outcome = harness_named(stage.harness).implement(work_item, working_copy, prompt, model=stage.model)
    duration_seconds = time.monotonic() - started
    measures = the_measures_of(outcome.events) if outcome.events else {}
    record = StageRecord(
        name=stage.name,
        harness=stage.harness,
        tokens=outcome.cost.tokens,
        usd=outcome.cost.usd,
        duration_seconds=duration_seconds,
        turns=measures.get("turns"),
        tool_calls=measures.get("tool_calls"),
        end_reason=measures.get("end_reason"),
    )
    if stage_directory is not None:
        record_stage(stage, work_item, prompt, outcome, working_copy, stage_directory, record)
    commit_snapshot(working_copy, f"stage {stage.name}")
    return record


def record_stage(
    stage: StageDefinition,
    work_item: WorkItem,
    prompt: str,
    outcome: StageOutcome,
    working_copy: Path,
    stage_directory: Path,
    record: StageRecord,
) -> None:
    stage_directory.mkdir(parents=True)
    (stage_directory / "diff.patch").write_bytes(the_staged_change(working_copy))
    (stage_directory / "events.jsonl").write_text("".join(json.dumps(event) + "\n" for event in outcome.events))
    (stage_directory / "stage.json").write_text(json.dumps(stage_json(stage, prompt)))


def stage_json(stage: StageDefinition, prompt: str) -> dict:
    resolved = {"name": stage.name, "harness": stage.harness}
    if stage.model is not None:
        resolved["model"] = stage.model
    if stage.prompt is not None:
        resolved["prompt"] = prompt
    return resolved


def new_run_id() -> str:
    return f"{socket.gethostname()}-{secrets.token_hex(4)}"


def write_record(record: RunRecord, results: Path) -> Path:
    path = results / record.experiment / record.pipeline / record.task / record.run_id / "record.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record.as_json(), indent=2))
    return path
