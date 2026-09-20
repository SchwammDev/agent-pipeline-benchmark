import json
import secrets
import shutil
import socket
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from agent_pipeline_benchmark.corpus import Task, WorkItem, load_task
from agent_pipeline_benchmark.definitions import ExperimentDefinition, PipelineDefinition, StageDefinition
from agent_pipeline_benchmark.harnesses import Harness, harness_named
from agent_pipeline_benchmark.hidden_tests import passing_test_ids, test_movements

HarnessResolver = Callable[[str], Harness]


@dataclass(frozen=True)
class StageRecord:
    name: str
    harness: str
    tokens: int
    usd: float


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
    work_items: tuple[WorkItemRecord, ...]

    def totals(self) -> dict:
        stages = [stage for item in self.work_items for stage in item.stages]
        return {
            "solved": sum(1 for item in self.work_items if item.solved),
            "tokens": sum(stage.tokens for stage in stages),
            "usd": sum(stage.usd for stage in stages),
        }

    def as_json(self) -> dict:
        return {
            "experiment": self.experiment,
            "pipeline": self.pipeline,
            "task": self.task,
            "run_id": self.run_id,
            "run_number": self.run_number,
            "work_items": [work_item_as_json(item) for item in self.work_items],
            "totals": self.totals(),
        }


def work_item_as_json(item: WorkItemRecord) -> dict:
    return {
        "name": item.name,
        "solved": item.solved,
        "progressed": item.progressed,
        "preserved": item.preserved,
        "stages": [stage_as_json(stage) for stage in item.stages],
    }


def stage_as_json(stage: StageRecord) -> dict:
    return {"name": stage.name, "harness": stage.harness, "tokens": stage.tokens, "usd": stage.usd}


def run_experiment(
    experiment: ExperimentDefinition, results: Path, *, harness_named: HarnessResolver = harness_named
) -> list[Path]:
    written_records = []
    for pipeline in experiment.pipelines:
        for task_name in experiment.tasks:
            task = load_task(experiment.corpus, task_name)
            for run_number in range(1, experiment.repeats + 1):
                record = run_pipeline_on_task(
                    experiment.name, pipeline, task, run_number, harness_named=harness_named, results=results
                )
                written_records.append(write_record(record, results))
    return written_records


def run_pipeline_on_task(
    experiment: str,
    pipeline: PipelineDefinition,
    task: Task,
    run_number: int,
    *,
    harness_named: HarnessResolver = harness_named,
    results: Path | None = None,
) -> RunRecord:
    run_id = new_run_id()
    with tempfile.TemporaryDirectory() as working_copy_root:
        working_copy = Path(working_copy_root)
        shutil.copytree(
            task.repository,
            working_copy,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns(".venv", "__pycache__"),
        )
        work_items = tuple(
            run_work_item(pipeline, work_item, working_copy, harness_named) for work_item in task.work_items
        )
        if results is not None:
            keep_working_copy(results, experiment, pipeline.name, task, run_id, working_copy)
    return RunRecord(
        experiment=experiment,
        pipeline=pipeline.name,
        task=task.name,
        run_id=run_id,
        run_number=run_number,
        work_items=work_items,
    )


def run_work_item(
    pipeline: PipelineDefinition, work_item: WorkItem, working_copy: Path, harness_named: HarnessResolver
) -> WorkItemRecord:
    before = passing_test_ids(work_item, working_copy)
    stages = tuple(run_stage(stage, work_item, working_copy, harness_named) for stage in pipeline.stages)
    after = passing_test_ids(work_item, working_copy)
    movements = test_movements(before, after)
    return WorkItemRecord(
        name=work_item.name,
        solved=movements.solved,
        progressed=movements.progressed,
        preserved=movements.preserved,
        stages=stages,
    )


def run_stage(
    stage: StageDefinition, work_item: WorkItem, working_copy: Path, harness_named: HarnessResolver
) -> StageRecord:
    cost = harness_named(stage.harness).implement(work_item, working_copy)
    return StageRecord(name=stage.name, harness=stage.harness, tokens=cost.tokens, usd=cost.usd)


def keep_working_copy(
    results: Path, experiment: str, pipeline: str, task: Task, run_id: str, working_copy: Path
) -> None:
    assert_no_hidden_tests_remain(task, working_copy)
    destination = results / experiment / pipeline / task.name / run_id / "working-copy"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(working_copy, destination)


def assert_no_hidden_tests_remain(task: Task, working_copy: Path) -> None:
    for work_item in task.work_items:
        for hidden_test in work_item.hidden_tests.rglob("*"):
            if hidden_test.is_file():
                leftover = working_copy / "tests" / hidden_test.relative_to(work_item.hidden_tests)
                if leftover.exists():
                    raise AssertionError(f"a hidden test file survived scoring: {leftover}")


def new_run_id() -> str:
    return f"{socket.gethostname()}-{secrets.token_hex(4)}"


def write_record(record: RunRecord, results: Path) -> Path:
    path = results / record.experiment / record.pipeline / record.task / record.run_id / "record.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record.as_json(), indent=2))
    return path
