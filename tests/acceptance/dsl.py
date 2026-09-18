import json
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Stage:
    name: str
    harness: str


@dataclass(frozen=True)
class Pipeline:
    name: str
    stages: list[Stage]


@dataclass(frozen=True)
class Experiment:
    name: str
    corpus: Path
    pipelines: list[Pipeline]
    tasks: list[str]
    repeats: int


def a_stage(name: str, *, harness: str) -> Stage:
    return Stage(name, harness)


def a_pipeline(name: str, *, stages: list[Stage]) -> Pipeline:
    return Pipeline(name, stages)


def an_experiment(name: str, *, tasks: list[str], corpus: Path, pipelines: list[Pipeline], repeats: int) -> Experiment:
    return Experiment(name, corpus, pipelines, tasks, repeats)


def run_experiment(experiment: Experiment, results: Path) -> None:
    from agent_pipeline_benchmark.cli import main

    with tempfile.TemporaryDirectory() as definitions:
        experiment_file = write_experiment(experiment, Path(definitions))
        main(["run", str(experiment_file), "--results", str(results)])


def the_run_record(results: Path, *, experiment: str, pipeline: str, task: str) -> dict:
    records = list((results / experiment / pipeline / task).glob("*/record.json"))
    assert len(records) == 1, f"expected exactly one run of {experiment}/{pipeline}/{task}, found {len(records)}"
    return json.loads(records[0].read_text())


def assert_every_work_item_was_solved(record: dict) -> None:
    verdicts = work_item_verdicts(record)
    unsolved = [name for name, solved in verdicts.items() if not solved]
    assert verdicts and not unsolved, f"unsolved work items: {unsolved or 'no work item recorded'}"


def assert_no_work_item_was_solved(record: dict) -> None:
    verdicts = work_item_verdicts(record)
    solved = [name for name, solved in verdicts.items() if solved]
    assert verdicts and not solved, f"solved work items: {solved or 'no work item recorded'}"


def assert_the_run_cost_nothing(record: dict) -> None:
    totals = record["totals"]
    assert totals["tokens"] == 0 and totals["usd"] == 0, f"the run cost {totals}"


def work_item_verdicts(record: dict) -> dict[str, bool]:
    return {item["name"]: item["solved"] for item in record["work_items"]}


def write_experiment(experiment: Experiment, directory: Path) -> Path:
    pipeline_files = [write_pipeline(pipeline, directory) for pipeline in experiment.pipelines]
    experiment_file = directory / f"{experiment.name}.toml"
    experiment_file.write_text(
        "\n".join(
            [
                f'name = "{experiment.name}"',
                f'corpus = {{ path = "{experiment.corpus}" }}',
                f"pipelines = {toml_strings(str(file) for file in pipeline_files)}",
                f"tasks = {toml_strings(experiment.tasks)}",
                f"repeats = {experiment.repeats}",
            ]
        )
    )
    return experiment_file


def write_pipeline(pipeline: Pipeline, directory: Path) -> Path:
    pipeline_file = directory / f"{pipeline.name}.toml"
    stages = [f'[[stage]]\nname = "{stage.name}"\nharness = "{stage.harness}"' for stage in pipeline.stages]
    pipeline_file.write_text("\n\n".join([f'name = "{pipeline.name}"', *stages]))
    return pipeline_file


def toml_strings(values: Iterable[str]) -> str:
    return json.dumps(list(values))
