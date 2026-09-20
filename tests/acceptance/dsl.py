import json
import subprocess
import sys
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Stage:
    name: str
    harness: str
    model: str | None = None
    prompt: str | None = None


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


def a_stage(name: str, *, harness: str, model: str | None = None, prompt: str | None = None) -> Stage:
    return Stage(name, harness, model, prompt)


def a_pipeline(name: str, *, stages: list[Stage]) -> Pipeline:
    return Pipeline(name, stages)


def an_experiment(name: str, *, tasks: list[str], corpus: Path, pipelines: list[Pipeline | str], repeats: int) -> Experiment:
    return Experiment(name, corpus, pipelines, tasks, repeats)


@dataclass(frozen=True)
class CommandResult:
    exit_code: int
    stdout: str
    stderr: str


def run_apb(*arguments: str) -> CommandResult:
    apb = Path(sys.executable).parent / "apb"
    completed = subprocess.run([str(apb), *arguments], capture_output=True, text=True)
    return CommandResult(completed.returncode, completed.stdout, completed.stderr)


def assert_the_command_exited_cleanly(command: CommandResult) -> None:
    assert command.exit_code == 0, f"expected the command to exit 0, it exited {command.exit_code}\n{command.stderr}"


def assert_the_command_failed(command: CommandResult) -> None:
    assert command.exit_code != 0, "expected the command to fail, it exited 0"


def assert_the_command_reported(command: CommandResult, *fragments: str) -> None:
    for fragment in fragments:
        assert fragment in command.stderr, f"expected {fragment!r} in the command output:\n{command.stderr}"


def assert_each_finished_run_path_was_printed(command: CommandResult, *, expected_runs: int) -> None:
    printed = [line for line in command.stdout.splitlines() if line.strip()]
    assert len(printed) == expected_runs, f"expected {expected_runs} finished run paths, printed: {printed}"
    for line in printed:
        path = Path(line)
        assert path.name == "record.json" and path.exists(), f"printed path is not an existing run record: {line}"


def assert_no_run_was_recorded(results: Path) -> None:
    records = list(results.rglob("record.json"))
    assert not records, f"runs were recorded before the definitions were validated: {records}"


def an_experiment_file_created_for(experiment: Experiment, *, in_directory: Path) -> Path:
    in_directory.mkdir(parents=True, exist_ok=True)
    return write_experiment(experiment, in_directory)


def introduce_a_typo_in(file: Path, *, field: str) -> None:
    text = file.read_text()
    corrupted = text.replace(field, typo(field))
    assert corrupted != text, f"could not introduce a typo in {field!r}: {text}"
    file.write_text(corrupted)


def typo(field: str) -> str:
    return field.replace("e", "c", 1)


def run_experiment(experiment: Experiment, results: Path) -> None:
    from agent_pipeline_benchmark.cli import main

    with tempfile.TemporaryDirectory() as definitions:
        experiment_file = write_experiment(experiment, Path(definitions))
        main(["run", str(experiment_file), "--results", str(results)])


def run_record_of(results: Path, *, experiment: str, pipeline: str, task: str) -> dict:
    records = list((results / experiment / pipeline / task).glob("*/record.json"))
    assert len(records) == 1, f"expected exactly one run of {experiment}/{pipeline}/{task}, found {len(records)}"
    return json.loads(records[0].read_text())


def assert_the_run_cost_nothing(record: dict) -> None:
    totals = record["totals"]
    assert totals["tokens"] == 0 and totals["usd"] == 0, f"the run cost {totals}"


def assert_the_work_item_was_solved(record: dict, *, work_item: str, progressed: int, preserved: int) -> None:
    assert_the_work_item_was_scored(
        record, work_item, solved=True, progressed=progressed, preserved=preserved
    )


def assert_the_work_item_was_not_solved(record: dict, *, work_item: str, progressed: int, preserved: int) -> None:
    assert_the_work_item_was_scored(
        record, work_item, solved=False, progressed=progressed, preserved=preserved
    )


def assert_the_work_item_was_scored(
    record: dict, work_item: str, *, solved: bool, progressed: int, preserved: int
) -> None:
    verdict = the_verdict_on(record, work_item)
    assert verdict["solved"] == solved, (
        f"expected {work_item!r} to be scored solved={solved}, the record reports {verdict}"
    )
    assert verdict["progressed"] == progressed, (
        f"expected {progressed} progressed tests for {work_item!r}, the record reports {verdict.get('progressed')}"
    )
    assert verdict["preserved"] == preserved, (
        f"expected {preserved} preserved tests for {work_item!r}, the record reports {verdict.get('preserved')}"
    )


def the_verdict_on(record: dict, work_item: str) -> dict:
    verdicts = {item["name"]: item for item in record["work_items"]}
    assert work_item in verdicts, f"no verdict was recorded for work item {work_item!r}"
    return verdicts[work_item]


def working_copy_of(results: Path, *, experiment: str, pipeline: str, task: str) -> Path:
    copies = list((results / experiment / pipeline / task).glob("*/working-copy"))
    assert len(copies) == 1, (
        f"expected exactly one kept working copy of {experiment}/{pipeline}/{task}, found {len(copies)}"
    )
    return copies[0]


def assert_no_hidden_test_file_remains(working_copy: Path, *, corpus: Path, task: str) -> None:
    for hidden_tests in sorted((corpus / task / "work-items").glob("*/tests")):
        for hidden_test in hidden_tests.rglob("*"):
            if hidden_test.is_file():
                leftover = working_copy / "tests" / hidden_test.relative_to(hidden_tests)
                assert not leftover.exists(), f"a hidden test file survived scoring: {leftover}"


def write_experiment(experiment: Experiment, directory: Path) -> Path:
    pipeline_references = [
        pipeline if isinstance(pipeline, str) else str(write_pipeline(pipeline, directory))
        for pipeline in experiment.pipelines
    ]
    experiment_file = directory / f"{experiment.name}.toml"
    experiment_file.write_text(
        "\n".join(
            [
                f'name = "{experiment.name}"',
                f'corpus = {{ path = "{experiment.corpus}" }}',
                f"pipelines = {toml_strings(pipeline_references)}",
                f"tasks = {toml_strings(experiment.tasks)}",
                f"repeats = {experiment.repeats}",
            ]
        )
    )
    return experiment_file


def write_pipeline(pipeline: Pipeline, directory: Path) -> Path:
    pipeline_file = directory / f"{pipeline.name}.toml"
    stages = ["\n".join(stage_lines(stage)) for stage in pipeline.stages]
    pipeline_file.write_text("\n\n".join([f'name = "{pipeline.name}"', *stages]))
    return pipeline_file


def stage_lines(stage: Stage) -> list[str]:
    lines = ["[[stage]]", f'name = "{stage.name}"', f'harness = "{stage.harness}"']
    if stage.model is not None:
        lines.append(f'model = "{stage.model}"')
    if stage.prompt is not None:
        lines.append(f'prompt = "{stage.prompt}"')
    return lines


def toml_strings(values: Iterable[str]) -> str:
    return json.dumps(list(values))
