import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from subprocess import CompletedProcess
from typing import Any

import pytest

from agent_pipeline_benchmark.definitions import ExperimentDefinition, PipelineDefinition, StageDefinition
from agent_pipeline_benchmark.environments import EnvironmentUnavailable, ExecutionEnvironment

from conftest import FAKE_AGENT_STREAM, FAKE_AGENT_VERSION, THE_UNPREPARABLE_REASON

LIUBAI_MODEL = "deepseek-v4-flash-284b"
IMPLEMENT_PROMPT = "implement.md"
TICKET_TEXT = "Greet a person by name"
TOKENS_ABOVE_ZERO = "above zero"


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


def an_experiment(name: str, *, tasks: list[str], corpus: Path, pipelines: list[Pipeline], repeats: int) -> Experiment:
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
    return the_run_record_in(the_run_directory_of(results, experiment=experiment, pipeline=pipeline, task=task))


def the_run_directory_of(results: Path, *, experiment: str, pipeline: str, task: str) -> Path:
    records = list((results / experiment / pipeline / task).glob("*/record.json"))
    assert len(records) == 1, f"expected exactly one run of {experiment}/{pipeline}/{task}, found {len(records)}"
    return records[0].parent


def the_run_record_in(run_directory: Path) -> dict:
    return json.loads((run_directory / "record.json").read_text())


def the_stage_directory_of(run_directory: Path, *, work_item: str, stage: str) -> Path:
    path = run_directory / "work-items" / work_item / "stages" / stage
    assert path.is_dir(), f"the run directory holds no stage directory for {work_item}/{stage}: {run_directory}"
    return path


def assert_the_liubai_run_solved_the_greeting_work_item(results: Path) -> None:
    record = run_record_of(results, experiment="skeleton", pipeline="bare", task="greeting")
    assert_the_work_item_was_solved(record, work_item="01-greet", progressed=2, preserved=1)
    assert_one_work_item_was_solved(record)


def assert_the_liubai_run_is_identified_in_its_record(results: Path, *, corpus: Path) -> None:
    record = run_record_of(results, experiment="skeleton", pipeline="bare", task="greeting")
    assert_the_record_identifies_the_run(
        record, experiment="skeleton", pipeline="bare", task="greeting", corpus=corpus
    )
    assert_the_record_was_produced_at_this_benchmark_commit(record)
    assert_the_record_names_the_harness_and_model(record, harness="liubai", model=LIUBAI_MODEL)
    assert_the_record_spans_the_run(record)
    assert_the_record_omits_fields_that_are_not_yet_measured(record)


def assert_the_liubai_run_left_every_file_a_scorer_needs_on_disk(results: Path, *, corpus: Path) -> None:
    run_directory = the_run_directory_of(results, experiment="skeleton", pipeline="bare", task="greeting")
    assert_the_run_directory_contains_exactly_the_recorded_files(run_directory)
    assert_the_stage_diff_is_exactly_the_reference_change(
        run_directory, corpus=corpus, task="greeting", work_item="01-greet"
    )
    assert_the_event_stream_shows_the_agent_finishing_the_stage(
        run_directory, work_item="01-greet", stage="01-implement"
    )
    assert_the_stage_record_resolves_the_stage(
        run_directory, harness="liubai", model=LIUBAI_MODEL, prompt_contains=TICKET_TEXT
    )
    assert_the_hidden_tests_passed_after_the_stage(
        run_directory, corpus=corpus, task="greeting", work_item="01-greet"
    )


def assert_one_work_item_was_solved(record: dict) -> None:
    solved = record["totals"]["solved"]
    assert solved == 1, f"the totals report {solved} solved work items"


def assert_the_record_omits_fields_that_are_not_yet_measured(record: dict) -> None:
    unmeasured = {"hook_events", "hook events", "static", "static_measures", "static measures", "image_digest", "image digest"}

    def walk(node: object) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                assert key not in unmeasured, f"the unmeasured field {key!r} appeared in the record"
                assert value is not None, f"the key {key!r} holds null; unmeasured fields must be absent"
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(record)


def the_strings_in(node: object) -> list[str]:
    if isinstance(node, str):
        return [node]
    if isinstance(node, dict):
        return [string for value in node.values() for string in the_strings_in(value)]
    if isinstance(node, list):
        return [string for item in node for string in the_strings_in(item)]
    return []


def assert_the_record_identifies_the_run(
    record: dict, *, experiment: str, pipeline: str, task: str, corpus: Path
) -> None:
    identity = record["identity"]
    assert identity["experiment"] == experiment, f"the record names experiment {identity['experiment']!r}"
    assert identity["pipeline"] == pipeline, f"the record names pipeline {identity['pipeline']!r}"
    assert identity["task"] == task, f"the record names task {identity['task']!r}"
    assert identity["run_number"] == 1, f"the record names run number {identity['run_number']!r}"
    assert str(corpus) in the_strings_in(identity), "the corpus location as given in the experiment file is missing"


def assert_the_record_was_produced_at_this_benchmark_commit(record: dict) -> None:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=Path(__file__).parents[2], capture_output=True, text=True, check=True
    ).stdout.strip()
    recorded = record["identity"]["benchmark_commit"]
    assert recorded == head, f"the record pins benchmark commit {recorded!r}, this checkout is at {head}"


def assert_the_record_names_the_harness_and_model(record: dict, *, harness: str, model: str) -> None:
    identity_strings = the_strings_in(record["identity"])
    assert any(string == harness or string.startswith(harness + " ") for string in identity_strings), (
        f"the identity does not name the {harness!r} harness: {identity_strings}"
    )
    assert any(model in string for string in identity_strings), (
        f"the identity does not name the {model!r} model: {identity_strings}"
    )


def the_moment(value: object) -> datetime:
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, tz=timezone.utc)
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed


def assert_the_record_spans_the_run(record: dict) -> None:
    identity = record["identity"]
    start, end = identity["start"], identity["end"]
    assert the_moment(end) > the_moment(start), f"the end time is not after the start time: {start} -> {end}"


def assert_the_run_directory_contains_exactly_the_recorded_files(run_directory: Path) -> None:
    recorded_files = [
        "record.json",
        "work-items/01-greet/stages/01-implement/stage.json",
        "work-items/01-greet/stages/01-implement/events.jsonl",
        "work-items/01-greet/stages/01-implement/diff.patch",
        "work-items/01-greet/scoring/hidden-tests.json",
    ]
    actual = sorted(str(path.relative_to(run_directory)) for path in run_directory.rglob("*") if path.is_file())
    assert actual == sorted(recorded_files), f"the run directory holds unexpected files: {actual}"


def assert_the_stage_diff_is_exactly_the_reference_change(
    run_directory: Path, *, corpus: Path, task: str, work_item: str
) -> None:
    stage_directory = the_stage_directory_of(run_directory, work_item=work_item, stage="01-implement")
    reference = corpus / task / "work-items" / work_item / "reference.diff"
    applied = (stage_directory / "diff.patch").read_text()
    assert applied == reference.read_text(), f"the stage diff is not exactly the reference change:\n{applied}"


def the_stage_events(run_directory: Path, *, work_item: str, stage: str) -> list[dict]:
    events_file = the_stage_directory_of(run_directory, work_item=work_item, stage=stage) / "events.jsonl"
    lines = [line for line in events_file.read_text().splitlines() if line.strip()]
    assert lines, "the event stream is empty"
    return [json_event_line(line) for line in lines]


def json_event_line(line: str) -> dict:
    try:
        return json.loads(line)
    except json.JSONDecodeError as error:
        raise AssertionError(f"an event line does not parse as JSON: {line!r}") from error


def assert_the_event_stream_shows_the_agent_finishing_the_stage(
    run_directory: Path, *, work_item: str, stage: str
) -> None:
    events = the_stage_events(run_directory, work_item=work_item, stage=stage)
    assistant_messages = [
        event
        for event in events
        if event.get("type") == "message_end" and event.get("message", {}).get("role") == "assistant"
    ]
    assert assistant_messages, "the event stream shows no model output"
    assert any("toolCall" in json.dumps(event) or "tool_call" in json.dumps(event) for event in events), (
        "the event stream shows no tool call"
    )
    assert any(event.get("type") == "agent_end" for event in events), "the event stream shows no normal finish"


def assert_the_stage_record_resolves_the_stage(
    run_directory: Path, *, harness: str, model: str, prompt_contains: str
) -> None:
    stage_file = the_stage_directory_of(run_directory, work_item="01-greet", stage="01-implement") / "stage.json"
    resolved = json.loads(stage_file.read_text())
    assert resolved["harness"] == harness, f"the stage record names harness {resolved['harness']!r}"
    assert resolved["model"] == model, f"the stage record names model {resolved['model']!r}"
    assert prompt_contains in resolved["prompt"], (
        f"the rendered prompt does not carry the ticket text {prompt_contains!r}: {resolved['prompt']!r}"
    )


def the_hidden_test_ids_in(directory: Path) -> set[str]:
    return {
        match.group(1)
        for test_file in directory.rglob("test_*.py")
        for match in re.finditer(r"def (test_\w+)", test_file.read_text())
    }


def the_recorded_test_verdicts(scoring: dict) -> dict[str, bool]:
    verdicts = {}

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            name = node.get("name") or node.get("id")
            if isinstance(name, str) and "passed" in node:
                verdicts[name] = bool(node["passed"])
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(scoring)
    return verdicts


def assert_the_hidden_tests_passed_after_the_stage(
    run_directory: Path, *, corpus: Path, task: str, work_item: str
) -> None:
    scoring_file = run_directory / "work-items" / work_item / "scoring" / "hidden-tests.json"
    scoring = json.loads(scoring_file.read_text())
    verdicts = the_recorded_test_verdicts(scoring)
    for test_id in sorted(the_hidden_test_ids_in(corpus / task / "work-items" / work_item / "tests")):
        matching = [passed for name, passed in verdicts.items() if name.endswith(test_id)]
        assert matching and matching[0], f"hidden test {test_id!r} was not recorded as passing: {verdicts}"


def assert_the_run_costs(record: dict, *, tokens: int | str, usd: float) -> None:
    totals = record["totals"]
    if tokens == TOKENS_ABOVE_ZERO:
        assert totals["tokens"] > 0, f"the run recorded no token usage: {totals}"
    else:
        assert totals["tokens"] == tokens, f"the run cost {totals['tokens']} tokens, expected {tokens}"
    assert totals["usd"] == usd, f"the run cost {totals['usd']} USD, expected {usd}"


def the_stage_measures_in(record: dict, *, work_item: str, stage: str) -> dict:
    item = the_verdict_on(record, work_item)
    stages = {stage_record["name"]: stage_record for stage_record in item["stages"]}
    assert stage in stages, f"no stage {stage!r} was recorded for work item {work_item!r}: {stages}"
    return stages[stage]


def the_stage_name_of(stage_directory_name: str) -> str:
    return stage_directory_name.split("-", 1)[1]


def assert_the_record_contains_the_measures_of_the_agents_run(
    record: dict, *, work_item: str, stage: str, turns: int, tool_calls: int, end_reason: str
) -> None:
    measures = the_stage_measures_in(record, work_item=work_item, stage=stage)
    assert measures["turns"] == turns, f"the stage recorded {measures.get('turns')} turns, expected {turns}: {measures}"
    assert measures["tool_calls"] == tool_calls, (
        f"the stage recorded {measures.get('tool_calls')} tool calls, expected {tool_calls}: {measures}"
    )
    assert measures["end_reason"] == end_reason, (
        f"the stage ended with {measures.get('end_reason')!r}, expected {end_reason!r}: {measures}"
    )
    assert measures["duration_seconds"] > 0, f"the stage recorded no wall-clock duration: {measures}"


def assert_the_recorded_measures_match_the_event_stream(run_directory: Path, *, work_item: str, stage: str) -> None:
    events = the_stage_events(run_directory, work_item=work_item, stage=stage)
    measures = the_stage_measures_in(
        the_run_record_in(run_directory), work_item=work_item, stage=the_stage_name_of(stage)
    )
    turns = sum(1 for event in events if event.get("type") == "turn_start")
    tool_calls = sum(1 for event in events if event.get("type") == "tool_execution_start")
    assert measures.get("turns") == turns, f"the stage recorded {measures.get('turns')} turns, the event stream shows {turns}: {measures}"
    assert measures.get("tool_calls") == tool_calls, (
        f"the stage recorded {measures.get('tool_calls')} tool calls, the event stream shows {tool_calls}: {measures}"
    )
    assert measures.get("end_reason") == "finished", (
        f"the agent finished normally in the event stream, the stage ended with {measures.get('end_reason')!r}: {measures}"
    )
    assert measures.get("duration_seconds", 0) > 0, f"the stage recorded no wall-clock duration: {measures}"


def assert_the_run_totals_sum_the_stage_measures(record: dict) -> None:
    stages = [stage for item in record["work_items"] for stage in item["stages"]]
    wall_clock = sum(stage.get("duration_seconds", 0.0) for stage in stages)
    totals = record["totals"]
    assert "duration_seconds" in totals, f"the totals do not carry the run's wall-clock duration: {totals}"
    assert abs(totals["duration_seconds"] - wall_clock) < 1e-6, (
        f"the totals wall-clock {totals['duration_seconds']} does not sum the stage durations {wall_clock}"
    )


def assert_the_record_names_the_version_the_harness_reports(record: dict, *, harness: str) -> None:
    reported = subprocess.run([harness, "--version"], capture_output=True, text=True, check=True).stdout.strip()
    versions = record["identity"].get("harness_versions", {})
    assert versions.get(harness) == reported, (
        f"the record does not name the version the {harness!r} harness reports ({reported!r}): {versions}"
    )


def assert_the_liubai_run_reports_the_measures_of_its_stage(results: Path) -> None:
    run_directory = the_run_directory_of(results, experiment="skeleton", pipeline="bare", task="greeting")
    assert_the_recorded_measures_match_the_event_stream(run_directory, work_item="01-greet", stage="01-implement")
    record = the_run_record_in(run_directory)
    assert_the_run_totals_sum_the_stage_measures(record)
    assert_the_record_names_the_version_the_harness_reports(record, harness="liubai")


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


def assert_the_run_is_free_of_hidden_tests(
    results: Path, *, corpus: Path, experiment: str, pipeline: str, task: str
) -> None:
    run_directory = the_run_directory_of(results, experiment=experiment, pipeline=pipeline, task=task)
    leaked = [
        (hidden_test.name, str(recorded.relative_to(run_directory)))
        for hidden_test in the_hidden_tests_of(corpus, task)
        for recorded in run_directory.rglob("*")
        if recorded.is_file() and recorded.read_bytes() == hidden_test.read_bytes()
    ]
    assert not leaked, f"hidden tests reached the recorded run: {leaked}"


def the_hidden_tests_of(corpus: Path, task: str) -> list[Path]:
    work_items = sorted((corpus / task / "work-items").iterdir())
    return [file for item in work_items for file in (item / "tests").rglob("*") if file.is_file()]


def write_experiment(experiment: Experiment, directory: Path) -> Path:
    pipeline_references = [str(write_pipeline(pipeline, directory)) for pipeline in experiment.pipelines]
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


def the_scripted_agent_experiment(corpus: Path) -> Experiment:
    stage = a_stage("implement", harness="scripted-agent")
    return an_experiment(
        "skeleton", tasks=["greeting"], corpus=corpus, pipelines=[a_pipeline("bare", stages=[stage])], repeats=1
    )


@dataclass
class ScriptedExecutionEnvironment(ExecutionEnvironment):
    agent_version: str = FAKE_AGENT_VERSION
    preparations: int = 0
    executed_commands: list[list[str]] = field(default_factory=list)
    preparation_error: str | None = None

    def prepare(self) -> None:
        if self.preparation_error is not None:
            raise EnvironmentUnavailable(self.preparation_error)
        self.preparations += 1

    def execute(self, command: Sequence[str], working_copy: Path) -> CompletedProcess:
        self.executed_commands.append(list(command))
        output = self.agent_version if "--version" in command else FAKE_AGENT_STREAM
        return CompletedProcess(command, 0, stdout=output.encode(), stderr=b"")


def a_preparable_execution_environment() -> ScriptedExecutionEnvironment:
    return ScriptedExecutionEnvironment()


def an_execution_environment_that_cannot_be_prepared() -> ScriptedExecutionEnvironment:
    return ScriptedExecutionEnvironment(preparation_error=THE_UNPREPARABLE_REASON)


def run_experiment_with(experiment: Experiment, results: Path, *, environment: ExecutionEnvironment) -> None:
    from agent_pipeline_benchmark.runner import run_experiment as the_run

    the_run(experiment_definition_of(experiment), results, environment=environment)  # ty: ignore[unknown-argument]


def experiment_definition_of(experiment: Experiment) -> ExperimentDefinition:
    return ExperimentDefinition(
        name=experiment.name,
        corpus=experiment.corpus,
        pipelines=tuple(
            PipelineDefinition(
                name=pipeline.name,
                stages=tuple(
                    StageDefinition(name=stage.name, harness=stage.harness, model=stage.model, prompt=stage.prompt)
                    for stage in pipeline.stages
                ),
            )
            for pipeline in experiment.pipelines
        ),
        tasks=tuple(experiment.tasks),
        repeats=experiment.repeats,
    )


def assert_the_environment_was_prepared_exactly_once(environment: ScriptedExecutionEnvironment) -> None:
    assert environment.preparations == 1, (
        f"the execution environment was prepared {environment.preparations} times, expected exactly once"
    )


def assert_the_run_ran_inside_the_prepared_environment(
    results: Path, *, environment: ScriptedExecutionEnvironment
) -> None:
    assert environment.executed_commands, "the execution environment executed nothing for the run's stages"
    record = run_record_of(results, experiment="skeleton", pipeline="bare", task="greeting")
    assert_the_record_names_the_version_the_environment_provides(record, environment=environment)
    run_directory = the_run_directory_of(results, experiment="skeleton", pipeline="bare", task="greeting")
    assert_the_stage_event_stream_shows_the_agent_running(run_directory)


def assert_the_record_names_the_version_the_environment_provides(
    record: dict, *, environment: ScriptedExecutionEnvironment
) -> None:
    versions = record["identity"].get("harness_versions", {})
    assert versions.get("scripted-agent") == environment.agent_version, (
        f"the record does not name the version the execution environment provides ({environment.agent_version!r}): {versions}"
    )


def assert_the_stage_event_stream_shows_the_agent_running(run_directory: Path) -> None:
    events = the_stage_events(run_directory, work_item="01-greet", stage="01-implement")
    assert any(event.get("type") == "session" for event in events), "the event stream shows no agent session"
    assert any(event.get("type") == "agent_end" for event in events), "the event stream shows no finished agent run"


def delete_the_run_directory(results: Path, *, experiment: str, pipeline: str, task: str) -> None:
    run_directory = the_run_directory_of(results, experiment=experiment, pipeline=pipeline, task=task)
    shutil.rmtree(run_directory)
