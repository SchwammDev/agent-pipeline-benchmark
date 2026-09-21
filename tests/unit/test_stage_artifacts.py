import json
import subprocess
from pathlib import Path

from agent_pipeline_benchmark.corpus import WorkItem, load_task
from agent_pipeline_benchmark.definitions import ExperimentDefinition, PipelineDefinition, StageDefinition
from agent_pipeline_benchmark.harnesses import Harness, StageOutcome, ZERO_COST
from agent_pipeline_benchmark.prompts import render_prompt
from agent_pipeline_benchmark.runner import HarnessResolver, run_experiment, run_pipeline_on_task, run_stage
from agent_pipeline_benchmark.snapshots import initialise_snapshot

CANNED_EVENTS = ({"type": "message_end", "message": {"role": "assistant"}}, {"type": "agent_end"})


def assert_the_stage_measures_recorded_in_the_record(record: dict) -> None:
    stage = record["stages"][0]
    assert stage["turns"] == 0
    assert stage["tool_calls"] == 0
    assert stage["end_reason"] == "finished"
    assert isinstance(stage["duration_seconds"], float) and stage["duration_seconds"] >= 0


def assert_the_stage_measures_omitted_in_the_record(record: dict) -> None:
    stage = record["stages"][0]
    assert "turns" not in stage
    assert "tool_calls" not in stage
    assert "end_reason" not in stage
    assert isinstance(stage["duration_seconds"], float) and stage["duration_seconds"] >= 0


class NoEventsHarness(Harness):
    def implement(
        self, work_item: WorkItem, working_copy: Path, prompt: str = "", model: str | None = None
    ) -> StageOutcome:
        return StageOutcome(cost=ZERO_COST)


class DiffApplyingHarness(Harness):
    def __init__(self) -> None:
        self.calls: list[tuple[WorkItem, Path, str]] = []
        self.commit_count_when_the_stage_ran: int | None = None

    def implement(
        self, work_item: WorkItem, working_copy: Path, prompt: str = "", model: str | None = None
    ) -> StageOutcome:
        self.calls.append((work_item, working_copy, prompt))
        apply_the_reference_diff(work_item, working_copy)
        plant_a_failing_pre_commit_hook(working_copy)
        self.commit_count_when_the_stage_ran = commit_count(working_copy)
        return StageOutcome(cost=ZERO_COST, events=CANNED_EVENTS)


def apply_the_reference_diff(work_item: WorkItem, working_copy: Path) -> None:
    subprocess.run(["git", "apply", str(work_item.reference_diff)], cwd=working_copy, capture_output=True, check=True)


def plant_a_failing_pre_commit_hook(working_copy: Path) -> None:
    hook = working_copy / ".git" / "hooks" / "pre-commit"
    hook.write_text("#!/bin/sh\nexit 1\n")
    hook.chmod(0o755)


def commit_count(working_copy: Path) -> int:
    output = run_git(working_copy, "rev-list", "--count", "HEAD")
    return int(output)


def run_git(working_copy: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments], cwd=working_copy, capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


def resolver_of(harness: Harness) -> HarnessResolver:
    def resolve(name: str) -> Harness:
        return harness

    return resolve


def a_fake_stage_directory(tmp_path: Path) -> Path:
    return tmp_path / "stages" / "implement"


def a_prompted_stage() -> StageDefinition:
    return StageDefinition(name="implement", harness="fake", model="deepseek-v4-flash", prompt="implement.md")


def a_plain_stage() -> StageDefinition:
    return StageDefinition(name="implement", harness="fake")


def test_the_working_copy_is_a_git_repository_with_one_initial_commit_before_the_first_stage(
    toy_corpus: Path,
) -> None:
    task = load_task(toy_corpus, "greeting")
    pipeline = PipelineDefinition(name="bare", stages=(a_plain_stage(),))
    harness = DiffApplyingHarness()

    run_pipeline_on_task("skeleton", pipeline, task, 1, corpus=toy_corpus, harness_named=resolver_of(harness))

    assert harness.commit_count_when_the_stage_ran == 1


def test_the_stage_diff_is_exactly_what_the_stage_changed(
    working_copy: Path, greet_work_item: WorkItem, tmp_path: Path
) -> None:
    initialise_snapshot(working_copy)
    stage_directory = a_fake_stage_directory(tmp_path)

    run_stage(a_plain_stage(), greet_work_item, working_copy, resolver_of(DiffApplyingHarness()), stage_directory)

    assert (stage_directory / "diff.patch").read_bytes() == greet_work_item.reference_diff.read_bytes()


def test_the_snapshot_commit_succeeds_despite_a_failing_pre_commit_hook(
    working_copy: Path, greet_work_item: WorkItem, tmp_path: Path
) -> None:
    initialise_snapshot(working_copy)
    stage_directory = a_fake_stage_directory(tmp_path)
    commits_before = commit_count(working_copy)

    run_stage(a_plain_stage(), greet_work_item, working_copy, resolver_of(DiffApplyingHarness()), stage_directory)

    assert commit_count(working_copy) == commits_before + 1


def test_the_stage_json_resolves_the_harness_model_and_rendered_prompt(
    working_copy: Path, greet_work_item: WorkItem, tmp_path: Path
) -> None:
    initialise_snapshot(working_copy)
    stage_directory = a_fake_stage_directory(tmp_path)
    harness = DiffApplyingHarness()

    run_stage(a_prompted_stage(), greet_work_item, working_copy, resolver_of(harness), stage_directory)

    resolved = json.loads((stage_directory / "stage.json").read_text())
    assert resolved == {
        "name": "implement",
        "harness": "fake",
        "model": "deepseek-v4-flash",
        "prompt": render_prompt("implement.md", greet_work_item),
    }
    assert harness.calls[0][2] == render_prompt("implement.md", greet_work_item)



def test_the_stage_json_omits_model_and_prompt_when_they_are_not_set(
    working_copy: Path, greet_work_item: WorkItem, tmp_path: Path
) -> None:
    initialise_snapshot(working_copy)
    stage_directory = a_fake_stage_directory(tmp_path)
    harness = DiffApplyingHarness()

    run_stage(a_plain_stage(), greet_work_item, working_copy, resolver_of(harness), stage_directory)

    resolved = json.loads((stage_directory / "stage.json").read_text())
    assert resolved == {"name": "implement", "harness": "fake"}
    assert harness.calls[0][2] == ""



def test_a_stage_records_a_wall_clock_duration(
    working_copy: Path, greet_work_item: WorkItem, tmp_path: Path
) -> None:
    initialise_snapshot(working_copy)
    stage_directory = a_fake_stage_directory(tmp_path)

    record = run_stage(a_plain_stage(), greet_work_item, working_copy, resolver_of(DiffApplyingHarness()), stage_directory)

    assert isinstance(record.duration_seconds, float) and record.duration_seconds > 0


def test_the_record_carries_the_measures_of_the_agents_run_per_stage(
    toy_corpus: Path,
) -> None:
    task = load_task(toy_corpus, "greeting")
    pipeline = PipelineDefinition(name="bare", stages=(a_plain_stage(),))

    record = run_pipeline_on_task(
        "skeleton", pipeline, task, 1, corpus=toy_corpus, harness_named=resolver_of(DiffApplyingHarness())
    )

    assert_the_stage_measures_recorded_in_the_record(record.as_json()["work_items"][0])


def test_the_record_omits_the_measures_when_the_stage_has_no_events(
    toy_corpus: Path,
) -> None:
    task = load_task(toy_corpus, "greeting")
    pipeline = PipelineDefinition(name="bare", stages=(a_plain_stage(),))

    record = run_pipeline_on_task(
        "skeleton", pipeline, task, 1, corpus=toy_corpus, harness_named=resolver_of(NoEventsHarness())
    )

    assert_the_stage_measures_omitted_in_the_record(record.as_json()["work_items"][0])


def test_the_event_stream_is_stored_raw_one_json_object_per_line(
    working_copy: Path, greet_work_item: WorkItem, tmp_path: Path
) -> None:
    initialise_snapshot(working_copy)
    stage_directory = a_fake_stage_directory(tmp_path)

    run_stage(a_plain_stage(), greet_work_item, working_copy, resolver_of(DiffApplyingHarness()), stage_directory)

    lines = (stage_directory / "events.jsonl").read_text().splitlines()
    assert lines == [json.dumps(event) for event in CANNED_EVENTS]


def test_the_run_directory_holds_no_working_copy_and_nothing_beyond_the_recorded_files(
    tmp_path: Path, toy_corpus: Path
) -> None:
    pipeline = PipelineDefinition(name="bare", stages=(StageDefinition(name="implement", harness="reference-solution"),))
    experiment = ExperimentDefinition(
        name="skeleton", corpus=toy_corpus, pipelines=(pipeline,), tasks=("greeting",), repeats=1
    )
    results = tmp_path / "results"

    written = run_experiment(experiment, results)

    run_directory = written[0].parent
    actual = sorted(str(path.relative_to(run_directory)) for path in run_directory.rglob("*") if path.is_file())
    assert actual == [
        "record.json",
        "work-items/01-greet/scoring/hidden-tests.json",
        "work-items/01-greet/stages/01-implement/diff.patch",
        "work-items/01-greet/stages/01-implement/events.jsonl",
        "work-items/01-greet/stages/01-implement/stage.json",
    ]
