import json
import subprocess
from datetime import datetime
from pathlib import Path

from helpers import snapshot_of

from agent_pipeline_benchmark.corpus import WorkItem, load_task
from agent_pipeline_benchmark.definitions import ExperimentDefinition, PipelineDefinition, StageDefinition
from agent_pipeline_benchmark.harnesses import Harness, StageCost, StageOutcome, ZERO_COST
from agent_pipeline_benchmark.runner import HarnessResolver, run_experiment, run_pipeline_on_task


def benchmark_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).parents[2],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


class RecordingHarness(Harness):
    def __init__(self, name: str, cost: StageCost, calls: list[tuple[str, str, Path]]) -> None:
        self.name = name
        self.cost = cost
        self.calls = calls

    def implement(
        self, work_item: WorkItem, working_copy: Path, prompt: str = "", model: str | None = None
    ) -> StageOutcome:
        self.calls.append((self.name, work_item.name, working_copy))
        return StageOutcome(cost=self.cost)


def resolver_of(harnesses: dict[str, Harness]) -> HarnessResolver:
    def resolve(name: str) -> Harness:
        return harnesses[name]

    return resolve


def assert_single_record_written_under(
    written: list[Path], results: Path, *, experiment: str, pipeline: str, task: str
) -> None:
    assert len(written) == 1
    relative = written[0].relative_to(results)
    assert relative.parts == (experiment, pipeline, task, relative.parts[3], "record.json")
    assert written[0].exists()


def assert_two_distinct_runs_numbered_one_and_two(written: list[Path]) -> None:
    assert len(written) == 2
    assert written[0].parent != written[1].parent
    run_numbers = sorted(json.loads(path.read_text())["identity"]["run_number"] for path in written)
    assert run_numbers == [1, 2]


def test_a_run_writes_its_record_under_experiment_pipeline_task_and_run_id(
    tmp_path: Path, toy_corpus: Path
) -> None:
    pipeline = PipelineDefinition(name="bare", stages=(StageDefinition(name="implement", harness="do-nothing"),))
    experiment = ExperimentDefinition(
        name="skeleton", corpus=toy_corpus, pipelines=(pipeline,), tasks=("greeting",), repeats=1
    )
    results = tmp_path / "results"

    written = run_experiment(experiment, results)

    assert_single_record_written_under(written, results, experiment="skeleton", pipeline="bare", task="greeting")


def test_repeats_write_two_runs_numbered_one_and_two_in_different_directories(
    tmp_path: Path, toy_corpus: Path
) -> None:
    pipeline = PipelineDefinition(name="bare", stages=(StageDefinition(name="implement", harness="do-nothing"),))
    experiment = ExperimentDefinition(
        name="skeleton", corpus=toy_corpus, pipelines=(pipeline,), tasks=("greeting",), repeats=2
    )
    results = tmp_path / "results"

    written = run_experiment(experiment, results)

    assert_two_distinct_runs_numbered_one_and_two(written)


def test_the_stages_of_a_pipeline_are_applied_in_order_to_the_same_working_copy(
    toy_corpus: Path,
) -> None:
    task = load_task(toy_corpus, "greeting")
    pipeline = PipelineDefinition(
        name="bare",
        stages=(StageDefinition(name="first", harness="harness-a"), StageDefinition(name="second", harness="harness-b")),
    )
    calls: list[tuple[str, str, Path]] = []
    resolve = resolver_of(
        {
            "harness-a": RecordingHarness("harness-a", ZERO_COST, calls),
            "harness-b": RecordingHarness("harness-b", ZERO_COST, calls),
        }
    )

    run_pipeline_on_task("skeleton", pipeline, task, 1, harness_named=resolve)

    harness_names_in_order = [name for name, _, _ in calls]
    working_copies_seen = {working_copy for _, _, working_copy in calls}
    assert harness_names_in_order == ["harness-a", "harness-b"]
    assert len(working_copies_seen) == 1


def test_totals_sum_tokens_and_usd_over_all_stages_and_work_items(toy_corpus: Path) -> None:
    task = load_task(toy_corpus, "greeting")
    pipeline = PipelineDefinition(
        name="bare",
        stages=(StageDefinition(name="first", harness="harness-a"), StageDefinition(name="second", harness="harness-b")),
    )
    resolve = resolver_of(
        {
            "harness-a": RecordingHarness("harness-a", StageCost(tokens=100, usd=0.5), []),
            "harness-b": RecordingHarness("harness-b", StageCost(tokens=50, usd=0.25), []),
        }
    )

    record = run_pipeline_on_task("skeleton", pipeline, task, 1, harness_named=resolve)

    assert record.totals() == {"solved": 0, "tokens": 150, "usd": 0.75}


def test_the_tasks_repository_is_untouched_by_a_run_with_the_reference_solution_harness(
    toy_corpus: Path,
) -> None:
    task = load_task(toy_corpus, "greeting")
    pipeline = PipelineDefinition(name="bare", stages=(StageDefinition(name="implement", harness="reference-solution"),))
    before = snapshot_of(task.repository)

    run_pipeline_on_task("skeleton", pipeline, task, 1)

    assert snapshot_of(task.repository) == before


def test_the_record_marks_the_work_item_solved_after_the_reference_solution(toy_corpus: Path) -> None:
    task = load_task(toy_corpus, "greeting")
    pipeline = PipelineDefinition(name="bare", stages=(StageDefinition(name="implement", harness="reference-solution"),))

    record = run_pipeline_on_task("skeleton", pipeline, task, 1)

    assert [item.solved for item in record.work_items] == [True]


def test_the_record_marks_the_work_item_unsolved_after_doing_nothing(toy_corpus: Path) -> None:
    task = load_task(toy_corpus, "greeting")
    pipeline = PipelineDefinition(name="bare", stages=(StageDefinition(name="implement", harness="do-nothing"),))

    record = run_pipeline_on_task("skeleton", pipeline, task, 1)

    assert [item.solved for item in record.work_items] == [False]


def test_the_records_json_has_the_shape_in_the_spec(toy_corpus: Path) -> None:
    task = load_task(toy_corpus, "greeting")
    pipeline = PipelineDefinition(name="bare", stages=(StageDefinition(name="implement", harness="reference-solution"),))

    record = run_pipeline_on_task("skeleton", pipeline, task, 1, corpus=toy_corpus)

    assert_the_record_keeps_the_spec_shape(record, corpus=toy_corpus, harnesses=["reference-solution"])


def assert_the_record_keeps_the_spec_shape(
    record: object, *, corpus: Path, harnesses: list[str]
) -> None:
    js = record.as_json()
    identity = js["identity"]
    assert set(js) == {"identity", "work_items", "totals"}
    assert_the_identity_group(identity, record, corpus=corpus, harnesses=harnesses)
    assert js["work_items"] == [
        {
            "name": "01-greet",
            "solved": True,
            "progressed": 2,
            "preserved": 1,
            "stages": [{"name": "implement", "harness": "reference-solution", "tokens": 0, "usd": 0.0}],
        }
    ]
    assert js["totals"] == {"solved": 1, "tokens": 0, "usd": 0.0}


def assert_the_identity_group(
    identity: dict, record: object, *, corpus: Path, harnesses: list[str]
) -> None:
    assert identity["experiment"] == "skeleton"
    assert identity["pipeline"] == "bare"
    assert identity["task"] == "greeting"
    assert identity["run_id"] == record.run_id
    assert identity["run_number"] == 1
    assert identity["corpus"] == str(corpus)
    assert identity["harnesses"] == harnesses
    assert "models" not in identity
    assert identity["benchmark_commit"] == benchmark_head()
    assert isinstance(identity["start"], str) and isinstance(identity["end"], str)


def test_the_identity_spans_the_run_end_is_strictly_after_start(toy_corpus: Path) -> None:
    task = load_task(toy_corpus, "greeting")
    pipeline = PipelineDefinition(name="bare", stages=(StageDefinition(name="implement", harness="reference-solution"),))

    record = run_pipeline_on_task("skeleton", pipeline, task, 1, corpus=toy_corpus)

    assert_the_run_spans_end_after_start(record.as_json()["identity"])


def assert_the_run_spans_end_after_start(identity: dict) -> None:
    start = datetime.fromisoformat(identity["start"])
    end = datetime.fromisoformat(identity["end"])
    assert start.tzinfo is not None and end.tzinfo is not None
    assert end > start


def test_the_identity_lists_the_harnesses_and_models_of_the_stages(toy_corpus: Path) -> None:
    task = load_task(toy_corpus, "greeting")
    pipeline = PipelineDefinition(
        name="bare",
        stages=(
            StageDefinition(name="first", harness="harness-a"),
            StageDefinition(name="second", harness="harness-b", model="model-b"),
        ),
    )
    resolve = resolver_of(
        {
            "harness-a": RecordingHarness("harness-a", ZERO_COST, []),
            "harness-b": RecordingHarness("harness-b", ZERO_COST, []),
        }
    )

    record = run_pipeline_on_task("skeleton", pipeline, task, 1, corpus=toy_corpus, harness_named=resolve)

    assert record.as_json()["identity"]["harnesses"] == ["harness-a", "harness-b"]
    assert record.as_json()["identity"]["models"] == ["model-b"]


def test_the_identity_omits_models_when_no_stage_has_one(toy_corpus: Path) -> None:
    task = load_task(toy_corpus, "greeting")
    pipeline = PipelineDefinition(name="bare", stages=(StageDefinition(name="implement", harness="do-nothing"),))

    record = run_pipeline_on_task("skeleton", pipeline, task, 1, corpus=toy_corpus)

    assert "models" not in record.as_json()["identity"]


def test_the_record_json_has_no_null_and_no_unmeasured_field(toy_corpus: Path) -> None:
    task = load_task(toy_corpus, "greeting")
    pipeline = PipelineDefinition(name="bare", stages=(StageDefinition(name="implement", harness="do-nothing"),))

    record = run_pipeline_on_task("skeleton", pipeline, task, 1, corpus=toy_corpus)

    assert_no_unmeasured(record.as_json())


def assert_no_unmeasured(node: object, *, unmeasured: set[str] | None = None) -> None:
    unmeasured = unmeasured or {"hook_events", "static_measures", "image_digest"}
    if isinstance(node, dict):
        for key, value in node.items():
            assert key not in unmeasured, f"the unmeasured field {key!r} appeared in the record"
            assert value is not None, f"the key {key!r} holds null"
            assert_no_unmeasured(value, unmeasured=unmeasured)
    elif isinstance(node, list):
        for item in node:
            assert_no_unmeasured(item, unmeasured=unmeasured)


def test_reference_solution_reports_two_progressed_and_one_preserved_test(toy_corpus: Path) -> None:
    task = load_task(toy_corpus, "greeting")
    pipeline = PipelineDefinition(name="bare", stages=(StageDefinition(name="implement", harness="reference-solution"),))

    record = run_pipeline_on_task("skeleton", pipeline, task, 1)

    assert [(item.progressed, item.preserved, item.solved) for item in record.work_items] == [(2, 1, True)]


def test_doing_nothing_reports_no_progress_but_preserves_the_package_test(toy_corpus: Path) -> None:
    task = load_task(toy_corpus, "greeting")
    pipeline = PipelineDefinition(name="bare", stages=(StageDefinition(name="implement", harness="do-nothing"),))

    record = run_pipeline_on_task("skeleton", pipeline, task, 1)

    assert [(item.progressed, item.preserved, item.solved) for item in record.work_items] == [(0, 1, False)]


def test_an_already_done_task_preserves_all_tests_without_progressing(toy_corpus: Path) -> None:
    task = load_task(toy_corpus, "already-done")
    pipeline = PipelineDefinition(name="bare", stages=(StageDefinition(name="implement", harness="do-nothing"),))

    record = run_pipeline_on_task("skeleton", pipeline, task, 1)

    assert [(item.progressed, item.preserved, item.solved) for item in record.work_items] == [(0, 3, False)]


def test_regression_progresses_hidden_tests_but_regresses_the_package_test(toy_corpus: Path) -> None:
    task = load_task(toy_corpus, "greeting")
    pipeline = PipelineDefinition(
        name="bare", stages=(StageDefinition(name="implement", harness="reference-solution-then-regression"),)
    )

    record = run_pipeline_on_task("skeleton", pipeline, task, 1)

    assert [(item.progressed, item.preserved, item.solved) for item in record.work_items] == [(2, 0, False)]


def test_the_recorded_stage_diff_holds_the_reference_solution(tmp_path: Path, toy_corpus: Path) -> None:
    task = load_task(toy_corpus, "greeting")
    pipeline = PipelineDefinition(name="bare", stages=(StageDefinition(name="implement", harness="reference-solution"),))
    results = tmp_path / "results"

    run_pipeline_on_task("skeleton", pipeline, task, 1, results=results)

    diff = the_recorded_stage_diff(results, task="greeting")
    reference = toy_corpus / "greeting" / "work-items" / "01-greet" / "reference.diff"
    assert diff.read_bytes() == reference.read_bytes()


def test_the_recorded_stage_diff_holds_the_regression_change(tmp_path: Path, toy_corpus: Path) -> None:
    task = load_task(toy_corpus, "greeting")
    pipeline = PipelineDefinition(
        name="bare", stages=(StageDefinition(name="implement", harness="reference-solution-then-regression"),)
    )
    results = tmp_path / "results"

    run_pipeline_on_task("skeleton", pipeline, task, 1, results=results)

    diff = the_recorded_stage_diff(results, task="greeting")
    assert "def test_broken" in diff.read_text()


def the_recorded_stage_diff(results: Path, *, task: str) -> Path:
    stage_directories = list(results.glob(f"skeleton/bare/{task}/*/work-items/01-greet/stages/01-implement"))
    assert len(stage_directories) == 1, f"expected exactly one recorded stage, found {len(stage_directories)}"
    return stage_directories[0] / "diff.patch"
