import json
from pathlib import Path

from helpers import snapshot_of

from agent_pipeline_benchmark.corpus import WorkItem, load_task
from agent_pipeline_benchmark.definitions import ExperimentDefinition, PipelineDefinition, StageDefinition
from agent_pipeline_benchmark.harnesses import Harness, StageCost, ZERO_COST
from agent_pipeline_benchmark.runner import HarnessResolver, run_experiment, run_pipeline_on_task


class RecordingHarness(Harness):
    def __init__(self, name: str, cost: StageCost, calls: list[tuple[str, str, Path]]) -> None:
        self.name = name
        self.cost = cost
        self.calls = calls

    def implement(self, work_item: WorkItem, working_copy: Path) -> StageCost:
        self.calls.append((self.name, work_item.name, working_copy))
        return self.cost


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
    run_numbers = sorted(json.loads(path.read_text())["run_number"] for path in written)
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

    record = run_pipeline_on_task("skeleton", pipeline, task, 1)

    assert record.as_json() == {
        "experiment": "skeleton",
        "pipeline": "bare",
        "task": "greeting",
        "run_id": record.run_id,
        "run_number": 1,
        "work_items": [
            {
                "name": "01-greet",
                "solved": True,
                "stages": [{"name": "implement", "harness": "reference-solution", "tokens": 0, "usd": 0.0}],
            }
        ],
        "totals": {"solved": 1, "tokens": 0, "usd": 0.0},
    }
