import json
from pathlib import Path

from agent_pipeline_benchmark.corpus import load_task
from agent_pipeline_benchmark.definitions import PipelineDefinition, StageDefinition
from agent_pipeline_benchmark.runner import run_pipeline_on_task

TOY_CORPUS = Path(__file__).parent.parent / "toy-corpus"
HIDDEN_TEST_NAMES = ["test_greets_the_given_name", "test_uses_the_name_exactly_as_given"]


def run_benchmark_with(tmp_path: Path, *, harness: str) -> Path:
    task = load_task(TOY_CORPUS, "greeting")
    pipeline = PipelineDefinition(name="bare", stages=(StageDefinition(name="implement", harness=harness),))
    results = tmp_path / "results"

    run_pipeline_on_task("skeleton", pipeline, task, 1, corpus=TOY_CORPUS, results=results)

    scoring_files = list(results.glob("skeleton/bare/greeting/*/work-items/01-greet/scoring/hidden-tests.json"))
    assert len(scoring_files) == 1, f"expected one scoring file, found {len(scoring_files)}"
    return scoring_files[0]


def the_recorded_verdicts(scoring: dict) -> dict[str, bool]:
    return {test["name"]: bool(test["passed"]) for test in scoring["tests"]}


def assert_every_hidden_test_recorded_as(scoring: dict, *, passed: bool) -> None:
    verdicts = the_recorded_verdicts(scoring)
    for hidden_test_id in HIDDEN_TEST_NAMES:
        matching = [verdict for name, verdict in verdicts.items() if name.endswith(hidden_test_id)]
        assert matching, f"hidden test {hidden_test_id!r} is missing from the verdicts: {verdicts}"
        assert matching[0] is passed, (
            f"hidden test {hidden_test_id!r} was recorded as {matching[0]}, expected {passed}: {verdicts}"
        )


def test_after_a_reference_solution_run_the_scoring_file_records_every_hidden_test_as_passed(
    tmp_path: Path,
) -> None:
    scoring_file = run_benchmark_with(tmp_path, harness="reference-solution")

    assert_every_hidden_test_recorded_as(json.loads(scoring_file.read_text()), passed=True)


def test_after_a_do_nothing_run_the_scoring_file_records_hidden_tests_as_not_passed(
    tmp_path: Path,
) -> None:
    scoring_file = run_benchmark_with(tmp_path, harness="do-nothing")

    assert_every_hidden_test_recorded_as(json.loads(scoring_file.read_text()), passed=False)


def test_the_scoring_file_sits_under_the_work_items_scoring_directory(tmp_path: Path) -> None:
    scoring_file = run_benchmark_with(tmp_path, harness="reference-solution")

    assert scoring_file == scoring_file.parent.parent / "scoring" / "hidden-tests.json"
    assert scoring_file.parent.parent.name == "01-greet"
