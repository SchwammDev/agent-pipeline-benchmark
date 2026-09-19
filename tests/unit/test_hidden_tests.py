from pathlib import Path

from agent_pipeline_benchmark.corpus import WorkItem
from agent_pipeline_benchmark.harnesses import ReferenceSolution
from agent_pipeline_benchmark.hidden_tests import hidden_tests_pass


def test_hidden_tests_pass_once_the_reference_solution_is_applied(
    working_copy: Path, greet_work_item: WorkItem
) -> None:
    ReferenceSolution().implement(greet_work_item, working_copy)

    assert hidden_tests_pass(greet_work_item, working_copy) is True


def test_hidden_tests_fail_on_an_untouched_working_copy(working_copy: Path, greet_work_item: WorkItem) -> None:
    assert hidden_tests_pass(greet_work_item, working_copy) is False


def test_copied_hidden_tests_are_removed_after_a_pass(working_copy: Path, greet_work_item: WorkItem) -> None:
    ReferenceSolution().implement(greet_work_item, working_copy)

    hidden_tests_pass(greet_work_item, working_copy)

    assert_hidden_test_files_absent(working_copy, greet_work_item)


def test_copied_hidden_tests_are_removed_after_a_failure(working_copy: Path, greet_work_item: WorkItem) -> None:
    hidden_tests_pass(greet_work_item, working_copy)

    assert_hidden_test_files_absent(working_copy, greet_work_item)


def assert_hidden_test_files_absent(working_copy: Path, work_item: WorkItem) -> None:
    for hidden_test_file in work_item.hidden_tests.rglob("*"):
        if hidden_test_file.is_file():
            copied_path = working_copy / "tests" / hidden_test_file.relative_to(work_item.hidden_tests)
            assert not copied_path.exists()
