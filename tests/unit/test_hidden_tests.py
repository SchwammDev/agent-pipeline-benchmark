from pathlib import Path

from agent_pipeline_benchmark.corpus import WorkItem
from agent_pipeline_benchmark.harnesses import ReferenceSolution
from agent_pipeline_benchmark.hidden_tests import TestMovements, passing_test_ids, test_movements


def test_passing_test_ids_on_an_untouched_working_copy_reports_only_the_package_test(
    working_copy: Path, greet_work_item: WorkItem
) -> None:
    assert passing_test_ids(greet_work_item, working_copy) == frozenset(
        {"tests.test_package::test_package_is_importable"}
    )


def test_passing_test_ids_includes_both_hidden_tests_after_the_reference_solution(
    working_copy: Path, greet_work_item: WorkItem
) -> None:
    ReferenceSolution().implement(greet_work_item, working_copy)

    passing = passing_test_ids(greet_work_item, working_copy)

    assert len(passing) == 3
    assert "tests.test_greet::test_greets_the_given_name" in passing
    assert "tests.test_greet::test_uses_the_name_exactly_as_given" in passing
    assert "tests.test_package::test_package_is_importable" in passing


def test_hidden_test_files_are_removed_after_scoring_a_failing_copy(
    working_copy: Path, greet_work_item: WorkItem
) -> None:
    passing_test_ids(greet_work_item, working_copy)

    assert_hidden_test_files_absent(working_copy, greet_work_item)


def test_hidden_test_files_are_removed_after_scoring_a_passing_copy(
    working_copy: Path, greet_work_item: WorkItem
) -> None:
    ReferenceSolution().implement(greet_work_item, working_copy)

    passing_test_ids(greet_work_item, working_copy)

    assert_hidden_test_files_absent(working_copy, greet_work_item)


def test_reference_solution_moves_both_hidden_tests_from_regressed_to_progressed(
    working_copy: Path, greet_work_item: WorkItem
) -> None:
    before = passing_test_ids(greet_work_item, working_copy)
    ReferenceSolution().implement(greet_work_item, working_copy)
    after = passing_test_ids(greet_work_item, working_copy)

    movements = test_movements(before, after)

    assert_both_hidden_tests_were_newly_progressed(movements)


def test_untouched_working_copy_preserves_but_never_progresses(
    working_copy: Path, greet_work_item: WorkItem
) -> None:
    before = passing_test_ids(greet_work_item, working_copy)
    after = passing_test_ids(greet_work_item, working_copy)

    movements = test_movements(before, after)

    assert_unchanged_copy_preserves_without_progressing(movements)


def test_work_item_already_satisfied_preserves_all_tests_without_progressing(
    already_done_working_copy: Path, already_done_work_item: WorkItem
) -> None:
    before = passing_test_ids(already_done_work_item, already_done_working_copy)
    after = passing_test_ids(already_done_work_item, already_done_working_copy)

    movements = test_movements(before, after)

    assert_all_tests_preserved_without_progress(movements)


def test_movements_reports_a_replaced_test_as_regressed_and_progressed() -> None:
    movements = test_movements(frozenset({"a"}), frozenset({"b"}))

    assert_a_replaced_test_counts_as_progress_and_regression(movements)


def test_movements_reports_an_added_test_as_progressed_only() -> None:
    movements = test_movements(frozenset({"a"}), frozenset({"a", "b"}))

    assert_an_added_test_counts_only_as_progress(movements)


def assert_hidden_test_files_absent(working_copy: Path, work_item: WorkItem) -> None:
    for hidden_test_file in work_item.hidden_tests.rglob("*"):
        if hidden_test_file.is_file():
            copied_path = working_copy / "tests" / hidden_test_file.relative_to(work_item.hidden_tests)
            assert not copied_path.exists()


def assert_both_hidden_tests_were_newly_progressed(movements: TestMovements) -> None:
    assert movements.progressed == 2
    assert movements.preserved == 1
    assert movements.regressed == 0
    assert movements.solved is True


def assert_unchanged_copy_preserves_without_progressing(movements: TestMovements) -> None:
    assert movements.progressed == 0
    assert movements.preserved == 1
    assert movements.regressed == 0
    assert movements.solved is False


def assert_all_tests_preserved_without_progress(movements: TestMovements) -> None:
    assert movements.progressed == 0
    assert movements.preserved == 3
    assert movements.regressed == 0
    assert movements.solved is False


def assert_a_replaced_test_counts_as_progress_and_regression(movements: TestMovements) -> None:
    assert movements.progressed == 1
    assert movements.preserved == 0
    assert movements.regressed == 1
    assert movements.solved is False


def assert_an_added_test_counts_only_as_progress(movements: TestMovements) -> None:
    assert movements.progressed == 1
    assert movements.preserved == 1
    assert movements.regressed == 0
    assert movements.solved is True
