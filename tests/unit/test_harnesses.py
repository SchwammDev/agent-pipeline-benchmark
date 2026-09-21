from pathlib import Path

import pytest
from helpers import snapshot_of

from agent_pipeline_benchmark.corpus import WorkItem
from agent_pipeline_benchmark.harnesses import (
    ZERO_COST,
    DoNothing,
    ReferenceSolution,
    ReferenceSolutionDoesNotApply,
    ReferenceSolutionThenRegression,
    harness_named,
)


def test_reference_solution_harness_leaves_the_reference_solution_in_the_working_copy(
    working_copy: Path, greet_work_item: WorkItem
) -> None:
    ReferenceSolution().implement(greet_work_item, working_copy)

    assert_file_contains(working_copy / "src" / "greeting" / "__init__.py", 'return f"Hello, {name}!"')


def test_do_nothing_harness_leaves_the_working_copy_unchanged(working_copy: Path, greet_work_item: WorkItem) -> None:
    before = snapshot_of(working_copy)

    DoNothing().implement(greet_work_item, working_copy)

    assert snapshot_of(working_copy) == before


def test_reference_solution_harness_reports_zero_cost(working_copy: Path, greet_work_item: WorkItem) -> None:
    outcome = ReferenceSolution().implement(greet_work_item, working_copy)

    assert outcome.cost == ZERO_COST


def test_do_nothing_harness_reports_zero_cost(working_copy: Path, greet_work_item: WorkItem) -> None:
    outcome = DoNothing().implement(greet_work_item, working_copy)

    assert outcome.cost == ZERO_COST


def test_a_reference_diff_that_does_not_apply_is_reported_with_the_work_item(
    working_copy: Path, greet_work_item: WorkItem
) -> None:
    (working_copy / "src" / "greeting" / "__init__.py").write_text("already conflicting content\n")

    with pytest.raises(ReferenceSolutionDoesNotApply, match="01-greet"):
        ReferenceSolution().implement(greet_work_item, working_copy)


def test_harness_named_reference_solution_returns_a_reference_solution_harness() -> None:
    assert isinstance(harness_named("reference-solution"), ReferenceSolution)


def test_harness_named_do_nothing_returns_a_do_nothing_harness() -> None:
    assert isinstance(harness_named("do-nothing"), DoNothing)


def test_reference_solution_then_regression_leaves_the_reference_solution_in_the_working_copy(
    working_copy: Path, greet_work_item: WorkItem
) -> None:
    ReferenceSolutionThenRegression().implement(greet_work_item, working_copy)

    assert_file_contains(working_copy / "src" / "greeting" / "__init__.py", 'return f"Hello, {name}!"')


def test_reference_solution_then_regression_replaces_the_repository_owned_test_file(
    working_copy: Path, greet_work_item: WorkItem
) -> None:
    ReferenceSolutionThenRegression().implement(greet_work_item, working_copy)

    test_file = working_copy / "tests" / "test_package.py"
    assert "def test_package_is_importable()" not in test_file.read_text()
    assert_file_contains(test_file, "def test_broken() -> None:\n    assert False")


def test_reference_solution_then_regression_reports_zero_cost(
    working_copy: Path, greet_work_item: WorkItem
) -> None:
    outcome = ReferenceSolutionThenRegression().implement(greet_work_item, working_copy)

    assert outcome.cost == ZERO_COST


def test_harness_named_reference_solution_then_regression_returns_a_reference_solution_then_regression_harness() -> None:
    assert isinstance(
        harness_named("reference-solution-then-regression"), ReferenceSolutionThenRegression
    )


def test_harness_named_with_an_unknown_name_reports_the_known_names() -> None:
    with pytest.raises(ValueError, match="reference-solution.*do-nothing|do-nothing.*reference-solution"):
        harness_named("magic-harness")


def assert_file_contains(path: Path, text: str) -> None:
    assert text in path.read_text()
