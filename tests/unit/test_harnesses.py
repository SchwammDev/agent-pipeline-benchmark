from pathlib import Path

import pytest
from helpers import snapshot_of

from agent_pipeline_benchmark.corpus import WorkItem
from agent_pipeline_benchmark.harnesses import (
    ZERO_COST,
    DoNothing,
    ReferenceSolution,
    ReferenceSolutionDoesNotApply,
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
    cost = ReferenceSolution().implement(greet_work_item, working_copy)

    assert cost == ZERO_COST


def test_do_nothing_harness_reports_zero_cost(working_copy: Path, greet_work_item: WorkItem) -> None:
    cost = DoNothing().implement(greet_work_item, working_copy)

    assert cost == ZERO_COST


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


def test_harness_named_with_an_unknown_name_reports_the_known_names() -> None:
    with pytest.raises(ValueError, match="reference-solution.*do-nothing|do-nothing.*reference-solution"):
        harness_named("magic-harness")


def assert_file_contains(path: Path, text: str) -> None:
    assert text in path.read_text()
