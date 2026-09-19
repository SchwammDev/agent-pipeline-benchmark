from pathlib import Path

import pytest

from agent_pipeline_benchmark.corpus import Task, WorkItem, load_task


def test_the_toy_greeting_task_loads_with_its_repository_and_one_work_item(toy_corpus: Path) -> None:
    task = load_task(toy_corpus, "greeting")

    assert_task_matches(
        task,
        name="greeting",
        repository=toy_corpus / "greeting" / "repo",
        work_item_names=["01-greet"],
    )


def test_a_work_items_three_parts_point_at_the_right_files(toy_corpus: Path) -> None:
    task = load_task(toy_corpus, "greeting")

    work_item = task.work_items[0]

    assert_paths_match(
        work_item,
        requirement=toy_corpus / "greeting" / "work-items" / "01-greet" / "requirement.md",
        hidden_tests=toy_corpus / "greeting" / "work-items" / "01-greet" / "tests",
        reference_diff=toy_corpus / "greeting" / "work-items" / "01-greet" / "reference.diff",
    )


def test_work_items_come_back_in_directory_order(tmp_path: Path) -> None:
    corpus = tmp_path
    a_task(corpus, "two-items", work_items=["02-second", "01-first"])

    task = load_task(corpus, "two-items")

    assert [item.name for item in task.work_items] == ["01-first", "02-second"]


def test_an_unknown_task_name_is_reported_with_the_name_and_corpus_path(tmp_path: Path) -> None:
    corpus = tmp_path

    with pytest.raises(FileNotFoundError) as error:
        load_task(corpus, "does-not-exist")

    assert "does-not-exist" in str(error.value)
    assert str(corpus) in str(error.value)


def test_a_work_item_missing_a_part_is_reported_by_part_name(tmp_path: Path) -> None:
    corpus = tmp_path
    a_task(corpus, "incomplete", work_items=["01-first"])
    (corpus / "incomplete" / "work-items" / "01-first" / "requirement.md").unlink()

    with pytest.raises(ValueError) as error:
        load_task(corpus, "incomplete")

    assert "requirement" in str(error.value)
    assert "01-first" in str(error.value)


def assert_task_matches(task: Task, *, name: str, repository: Path, work_item_names: list[str]) -> None:
    actual = (task.name, task.repository, [item.name for item in task.work_items])
    expected = (name, repository, work_item_names)
    assert actual == expected


def assert_paths_match(
    work_item: WorkItem, *, requirement: Path, hidden_tests: Path, reference_diff: Path
) -> None:
    actual = (work_item.requirement, work_item.hidden_tests, work_item.reference_diff)
    expected = (requirement, hidden_tests, reference_diff)
    assert actual == expected


def a_task(corpus: Path, name: str, *, work_items: list[str]) -> Path:
    task_dir = corpus / name
    (task_dir / "repo").mkdir(parents=True)
    for work_item in work_items:
        a_work_item(task_dir, work_item)
    return task_dir


def a_work_item(task_dir: Path, name: str) -> Path:
    work_item_dir = task_dir / "work-items" / name
    (work_item_dir / "tests").mkdir(parents=True)
    (work_item_dir / "requirement.md").write_text("a requirement")
    (work_item_dir / "reference.diff").write_text("a diff")
    return work_item_dir
