from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class WorkItem:
    name: str
    requirement: Path
    hidden_tests: Path
    reference_diff: Path


@dataclass(frozen=True)
class Task:
    name: str
    repository: Path
    work_items: tuple[WorkItem, ...]


def load_task(corpus: Path, name: str) -> Task:
    task_dir = corpus / name
    if not task_dir.is_dir():
        raise FileNotFoundError(f"task {name!r} not found in corpus {corpus}")
    work_items_dir = task_dir / "work-items"
    work_item_dirs = sorted(path for path in work_items_dir.iterdir() if path.is_dir())
    return Task(
        name=name,
        repository=task_dir / "repo",
        work_items=tuple(load_work_item(work_item_dir) for work_item_dir in work_item_dirs),
    )


def load_work_item(work_item_dir: Path) -> WorkItem:
    return WorkItem(
        name=work_item_dir.name,
        requirement=required_part(work_item_dir, "requirement.md", "requirement"),
        hidden_tests=required_part(work_item_dir, "tests", "hidden_tests"),
        reference_diff=required_part(work_item_dir, "reference.diff", "reference_diff"),
    )


def required_part(work_item_dir: Path, filename: str, part_name: str) -> Path:
    path = work_item_dir / filename
    if not path.exists():
        raise ValueError(f"work item {work_item_dir.name!r} is missing its {part_name!r} part")
    return path
