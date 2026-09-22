import shutil
from pathlib import Path

import pytest

from agent_pipeline_benchmark.corpus import WorkItem
from agent_pipeline_benchmark.development_environments import UVDevelopmentEnvironment

TOY_CORPUS = Path(__file__).parent.parent / "toy-corpus"
GREETING_WORK_ITEM = TOY_CORPUS / "greeting" / "work-items" / "01-greet"
GREETING_REPO = TOY_CORPUS / "greeting" / "repo"
ALREADY_DONE_WORK_ITEM = TOY_CORPUS / "already-done" / "work-items" / "01-greet"
ALREADY_DONE_REPO = TOY_CORPUS / "already-done" / "repo"


@pytest.fixture
def toy_corpus() -> Path:
    return TOY_CORPUS


@pytest.fixture
def working_copy(tmp_path: Path) -> Path:
    destination = tmp_path / "working-copy"
    shutil.copytree(GREETING_REPO, destination, ignore=shutil.ignore_patterns(".venv", "__pycache__"))
    UVDevelopmentEnvironment(destination).prepare()
    return destination


@pytest.fixture
def development_environment(working_copy: Path) -> UVDevelopmentEnvironment:
    return UVDevelopmentEnvironment(working_copy)


@pytest.fixture
def greet_work_item() -> WorkItem:
    return WorkItem(
        name="01-greet",
        requirement=GREETING_WORK_ITEM / "requirement.md",
        hidden_tests=GREETING_WORK_ITEM / "tests",
        reference_diff=GREETING_WORK_ITEM / "reference.diff",
    )


@pytest.fixture
def already_done_working_copy(tmp_path: Path) -> Path:
    destination = tmp_path / "working-copy"
    shutil.copytree(
        ALREADY_DONE_REPO, destination, ignore=shutil.ignore_patterns(".venv", "__pycache__")
    )
    UVDevelopmentEnvironment(destination).prepare()
    return destination


@pytest.fixture
def already_done_development_environment(already_done_working_copy: Path) -> UVDevelopmentEnvironment:
    return UVDevelopmentEnvironment(already_done_working_copy)


@pytest.fixture
def already_done_work_item() -> WorkItem:
    return WorkItem(
        name="01-greet",
        requirement=ALREADY_DONE_WORK_ITEM / "requirement.md",
        hidden_tests=ALREADY_DONE_WORK_ITEM / "tests",
        reference_diff=ALREADY_DONE_WORK_ITEM / "reference.diff",
    )
