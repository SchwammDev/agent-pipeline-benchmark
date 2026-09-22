import shutil
from collections.abc import Callable
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


@pytest.fixture(scope="session")
def a_prepared_copy_of(tmp_path_factory: pytest.TempPathFactory) -> Callable[[Path], Path]:
    def prepare(repository: Path) -> Path:
        destination = tmp_path_factory.mktemp("prepared") / repository.name
        shutil.copytree(repository, destination, ignore=shutil.ignore_patterns(".venv", "__pycache__"))
        UVDevelopmentEnvironment(destination).prepare()
        return destination

    return prepare


@pytest.fixture(scope="session")
def prepared_greeting_copy(a_prepared_copy_of: Callable[[Path], Path]) -> Path:
    return a_prepared_copy_of(GREETING_REPO)


@pytest.fixture(scope="session")
def prepared_already_done_copy(a_prepared_copy_of: Callable[[Path], Path]) -> Path:
    return a_prepared_copy_of(ALREADY_DONE_REPO)


def a_working_copy_from(tmp_path: Path, prepared_copy: Path) -> Path:
    destination = tmp_path / "working-copy"
    shutil.copytree(prepared_copy, destination, ignore=shutil.ignore_patterns(".venv"))
    (destination / ".venv").symlink_to(prepared_copy / ".venv", target_is_directory=True)
    return destination


@pytest.fixture
def working_copy(tmp_path: Path, prepared_greeting_copy: Path) -> Path:
    return a_working_copy_from(tmp_path, prepared_greeting_copy)


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
def already_done_working_copy(tmp_path: Path, prepared_already_done_copy: Path) -> Path:
    return a_working_copy_from(tmp_path, prepared_already_done_copy)


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
