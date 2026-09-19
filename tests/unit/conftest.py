import shutil
from pathlib import Path

import pytest

from agent_pipeline_benchmark.corpus import WorkItem

TOY_CORPUS = Path(__file__).parent.parent / "toy-corpus"
GREETING_WORK_ITEM = TOY_CORPUS / "greeting" / "work-items" / "01-greet"
GREETING_REPO = TOY_CORPUS / "greeting" / "repo"


@pytest.fixture
def toy_corpus() -> Path:
    return TOY_CORPUS


@pytest.fixture
def working_copy(tmp_path: Path) -> Path:
    destination = tmp_path / "working-copy"
    shutil.copytree(GREETING_REPO, destination, ignore=shutil.ignore_patterns(".venv", "__pycache__"))
    return destination


@pytest.fixture
def greet_work_item() -> WorkItem:
    return WorkItem(
        name="01-greet",
        requirement=GREETING_WORK_ITEM / "requirement.md",
        hidden_tests=GREETING_WORK_ITEM / "tests",
        reference_diff=GREETING_WORK_ITEM / "reference.diff",
    )
