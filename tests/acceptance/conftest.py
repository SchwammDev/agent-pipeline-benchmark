from pathlib import Path

import pytest

FIXTURE_CORPUS = Path(__file__).parent.parent / "fixtures" / "corpus"


@pytest.fixture
def corpus_of_fixture_tasks() -> Path:
    return FIXTURE_CORPUS


@pytest.fixture
def results(tmp_path: Path) -> Path:
    return tmp_path / "results"
