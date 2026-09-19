from pathlib import Path

import pytest

TOY_CORPUS = Path(__file__).parent.parent / "toy-corpus"


@pytest.fixture
def toy_corpus() -> Path:
    return TOY_CORPUS


@pytest.fixture
def results(tmp_path: Path) -> Path:
    return tmp_path / "results"


@pytest.fixture
def definitions(tmp_path: Path) -> Path:
    return tmp_path / "definitions"
