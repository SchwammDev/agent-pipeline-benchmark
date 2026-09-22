import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from agent_pipeline_benchmark import hidden_tests
from agent_pipeline_benchmark.corpus import WorkItem
from agent_pipeline_benchmark.development_environments import UVDevelopmentEnvironment


@pytest.fixture(autouse=True)
def cold_scoring_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hidden_tests, "SCORING_CACHE", {})


def test_scoring_runs_pytest_in_the_working_copys_own_environment_not_a_nested_uv(
    working_copy: Path, greet_work_item: WorkItem, monkeypatch: pytest.MonkeyPatch
) -> None:
    UVDevelopmentEnvironment(working_copy).prepare()
    commands: list[list[str]] = []

    real_run = subprocess.run

    def record(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
        commands.append(command)
        return real_run(command, **kwargs)

    monkeypatch.setattr(hidden_tests.subprocess, "run", record)

    hidden_tests.score_hidden_tests(greet_work_item, working_copy)

    assert_pytest_ran_once_in_the_working_copys_own_environment(commands, working_copy)


def assert_pytest_ran_once_in_the_working_copys_own_environment(
    commands: list[list[str]], working_copy: Path
) -> None:
    assert len(commands) == 1, f"expected exactly one scoring command, got {commands}"
    [pytest_command] = commands
    venv_python = working_copy / ".venv" / "bin" / "python"
    assert pytest_command[:3] == [str(venv_python), "-m", "pytest"], pytest_command


def test_scoring_a_second_identical_copy_reuses_the_first_verdicts(
    tmp_path: Path, working_copy: Path, greet_work_item: WorkItem, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepare = UVDevelopmentEnvironment(working_copy)
    prepare.prepare()
    second_copy = tmp_path / "second-copy"
    shutil.copytree(working_copy, second_copy)
    suite_runs = []

    real_suite = hidden_tests.run_full_suite

    def counting_suite(junit_xml: Path, working_copy: Path) -> subprocess.CompletedProcess:
        suite_runs.append(working_copy)
        return real_suite(junit_xml, working_copy)

    monkeypatch.setattr(hidden_tests, "run_full_suite", counting_suite)

    first = hidden_tests.score_hidden_tests(greet_work_item, working_copy)
    second = hidden_tests.score_hidden_tests(greet_work_item, second_copy)

    assert suite_runs == [working_copy]
    assert second == first
