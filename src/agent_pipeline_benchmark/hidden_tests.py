import shutil
import subprocess
from pathlib import Path

from agent_pipeline_benchmark.corpus import WorkItem
from agent_pipeline_benchmark.subprocesses import environment_without_virtualenv


def hidden_tests_pass(work_item: WorkItem, working_copy: Path) -> bool:
    destination = working_copy / "tests"
    copied_files = copy_hidden_tests(work_item.hidden_tests, destination)
    try:
        result = run_pytest(copied_files, working_copy)
        return result.returncode == 0
    finally:
        remove_copied_files(copied_files)


def copy_hidden_tests(hidden_tests: Path, destination: Path) -> list[Path]:
    copied_files = []
    for source in hidden_tests.rglob("*"):
        if source.is_file():
            target = destination / source.relative_to(hidden_tests)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            copied_files.append(target)
    return copied_files


def run_pytest(test_files: list[Path], working_copy: Path) -> subprocess.CompletedProcess:
    relative_paths = [str(path.relative_to(working_copy)) for path in test_files]
    return subprocess.run(
        ["uv", "run", "pytest", "-q", *relative_paths],
        cwd=working_copy,
        env=environment_without_virtualenv(),
        capture_output=True,
        check=False,
    )


def remove_copied_files(copied_files: list[Path]) -> None:
    for path in copied_files:
        path.unlink(missing_ok=True)
