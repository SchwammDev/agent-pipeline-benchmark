import subprocess
from pathlib import Path

from agent_pipeline_benchmark.subprocesses import environment_without_virtualenv

GIT_IDENTITY_NAME = "agent-pipeline-benchmark"
GIT_IDENTITY_EMAIL = "benchmark@agent-pipeline-benchmark.invalid"


def prepare_environment(working_copy: Path) -> None:
    subprocess.run(
        ["uv", "sync", "--frozen"],
        cwd=working_copy,
        env=environment_without_virtualenv(),
        capture_output=True,
        check=True,
    )


def initialise_snapshot(working_copy: Path) -> None:
    run_git(working_copy, "init")
    run_git(working_copy, "config", "user.name", GIT_IDENTITY_NAME)
    run_git(working_copy, "config", "user.email", GIT_IDENTITY_EMAIL)
    stage_everything(working_copy)
    commit_snapshot(working_copy, "initial snapshot")


def the_staged_change(working_copy: Path) -> bytes:
    stage_everything(working_copy)
    return run_git(working_copy, "diff", "--cached")


def stage_everything(working_copy: Path) -> None:
    run_git(working_copy, "add", "-A")


def commit_snapshot(working_copy: Path, message: str) -> None:
    run_git(working_copy, "commit", "--no-verify", "--allow-empty", "-m", message)


def run_git(working_copy: Path, *arguments: str) -> bytes:
    result = subprocess.run(
        ["git", *arguments],
        cwd=working_copy,
        capture_output=True,
        check=True,
    )
    return result.stdout
