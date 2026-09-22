import subprocess
from pathlib import Path

from agent_pipeline_benchmark.development_environments import DevelopmentEnvironment

EMPTY_SUITE_JUNIT = (
    "<?xml version='1.0' encoding='utf-8'?>\n"
    "<testsuites><testsuite name='pytest' errors='0' failures='0' skipped='0' tests='0' time='0'/></testsuites>\n"
)


class EmptySuiteEnvironment(DevelopmentEnvironment):
    def prepare(self) -> None: ...

    def run(self, command: list[str]) -> subprocess.CompletedProcess:
        junit_xml = Path(command[command.index("--junitxml") + 1])
        junit_xml.write_text(EMPTY_SUITE_JUNIT)
        return subprocess.CompletedProcess(command, returncode=0)


def new_empty_suite_environment(working_copy: Path) -> EmptySuiteEnvironment:
    return EmptySuiteEnvironment(working_copy)


def snapshot_of(directory: Path) -> dict[Path, str]:
    return {
        path.relative_to(directory): path.read_text()
        for path in directory.rglob("*")
        if path.is_file() and not is_ignored_by_a_snapshot(path)
    }


def is_ignored_by_a_snapshot(path: Path) -> bool:
    return any(part in (".venv", "__pycache__", ".git") for part in path.parts)


def a_pipeline_file(directory: Path, name: str, *, stages: list[tuple[str, str]]) -> Path:
    pipeline_file = directory / f"{name}.toml"
    stage_blocks = [f'[[stage]]\nname = "{stage_name}"\nharness = "{harness}"' for stage_name, harness in stages]
    pipeline_file.write_text("\n\n".join([f'name = "{name}"', *stage_blocks]))
    return pipeline_file


def an_experiment_file(
    directory: Path,
    name: str,
    *,
    corpus: Path | str,
    pipelines: list[Path | str],
    tasks: list[str],
    repeats: int,
) -> Path:
    experiment_file = directory / f"{name}.toml"
    pipeline_entries = ", ".join(f'"{pipeline}"' for pipeline in pipelines)
    task_entries = ", ".join(f'"{task}"' for task in tasks)
    experiment_file.write_text(
        "\n".join(
            [
                f'name = "{name}"',
                f'corpus = {{ path = "{corpus}" }}',
                f"pipelines = [{pipeline_entries}]",
                f"tasks = [{task_entries}]",
                f"repeats = {repeats}",
            ]
        )
    )
    return experiment_file
