import subprocess
from abc import ABC, abstractmethod
from pathlib import Path

from agent_pipeline_benchmark.subprocesses import environment_without_virtualenv


class DevelopmentEnvironment(ABC):
    def __init__(self, working_copy: Path) -> None:
        self.working_copy = working_copy

    @abstractmethod
    def prepare(self) -> None: ...

    @abstractmethod
    def run(self, command: list[str]) -> subprocess.CompletedProcess: ...


class UVDevelopmentEnvironment(DevelopmentEnvironment):
    def prepare(self) -> None:
        subprocess.run(
            ["uv", "sync", "--frozen"],
            cwd=self.working_copy,
            env=environment_without_virtualenv(),
            capture_output=True,
            check=True,
        )

    def run(self, command: list[str]) -> subprocess.CompletedProcess:
        return subprocess.run(
            self.resolved(command),
            cwd=self.working_copy,
            capture_output=True,
            check=False,
        )

    def resolved(self, command: list[str]) -> list[str]:
        if command[0] == "python":
            return [str(self.working_copy / ".venv" / "bin" / "python"), *command[1:]]
        return command


def new_uv_development_environment(working_copy: Path) -> DevelopmentEnvironment:
    return UVDevelopmentEnvironment(working_copy)
