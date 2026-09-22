import subprocess
from abc import ABC, abstractmethod
from pathlib import Path

from agent_pipeline_benchmark.subprocesses import environment_without_virtualenv


class DevelopmentEnvironment(ABC):
    def __init__(self, working_copy: Path) -> None:
        self.working_copy = working_copy

    @abstractmethod
    def prepare(self) -> None: ...


class UVDevelopmentEnvironment(DevelopmentEnvironment):
    def prepare(self) -> None:
        subprocess.run(
            ["uv", "sync", "--frozen"],
            cwd=self.working_copy,
            env=environment_without_virtualenv(),
            capture_output=True,
            check=True,
        )


def new_uv_development_environment(working_copy: Path) -> DevelopmentEnvironment:
    return UVDevelopmentEnvironment(working_copy)
