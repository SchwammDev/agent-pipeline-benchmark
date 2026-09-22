from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from subprocess import CompletedProcess


class EnvironmentUnavailable(RuntimeError):
    pass


class ExecutionEnvironment(ABC):
    @abstractmethod
    def prepare(self) -> None: ...

    @abstractmethod
    def execute(self, command: Sequence[str], working_copy: Path) -> CompletedProcess: ...
