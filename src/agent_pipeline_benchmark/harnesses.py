import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from agent_pipeline_benchmark.corpus import WorkItem
from agent_pipeline_benchmark.subprocesses import environment_without_virtualenv


@dataclass(frozen=True)
class StageCost:
    tokens: int
    usd: float


ZERO_COST = StageCost(tokens=0, usd=0.0)


class ReferenceSolutionDoesNotApply(RuntimeError):
    def __init__(self, work_item_name: str, stderr: str) -> None:
        super().__init__(f"reference solution for {work_item_name!r} does not apply: {stderr}")


class Harness(ABC):
    @abstractmethod
    def implement(self, work_item: WorkItem, working_copy: Path) -> StageCost: ...


class ReferenceSolution(Harness):
    def implement(self, work_item: WorkItem, working_copy: Path) -> StageCost:
        result = subprocess.run(
            ["git", "apply", str(work_item.reference_diff)],
            cwd=working_copy,
            env=environment_without_virtualenv(),
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            raise ReferenceSolutionDoesNotApply(work_item.name, result.stderr.decode())
        return ZERO_COST


class DoNothing(Harness):
    def implement(self, work_item: WorkItem, working_copy: Path) -> StageCost:
        return ZERO_COST


class ReferenceSolutionThenRegression(Harness):
    BROKEN_TEST = "def test_broken() -> None:\n    assert False\n"

    def implement(self, work_item: WorkItem, working_copy: Path) -> StageCost:
        ReferenceSolution().implement(work_item, working_copy)
        for test_file in (working_copy / "tests").glob("test_*.py"):
            test_file.write_text(self.BROKEN_TEST)
        for test_file in (working_copy / "tests").glob("*_test.py"):
            test_file.write_text(self.BROKEN_TEST)
        return ZERO_COST


KNOWN_HARNESSES: dict[str, type[Harness]] = {
    "reference-solution": ReferenceSolution,
    "reference-solution-then-regression": ReferenceSolutionThenRegression,
    "do-nothing": DoNothing,
}


def harness_named(name: str) -> Harness:
    if name not in KNOWN_HARNESSES:
        known = ", ".join(sorted(KNOWN_HARNESSES))
        raise ValueError(f"unknown harness {name!r}, known harnesses: {known}")
    return KNOWN_HARNESSES[name]()
