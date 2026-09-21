import json
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


@dataclass(frozen=True)
class StageOutcome:
    cost: StageCost
    events: tuple[dict, ...] = ()


class ReferenceSolutionDoesNotApply(RuntimeError):
    def __init__(self, work_item_name: str, stderr: str) -> None:
        super().__init__(f"reference solution for {work_item_name!r} does not apply: {stderr}")


class Harness(ABC):
    @abstractmethod
    def implement(
        self, work_item: WorkItem, working_copy: Path, prompt: str = "", model: str | None = None
    ) -> StageOutcome: ...


class ReferenceSolution(Harness):
    def implement(
        self, work_item: WorkItem, working_copy: Path, prompt: str = "", model: str | None = None
    ) -> StageOutcome:
        result = subprocess.run(
            ["git", "apply", str(work_item.reference_diff)],
            cwd=working_copy,
            env=environment_without_virtualenv(),
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            raise ReferenceSolutionDoesNotApply(work_item.name, result.stderr.decode())
        return StageOutcome(cost=ZERO_COST)


class DoNothing(Harness):
    def implement(
        self, work_item: WorkItem, working_copy: Path, prompt: str = "", model: str | None = None
    ) -> StageOutcome:
        return StageOutcome(cost=ZERO_COST)


class ReferenceSolutionThenRegression(Harness):
    BROKEN_TEST = "def test_broken() -> None:\n    assert False\n"

    def implement(
        self, work_item: WorkItem, working_copy: Path, prompt: str = "", model: str | None = None
    ) -> StageOutcome:
        ReferenceSolution().implement(work_item, working_copy)
        for test_file in (working_copy / "tests").glob("test_*.py"):
            test_file.write_text(self.BROKEN_TEST)
        for test_file in (working_copy / "tests").glob("*_test.py"):
            test_file.write_text(self.BROKEN_TEST)
        return StageOutcome(cost=ZERO_COST)


class LiubaiMisconfigured(RuntimeError):
    pass


class LiubaiFailed(RuntimeError):
    def __init__(self, stderr: str) -> None:
        super().__init__(f"liubai exited with an error: {stderr}")


class Liubai(Harness):
    def implement(
        self, work_item: WorkItem, working_copy: Path, prompt: str = "", model: str | None = None
    ) -> StageOutcome:
        if not model:
            raise LiubaiMisconfigured("the liubai harness requires a model on its stage")
        if not prompt:
            raise LiubaiMisconfigured("the liubai harness requires a prompt on its stage")
        result = subprocess.run(
            ["liubai", "--mode", "json", "--print", "--no-session", "--model", model, "--", prompt],
            cwd=working_copy,
            env=environment_without_virtualenv(),
            capture_output=True,
            check=False,
        )
        events = the_events_in(result.stdout)
        if result.returncode != 0:
            raise LiubaiFailed(result.stderr.decode())
        return StageOutcome(cost=the_cost_of(events), events=events)


def the_events_in(stdout: bytes) -> tuple[dict, ...]:
    events = []
    for line in stdout.decode().splitlines():
        text = line.strip()
        if not text:
            continue
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            events.append(parsed)
    return tuple(events)


def the_usage_of(event: dict) -> dict:
    usage = event.get("usage")
    if not isinstance(usage, dict):
        message = event.get("message")
        if isinstance(message, dict) and isinstance(message.get("usage"), dict):
            usage = message["usage"]
    return usage if isinstance(usage, dict) else {}


def the_cost_of(events: tuple[dict, ...]) -> StageCost:
    tokens = 0
    for event in events:
        usage = the_usage_of(event)
        if usage:
            tokens = sum(usage.get(field) or 0 for field in ("input", "output", "cacheRead", "cacheWrite"))
    return StageCost(tokens=tokens, usd=0.0)


KNOWN_HARNESSES: dict[str, type[Harness]] = {
    "reference-solution": ReferenceSolution,
    "reference-solution-then-regression": ReferenceSolutionThenRegression,
    "do-nothing": DoNothing,
    "liubai": Liubai,
}


def harness_named(name: str) -> Harness:
    if name not in KNOWN_HARNESSES:
        known = ", ".join(sorted(KNOWN_HARNESSES))
        raise ValueError(f"unknown harness {name!r}, known harnesses: {known}")
    return KNOWN_HARNESSES[name]()
