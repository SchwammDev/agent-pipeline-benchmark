from collections.abc import Sequence
from pathlib import Path
from subprocess import CompletedProcess

import pytest
from helpers import snapshot_of

from agent_pipeline_benchmark.corpus import WorkItem
from agent_pipeline_benchmark.environments import ExecutionEnvironment
from agent_pipeline_benchmark.harnesses import (
    ZERO_COST,
    DoNothing,
    Harness,
    Liubai,
    LiubaiFailed,
    LiubaiMisconfigured,
    ReferenceSolution,
    ReferenceSolutionDoesNotApply,
    ReferenceSolutionThenRegression,
    ScriptedAgent,
    ScriptedAgentFailed,
    StageCost,
    harness_named,
)

FAKE_STREAM = "\n".join(
    [
        '{"type":"agent_start"}',
        '{"type":"message_end","message":{"role":"assistant","content":[],"usage":{"input":30,"output":40,"cacheRead":0,"cacheWrite":0}}}',
        '{"type":"agent_end","messages":[]}',
    ]
)
FAKE_STREAM_TOKENS = 70
LIUBAI_MODEL = "deepseek-v4-flash-284b"


class FakeExecutionEnvironment(ExecutionEnvironment):
    def __init__(
        self,
        *,
        version: str = "",
        stream: bytes = FAKE_STREAM.encode(),
        returncode: int = 0,
        stderr: bytes = b"",
    ) -> None:
        self.executed_commands: list[list[str]] = []
        self.working_copies: list[Path] = []
        self._version = version
        self._stream = stream
        self._returncode = returncode
        self._stderr = stderr

    def prepare(self) -> None:
        pass

    def execute(self, command: Sequence[str], working_copy: Path) -> CompletedProcess:
        self.executed_commands.append(list(command))
        self.working_copies.append(working_copy)
        stdout = self._version.encode() if "--version" in command else self._stream
        return CompletedProcess(command, self._returncode, stdout=stdout, stderr=self._stderr)


def test_reference_solution_harness_leaves_the_reference_solution_in_the_working_copy(
    working_copy: Path, greet_work_item: WorkItem
) -> None:
    ReferenceSolution().implement(greet_work_item, working_copy)

    assert_file_contains(working_copy / "src" / "greeting" / "__init__.py", 'return f"Hello, {name}!"')


def test_do_nothing_harness_leaves_the_working_copy_unchanged(working_copy: Path, greet_work_item: WorkItem) -> None:
    before = snapshot_of(working_copy)

    DoNothing().implement(greet_work_item, working_copy)

    assert snapshot_of(working_copy) == before


def test_reference_solution_harness_reports_zero_cost(working_copy: Path, greet_work_item: WorkItem) -> None:
    outcome = ReferenceSolution().implement(greet_work_item, working_copy)

    assert outcome.cost == ZERO_COST


def test_do_nothing_harness_reports_zero_cost(working_copy: Path, greet_work_item: WorkItem) -> None:
    outcome = DoNothing().implement(greet_work_item, working_copy)

    assert outcome.cost == ZERO_COST


def test_a_reference_diff_that_does_not_apply_is_reported_with_the_work_item(
    working_copy: Path, greet_work_item: WorkItem
) -> None:
    (working_copy / "src" / "greeting" / "__init__.py").write_text("already conflicting content\n")

    with pytest.raises(ReferenceSolutionDoesNotApply, match="01-greet"):
        ReferenceSolution().implement(greet_work_item, working_copy)


def test_harness_named_reference_solution_returns_a_reference_solution_harness() -> None:
    assert isinstance(harness_named("reference-solution"), ReferenceSolution)


def test_harness_named_do_nothing_returns_a_do_nothing_harness() -> None:
    assert isinstance(harness_named("do-nothing"), DoNothing)


def test_reference_solution_then_regression_leaves_the_reference_solution_in_the_working_copy(
    working_copy: Path, greet_work_item: WorkItem
) -> None:
    ReferenceSolutionThenRegression().implement(greet_work_item, working_copy)

    assert_file_contains(working_copy / "src" / "greeting" / "__init__.py", 'return f"Hello, {name}!"')


def test_reference_solution_then_regression_replaces_the_repository_owned_test_file(
    working_copy: Path, greet_work_item: WorkItem
) -> None:
    ReferenceSolutionThenRegression().implement(greet_work_item, working_copy)

    test_file = working_copy / "tests" / "test_package.py"
    assert "def test_package_is_importable()" not in test_file.read_text()
    assert_file_contains(test_file, "def test_broken() -> None:\n    assert False")


def test_reference_solution_then_regression_reports_zero_cost(
    working_copy: Path, greet_work_item: WorkItem
) -> None:
    outcome = ReferenceSolutionThenRegression().implement(greet_work_item, working_copy)

    assert outcome.cost == ZERO_COST


def test_harness_named_reference_solution_then_regression_returns_a_reference_solution_then_regression_harness() -> None:
    assert isinstance(
        harness_named("reference-solution-then-regression"), ReferenceSolutionThenRegression
    )


def test_harness_named_with_an_unknown_name_reports_the_known_names() -> None:
    with pytest.raises(ValueError, match="reference-solution.*do-nothing|do-nothing.*reference-solution"):
        harness_named("magic-harness")


def test_a_host_side_harness_inside_an_environment_is_unchanged() -> None:
    do_nothing = DoNothing()

    assert do_nothing.inside(FakeExecutionEnvironment()) is do_nothing


def test_liubai_inside_an_execution_environment_executes_liubai_through_it(
    working_copy: Path, greet_work_item: WorkItem
) -> None:
    environment = FakeExecutionEnvironment()

    assert_the_harness_ran_inside_the_environment(
        environment,
        Liubai().inside(environment),
        working_copy,
        greet_work_item,
        model=LIUBAI_MODEL,
        prompt="Implement greet.",
        expected_command=[
            "liubai", "--mode", "json", "--print", "--no-session", "--model", LIUBAI_MODEL, "--", "Implement greet."
        ],
        expected_tokens=FAKE_STREAM_TOKENS,
    )


def test_the_liubai_version_inside_an_execution_environment_is_probed_through_it(working_copy: Path) -> None:
    environment = FakeExecutionEnvironment(version="0.4.2")

    version = Liubai().inside(environment).version(working_copy)

    assert version == "0.4.2"
    assert environment.executed_commands == [["liubai", "--version"]]


def test_liubai_inside_still_raises_liubai_failed_on_nonzero_exit(
    working_copy: Path, greet_work_item: WorkItem
) -> None:
    environment = FakeExecutionEnvironment(returncode=1, stderr=b"model overloaded", stream=b"")

    with pytest.raises(LiubaiFailed, match="model overloaded"):
        Liubai().inside(environment).implement(
            greet_work_item, working_copy, "Implement greet.", model=LIUBAI_MODEL
        )


def test_liubai_inside_still_raises_liubai_misconfigured_for_missing_model_or_prompt(
    working_copy: Path, greet_work_item: WorkItem
) -> None:
    environment = FakeExecutionEnvironment()
    harness = Liubai().inside(environment)

    with pytest.raises(LiubaiMisconfigured):
        harness.implement(greet_work_item, working_copy, "", model=LIUBAI_MODEL)
    with pytest.raises(LiubaiMisconfigured):
        harness.implement(greet_work_item, working_copy, "Implement greet.")

    assert environment.executed_commands == []


def test_harness_named_scripted_agent_resolves_the_scripted_agent_harness() -> None:
    assert isinstance(harness_named("scripted-agent"), ScriptedAgent)


def test_the_scripted_agent_harness_executes_the_scripted_agent_command_inside_the_environment_with_model_and_prompt(
    working_copy: Path, greet_work_item: WorkItem
) -> None:
    environment = FakeExecutionEnvironment()
    harness = harness_named("scripted-agent").inside(environment)

    harness.implement(greet_work_item, working_copy, "Implement greet.", model=LIUBAI_MODEL)

    assert environment.executed_commands == [
        ["scripted-agent", "--mode", "json", "--print", "--no-session", "--model", LIUBAI_MODEL, "--", "Implement greet."]
    ]
    assert environment.working_copies == [working_copy]


def test_a_scripted_agent_stage_without_a_model_runs_without_the_model_flag(
    working_copy: Path, greet_work_item: WorkItem
) -> None:
    environment = FakeExecutionEnvironment()
    harness = harness_named("scripted-agent").inside(environment)

    harness.implement(greet_work_item, working_copy)

    assert environment.executed_commands == [["scripted-agent", "--mode", "json", "--print", "--no-session"]]
    assert environment.working_copies == [working_copy]


def test_the_scripted_agent_harness_parses_the_event_stream_and_cost_from_the_environment_stdout(
    working_copy: Path, greet_work_item: WorkItem
) -> None:
    outcome = harness_named("scripted-agent").inside(FakeExecutionEnvironment()).implement(
        greet_work_item, working_copy, "Implement greet.", model=LIUBAI_MODEL
    )

    assert [event["type"] for event in outcome.events] == ["agent_start", "message_end", "agent_end"]
    assert outcome.cost == StageCost(tokens=FAKE_STREAM_TOKENS, usd=0.0)


def test_the_scripted_agent_harness_reports_a_nonzero_exit_as_scripted_agent_failed_with_stderr(
    working_copy: Path, greet_work_item: WorkItem
) -> None:
    environment = FakeExecutionEnvironment(returncode=1, stderr=b"script failed", stream=b"")

    with pytest.raises(ScriptedAgentFailed, match="script failed"):
        harness_named("scripted-agent").inside(environment).implement(
            greet_work_item, working_copy, "Implement greet.", model=LIUBAI_MODEL
        )


def test_the_scripted_agent_harness_version_is_probed_through_the_environment(working_copy: Path) -> None:
    environment = FakeExecutionEnvironment(version="0.1.0")

    version = harness_named("scripted-agent").inside(environment).version(working_copy)

    assert version == "0.1.0"
    assert environment.executed_commands == [["scripted-agent", "--version"]]


def test_the_scripted_agent_harness_inside_an_environment_is_the_harness_itself(
    working_copy: Path, greet_work_item: WorkItem
) -> None:
    environment = FakeExecutionEnvironment()
    inside = harness_named("scripted-agent").inside(environment)

    inside.implement(greet_work_item, working_copy, "", model=LIUBAI_MODEL)

    assert isinstance(inside, ScriptedAgent)
    assert environment.executed_commands == [
        ["scripted-agent", "--mode", "json", "--print", "--no-session", "--model", LIUBAI_MODEL]
    ]


def assert_the_harness_ran_inside_the_environment(
    environment: FakeExecutionEnvironment,
    harness: Harness,
    working_copy: Path,
    work_item: WorkItem,
    *,
    model: str,
    prompt: str,
    expected_command: list[str],
    expected_tokens: int,
) -> None:
    outcome = harness.implement(work_item, working_copy, prompt, model=model)

    assert environment.executed_commands == [expected_command]
    assert environment.working_copies == [working_copy]
    assert [event["type"] for event in outcome.events] == ["agent_start", "message_end", "agent_end"]
    assert outcome.cost == StageCost(tokens=expected_tokens, usd=0.0)


def assert_file_contains(path: Path, text: str) -> None:
    assert text in path.read_text()
