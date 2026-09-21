import os
from pathlib import Path

import pytest

from agent_pipeline_benchmark.corpus import WorkItem
from agent_pipeline_benchmark.harnesses import (
    StageCost,
    Liubai,
    LiubaiFailed,
    LiubaiMisconfigured,
    harness_named,
)

FAKE_STREAM = "\n".join(
    [
        '{"type":"session","version":3,"id":"fake-session","timestamp":"2026-02-14T00:00:00Z","cwd":"/repo"}',
        '{"type":"agent_start"}',
        '{"type":"message_update","usage":{"input":10,"output":20,"cacheRead":5,"cacheWrite":0},'
        '"assistantMessageEvent":{"type":"text_delta","contentIndex":0,"delta":"thinking"}}',
        '{"type":"message_end","message":{"role":"assistant","content":['
        '{"type":"toolCall","id":"t1","name":"edit","arguments":{}}],'
        '"usage":{"input":30,"output":40,"cacheRead":0,"cacheWrite":0}}}',
        '{"type":"agent_end","messages":[]}',
    ]
)
FAKE_STREAM_TOKENS = 30 + 40 + 0 + 0
LIUBAI_MODEL = "deepseek-v4-flash-284b"


@pytest.fixture
def fake_liubai(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    executable = bin_dir / "liubai"
    executable.write_text(
        "#!/bin/sh\n"
        'printf "%s\\n" "$@" > "$FAKE_LIUBAI_ARGS"\n'
        'pwd > "$FAKE_LIUBAI_CWD"\n'
        f"cat <<'EOF'\n{FAKE_STREAM}\nEOF\n"
    )
    executable.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_LIUBAI_ARGS", str(tmp_path / "args.txt"))
    monkeypatch.setenv("FAKE_LIUBAI_CWD", str(tmp_path / "cwd.txt"))
    return tmp_path


@pytest.fixture
def failing_liubai(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    executable = bin_dir / "liubai"
    executable.write_text('#!/bin/sh\necho "model overloaded" >&2\nexit 1\n')
    executable.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ['PATH']}")
    return tmp_path


def test_the_liubai_harness_runs_liubai_in_the_working_copy_with_the_model_and_the_prompt(
    fake_liubai: Path, working_copy: Path, greet_work_item: WorkItem
) -> None:
    Liubai().implement(greet_work_item, working_copy, "Implement greet.", model=LIUBAI_MODEL)

    argv = (fake_liubai / "args.txt").read_text().splitlines()
    assert argv == [
        "--mode",
        "json",
        "--print",
        "--no-session",
        "--model",
        LIUBAI_MODEL,
        "--",
        "Implement greet.",
    ]
    assert (fake_liubai / "cwd.txt").read_text().strip() == str(working_copy)


def test_the_stage_events_carry_the_agent_event_stream_in_order(
    fake_liubai: Path, working_copy: Path, greet_work_item: WorkItem
) -> None:
    outcome = Liubai().implement(greet_work_item, working_copy, "Implement greet.", model=LIUBAI_MODEL)

    assert [event["type"] for event in outcome.events] == [
        "session",
        "agent_start",
        "message_update",
        "message_end",
        "agent_end",
    ]


def test_the_stage_cost_counts_the_tokens_of_the_last_reported_usage_and_zero_usd(
    fake_liubai: Path, working_copy: Path, greet_work_item: WorkItem
) -> None:
    outcome = Liubai().implement(greet_work_item, working_copy, "Implement greet.", model=LIUBAI_MODEL)

    assert outcome.cost == StageCost(tokens=FAKE_STREAM_TOKENS, usd=0.0)


def test_a_nonzero_liubai_exit_is_reported_with_its_stderr(
    failing_liubai: Path, working_copy: Path, greet_work_item: WorkItem
) -> None:
    with pytest.raises(LiubaiFailed, match="model overloaded"):
        Liubai().implement(greet_work_item, working_copy, "Implement greet.", model=LIUBAI_MODEL)


def test_a_stage_without_a_model_is_reported(
    fake_liubai: Path, working_copy: Path, greet_work_item: WorkItem
) -> None:
    with pytest.raises(LiubaiMisconfigured, match="model"):
        Liubai().implement(greet_work_item, working_copy, "Implement greet.", model=None)


def test_a_stage_without_a_prompt_is_reported(
    fake_liubai: Path, working_copy: Path, greet_work_item: WorkItem
) -> None:
    with pytest.raises(LiubaiMisconfigured, match="prompt"):
        Liubai().implement(greet_work_item, working_copy, "", model=LIUBAI_MODEL)


def test_harness_named_liubai_resolves_the_liubai_harness() -> None:
    assert isinstance(harness_named("liubai"), Liubai)
