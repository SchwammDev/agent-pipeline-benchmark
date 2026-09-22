import os
from pathlib import Path

import pytest

TOY_CORPUS = Path(__file__).parent.parent / "toy-corpus"

FAKE_AGENT_VERSION = "0.4.2"

FAKE_AGENT_STREAM = "\n".join(
    [
        '{"type":"session","version":3,"id":"fake-session","timestamp":"2026-02-14T00:00:00Z","cwd":"/repo"}',
        '{"type":"agent_start"}',
        '{"type":"turn_start"}',
        '{"type":"tool_execution_start","toolCallId":"t1","toolName":"bash","args":{"command":["ls"]}}',
        '{"type":"tool_execution_end","toolCallId":"t1","toolName":"bash","result":{"output":["src"],"isError":false}}',
        '{"type":"message_end","message":{"role":"assistant","content":[{"type":"toolCall","id":"t1","name":"edit","arguments":{}}]}}',
        '{"type":"turn_end","message":{"role":"assistant","content":[{"type":"toolCall","id":"t1","name":"edit","arguments":{}}]}}',
        '{"type":"turn_start"}',
        '{"type":"tool_execution_start","toolCallId":"t2","toolName":"write","args":{"path":"src/greeting/__init__.py"}}',
        '{"type":"tool_execution_end","toolCallId":"t2","toolName":"write","result":{"output":[],"isError":false}}',
        '{"type":"message_end","message":{"role":"assistant","content":[{"type":"text","text":"Implemented the greet function."}]}}',
        '{"type":"turn_end","message":{"role":"assistant","content":[{"type":"text","text":"Implemented the greet function."}]}}',
        '{"type":"agent_end","messages":[]}',
    ]
)


THE_UNPREPARABLE_REASON = "the container runtime cannot be reached"


@pytest.fixture
def toy_corpus() -> Path:
    return TOY_CORPUS


@pytest.fixture
def results(tmp_path: Path) -> Path:
    return tmp_path / "results"


@pytest.fixture
def definitions(tmp_path: Path) -> Path:
    return tmp_path / "definitions"


def the_fake_liubai_script() -> str:
    return "\n".join(
        [
            "#!/bin/sh",
            'if [ "$1" = "--version" ]; then',
            f'  echo "{FAKE_AGENT_VERSION}"',
            "  exit 0",
            "fi",
            "cat <<'EOF'",
            FAKE_AGENT_STREAM,
            "EOF",
            "",
        ]
    )


@pytest.fixture
def fake_liubai(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    executable = bin_dir / "liubai"
    executable.write_text(the_fake_liubai_script())
    executable.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ['PATH']}")
    return bin_dir
