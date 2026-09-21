from pathlib import Path

import pytest

from agent_pipeline_benchmark.corpus import WorkItem
from agent_pipeline_benchmark.prompts import render_prompt


def test_the_rendered_prompt_carries_the_ticket_text_from_the_requirement(
    greet_work_item: WorkItem, tmp_path: Path
) -> None:
    template_dir = a_template_dir(tmp_path, "implement.md", "Do: {{ticket}}")

    rendered = render_prompt("implement.md", greet_work_item, prompts_dir=template_dir)

    ticket_text = greet_work_item.requirement.read_text()
    assert ticket_text in rendered


def test_the_placeholder_is_fully_replaced(
    greet_work_item: WorkItem, tmp_path: Path
) -> None:
    template_dir = a_template_dir(tmp_path, "implement.md", "Do: {{ticket}}")

    rendered = render_prompt("implement.md", greet_work_item, prompts_dir=template_dir)

    assert "{{ticket}}" not in rendered


def test_a_missing_template_is_reported_by_name(greet_work_item: WorkItem, tmp_path: Path) -> None:
    template_dir = tmp_path

    with pytest.raises(ValueError) as error:
        render_prompt("missing.md", greet_work_item, prompts_dir=template_dir)

    assert "missing.md" in str(error.value)


def a_template_dir(tmp_path: Path, name: str, content: str) -> Path:
    template_dir = tmp_path / "prompts"
    template_dir.mkdir()
    (template_dir / name).write_text(content)
    return template_dir
