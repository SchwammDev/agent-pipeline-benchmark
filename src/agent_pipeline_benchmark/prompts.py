from pathlib import Path

from agent_pipeline_benchmark.corpus import WorkItem

PROMPTS_DIR = Path(__file__).parents[2] / "prompts"


def render_prompt(
    template: str, work_item: WorkItem, prompts_dir: Path | None = None
) -> str:
    prompts_dir = prompts_dir or PROMPTS_DIR
    template_path = prompts_dir / template
    if not template_path.is_file():
        raise ValueError(f"prompt template {template!r} not found in {prompts_dir}")
    template_text = template_path.read_text()
    return template_text.replace("{{ticket}}", work_item.requirement.read_text())
