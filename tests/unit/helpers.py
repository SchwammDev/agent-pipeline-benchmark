from pathlib import Path


def snapshot_of(directory: Path) -> dict[Path, str]:
    return {path.relative_to(directory): path.read_text() for path in directory.rglob("*") if path.is_file()}


def a_pipeline_file(directory: Path, name: str, *, stages: list[tuple[str, str]]) -> Path:
    pipeline_file = directory / f"{name}.toml"
    stage_blocks = [f'[[stage]]\nname = "{stage_name}"\nharness = "{harness}"' for stage_name, harness in stages]
    pipeline_file.write_text("\n\n".join([f'name = "{name}"', *stage_blocks]))
    return pipeline_file


def an_experiment_file(
    directory: Path,
    name: str,
    *,
    corpus: Path | str,
    pipelines: list[Path | str],
    tasks: list[str],
    repeats: int,
) -> Path:
    experiment_file = directory / f"{name}.toml"
    pipeline_entries = ", ".join(f'"{pipeline}"' for pipeline in pipelines)
    task_entries = ", ".join(f'"{task}"' for task in tasks)
    experiment_file.write_text(
        "\n".join(
            [
                f'name = "{name}"',
                f'corpus = {{ path = "{corpus}" }}',
                f"pipelines = [{pipeline_entries}]",
                f"tasks = [{task_entries}]",
                f"repeats = {repeats}",
            ]
        )
    )
    return experiment_file
