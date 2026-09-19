import os
import tomllib
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError


class StageDefinition(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    harness: str
    model: str | None = None
    prompt: str | None = None


class CorpusDefinition(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    path: str


class PipelineFile(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    stages: tuple[StageDefinition, ...] = Field(default=(), alias="stage")


class PipelineDefinition(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    stages: tuple[StageDefinition, ...]


class ExperimentFile(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    corpus: CorpusDefinition
    pipelines: tuple[str, ...]
    tasks: tuple[str, ...]
    repeats: int


class ExperimentDefinition(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    corpus: Path
    pipelines: tuple[PipelineDefinition, ...]
    tasks: tuple[str, ...]
    repeats: int


def load_pipeline(path: Path) -> PipelineDefinition:
    document = load_toml(path)
    file = parse_pipeline_file(document, path)
    return PipelineDefinition(name=file.name, stages=file.stages)


def load_experiment(path: Path) -> ExperimentDefinition:
    document = load_toml(path)
    file = parse_experiment_file(document, path)
    directory = path.parent
    corpus_path = resolve(directory, Path(file.corpus.path))
    pipeline_paths = [resolve(directory, Path(pipeline)) for pipeline in file.pipelines]
    return ExperimentDefinition(
        name=file.name,
        corpus=corpus_path,
        pipelines=tuple(load_pipeline(pipeline_path) for pipeline_path in pipeline_paths),
        tasks=file.tasks,
        repeats=file.repeats,
    )


def parse_pipeline_file(document: dict, path: Path) -> PipelineFile:
    try:
        return PipelineFile.model_validate(document)
    except ValidationError as error:
        raise pipeline_error(error, document, path) from error


def parse_experiment_file(document: dict, path: Path) -> ExperimentFile:
    try:
        return ExperimentFile.model_validate(document)
    except ValidationError as error:
        raise ValueError(f"{path}: {error}") from error


def pipeline_error(error: ValidationError, document: dict, path: Path) -> ValueError:
    messages = [stage_error_message(document, issue) for issue in error.errors()]
    return ValueError(f"{path}: " + "; ".join(messages))


def stage_error_message(document: dict, issue: dict) -> str:
    location = issue["loc"]
    if location and location[0] == "stage":
        field = stage_field(location)
        return f"stage {stage_label(document, location[1])!r} field {field!r}: {issue['msg']}"
    field = ".".join(str(part) for part in location)
    return f"field {field!r}: {issue['msg']}"


def stage_label(document: dict, index: int) -> str:
    stages = document.get("stage", [])
    if index < len(stages) and "name" in stages[index]:
        return stages[index]["name"]
    return f"#{index}"


def stage_field(location: tuple) -> str:
    return str(location[2]) if len(location) > 2 else ""


def resolve(directory: Path, path: Path) -> Path:
    resolved = path if path.is_absolute() else directory / path
    return Path(os.path.normpath(resolved))


def load_toml(path: Path) -> dict:
    with path.open("rb") as file:
        return tomllib.load(file)
