import os
import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class StageDefinition:
    name: str
    harness: str


@dataclass(frozen=True)
class PipelineDefinition:
    name: str
    stages: tuple[StageDefinition, ...]


@dataclass(frozen=True)
class ExperimentDefinition:
    name: str
    corpus: Path
    pipelines: tuple[PipelineDefinition, ...]
    tasks: tuple[str, ...]
    repeats: int


def load_pipeline(path: Path) -> PipelineDefinition:
    document = load_toml(path)
    name = required(document, "name", path)
    stages = tuple(load_stage(stage, path) for stage in document.get("stage", []))
    return PipelineDefinition(name, stages)


def load_experiment(path: Path) -> ExperimentDefinition:
    document = load_toml(path)
    directory = path.parent
    name = required(document, "name", path)
    corpus = required(document, "corpus", path)
    pipelines = required(document, "pipelines", path)
    tasks = required(document, "tasks", path)
    repeats = required(document, "repeats", path)
    corpus_path = resolve(directory, corpus_path_of(corpus, path))
    pipeline_paths = [resolve(directory, Path(pipeline)) for pipeline in pipelines]
    return ExperimentDefinition(
        name=name,
        corpus=corpus_path,
        pipelines=tuple(load_pipeline(pipeline_path) for pipeline_path in pipeline_paths),
        tasks=tuple(tasks),
        repeats=repeats,
    )


def load_stage(stage: dict, path: Path) -> StageDefinition:
    name = required(stage, "name", path)
    harness = required(stage, "harness", path)
    return StageDefinition(name, harness)


def corpus_path_of(corpus: dict, path: Path) -> Path:
    return Path(required(corpus, "path", path))


def resolve(directory: Path, path: Path) -> Path:
    resolved = path if path.is_absolute() else directory / path
    return Path(os.path.normpath(resolved))


def load_toml(path: Path) -> dict:
    with path.open("rb") as file:
        return tomllib.load(file)


def required(document: dict, key: str, path: Path) -> object:
    if key not in document:
        raise ValueError(f"missing required key {key!r} in {path}")
    return document[key]
