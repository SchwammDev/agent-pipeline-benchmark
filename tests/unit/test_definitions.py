from pathlib import Path

import pytest
from helpers import a_pipeline_file, an_experiment_file

from agent_pipeline_benchmark.definitions import ExperimentDefinition, load_experiment, load_pipeline


def test_a_pipeline_file_yields_its_name_and_stages_in_order(tmp_path: Path) -> None:
    pipeline_file = a_pipeline_file(
        tmp_path,
        "bare",
        stages=[("implement", "reference-solution"), ("cleanup", "do-nothing")],
    )

    pipeline = load_pipeline(pipeline_file)

    assert pipeline.name == "bare"
    assert [(stage.name, stage.harness) for stage in pipeline.stages] == [
        ("implement", "reference-solution"),
        ("cleanup", "do-nothing"),
    ]


def test_an_experiment_file_yields_its_name_corpus_tasks_repeats_and_pipelines(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    pipeline_file = a_pipeline_file(tmp_path, "bare", stages=[("implement", "reference-solution")])
    experiment_file = an_experiment_file(
        tmp_path,
        "skeleton",
        corpus=corpus,
        pipelines=[pipeline_file],
        tasks=["greeting"],
        repeats=3,
    )

    experiment = load_experiment(experiment_file)

    assert_experiment_matches(
        experiment,
        name="skeleton",
        corpus=corpus,
        tasks=("greeting",),
        repeats=3,
        pipeline_names=["bare"],
    )


def test_a_relative_pipeline_path_resolves_against_the_experiment_files_directory(tmp_path: Path) -> None:
    definitions_dir = tmp_path / "definitions"
    definitions_dir.mkdir()
    a_pipeline_file(definitions_dir, "bare", stages=[("implement", "reference-solution")])
    experiment_file = an_experiment_file(
        definitions_dir,
        "skeleton",
        corpus=tmp_path / "corpus",
        pipelines=["bare.toml"],
        tasks=["greeting"],
        repeats=1,
    )

    experiment = load_experiment(experiment_file)

    assert [pipeline.name for pipeline in experiment.pipelines] == ["bare"]


def test_a_relative_corpus_path_resolves_against_the_experiment_files_directory(tmp_path: Path) -> None:
    definitions_dir = tmp_path / "definitions"
    definitions_dir.mkdir()
    pipeline_file = a_pipeline_file(definitions_dir, "bare", stages=[("implement", "reference-solution")])
    experiment_file = an_experiment_file(
        definitions_dir,
        "skeleton",
        corpus="../corpus",
        pipelines=[pipeline_file],
        tasks=["greeting"],
        repeats=1,
    )

    experiment = load_experiment(experiment_file)

    assert experiment.corpus == tmp_path / "corpus"


def test_a_stage_with_model_and_prompt_loads_with_those_values(tmp_path: Path) -> None:
    pipeline_file = tmp_path / "full.toml"
    pipeline_file.write_text(
        '\n'.join(
            [
                'name = "full"',
                '',
                '[[stage]]',
                'name = "implement"',
                'harness = "reference-solution"',
                'model = "claude-sonnet-5"',
                'prompt = "implement.md"',
            ]
        )
    )

    pipeline = load_pipeline(pipeline_file)

    stage = pipeline.stages[0]
    assert stage.model == "claude-sonnet-5"
    assert stage.prompt == "implement.md"


def test_a_stage_without_model_or_prompt_loads_with_both_as_none(tmp_path: Path) -> None:
    pipeline_file = a_pipeline_file(tmp_path, "bare", stages=[("implement", "reference-solution")])

    pipeline = load_pipeline(pipeline_file)

    stage = pipeline.stages[0]
    assert stage.model is None
    assert stage.prompt is None


def test_a_pipeline_missing_a_required_key_is_reported_with_the_key_and_file(tmp_path: Path) -> None:
    pipeline_file = tmp_path / "broken.toml"
    pipeline_file.write_text('[[stage]]\nname = "implement"\nharness = "reference-solution"\n')

    with pytest.raises(ValueError) as error:
        load_pipeline(pipeline_file)

    assert "name" in str(error.value)
    assert "broken.toml" in str(error.value)


def test_an_experiment_missing_a_required_key_is_reported_with_the_key_and_file(tmp_path: Path) -> None:
    experiment_file = tmp_path / "broken.toml"
    experiment_file.write_text(
        '\n'.join(
            [
                'name = "skeleton"',
                'corpus = { path = "corpus" }',
                'pipelines = []',
                'tasks = ["greeting"]',
            ]
        )
    )

    with pytest.raises(ValueError) as error:
        load_experiment(experiment_file)

    assert "repeats" in str(error.value)
    assert "broken.toml" in str(error.value)


def test_an_unknown_stage_field_is_reported_with_the_file_the_stage_and_the_field(tmp_path: Path) -> None:
    pipeline_file = tmp_path / "broken.toml"
    pipeline_file.write_text(
        '\n'.join(
            [
                'name = "bare-pi"',
                '',
                '[[stage]]',
                'name = "implement"',
                'harness = "pi"',
                'modle = "x"',
            ]
        )
    )

    with pytest.raises(ValueError) as error:
        load_pipeline(pipeline_file)

    assert_error_message_names(error.value, "broken.toml", "implement", "modle")


def test_an_unknown_experiment_field_is_reported_with_the_file_and_the_field(tmp_path: Path) -> None:
    experiment_file = tmp_path / "broken.toml"
    experiment_file.write_text(
        '\n'.join(
            [
                'name = "skeleton"',
                'corpus = { path = "corpus" }',
                'pipelines = []',
                'tasks = ["greeting"]',
                'repeats = 1',
                'concurrency = 4',
            ]
        )
    )

    with pytest.raises(ValueError) as error:
        load_experiment(experiment_file)

    assert_error_message_names(error.value, "broken.toml", "concurrency")


def test_a_wrong_type_for_repeats_is_reported_with_the_file_and_the_field(tmp_path: Path) -> None:
    experiment_file = tmp_path / "broken.toml"
    experiment_file.write_text(
        '\n'.join(
            [
                'name = "skeleton"',
                'corpus = { path = "corpus" }',
                'pipelines = []',
                'tasks = ["greeting"]',
                'repeats = "three"',
            ]
        )
    )

    with pytest.raises(ValueError) as error:
        load_experiment(experiment_file)

    assert_error_message_names(error.value, "broken.toml", "repeats")


def assert_error_message_names(error: ValueError, *fragments: str) -> None:
    message = str(error)
    for fragment in fragments:
        assert fragment in message


def assert_experiment_matches(
    experiment: ExperimentDefinition,
    *,
    name: str,
    corpus: Path,
    tasks: tuple[str, ...],
    repeats: int,
    pipeline_names: list[str],
) -> None:
    actual = (
        experiment.name,
        experiment.corpus,
        experiment.tasks,
        experiment.repeats,
        [pipeline.name for pipeline in experiment.pipelines],
    )
    expected = (name, corpus, tasks, repeats, pipeline_names)
    assert actual == expected
