from pathlib import Path

import pytest
from helpers import a_pipeline_file, an_experiment_file

from agent_pipeline_benchmark.cli import main


def test_running_an_experiment_from_the_command_line_leaves_a_record_under_results(
    tmp_path: Path, toy_corpus: Path
) -> None:
    pipeline_file = a_pipeline_file(tmp_path, "bare", stages=[("implement", "do-nothing")])
    experiment_file = an_experiment_file(
        tmp_path,
        "skeleton",
        corpus=toy_corpus,
        pipelines=[pipeline_file],
        tasks=["greeting"],
        repeats=1,
    )
    results = tmp_path / "results"

    main(["run", str(experiment_file), "--results", str(results)])

    records = list(results.glob("skeleton/bare/greeting/*/record.json"))
    assert len(records) == 1


def test_running_an_experiment_prints_an_existing_record_path_per_finished_run(
    capsys: pytest.CaptureFixture[str], tmp_path: Path, toy_corpus: Path
) -> None:
    pipeline_file = a_pipeline_file(tmp_path, "bare", stages=[("implement", "do-nothing")])
    experiment_file = an_experiment_file(
        tmp_path,
        "skeleton",
        corpus=toy_corpus,
        pipelines=[pipeline_file],
        tasks=["greeting"],
        repeats=1,
    )
    results = tmp_path / "results"

    main(["run", str(experiment_file), "--results", str(results)])

    assert_each_finished_run_path_was_printed(capsys.readouterr().out, expected_count=1)


def assert_each_finished_run_path_was_printed(output: str, expected_count: int) -> None:
    lines = [line.strip() for line in output.splitlines()]
    assert len(lines) == expected_count
    for line in lines:
        printed = Path(line)
        assert printed.name == "record.json"
        assert printed.is_file()


def test_a_missing_experiment_file_makes_run_exit_non_zero_and_record_nothing(
    tmp_path: Path
) -> None:
    results = tmp_path / "results"
    missing = tmp_path / "missing.toml"

    with pytest.raises(SystemExit) as error:
        main(["run", str(missing), "--results", str(results)])

    assert_the_command_failed(error.value, naming="missing.toml")
    assert_no_run_was_recorded(results)


def test_a_typoed_pipeline_field_makes_run_exit_non_zero_and_record_nothing(
    tmp_path: Path, toy_corpus: Path
) -> None:
    pipeline_file = tmp_path / "broken-pipeline.toml"
    pipeline_file.write_text(
        "\n".join(
            [
                'name = "broken-pipeline"',
                "",
                "[[stage]]",
                'name = "implement"',
                'harness = "do-nothing"',
                'modle = "x"',
            ]
        )
    )
    experiment_file = an_experiment_file(
        tmp_path,
        "skeleton",
        corpus=toy_corpus,
        pipelines=[pipeline_file],
        tasks=["greeting"],
        repeats=1,
    )
    results = tmp_path / "results"

    with pytest.raises(SystemExit) as error:
        main(["run", str(experiment_file), "--results", str(results)])

    assert_the_command_failed(error.value, naming=("broken-pipeline.toml", "implement", "modle"))
    assert_no_run_was_recorded(results)


def assert_the_command_failed(error: SystemExit, *, naming: str | tuple[str, ...]) -> None:
    fragments = (naming,) if isinstance(naming, str) else naming
    assert error.code != 0
    for fragment in fragments:
        assert fragment in str(error)


def assert_no_run_was_recorded(results: Path) -> None:
    assert not (results.exists() and list(results.rglob("record.json")))
