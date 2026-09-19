from pathlib import Path

import pytest
from dsl import (
    a_pipeline,
    a_stage,
    an_experiment,
    an_experiment_file_created_for,
    assert_each_finished_run_path_was_printed,
    assert_no_run_was_recorded,
    assert_the_command_exited_cleanly,
    assert_the_command_failed,
    assert_the_command_reported,
    introduce_a_typo_in,
    run_apb,
)


@pytest.mark.xfail(strict=True, reason="the apb entry point does not exist yet")
def test_running_an_experiment_prints_each_finished_run_and_exits_cleanly(
    toy_corpus: Path, definitions: Path, results: Path
) -> None:
    pipeline = a_pipeline(
        "bare-pi", stages=[a_stage("implement", harness="reference-solution", model="claude-sonnet-5", prompt="implement.md")]
    )
    experiment = an_experiment(
        "skeleton", tasks=["greeting"], corpus=toy_corpus, pipelines=[pipeline], repeats=1
    )
    experiment_file = an_experiment_file_created_for(experiment, in_directory=definitions)

    command = run_apb("run", str(experiment_file), "--results", str(results))

    assert_the_command_exited_cleanly(command)
    assert_each_finished_run_path_was_printed(command, expected_runs=1)


@pytest.mark.xfail(strict=True, reason="unknown fields are not rejected at the command boundary yet")
def test_a_typo_in_a_stage_field_stops_the_run_before_anything_is_recorded(
    toy_corpus: Path, definitions: Path, results: Path
) -> None:
    pipeline = a_pipeline("bare-pi", stages=[a_stage("implement", harness="pi")])
    experiment = an_experiment(
        "skeleton", tasks=["greeting"], corpus=toy_corpus, pipelines=[pipeline], repeats=1
    )
    experiment_file = an_experiment_file_created_for(experiment, in_directory=definitions)
    pipeline_file = definitions / "bare-pi.toml"
    introduce_a_typo_in(pipeline_file, field="harness")

    command = run_apb("run", str(experiment_file), "--results", str(results))

    assert_the_command_failed(command)
    assert_the_command_reported(command, "bare-pi.toml", "implement", "harncss")
    assert_no_run_was_recorded(results)


@pytest.mark.xfail(strict=True, reason="a missing experiment file is not reported as a clean failure yet")
def test_a_missing_experiment_file_fails_the_command_without_touching_results(results: Path) -> None:
    command = run_apb("run", "experiments/no-such-experiment.toml", "--results", str(results))

    assert_the_command_failed(command)
    assert_the_command_reported(command, "no-such-experiment.toml")
    assert_no_run_was_recorded(results)
