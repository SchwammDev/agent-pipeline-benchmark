from pathlib import Path

import pytest

from agent_pipeline_benchmark.environments import EnvironmentUnavailable
from dsl import (
    FAKE_AGENT_VERSION,
    IMPLEMENT_PROMPT,
    LIUBAI_MODEL,
    a_dockerfile_baking_in_the_fake_liubai,
    a_pipeline,
    a_preparable_execution_environment,
    a_stage,
    an_execution_environment_that_cannot_be_prepared,
    an_experiment,
    an_experiment_file_created_for,
    assert_each_finished_run_path_was_printed,
    assert_no_run_was_recorded,
    assert_the_command_exited_cleanly,
    assert_the_environment_was_prepared_exactly_once,
    assert_the_run_ran_inside_the_container,
    assert_the_run_ran_inside_the_prepared_environment,
    delete_the_run_directory,
    run_apb,
    run_experiment_with,
    the_scripted_agent_experiment,
)


@pytest.mark.docker
def test_a_run_executes_its_stages_inside_the_container_built_from_the_experiments_dockerfile(
    toy_corpus: Path, definitions: Path, results: Path
) -> None:
    dockerfile = a_dockerfile_baking_in_the_fake_liubai(in_directory=definitions)
    pipeline = a_pipeline(
        "bare", stages=[a_stage("implement", harness="liubai", model=LIUBAI_MODEL, prompt=IMPLEMENT_PROMPT)]
    )
    experiment = an_experiment(
        "skeleton",
        tasks=["greeting"],
        corpus=toy_corpus,
        pipelines=[pipeline],
        repeats=1,
        environment={"dockerfile": str(dockerfile)},
    )
    experiment_file = an_experiment_file_created_for(experiment, in_directory=definitions)

    command = run_apb("run", str(experiment_file), "--results", str(results))

    assert_the_command_exited_cleanly(command)
    assert_each_finished_run_path_was_printed(command, expected_runs=1)
    assert_the_run_ran_inside_the_container(results, liubai_version=FAKE_AGENT_VERSION)


def test_a_run_prepares_the_execution_environment_for_its_stages(
    toy_corpus: Path,
    results: Path,
) -> None:
    environment = a_preparable_execution_environment()

    run_experiment_with(the_scripted_agent_experiment(toy_corpus), results, environment=environment)

    assert_the_environment_was_prepared_exactly_once(environment)
    assert_the_run_ran_inside_the_prepared_environment(results, environment=environment)


def test_a_later_run_reuses_the_prepared_execution_environment(
    toy_corpus: Path,
    results: Path,
) -> None:
    environment = a_preparable_execution_environment()
    run_experiment_with(the_scripted_agent_experiment(toy_corpus), results, environment=environment)
    delete_the_run_directory(results, experiment="skeleton", pipeline="bare", task="greeting")

    run_experiment_with(the_scripted_agent_experiment(toy_corpus), results, environment=environment)

    assert_the_environment_was_prepared_exactly_once(environment)
    assert_the_run_ran_inside_the_prepared_environment(results, environment=environment)


def test_a_run_refuses_to_start_when_its_execution_environment_cannot_be_prepared(
    toy_corpus: Path,
    results: Path,
) -> None:
    environment = an_execution_environment_that_cannot_be_prepared()

    with pytest.raises(EnvironmentUnavailable):
        run_experiment_with(the_scripted_agent_experiment(toy_corpus), results, environment=environment)

    assert_no_run_was_recorded(results)
