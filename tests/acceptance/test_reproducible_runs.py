from pathlib import Path

import pytest

from agent_pipeline_benchmark.environments import EnvironmentUnavailable
from dsl import (
    a_preparable_execution_environment,
    an_execution_environment_that_cannot_be_prepared,
    assert_the_environment_was_prepared_exactly_once,
    assert_the_run_ran_inside_the_prepared_environment,
    assert_no_run_was_recorded,
    delete_the_run_directory,
    run_experiment_with,
    the_scripted_agent_experiment,
)


@pytest.mark.xfail(strict=True, reason="issue #10: runs execute inside a prepared execution environment")
def test_a_run_prepares_the_execution_environment_for_its_stages(
    toy_corpus: Path,
    results: Path,
) -> None:
    environment = a_preparable_execution_environment()

    run_experiment_with(the_scripted_agent_experiment(toy_corpus), results, environment=environment)

    assert_the_environment_was_prepared_exactly_once(environment)
    assert_the_run_ran_inside_the_prepared_environment(results, environment=environment)


@pytest.mark.xfail(strict=True, reason="issue #10: runs execute inside a prepared execution environment")
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


@pytest.mark.xfail(strict=True, reason="issue #10: an unpreparable environment must refuse the run")
def test_a_run_refuses_to_start_when_its_execution_environment_cannot_be_prepared(
    toy_corpus: Path,
    results: Path,
) -> None:
    environment = an_execution_environment_that_cannot_be_prepared()

    with pytest.raises(EnvironmentUnavailable):
        run_experiment_with(the_scripted_agent_experiment(toy_corpus), results, environment=environment)

    assert_no_run_was_recorded(results)
