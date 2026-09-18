from pathlib import Path

import pytest
from dsl import (
    a_pipeline,
    a_stage,
    an_experiment,
    assert_every_work_item_was_solved,
    assert_no_work_item_was_solved,
    assert_the_run_cost_nothing,
    run_experiment,
    the_run_record,
)

HARNESSES_THAT_SOLVE_THE_FIXTURE_TASK = ["reference-solution"]
CONTROL_HARNESSES = ["reference-solution", "do-nothing"]

pending = pytest.mark.xfail(strict=True, reason="issue #3 in progress")


@pending
@pytest.mark.parametrize("harness", HARNESSES_THAT_SOLVE_THE_FIXTURE_TASK)
def test_an_experiment_records_the_work_item_as_solved_when_the_stage_harness_solves_it(
    harness: str, corpus_of_fixture_tasks: Path, results: Path
) -> None:
    pipeline = a_pipeline("bare", stages=[a_stage("implement", harness=harness)])
    experiment = an_experiment("skeleton", tasks=["greeting"], corpus=corpus_of_fixture_tasks, pipelines=[pipeline], repeats=1)

    run_experiment(experiment, results)

    record = the_run_record(results, experiment="skeleton", pipeline="bare", task="greeting")
    assert_every_work_item_was_solved(record)


@pending
def test_an_experiment_records_the_work_item_as_unsolved_when_the_stage_harness_does_nothing(
    corpus_of_fixture_tasks: Path, results: Path
) -> None:
    pipeline = a_pipeline("bare", stages=[a_stage("implement", harness="do-nothing")])
    experiment = an_experiment("skeleton", tasks=["greeting"], corpus=corpus_of_fixture_tasks, pipelines=[pipeline], repeats=1)

    run_experiment(experiment, results)

    record = the_run_record(results, experiment="skeleton", pipeline="bare", task="greeting")
    assert_no_work_item_was_solved(record)


@pending
@pytest.mark.parametrize("control", CONTROL_HARNESSES)
def test_an_experiment_records_no_cost_when_the_stage_harness_is_a_control(
    control: str, corpus_of_fixture_tasks: Path, results: Path
) -> None:
    pipeline = a_pipeline("bare", stages=[a_stage("implement", harness=control)])
    experiment = an_experiment("skeleton", tasks=["greeting"], corpus=corpus_of_fixture_tasks, pipelines=[pipeline], repeats=1)

    run_experiment(experiment, results)

    record = the_run_record(results, experiment="skeleton", pipeline="bare", task="greeting")
    assert_the_run_cost_nothing(record)
