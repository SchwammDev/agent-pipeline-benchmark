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
    run_record_of,
)

HARNESSES_THAT_TRY_TO_SOLVE_THE_WORK_ITEM = ["reference-solution"]
CONTROL_HARNESSES = ["reference-solution", "do-nothing"]

pending = pytest.mark.xfail(strict=True, reason="issue #3 in progress")


@pending
@pytest.mark.parametrize("harness", HARNESSES_THAT_TRY_TO_SOLVE_THE_WORK_ITEM)
def test_an_experiment_using_a_toy_work_item_shows_that_every_harness_that_tries_solves_it(
    harness: str, toy_corpus: Path, results: Path
) -> None:
    pipeline = a_pipeline("bare", stages=[a_stage("implement", harness=harness)])
    experiment = an_experiment("skeleton", tasks=["greeting"], corpus=toy_corpus, pipelines=[pipeline], repeats=1)

    run_experiment(experiment, results)

    record = run_record_of(results, experiment="skeleton", pipeline="bare", task="greeting")
    assert_every_work_item_was_solved(record)


@pending
def test_an_experiment_shows_that_even_a_toy_work_item_stays_unsolved_when_the_harness_does_nothing(
    toy_corpus: Path, results: Path
) -> None:
    pipeline = a_pipeline("bare", stages=[a_stage("implement", harness="do-nothing")])
    experiment = an_experiment("skeleton", tasks=["greeting"], corpus=toy_corpus, pipelines=[pipeline], repeats=1)

    run_experiment(experiment, results)

    record = run_record_of(results, experiment="skeleton", pipeline="bare", task="greeting")
    assert_no_work_item_was_solved(record)


@pending
@pytest.mark.parametrize("control", CONTROL_HARNESSES)
def test_a_control_harness_costs_nothing_when_running_experiments(
    control: str, toy_corpus: Path, results: Path
) -> None:
    pipeline = a_pipeline("bare", stages=[a_stage("implement", harness=control)])
    experiment = an_experiment("skeleton", tasks=["greeting"], corpus=toy_corpus, pipelines=[pipeline], repeats=1)

    run_experiment(experiment, results)

    record = run_record_of(results, experiment="skeleton", pipeline="bare", task="greeting")
    assert_the_run_cost_nothing(record)
