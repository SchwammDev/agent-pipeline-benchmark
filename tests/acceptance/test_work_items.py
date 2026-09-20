from pathlib import Path

import pytest
from dsl import (
    a_pipeline,
    a_stage,
    an_experiment,
    assert_no_hidden_test_file_remains,
    assert_the_run_cost_nothing,
    assert_the_work_item_was_not_solved,
    assert_the_work_item_was_solved,
    run_experiment,
    run_record_of,
    working_copy_of,
)

CONTROL_HARNESSES = ["reference-solution", "do-nothing"]


@pytest.mark.xfail(strict=True)
def test_a_solved_work_item_reports_its_progressed_and_preserved_tests(
    toy_corpus: Path, results: Path
) -> None:
    pipeline = a_pipeline("bare", stages=[a_stage("implement", harness="reference-solution")])
    experiment = an_experiment("skeleton", tasks=["greeting"], corpus=toy_corpus, pipelines=[pipeline], repeats=1)

    run_experiment(experiment, results)

    record = run_record_of(results, experiment="skeleton", pipeline="bare", task="greeting")
    assert_the_work_item_was_solved(record, work_item="01-greet", progressed=2, preserved=1)


@pytest.mark.xfail(strict=True)
def test_an_untouched_work_item_is_unsolved_and_reports_zero_progressed_tests(
    toy_corpus: Path, results: Path
) -> None:
    pipeline = a_pipeline("bare", stages=[a_stage("implement", harness="do-nothing")])
    experiment = an_experiment("skeleton", tasks=["greeting"], corpus=toy_corpus, pipelines=[pipeline], repeats=1)

    run_experiment(experiment, results)

    record = run_record_of(results, experiment="skeleton", pipeline="bare", task="greeting")
    assert_the_work_item_was_not_solved(record, work_item="01-greet", progressed=0, preserved=1)


@pytest.mark.xfail(strict=True)
def test_a_work_item_whose_repository_already_satisfies_the_hidden_tests_stays_unsolved(
    toy_corpus: Path, results: Path
) -> None:
    pipeline = a_pipeline("bare", stages=[a_stage("implement", harness="reference-solution")])
    experiment = an_experiment("skeleton", tasks=["already-done"], corpus=toy_corpus, pipelines=[pipeline], repeats=1)

    run_experiment(experiment, results)

    record = run_record_of(results, experiment="skeleton", pipeline="bare", task="already-done")
    assert_the_work_item_was_not_solved(record, work_item="01-greet", progressed=0, preserved=3)


@pytest.mark.xfail(strict=True)
def test_a_work_item_that_breaks_the_repositorys_own_tests_stays_unsolved(
    toy_corpus: Path, results: Path
) -> None:
    pipeline = a_pipeline("bare", stages=[a_stage("implement", harness="reference-solution-then-regression")])
    experiment = an_experiment("skeleton", tasks=["greeting"], corpus=toy_corpus, pipelines=[pipeline], repeats=1)

    run_experiment(experiment, results)

    record = run_record_of(results, experiment="skeleton", pipeline="bare", task="greeting")
    assert_the_work_item_was_not_solved(record, work_item="01-greet", progressed=2, preserved=0)


@pytest.mark.xfail(strict=True)
def test_the_kept_working_copy_is_free_of_hidden_tests_after_scoring(
    toy_corpus: Path, results: Path
) -> None:
    pipeline = a_pipeline("bare", stages=[a_stage("implement", harness="reference-solution")])
    experiment = an_experiment("skeleton", tasks=["greeting"], corpus=toy_corpus, pipelines=[pipeline], repeats=1)

    run_experiment(experiment, results)

    working_copy = working_copy_of(results, experiment="skeleton", pipeline="bare", task="greeting")
    assert_no_hidden_test_file_remains(working_copy, corpus=toy_corpus, task="greeting")


@pytest.mark.parametrize("control", CONTROL_HARNESSES)
def test_a_control_harness_costs_nothing_when_running_experiments(
    control: str, toy_corpus: Path, results: Path
) -> None:
    pipeline = a_pipeline("bare", stages=[a_stage("implement", harness=control)])
    experiment = an_experiment("skeleton", tasks=["greeting"], corpus=toy_corpus, pipelines=[pipeline], repeats=1)

    run_experiment(experiment, results)

    record = run_record_of(results, experiment="skeleton", pipeline="bare", task="greeting")
    assert_the_run_cost_nothing(record)
