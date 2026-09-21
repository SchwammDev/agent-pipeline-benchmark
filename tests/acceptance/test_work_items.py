from pathlib import Path

import pytest
from dsl import (
    IMPLEMENT_PROMPT,
    LIUBAI_MODEL,
    TOKENS_ABOVE_ZERO,
    a_pipeline,
    a_stage,
    an_experiment,
    assert_the_liubai_run_is_identified_in_its_record,
    assert_the_liubai_run_left_every_file_a_scorer_needs_on_disk,
    assert_the_liubai_run_reports_the_measures_of_its_stage,
    assert_the_liubai_run_solved_the_greeting_work_item,
    assert_the_run_costs,
    assert_the_run_is_free_of_hidden_tests,
    assert_the_record_contains_the_measures_of_the_agents_run,
    assert_the_record_names_the_version_the_harness_reports,
    assert_the_run_totals_sum_the_stage_measures,
    assert_the_stage_diff_is_exactly_the_reference_change,
    assert_the_work_item_was_not_solved,
    assert_the_work_item_was_solved,
    run_experiment,
    run_record_of,
    the_run_directory_of,
)

CONTROL_HARNESSES = ["reference-solution", "do-nothing"]


def test_a_solved_work_item_reports_its_progressed_and_preserved_tests(
    toy_corpus: Path, results: Path
) -> None:
    pipeline = a_pipeline("bare", stages=[a_stage("implement", harness="reference-solution")])
    experiment = an_experiment("skeleton", tasks=["greeting"], corpus=toy_corpus, pipelines=[pipeline], repeats=1)

    run_experiment(experiment, results)

    record = run_record_of(results, experiment="skeleton", pipeline="bare", task="greeting")
    assert_the_work_item_was_solved(record, work_item="01-greet", progressed=2, preserved=1)


def test_an_untouched_work_item_is_unsolved_and_reports_zero_progressed_tests(
    toy_corpus: Path, results: Path
) -> None:
    pipeline = a_pipeline("bare", stages=[a_stage("implement", harness="do-nothing")])
    experiment = an_experiment("skeleton", tasks=["greeting"], corpus=toy_corpus, pipelines=[pipeline], repeats=1)

    run_experiment(experiment, results)

    record = run_record_of(results, experiment="skeleton", pipeline="bare", task="greeting")
    assert_the_work_item_was_not_solved(record, work_item="01-greet", progressed=0, preserved=1)


def test_a_work_item_whose_repository_already_satisfies_the_hidden_tests_stays_unsolved(
    toy_corpus: Path, results: Path
) -> None:
    pipeline = a_pipeline("bare", stages=[a_stage("implement", harness="reference-solution")])
    experiment = an_experiment("skeleton", tasks=["already-done"], corpus=toy_corpus, pipelines=[pipeline], repeats=1)

    run_experiment(experiment, results)

    record = run_record_of(results, experiment="skeleton", pipeline="bare", task="already-done")
    assert_the_work_item_was_not_solved(record, work_item="01-greet", progressed=0, preserved=3)


def test_a_work_item_that_breaks_the_repositorys_own_tests_stays_unsolved(
    toy_corpus: Path, results: Path
) -> None:
    pipeline = a_pipeline("bare", stages=[a_stage("implement", harness="reference-solution-then-regression")])
    experiment = an_experiment("skeleton", tasks=["greeting"], corpus=toy_corpus, pipelines=[pipeline], repeats=1)

    run_experiment(experiment, results)

    record = run_record_of(results, experiment="skeleton", pipeline="bare", task="greeting")
    assert_the_work_item_was_not_solved(record, work_item="01-greet", progressed=2, preserved=0)


def test_the_recorded_stage_diff_is_exactly_the_reference_change(
    toy_corpus: Path, results: Path
) -> None:
    pipeline = a_pipeline("bare", stages=[a_stage("implement", harness="reference-solution")])
    experiment = an_experiment("skeleton", tasks=["greeting"], corpus=toy_corpus, pipelines=[pipeline], repeats=1)

    run_experiment(experiment, results)

    run_directory = the_run_directory_of(results, experiment="skeleton", pipeline="bare", task="greeting")
    assert_the_stage_diff_is_exactly_the_reference_change(
        run_directory, corpus=toy_corpus, task="greeting", work_item="01-greet"
    )


def test_the_run_is_free_of_hidden_tests_after_scoring(toy_corpus: Path, results: Path) -> None:
    pipeline = a_pipeline("bare", stages=[a_stage("implement", harness="reference-solution")])
    experiment = an_experiment("skeleton", tasks=["greeting"], corpus=toy_corpus, pipelines=[pipeline], repeats=1)

    run_experiment(experiment, results)

    assert_the_run_is_free_of_hidden_tests(results, corpus=toy_corpus, experiment="skeleton", pipeline="bare", task="greeting")


HARNESS_RUNS = [
    pytest.param("reference-solution", None, None, 0, 0.0, id="reference-solution"),
    pytest.param("do-nothing", None, None, 0, 0.0, id="do-nothing"),
    pytest.param(
        "liubai",
        LIUBAI_MODEL,
        IMPLEMENT_PROMPT,
        TOKENS_ABOVE_ZERO,
        0.0,
        marks=[pytest.mark.liubai],
        id="liubai",
    ),
]


@pytest.mark.parametrize(
    ("harness", "model", "prompt", "expected_tokens", "expected_usd"), HARNESS_RUNS
)
def test_a_harness_run_reports_the_cost_of_its_stage(
    harness: str,
    model: str | None,
    prompt: str | None,
    expected_tokens: int | str,
    expected_usd: float,
    toy_corpus: Path,
    results: Path,
) -> None:
    stage = a_stage("implement", harness=harness, model=model, prompt=prompt)
    pipeline = a_pipeline("bare", stages=[stage])
    experiment = an_experiment("skeleton", tasks=["greeting"], corpus=toy_corpus, pipelines=[pipeline], repeats=1)

    run_experiment(experiment, results)

    record = run_record_of(results, experiment="skeleton", pipeline="bare", task="greeting")
    assert_the_run_costs(record, tokens=expected_tokens, usd=expected_usd)


def test_the_record_contains_the_measures_of_the_agents_run(
    fake_liubai: Path, toy_corpus: Path, results: Path
) -> None:
    stage = a_stage("implement", harness="liubai", model=LIUBAI_MODEL, prompt=IMPLEMENT_PROMPT)
    pipeline = a_pipeline("bare", stages=[stage])
    experiment = an_experiment("skeleton", tasks=["greeting"], corpus=toy_corpus, pipelines=[pipeline], repeats=1)

    run_experiment(experiment, results)

    record = run_record_of(results, experiment="skeleton", pipeline="bare", task="greeting")
    assert_the_record_contains_the_measures_of_the_agents_run(
        record, work_item="01-greet", stage="implement", turns=2, tool_calls=2, end_reason="finished"
    )


def test_the_run_totals_sum_the_measures_of_its_stages(
    fake_liubai: Path, toy_corpus: Path, results: Path
) -> None:
    pipeline = a_pipeline(
        "bare",
        stages=[
            a_stage("implement", harness="liubai", model=LIUBAI_MODEL, prompt=IMPLEMENT_PROMPT),
            a_stage("review", harness="liubai", model=LIUBAI_MODEL, prompt=IMPLEMENT_PROMPT),
        ],
    )
    experiment = an_experiment("skeleton", tasks=["greeting"], corpus=toy_corpus, pipelines=[pipeline], repeats=1)

    run_experiment(experiment, results)

    record = run_record_of(results, experiment="skeleton", pipeline="bare", task="greeting")
    assert_the_run_totals_sum_the_stage_measures(record)


def test_the_record_names_the_version_the_harness_reports(
    fake_liubai: Path, toy_corpus: Path, results: Path
) -> None:
    stage = a_stage("implement", harness="liubai", model=LIUBAI_MODEL, prompt=IMPLEMENT_PROMPT)
    pipeline = a_pipeline("bare", stages=[stage])
    experiment = an_experiment("skeleton", tasks=["greeting"], corpus=toy_corpus, pipelines=[pipeline], repeats=1)

    run_experiment(experiment, results)

    record = run_record_of(results, experiment="skeleton", pipeline="bare", task="greeting")
    assert_the_record_names_the_version_the_harness_reports(record, harness="liubai")


@pytest.mark.liubai
def test_the_liubai_harness_solves_the_greeting_work_item(toy_corpus: Path, results: Path) -> None:
    stage = a_stage("implement", harness="liubai", model=LIUBAI_MODEL, prompt=IMPLEMENT_PROMPT)
    pipeline = a_pipeline("bare", stages=[stage])
    experiment = an_experiment("skeleton", tasks=["greeting"], corpus=toy_corpus, pipelines=[pipeline], repeats=1)

    run_experiment(experiment, results)

    assert_the_liubai_run_solved_the_greeting_work_item(results)
    assert_the_liubai_run_is_identified_in_its_record(results, corpus=toy_corpus)
    assert_the_liubai_run_reports_the_measures_of_its_stage(results)
    assert_the_liubai_run_left_every_file_a_scorer_needs_on_disk(results, corpus=toy_corpus)
