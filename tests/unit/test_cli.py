from pathlib import Path

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
