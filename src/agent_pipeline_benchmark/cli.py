import argparse
from pathlib import Path

from agent_pipeline_benchmark.container_environments import new_container_environment
from agent_pipeline_benchmark.definitions import load_experiment
from agent_pipeline_benchmark.runner import run_experiment


def main(argv: list[str] | None = None) -> None:
    arguments = build_parser().parse_args(argv)
    arguments.handler(arguments)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="agent-pipeline-benchmark")
    subcommands = parser.add_subparsers(required=True)
    add_run_subcommand(subcommands)
    return parser


def add_run_subcommand(subcommands: argparse._SubParsersAction) -> None:
    run_parser = subcommands.add_parser("run")
    run_parser.add_argument("experiment", type=Path)
    run_parser.add_argument("--results", type=Path, required=True)
    run_parser.set_defaults(handler=run_command)


def run_command(arguments: argparse.Namespace) -> None:
    try:
        experiment = load_experiment(arguments.experiment)
    except (ValueError, FileNotFoundError) as error:
        raise SystemExit(error) from error
    environment = None
    if experiment.environment is not None:
        environment = new_container_environment(experiment.environment.dockerfile)
    for record in run_experiment(experiment, arguments.results, environment=environment):
        print(record)
