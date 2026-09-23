import subprocess
from pathlib import Path

import pytest

from agent_pipeline_benchmark import container_environments

pytestmark = pytest.mark.docker
from agent_pipeline_benchmark.container_environments import DockerExecutionEnvironment
from agent_pipeline_benchmark.environments import EnvironmentUnavailable


@pytest.fixture(scope="module")
def alpine_dockerfile(tmp_path_factory: pytest.TempPathFactory) -> Path:
    directory = tmp_path_factory.mktemp("container-dockerfiles")
    dockerfile = directory / "Dockerfile"
    dockerfile.write_text("FROM alpine\n")
    return dockerfile


@pytest.fixture
def prepared_environment(alpine_dockerfile: Path) -> DockerExecutionEnvironment:
    environment = DockerExecutionEnvironment(alpine_dockerfile)
    environment.prepare()
    return environment


def a_working_copy_with(tmp_path: Path, marker_name: str) -> Path:
    working_copy = tmp_path / "working-copy"
    working_copy.mkdir()
    (working_copy / marker_name).write_text("here")
    return working_copy


def test_preparing_a_container_environment_builds_an_image_from_its_dockerfile(
    alpine_dockerfile: Path,
) -> None:
    environment = DockerExecutionEnvironment(alpine_dockerfile)

    environment.prepare()

    assert environment.image
    inspect = subprocess.run(
        ["docker", "image", "inspect", environment.image], capture_output=True, check=False
    )
    assert inspect.returncode == 0


def test_preparing_twice_builds_only_once(alpine_dockerfile: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    first = DockerExecutionEnvironment(alpine_dockerfile)
    first.prepare()
    build_calls: list[str] = []
    real_build = container_environments.docker_build

    def counting_build(tag: str, dockerfile_directory: Path) -> subprocess.CompletedProcess:
        build_calls.append(tag)
        return real_build(tag, dockerfile_directory)

    monkeypatch.setattr(container_environments, "docker_build", counting_build)

    second = DockerExecutionEnvironment(alpine_dockerfile)
    second.prepare()

    assert build_calls == []
    assert second.image == first.image


def assert_the_command_saw_the_working_copy(result: subprocess.CompletedProcess, working_copy: Path) -> None:
    stdout = result.stdout.decode()
    assert stdout.startswith(f"{working_copy}\n")
    assert "marker.txt" in stdout


def test_executing_a_command_runs_it_inside_the_container_with_the_working_copy_mounted(
    prepared_environment: DockerExecutionEnvironment, tmp_path: Path
) -> None:
    working_copy = a_working_copy_with(tmp_path, "marker.txt")

    result = prepared_environment.execute(["/bin/sh", "-c", "pwd && ls"], working_copy)

    assert_the_command_saw_the_working_copy(result, working_copy)


def test_executing_a_failing_command_reports_its_nonzero_exit_and_stderr(
    prepared_environment: DockerExecutionEnvironment, tmp_path: Path
) -> None:
    working_copy = a_working_copy_with(tmp_path, "marker.txt")

    result = prepared_environment.execute(
        ["/bin/sh", "-c", "echo boom >&2; exit 3"], working_copy
    )

    assert result.returncode == 3
    assert "boom" in result.stderr.decode()


def test_executing_when_docker_is_missing_raises_environment_unavailable(
    alpine_dockerfile: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    environment = DockerExecutionEnvironment(alpine_dockerfile)
    environment.prepare()
    working_copy = a_working_copy_with(tmp_path, "marker.txt")
    monkeypatch.setattr(
        container_environments, "docker_executable", lambda: str(tmp_path / "missing-docker")
    )

    with pytest.raises(EnvironmentUnavailable):
        environment.execute(["/bin/sh", "-c", "echo hi"], working_copy)


def test_preparing_with_a_dockerfile_that_cannot_build_raises_environment_unavailable(
    tmp_path: Path,
) -> None:
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text("FROM nonexistent-base-image-that-cannot-be-pulled\n")
    environment = DockerExecutionEnvironment(dockerfile)

    with pytest.raises(EnvironmentUnavailable):
        environment.prepare()
