import json
import os
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


def test_executing_runs_the_command_as_the_invoking_user(
    prepared_environment: DockerExecutionEnvironment, tmp_path: Path
) -> None:
    working_copy = a_working_copy_with(tmp_path, "marker.txt")

    result = prepared_environment.execute(["id", "-u"], working_copy)

    assert result.stdout.decode().strip() == str(os.getuid())


def test_executing_a_failing_command_reports_its_nonzero_exit_and_stderr(
    prepared_environment: DockerExecutionEnvironment, tmp_path: Path
) -> None:
    working_copy = a_working_copy_with(tmp_path, "marker.txt")

    result = prepared_environment.execute(
        ["/bin/sh", "-c", "echo boom >&2; exit 3"], working_copy
    )

    assert result.returncode == 3
    assert "boom" in result.stderr.decode()


def test_executing_forwards_the_exported_provider_key_into_the_container(
    prepared_environment: DockerExecutionEnvironment, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    secret = "aqueduct-key-from-the-host-shell"
    monkeypatch.setenv("TU_WIEN_AQUEDUCT_API_KEY", secret)
    working_copy = a_working_copy_with(tmp_path, "marker.txt")

    result = prepared_environment.execute(
        ["/bin/sh", "-c", 'printf %s "$TU_WIEN_AQUEDUCT_API_KEY"'], working_copy
    )

    assert result.stdout.decode().strip() == secret


def test_executing_without_an_exported_provider_key_forwards_nothing(
    prepared_environment: DockerExecutionEnvironment, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("TU_WIEN_AQUEDUCT_API_KEY", raising=False)
    working_copy = a_working_copy_with(tmp_path, "marker.txt")

    result = prepared_environment.execute(
        ["/bin/sh", "-c", "echo key=${TU_WIEN_AQUEDUCT_API_KEY:-absent}"], working_copy
    )

    assert "key=absent" in result.stdout.decode()


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


@pytest.fixture(scope="module")
def base_image(tmp_path_factory: pytest.TempPathFactory) -> DockerExecutionEnvironment:
    config_directory = tmp_path_factory.mktemp("agent-config")
    (config_directory / "models.json").write_text(the_aqueduct_models_json())
    dockerfile = Path(__file__).parents[2] / "image" / "Dockerfile"
    environment = DockerExecutionEnvironment(dockerfile, config_directory=config_directory)
    environment.prepare()
    return environment


BASE_IMAGE_LIUBAI_VERSION = "0.87.1"


def the_aqueduct_models_json() -> str:
    provider = {
        "providers": {
            "aqueduct": {
                "name": "aqueduct",
                "baseUrl": "https://aqueduct.invalid",
                "apiKey": "$TU_WIEN_AQUEDUCT_API_KEY",
                "api": "openai-responses",
                "models": [
                    {
                        "id": "deepseek-v4-flash-284b",
                        "name": "DeepSeek V4 Flash",
                        "input": ["text"],
                        "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
                        "contextWindow": 393216,
                        "maxTokens": 8192,
                    }
                ],
            }
        }
    }
    return json.dumps(provider)


def test_the_base_image_provides_liubai_reporting_the_pinned_engine_version(
    base_image: DockerExecutionEnvironment, tmp_path: Path
) -> None:
    working_copy = tmp_path / "working-copy"
    working_copy.mkdir()

    result = base_image.execute(["liubai", "--version"], working_copy)

    assert result.stdout.decode().strip() == BASE_IMAGE_LIUBAI_VERSION


def test_the_base_image_carries_the_aqueduct_model_catalog(
    base_image: DockerExecutionEnvironment, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TU_WIEN_AQUEDUCT_API_KEY", "catalog-probe-key")
    working_copy = tmp_path / "working-copy"
    working_copy.mkdir()

    result = base_image.execute(["liubai", "--list-models"], working_copy)

    assert "deepseek-v4-flash-284b" in result.stdout.decode()
