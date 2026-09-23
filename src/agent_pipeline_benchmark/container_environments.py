import hashlib
import subprocess
from collections.abc import Sequence
from pathlib import Path

from agent_pipeline_benchmark.environments import EnvironmentUnavailable, ExecutionEnvironment


_BUILT_IMAGES: dict[str, str] = {}


def image_tag_for(dockerfile: Path) -> str:
    content = dockerfile.read_bytes()
    return "apb:" + hashlib.sha256(content).hexdigest()[:16]


def docker_executable() -> str:
    return "docker"


def docker_build(tag: str, dockerfile_directory: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["docker", "build", "-q", "-t", tag, str(dockerfile_directory)],
        capture_output=True,
        text=True,
        check=False,
    )


def docker_image_id(tag: str) -> str:
    inspected = subprocess.run(
        ["docker", "image", "inspect", tag, "-f", "{{.Id}}"],
        capture_output=True,
        text=True,
        check=False,
    )
    if inspected.returncode != 0:
        raise EnvironmentUnavailable(inspected.stderr.strip())
    return inspected.stdout.strip()


def resolve_image_for(dockerfile: Path) -> str:
    tag = image_tag_for(dockerfile)
    if tag not in _BUILT_IMAGES:
        built = docker_build(tag, dockerfile.parent)
        if built.returncode != 0:
            raise EnvironmentUnavailable(built.stderr.strip())
        _BUILT_IMAGES[tag] = docker_image_id(tag)
    return _BUILT_IMAGES[tag]


class DockerExecutionEnvironment(ExecutionEnvironment):
    def __init__(self, dockerfile: Path) -> None:
        self.dockerfile = dockerfile

    def prepare(self) -> None:
        self.image = resolve_image_for(self.dockerfile)

    def execute(self, command: Sequence[str], working_copy: Path) -> subprocess.CompletedProcess:
        try:
            return subprocess.run(
                [
                    docker_executable(),
                    "run",
                    "--rm",
                    "-v",
                    f"{working_copy}:{working_copy}",
                    "-w",
                    str(working_copy),
                    self.image,
                    *command,
                ],
                capture_output=True,
                check=False,
            )
        except FileNotFoundError as missing:
            raise EnvironmentUnavailable(f"{missing}: docker is not available") from missing


def new_container_environment(dockerfile: Path) -> ExecutionEnvironment:
    return DockerExecutionEnvironment(dockerfile)
