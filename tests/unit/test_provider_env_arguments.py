from agent_pipeline_benchmark.container_environments import PROVIDER_ENV_VARS, provider_env_arguments

import pytest


def test_an_exported_provider_key_becomes_a_valueless_env_argument() -> None:
    exported = {PROVIDER_ENV_VARS[0]: "anything"}

    arguments = provider_env_arguments(exported)

    assert arguments == ["-e", PROVIDER_ENV_VARS[0]]


def test_an_unexported_provider_key_becomes_no_argument() -> None:
    arguments = provider_env_arguments({})

    assert arguments == []


def test_without_explicit_env_the_host_environment_is_the_source(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(PROVIDER_ENV_VARS[0], "anything")

    arguments = provider_env_arguments()

    assert arguments == ["-e", PROVIDER_ENV_VARS[0]]
