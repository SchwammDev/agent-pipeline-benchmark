import pytest
from pathlib import Path

from agent_pipeline_benchmark import container_environments


def test_a_present_config_file_is_mounted_read_only_into_the_agent_directory(tmp_path: Path) -> None:
    (tmp_path / "models.json").write_text("{}")

    arguments = container_environments.config_mount_arguments(tmp_path)

    assert arguments == [
        "-v",
        f"{tmp_path / 'models.json'}:/root/.pi/agent/models.json:ro",
    ]


def test_a_missing_config_file_is_not_mounted(tmp_path: Path) -> None:
    arguments = container_environments.config_mount_arguments(tmp_path)

    assert arguments == []


def test_settings_json_is_mounted_alongside_models_json(tmp_path: Path) -> None:
    (tmp_path / "models.json").write_text("{}")
    (tmp_path / "settings.json").write_text("{}")

    arguments = container_environments.config_mount_arguments(tmp_path)

    assert arguments == [
        "-v",
        f"{tmp_path / 'models.json'}:/root/.pi/agent/models.json:ro",
        "-v",
        f"{tmp_path / 'settings.json'}:/root/.pi/agent/settings.json:ro",
    ]


def test_without_explicit_directory_the_config_source_is_apb_in_the_home_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / ".apb").mkdir()
    (tmp_path / ".apb" / "models.json").write_text("{}")

    arguments = container_environments.config_mount_arguments()

    assert arguments == [
        "-v",
        f"{tmp_path / '.apb' / 'models.json'}:/root/.pi/agent/models.json:ro",
    ]
