from pathlib import Path

import platformdirs
import pytest
from pydantic import ValidationError

from clipsdex_engine import config
from clipsdex_engine.config import PRESETS, Preset, Settings


def test_default_data_dir_is_platformdirs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(config.DATA_DIR_ENV, raising=False)
    expected = platformdirs.user_data_dir("ClipsDex", appauthor=False, roaming=False)
    assert config.data_dir() == Path(expected)


def test_data_dir_override_with_cyrillic_and_spaces(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "Андрија подаци" / "ClipsDex"
    monkeypatch.setenv(config.DATA_DIR_ENV, str(root))

    paths = config.data_paths().ensure()

    assert paths.root == root.resolve()
    for folder in (paths.root, paths.artifacts, paths.videos, paths.logs):
        assert folder.is_dir()
    assert paths.db.parent == paths.root


def test_ensure_is_idempotent(tmp_path: Path) -> None:
    paths = config.DataPaths(tmp_path / "data")
    paths.ensure()
    paths.ensure()
    assert paths.logs.is_dir()


def test_every_preset_is_defined() -> None:
    assert set(PRESETS) == set(Preset)
    assert [PRESETS[p].max_frames for p in Preset] == [2, 4, 8]


def test_balanced_uses_one_model_for_vision_and_text() -> None:
    balanced = PRESETS[Preset.BALANCED]
    assert balanced.vision == balanced.text


def test_default_settings_are_safe() -> None:
    settings = Settings()
    assert settings.preset is Preset.BALANCED
    assert settings.delete_after_analysis is True
    assert settings.downloader_enabled is False
    assert settings.models() == PRESETS[Preset.BALANCED]


def test_overrides_apply_on_top_of_preset() -> None:
    settings = Settings(preset=Preset.FAST, max_frames=3, text_model="qwen2.5:7b")
    models = settings.models()
    assert models.max_frames == 3
    assert models.text == "qwen2.5:7b"
    assert models.vision == PRESETS[Preset.FAST].vision


def test_settings_round_trip_through_json() -> None:
    settings = Settings(preset=Preset.DETAILED, delete_after_analysis=False)
    assert Settings.model_validate_json(settings.model_dump_json()) == settings


@pytest.mark.parametrize("bad", [{"max_frames": 0}, {"preset": "turbo"}])
def test_invalid_settings_are_rejected(bad: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        Settings.model_validate(bad)
