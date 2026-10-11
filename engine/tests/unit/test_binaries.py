from pathlib import Path

import pytest
from typer.testing import CliRunner

from clipsdex_engine import binaries
from clipsdex_engine.cli import app


@pytest.fixture
def isolated_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Clear overrides and point PATH at an empty folder, so real installs are not found."""
    for name in binaries.BINARIES:
        monkeypatch.delenv(binaries.override_env_var(name), raising=False)
    monkeypatch.delenv(binaries.BIN_DIR_ENV, raising=False)
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))
    return tmp_path


def make_fake_binary(folder: Path, name: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / binaries._executable_name(name)
    path.write_bytes(b"")
    path.chmod(0o755)
    return path


def test_env_override_wins(isolated_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Cyrillic and spaces on purpose: Windows usernames and folders can contain them.
    fake = make_fake_binary(isolated_env / "Мој фолдер" / "tools", "ffmpeg")
    bundled = make_fake_binary(isolated_env / "bundled", "ffmpeg")
    monkeypatch.setenv("CLIPSDEX_FFMPEG", str(fake))
    monkeypatch.setenv(binaries.BIN_DIR_ENV, str(bundled.parent))

    resolved = binaries.resolve("ffmpeg")

    assert resolved.path == fake
    assert resolved.source is binaries.Source.ENV


def test_broken_env_override_is_reported_not_skipped(
    isolated_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    make_fake_binary(isolated_env / "bundled", "ffmpeg")
    monkeypatch.setenv(binaries.BIN_DIR_ENV, str(isolated_env / "bundled"))
    monkeypatch.setenv("CLIPSDEX_FFMPEG", str(isolated_env / "does-not-exist.exe"))

    resolved = binaries.resolve("ffmpeg")

    assert not resolved.found
    assert resolved.problem is not None and "CLIPSDEX_FFMPEG" in resolved.problem


def test_bundled_dir_before_path(isolated_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bundled = make_fake_binary(isolated_env / "app bin", "fpcalc")
    on_path = make_fake_binary(isolated_env / "path dir", "fpcalc")
    monkeypatch.setenv(binaries.BIN_DIR_ENV, str(bundled.parent))
    monkeypatch.setenv("PATH", str(on_path.parent))

    resolved = binaries.resolve("fpcalc")

    assert resolved.path == bundled
    assert resolved.source is binaries.Source.BUNDLED


def test_falls_back_to_path(isolated_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    on_path = make_fake_binary(isolated_env / "Ђорђе tools", "ffprobe")
    monkeypatch.setenv(binaries.BIN_DIR_ENV, str(isolated_env / "missing-bundle"))
    monkeypatch.setenv("PATH", str(on_path.parent))

    resolved = binaries.resolve("ffprobe")

    assert resolved.path is not None
    assert resolved.path.resolve() == on_path.resolve()
    assert resolved.source is binaries.Source.PATH


def test_require_raises_readable_error(isolated_env: Path) -> None:
    with pytest.raises(binaries.BinaryNotFoundError, match=r"winget install AcoustID\.Chromaprint"):
        binaries.require("fpcalc")


def test_version_of_unrunnable_file_is_none(isolated_env: Path) -> None:
    fake = make_fake_binary(isolated_env / "tools", "ffmpeg")
    assert binaries.version(fake) is None


def test_doctor_reports_missing_and_exits_1(isolated_env: Path) -> None:
    result = CliRunner().invoke(app, ["doctor"])

    assert result.exit_code == 1
    for name in binaries.BINARIES:
        assert f"[MISSING] {name}" in result.output


def test_doctor_reports_found(isolated_env: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for name in binaries.BINARIES:
        make_fake_binary(isolated_env / "bundled", name)
    monkeypatch.setenv(binaries.BIN_DIR_ENV, str(isolated_env / "bundled"))

    result = CliRunner().invoke(app, ["doctor"])

    assert result.exit_code == 0
    for name in binaries.BINARIES:
        assert f"[OK]      {name} (bundled)" in result.output
