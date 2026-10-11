"""Shared pytest fixtures."""

import pytest

from synthetic import SyntheticMedia, make_synthetic_media


@pytest.fixture(scope="session")
def synthetic_media(tmp_path_factory: pytest.TempPathFactory) -> SyntheticMedia:
    return make_synthetic_media(tmp_path_factory)
