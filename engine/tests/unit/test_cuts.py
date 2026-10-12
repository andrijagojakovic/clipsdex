import numpy as np
import pytest

from clipsdex_engine import media
from clipsdex_engine.profile import cuts
from synthetic import SyntheticMedia, needs_ffmpeg


def solid(bgr: tuple[int, int, int], n: int) -> list[np.ndarray]:
    return [np.full((32, 48, 3), bgr, dtype=np.uint8) for _ in range(n)]


def test_no_frames_no_cuts() -> None:
    assert cuts.count_cuts_in_frames([]) == 0


def test_static_shot_has_no_cuts() -> None:
    assert cuts.count_cuts_in_frames(solid((0, 0, 255), 30)) == 0


def test_two_hard_cuts() -> None:
    frames = solid((0, 0, 255), 15) + solid((20, 20, 20), 15) + solid((255, 255, 255), 15)
    assert cuts.count_cuts_in_frames(frames) == 2


def test_hue_only_change_is_below_threshold() -> None:
    # Pure red -> pure green changes hue only (score ~20 < 27): not a cut by design.
    assert cuts.count_cuts_in_frames(solid((0, 0, 255), 15) + solid((0, 255, 0), 15)) == 0


def test_flash_shorter_than_min_scene_is_ignored() -> None:
    frames = solid((0, 0, 255), 15) + solid((255, 255, 255), 2) + solid((0, 0, 255), 15)
    assert cuts.count_cuts_in_frames(frames) <= 1


@pytest.mark.parametrize(
    ("size", "expected"),
    [((1080, 1920), (256, 456)), ((1920, 1080), (256, 144)), ((100, 50), (100, 50))],
)
def test_analysis_size(size: tuple[int, int], expected: tuple[int, int]) -> None:
    assert media.analysis_size(*size) == expected


@needs_ffmpeg
def test_synthetic_video_cuts(synthetic_media: SyntheticMedia) -> None:
    assert cuts.count_cuts(synthetic_media.three_shots, 320, 568) == 2
    assert cuts.count_cuts(synthetic_media.silent, 160, 120) == 0


@needs_ffmpeg
def test_frames_from_ffmpeg(synthetic_media: SyntheticMedia) -> None:
    frames = list(media.iter_frames(synthetic_media.silent, 80, 60, fps=10))
    assert len(frames) == 10  # 1 s at 10 fps
    assert frames[0].shape == (60, 80, 3)
    assert frames[0][..., 0].mean() > 200  # blue channel (BGR)


@needs_ffmpeg
def test_stopping_early_does_not_hang(synthetic_media: SyntheticMedia) -> None:
    frames = media.iter_frames(synthetic_media.three_shots, 64, 112, fps=30)
    first = next(frames)
    frames.close()  # pyright: ignore[reportAttributeAccessIssue]  (it is a generator)
    assert first.shape == (112, 64, 3)


@needs_ffmpeg
def test_unreadable_file_raises(synthetic_media: SyntheticMedia) -> None:
    with pytest.raises(media.MediaError):
        list(media.iter_frames(synthetic_media.not_video, 64, 64, fps=10))
