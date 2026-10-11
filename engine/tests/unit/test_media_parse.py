from clipsdex_engine.media import parse_probe


def test_video_with_audio() -> None:
    info = parse_probe(
        {
            "format": {"duration": "12.345678"},
            "streams": [
                {"codec_type": "video", "width": 1080, "height": 1920},
                {"codec_type": "audio"},
            ],
        }
    )
    assert (info.duration_s, info.width, info.height) == (12.346, 1080, 1920)
    assert info.has_video and info.has_audio


def test_rotated_phone_video_swaps_size() -> None:
    info = parse_probe(
        {
            "format": {"duration": "5"},
            "streams": [
                {
                    "codec_type": "video",
                    "width": 1920,
                    "height": 1080,
                    "side_data_list": [{"side_data_type": "Display Matrix", "rotation": -90}],
                }
            ],
        }
    )
    assert (info.width, info.height) == (1080, 1920)
    assert not info.has_audio


def test_old_style_rotate_tag() -> None:
    info = parse_probe(
        {
            "streams": [
                {"codec_type": "video", "width": 640, "height": 360, "tags": {"rotate": "270"}}
            ]
        }
    )
    assert (info.width, info.height) == (360, 640)


def test_cover_art_is_not_video() -> None:
    info = parse_probe(
        {
            "format": {"duration": "180"},
            "streams": [
                {"codec_type": "audio"},
                {
                    "codec_type": "video",
                    "width": 600,
                    "height": 600,
                    "disposition": {"attached_pic": 1},
                },
            ],
        }
    )
    assert not info.has_video
    assert info.width is None


def test_missing_duration() -> None:
    info = parse_probe({"format": {"duration": "N/A"}, "streams": [{"codec_type": "audio"}]})
    assert info.duration_s is None
