"""Unit tests for the collector's pure helpers and parsing logic.

No network access: every test works on canned API payloads.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pytest

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE_ROOT))
sys.path.insert(0, str(WORKSPACE_ROOT / "services" / "01_youtube_collector"))

from api_client import (  # noqa: E402
    DislikeData,
    TranscriptData,
    YouTubeDataClient,
    count_words,
    parse_duration_to_seconds,
    parse_youtube_datetime,
)


@pytest.mark.parametrize(
    ("iso", "expected"),
    [
        ("PT15M33S", 933),
        ("PT1H2M3S", 3723),
        ("PT59S", 59),
        ("PT1M", 60),
        ("P1DT2H", 93600),
        ("P0D", None),      # unfinished live stream
        ("", None),
        (None, None),
        ("garbage", None),
    ],
)
def test_parse_duration(iso, expected):
    assert parse_duration_to_seconds(iso) == expected


def test_parse_datetime_normalises_to_naive_utc():
    parsed = parse_youtube_datetime("2026-08-01T14:30:00Z")
    assert parsed == datetime(2026, 8, 1, 14, 30)
    assert parsed.tzinfo is None


def test_parse_datetime_converts_offset():
    # 17:30 at +03:00 is 14:30 UTC.
    assert parse_youtube_datetime("2026-08-01T17:30:00+03:00") == datetime(2026, 8, 1, 14, 30)


def test_parse_datetime_handles_junk():
    assert parse_youtube_datetime("not a date") is None
    assert parse_youtube_datetime(None) is None


def test_count_words_handles_unicode():
    assert count_words("merhaba dünya şükrü") == 3
    assert count_words("it's a test") == 3
    assert count_words("") is None
    assert count_words(None) is None


def test_words_per_minute():
    t = TranscriptData(text="x", word_count=1500)
    assert t.words_per_minute(600) == 150.0
    assert t.words_per_minute(0) is None       # no division by zero
    assert t.words_per_minute(None) is None
    assert TranscriptData().words_per_minute(600) is None


def test_like_dislike_ratio_prefers_official_like_count():
    d = DislikeData(dislikes=50, likes=800)
    assert d.like_dislike_ratio(900) == 18.0    # official 900 wins over 800
    assert d.like_dislike_ratio(None) == 16.0   # falls back to the API's likes
    assert DislikeData(dislikes=0, likes=10).like_dislike_ratio(10) is None
    assert DislikeData().like_dislike_ratio(10) is None


class _FakeResponse:
    """Minimal stand-in for ``requests.Response`` in parser tests."""

    def __init__(self, payload):
        self._payload = payload

    def json(self):
        if self._payload is None:
            raise ValueError("not json")
        return self._payload


def test_dislike_all_zero_payload_means_unknown():
    """RYD answers 200 with zeros for videos it has never seen."""
    from api_client import DislikeClient

    unknown = DislikeClient._parse(_FakeResponse({"dislikes": 0, "likes": 0, "viewCount": 0}), "x")
    assert unknown.dislikes is None

    genuine = DislikeClient._parse(_FakeResponse({"dislikes": 0, "likes": 67, "viewCount": 1266}), "x")
    assert genuine.dislikes == 0        # real video that truly has no dislikes
    assert genuine.likes == 67

    assert DislikeClient._parse(_FakeResponse(None), "x").dislikes is None


def test_parse_video_full_payload():
    item = {
        "id": "abc123",
        "snippet": {
            "channelId": "UC_test",
            "publishedAt": "2026-08-01T10:00:00Z",
            "title": "Başlık",
            "description": "açıklama",
            "tags": ["a", "b"],
            "categoryId": "27",
            "thumbnails": {"maxres": {"url": "https://i.ytimg.com/max.jpg"}},
        },
        "statistics": {"viewCount": "1000", "likeCount": "90", "commentCount": "10"},
        "contentDetails": {"duration": "PT10M"},
    }
    v = YouTubeDataClient._parse_video(item)
    assert v.video_id == "abc123"
    assert v.category_id == 27
    assert v.tags == ["a", "b"]
    assert v.duration_seconds == 600
    assert v.view_count == 1000
    assert v.thumbnail_url_maxres.endswith("max.jpg")
    assert v.is_shorts is False


def test_parse_video_falls_back_through_thumbnail_sizes():
    item = {
        "id": "x",
        "snippet": {"channelId": "UC", "thumbnails": {"high": {"url": "https://i/high.jpg"}}},
        "statistics": {},
        "contentDetails": {"duration": "PT30S"},
    }
    v = YouTubeDataClient._parse_video(item)
    assert v.thumbnail_url_maxres == "https://i/high.jpg"
    assert v.is_shorts is True          # under 60s: certain
    assert v.tags is None


def test_shorts_classification_admits_the_ambiguous_band():
    """Since the 3-minute limit, 61-180s cannot be settled by duration alone."""
    from api_client import classify_shorts_by_duration

    assert classify_shorts_by_duration(30) is True
    assert classify_shorts_by_duration(60) is True
    assert classify_shorts_by_duration(61) is None      # needs the probe
    assert classify_shorts_by_duration(167) is None     # a real Short in our data
    assert classify_shorts_by_duration(181) is False
    assert classify_shorts_by_duration(2354) is False
    assert classify_shorts_by_duration(None) is None


def test_hidden_like_count_stays_none_not_zero():
    """A creator hiding likes means unknown, which must not become 0."""
    item = {
        "id": "x",
        "snippet": {"channelId": "UC", "thumbnails": {}},
        "statistics": {"viewCount": "500"},   # likeCount absent
        "contentDetails": {"duration": "PT5M"},
    }
    v = YouTubeDataClient._parse_video(item)
    assert v.view_count == 500
    assert v.like_count is None
    assert v.comment_count is None
