"""Tests for the AI analyzer service: schemas, payloads and column mapping.

No network access -- the OpenAI call itself is never made. What is tested is
everything around it, which is where the bugs actually live: schema
compatibility with strict Structured Outputs, prompt truncation, score
clamping, and how model output maps onto database columns.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE_ROOT))
sys.path.insert(0, str(WORKSPACE_ROOT / "services" / "02_ai_analyzer"))

from ai_client import clamp_score, supports_temperature  # noqa: E402
from analyze_text import VideoTextAnalysis, build_messages  # noqa: E402
from analyze_thumbnail import (  # noqa: E402
    ThumbnailAnalysis,
    apply_analysis,
    candidate_urls,
)
from analyze_thumbnail import build_messages as build_vision_messages  # noqa: E402

from core.models import Video  # noqa: E402


def _video(**kwargs) -> Video:
    defaults = {"video_id": "vid123", "channel_id": "UC_test", "title": "Test Başlık"}
    return Video(**{**defaults, **kwargs})


# --------------------------------------------------------------------------- #
# Structured Outputs compatibility                                             #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("schema", [VideoTextAnalysis, ThumbnailAnalysis])
def test_schema_has_no_keywords_strict_mode_rejects(schema):
    """Strict Structured Outputs rejects numeric bounds anywhere in the schema.

    A `Field(ge=1, le=10)` would emit `minimum`/`maximum` and make every API
    call fail with a 400, so this guards against reintroducing them.
    """
    json_schema = schema.model_json_schema()
    forbidden = {"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf"}

    def walk(node):
        if isinstance(node, dict):
            assert not (forbidden & node.keys()), f"forbidden keyword in {node}"
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(json_schema)


def test_emotion_tone_is_a_closed_enum():
    """Categorical fields must compile to an enum, not a free-text string."""
    schema = VideoTextAnalysis.model_json_schema()
    tone = schema["$defs"]["EmotionTone"] if "$defs" in schema else schema["properties"]["emotion_tone"]
    assert "enum" in str(tone) or "const" in str(tone)


def test_clamp_score_enforces_the_documented_range():
    assert clamp_score(11.0) == 10.0      # model overshoot
    assert clamp_score(0.0) == 1.0        # model undershoot
    assert clamp_score(7.5) == 7.5
    assert clamp_score(None) is None


def test_temperature_support_detection():
    assert supports_temperature("gpt-4o-mini")
    assert supports_temperature("gpt-4.1")
    assert not supports_temperature("gpt-5.1")   # reasoning models reject it
    assert not supports_temperature("o3-mini")


# --------------------------------------------------------------------------- #
# Text payload                                                                 #
# --------------------------------------------------------------------------- #
def test_transcript_is_truncated_to_control_cost():
    video = _video(full_transcript="x" * 50_000, duration_seconds=600)
    messages = build_messages(video, max_chars=3000)
    user = messages[1]["content"]
    assert user.count("x") == 3000
    assert "Test Başlık" in user


def test_missing_transcript_still_produces_a_valid_payload():
    video = _video(full_transcript=None)
    messages = build_messages(video, max_chars=3000)
    assert "no transcript text available" in messages[1]["content"]
    assert messages[0]["role"] == "system"


# --------------------------------------------------------------------------- #
# Thumbnail URLs                                                               #
# --------------------------------------------------------------------------- #
def test_candidate_urls_puts_stored_url_first_then_fallbacks():
    video = _video(thumbnail_url_maxres="https://i.ytimg.com/vi/vid123/maxresdefault.jpg")
    urls = candidate_urls(video)
    assert urls[0] == "https://i.ytimg.com/vi/vid123/maxresdefault.jpg"
    assert any("hqdefault" in u for u in urls)      # always-present fallback
    assert len(urls) == len(set(urls))              # no duplicates


def test_candidate_urls_works_without_a_stored_url():
    urls = candidate_urls(_video(thumbnail_url_maxres=None))
    assert urls and all(u.startswith("https://i.ytimg.com/vi/vid123/") for u in urls)


def test_vision_payload_carries_both_title_and_image():
    video = _video(title="Kapak Testi")
    messages = build_vision_messages(video, "https://example.invalid/t.jpg", detail="low")
    parts = messages[1]["content"]
    assert parts[0]["text"].endswith("Kapak Testi")
    assert parts[1]["image_url"]["url"] == "https://example.invalid/t.jpg"
    assert parts[1]["image_url"]["detail"] == "low"


# --------------------------------------------------------------------------- #
# Column mapping                                                               #
# --------------------------------------------------------------------------- #
def test_no_face_maps_to_null_not_the_string_none():
    """'None' is how the enum says "no face"; the column must store NULL."""
    video = _video()
    analysis = ThumbnailAnalysis(
        thumbnail_has_face=False,
        thumbnail_face_emotion="None",
        thumbnail_text="",
        title_thumbnail_synergy=5.0,
        reasoning="no face",
    )
    apply_analysis(video, analysis, "gpt-4o-2024-08-06")
    assert video.thumbnail_has_face is False
    assert video.thumbnail_face_emotion is None
    assert video.thumbnail_text is None          # empty OCR is absence, not ""


def test_vision_does_not_touch_the_text_analyser_columns():
    """Both services write to one row; neither may claim the other's metadata."""
    video = _video(ai_model_version="gpt-4o-mini-2024-07-18", hook_score=8.0)
    analysis = ThumbnailAnalysis(
        thumbnail_has_face=True,
        thumbnail_face_emotion="Happy",
        thumbnail_text="  SHOCKING   result  ",
        title_thumbnail_synergy=42.0,        # deliberately out of range
        reasoning="ok",
    )
    apply_analysis(video, analysis, "gpt-4o-2024-08-06")

    assert video.ai_model_version == "gpt-4o-mini-2024-07-18"   # untouched
    assert video.hook_score == 8.0                              # untouched
    assert video.vision_model_version == "gpt-4o-2024-08-06"
    assert video.title_thumbnail_synergy == 10.0                # clamped
    assert video.thumbnail_text == "SHOCKING result"            # whitespace collapsed
