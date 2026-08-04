"""Tests for the IP-block path in transcript collection.

An IP block is not a property of a video, so it must never be recorded as
"we checked and there are no captions". Getting this wrong silently discards
transcripts for every video in the queue over a block that lifts in a few hours.
"""

from __future__ import annotations

import sys
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE_ROOT))
sys.path.insert(0, str(WORKSPACE_ROOT / "services" / "01_youtube_collector"))

from api_client import (  # noqa: E402
    BLOCK_ABORT_THRESHOLD,
    BLOCKING_ERRORS,
    TranscriptClient,
    TranscriptData,
)
from fetch_transcripts_and_dislikes import RunStats, enrich_transcript  # noqa: E402
from youtube_transcript_api import (  # noqa: E402
    IpBlocked,
    NoTranscriptFound,
    TranscriptsDisabled,
)

from core.models import Video  # noqa: E402


class _StubClient:
    """Returns canned TranscriptData, so no network is touched."""

    def __init__(self, result: TranscriptData):
        self.result = result
        self.calls = 0

    def fetch(self, _video_id: str) -> TranscriptData:
        self.calls += 1
        return self.result


def _video() -> Video:
    return Video(video_id="v1", channel_id="UC_test", duration_seconds=600)


def test_block_leaves_the_row_queued():
    """A blocked fetch must not stamp the marker -- the row stays in the queue."""
    video = _video()
    stats = RunStats()

    blocked = enrich_transcript(video, _StubClient(TranscriptData(blocked=True)), stats)

    assert blocked is True
    assert video.transcript_checked_at is None      # the crucial assertion
    assert stats.blocked == 1
    assert stats.transcripts_missing == 0           # not counted as "no captions"


def test_genuine_miss_marks_the_row_as_checked():
    """Captions genuinely disabled: stamp it so we stop asking."""
    video = _video()
    stats = RunStats()

    blocked = enrich_transcript(video, _StubClient(TranscriptData()), stats)

    assert blocked is False
    assert video.transcript_checked_at is not None
    assert stats.transcripts_missing == 1
    assert stats.blocked == 0


def test_success_stores_text_and_derived_metrics():
    video = _video()
    stats = RunStats()
    result = TranscriptData(text="word " * 1500, word_count=1500, language="en")

    blocked = enrich_transcript(video, _StubClient(result), stats)

    assert blocked is False
    assert video.word_count == 1500
    assert video.words_per_minute == 150.0          # 1500 words / 10 minutes
    assert video.transcript_checked_at is not None
    assert stats.transcripts_ok == 1


def test_blocking_errors_are_classified_separately_from_missing_captions():
    """IpBlocked is environmental; TranscriptsDisabled belongs to the video."""
    assert IpBlocked in BLOCKING_ERRORS
    assert TranscriptsDisabled not in BLOCKING_ERRORS
    assert NoTranscriptFound not in BLOCKING_ERRORS


def test_client_reports_blocked_rather_than_empty(monkeypatch):
    """A raised IpBlocked must surface as blocked=True, not a silent miss."""
    client = TranscriptClient(min_interval=0.0)

    def _raise(*_args, **_kwargs):
        raise IpBlocked("vid")

    monkeypatch.setattr(client._api, "fetch", _raise)
    result = client.fetch("vid")

    assert result.blocked is True
    assert result.text is None


def test_abort_threshold_is_small_enough_to_matter():
    """The sweep must give up early, not after hammering the whole queue."""
    assert 2 <= BLOCK_ABORT_THRESHOLD <= 10
