"""Tests for the target-variable calculation.

The property that matters most is the absence of leakage: a video must never
contribute to the baseline it is measured against. That mistake produces
plausible-looking numbers and a model that scores well in training and is
worthless in production, so it is tested directly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE_ROOT))
sys.path.insert(0, str(WORKSPACE_ROOT / "services" / "03_predictor_engine"))

from calculate_targets import (  # noqa: E402
    BASELINE_WINDOW,
    MIN_HISTORY,
    _targets_sql,
    shifted_rolling_mean,
)


def _frame(views: list[int], channel: str = "UC_a") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "video_id": [f"v{i:02d}" for i in range(len(views))],
            "channel_id": [channel] * len(views),
            "published_at": pd.date_range("2026-01-01", periods=len(views), freq="D"),
            "view_count": views,
        }
    )


def test_video_is_excluded_from_its_own_baseline():
    """The leakage test: a huge video must not lift its own baseline."""
    frame = _frame([100, 100, 100, 100, 1_000_000])
    baseline = shifted_rolling_mean(frame, window_size=10, min_history=3)

    # The last video has 1M views but its baseline is the four 100s before it.
    assert baseline.iloc[4] == 100.0
    assert frame["view_count"].iloc[4] / baseline.iloc[4] == 10_000.0


def test_first_videos_have_no_baseline_until_min_history():
    """Fewer than min_history earlier videos means no trustworthy baseline."""
    frame = _frame([10, 20, 30, 40, 50])
    baseline = shifted_rolling_mean(frame, window_size=10, min_history=3)

    assert baseline.isna().tolist()[:3] == [True, True, True]   # 0, 1, 2 predecessors
    assert baseline.iloc[3] == pytest.approx(20.0)              # mean(10,20,30)
    assert baseline.iloc[4] == pytest.approx(25.0)              # mean(10,20,30,40)


def test_window_is_capped_at_ten_preceding_videos():
    """Video 12 must average videos 2-11, not the whole back catalogue."""
    frame = _frame([1] * 2 + [100] * 10 + [0])
    baseline = shifted_rolling_mean(frame, window_size=10, min_history=3)

    # Last row: the 10 immediately preceding are all 100 -- the two 1s drop out.
    assert baseline.iloc[12] == pytest.approx(100.0)


def test_channels_do_not_contaminate_each_other():
    """A big channel's views must never enter a small channel's baseline."""
    big = _frame([1_000_000] * 5, channel="UC_big")
    small = _frame([100, 200, 300, 400], channel="UC_small")
    combined = pd.concat([big, small], ignore_index=True)

    baseline = shifted_rolling_mean(combined, window_size=10, min_history=3)
    small_rows = combined["channel_id"] == "UC_small"

    assert baseline[small_rows].iloc[3] == pytest.approx(200.0)   # mean(100,200,300)
    assert baseline[small_rows].max() < 1000


def test_ordering_is_by_publication_not_row_order():
    """Rows arriving out of order must still produce a chronological baseline."""
    frame = _frame([10, 20, 30, 40])
    shuffled = frame.iloc[[3, 1, 0, 2]].reset_index(drop=True)

    baseline = shifted_rolling_mean(shuffled, window_size=10, min_history=3)
    newest = shuffled["video_id"] == "v03"

    assert baseline[newest].iloc[0] == pytest.approx(20.0)   # mean(10,20,30)


def test_window_bound_is_a_validated_literal():
    """T-SQL rejects a parameter in ROWS BETWEEN, so the value is interpolated.

    It must therefore be impossible for anything but an integer to get in.
    """
    assert "ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING" in _targets_sql(10)
    assert "ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING" in _targets_sql(5)

    with pytest.raises(ValueError):
        _targets_sql(0)
    with pytest.raises((ValueError, TypeError)):
        _targets_sql("10; DROP TABLE videos")


def test_sql_uses_a_strictly_preceding_window():
    """Guards the one clause whose corruption would be invisible downstream."""
    sql = _targets_sql(BASELINE_WINDOW)
    assert "AND 1 PRECEDING" in sql
    assert "CURRENT ROW" not in sql
    assert "PARTITION BY channel_id" in sql


def test_defaults_match_the_documented_business_rules():
    assert BASELINE_WINDOW == 10
    assert MIN_HISTORY == 3
