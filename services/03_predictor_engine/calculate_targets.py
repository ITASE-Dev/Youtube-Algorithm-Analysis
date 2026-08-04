"""Compute the target variables the ML model will learn to predict.

Three columns, all derived from data already in MSSQL -- no network, no API::

    engagement_rate           (like_count + comment_count) / view_count
    recent_channel_avg_views  mean views of the 10 videos published before this
                              one on the same channel
    performance_ratio         view_count / recent_channel_avg_views   <- TARGET

Usage (from the repo root, venv active)::

    python services/03_predictor_engine/calculate_targets.py
    python services/03_predictor_engine/calculate_targets.py --verify
    python services/03_predictor_engine/calculate_targets.py --dry-run
    python services/03_predictor_engine/calculate_targets.py --min-history 5

Why a shifted window
--------------------
The window is ``ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING`` -- strictly earlier
videos, never the video itself. Including the current row would leak the answer
into its own baseline: a video's views would inflate the average it is measured
against, compressing every ratio toward 1.0 and teaching the model a
relationship that does not exist at prediction time. This is the single most
important line in the file.

Baselines are computed only from videos **present in this database**. Collect a
channel's back catalogue in publication order; the oldest videos of any channel
will always lack history and keep a NULL baseline.
"""

from __future__ import annotations

import argparse
import logging
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator, Optional

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from sqlalchemy import func, select, text  # noqa: E402
from sqlalchemy.exc import SQLAlchemyError  # noqa: E402
from sqlalchemy.orm import Session  # noqa: E402

from core.config import settings  # noqa: E402
from core.database import get_db, get_engine, healthcheck  # noqa: E402
from core.models import Video  # noqa: E402

logger = logging.getLogger("targets")

#: How many earlier videos form the baseline window.
BASELINE_WINDOW = 10

#: Fewer earlier videos than this and the baseline is too noisy to trust.
MIN_HISTORY = 3


@dataclass
class RunStats:
    total_videos: int = 0
    engagement_computed: int = 0
    baseline_computed: int = 0
    baseline_insufficient: int = 0
    ratio_computed: int = 0

    def log_summary(self) -> None:
        logger.info("=" * 62)
        logger.info("Videos in scope        : %d", self.total_videos)
        logger.info("engagement_rate set    : %d", self.engagement_computed)
        logger.info("baseline set           : %d", self.baseline_computed)
        logger.info("baseline skipped       : %d (too little channel history)",
                    self.baseline_insufficient)
        logger.info("performance_ratio set  : %d", self.ratio_computed)
        logger.info("=" * 62)


@contextmanager
def managed_session() -> Iterator[Session]:
    """Drive :func:`core.database.get_db` as a context manager."""
    generator = get_db()
    session = next(generator)
    try:
        yield session
    except Exception:
        session.rollback()
        raise
    finally:
        generator.close()


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


# --------------------------------------------------------------------------- #
# SQL                                                                          #
# --------------------------------------------------------------------------- #
# One set-based UPDATE beats row-by-row by orders of magnitude, and MSSQL cannot
# use a window function directly in SET -- hence the CTE plus join.
#
# * ORDER BY published_at, video_id -- the id breaks ties so two videos posted
#   in the same second produce a stable, reproducible ordering.
# * COUNT(view_count) counts non-NULL views only, so a video whose history is
#   mostly hidden-count rows is correctly treated as having thin history.
# * NULLIF(...,0) turns a zero baseline into NULL instead of a divide-by-zero.
#
# The window bound is interpolated rather than bound as a parameter: T-SQL
# requires a literal in `ROWS BETWEEN n PRECEDING` and rejects `@P1` there.
# `_targets_sql` therefore forces the value through int() before it can reach
# the string, so the interpolation cannot carry anything but a number.
TARGETS_SQL_TEMPLATE = """
WITH baseline AS (
    SELECT
        video_id,
        AVG(CAST(view_count AS FLOAT)) OVER (
            PARTITION BY channel_id
            ORDER BY published_at, video_id
            ROWS BETWEEN {window_size:d} PRECEDING AND 1 PRECEDING
        ) AS avg_views,
        COUNT(view_count) OVER (
            PARTITION BY channel_id
            ORDER BY published_at, video_id
            ROWS BETWEEN {window_size:d} PRECEDING AND 1 PRECEDING
        ) AS history_count
    FROM videos
    WHERE published_at IS NOT NULL
)
UPDATE v
SET
    engagement_rate = CASE
        WHEN v.view_count IS NULL THEN NULL
        WHEN v.view_count = 0 THEN 0.0
        ELSE (COALESCE(v.like_count, 0) + COALESCE(v.comment_count, 0))
             / CAST(v.view_count AS FLOAT)
    END,
    recent_channel_avg_views = CASE
        WHEN b.history_count >= :min_history THEN b.avg_views
        ELSE NULL
    END,
    performance_ratio = CASE
        WHEN b.history_count >= :min_history AND v.view_count IS NOT NULL
        THEN v.view_count / NULLIF(b.avg_views, 0)
        ELSE NULL
    END,
    targets_computed_at = :computed_at
FROM videos v
JOIN baseline b ON b.video_id = v.video_id
"""


def _targets_sql(window_size: int) -> str:
    """Return the UPDATE with the window bound baked in as a literal integer."""
    size = int(window_size)
    if size < 1:
        raise ValueError("window size must be at least 1")
    return TARGETS_SQL_TEMPLATE.format(window_size=size)


def compute_targets(
    session: Session,
    *,
    window_size: int = BASELINE_WINDOW,
    min_history: int = MIN_HISTORY,
) -> int:
    """Run the bulk UPDATE. Returns the number of rows touched."""
    result = session.execute(
        text(_targets_sql(window_size)),
        {
            "min_history": min_history,
            "computed_at": _utcnow(),
        },
    )
    return result.rowcount or 0


def gather_stats(session: Session) -> RunStats:
    """Count what actually landed in the table."""
    def count(*conditions) -> int:
        stmt = select(func.count()).select_from(Video)
        for condition in conditions:
            stmt = stmt.where(condition)
        return session.execute(stmt).scalar() or 0

    stats = RunStats()
    stats.total_videos = count(Video.published_at.is_not(None))
    stats.engagement_computed = count(Video.engagement_rate.is_not(None))
    stats.baseline_computed = count(Video.recent_channel_avg_views.is_not(None))
    stats.baseline_insufficient = count(
        Video.published_at.is_not(None), Video.recent_channel_avg_views.is_(None)
    )
    stats.ratio_computed = count(Video.performance_ratio.is_not(None))
    return stats


def log_distribution(session: Session) -> None:
    """Log where the target lands, since its shape decides the model's job."""
    rows = session.execute(
        select(Video.performance_ratio).where(Video.performance_ratio.is_not(None))
    ).scalars().all()
    if not rows:
        logger.warning("No performance_ratio values yet -- nothing to describe.")
        return

    values = sorted(rows)
    n = len(values)

    def percentile(p: float) -> float:
        return values[min(n - 1, int(p * n))]

    over = sum(1 for v in values if v > 1.0)
    logger.info("performance_ratio over %d video(s):", n)
    logger.info("  min %.2f | p25 %.2f | median %.2f | p75 %.2f | max %.2f",
                values[0], percentile(0.25), percentile(0.50), percentile(0.75), values[-1])
    logger.info("  above channel average: %d (%.0f%%)", over, 100 * over / n)


def warn_about_young_videos(session: Session, *, days: int = 30) -> None:
    """Flag recently published videos, whose ratios are biased downward.

    ``view_count`` is today's number for every row, but the baseline videos have
    had months to accumulate while a video posted last week has had days. Its
    ratio is therefore low for reasons that have nothing to do with quality.
    The rows are still computed -- filter them at training time.
    """
    cutoff = _utcnow() - timedelta(days=days)
    young = session.execute(
        select(func.count()).select_from(Video).where(
            Video.performance_ratio.is_not(None), Video.published_at > cutoff
        )
    ).scalar() or 0
    if young:
        logger.warning(
            "%d video(s) published in the last %d days have a performance_ratio. "
            "Their views are still accumulating while the baseline videos are mature, "
            "so those ratios read artificially low -- exclude them when training.",
            young, days,
        )


# --------------------------------------------------------------------------- #
# Independent verification                                                     #
# --------------------------------------------------------------------------- #
def shifted_rolling_mean(frame, *, window_size: int, min_history: int):
    """Mean views of the preceding ``window_size`` videos, per channel.

    ``frame`` must have ``channel_id``, ``published_at``, ``video_id`` and
    ``view_count``. The result is aligned to ``frame``'s index.

    ``shift(1)`` is what excludes each video from its own baseline. Without it
    the target leaks: a video's own views would inflate the average it is
    compared against, so every ratio would drift toward 1.0 and the model would
    learn a relationship that cannot exist at prediction time.
    """
    ordered = frame.sort_values(["channel_id", "published_at", "video_id"])
    return (
        ordered.groupby("channel_id")["view_count"]
        .transform(lambda s: s.shift(1).rolling(window=window_size, min_periods=min_history).mean())
        .reindex(frame.index)
    )


def verify_with_pandas(*, window_size: int, min_history: int, tolerance: float = 1e-6) -> bool:
    """Recompute the baseline in pandas and compare against the stored values.

    Deliberately a second implementation rather than a repeat of the same SQL:
    an off-by-one in the window (including the current row) is invisible in the
    output but silently destroys the target, so it is worth catching with an
    independent shifted rolling mean.
    """
    import pandas as pd

    frame = pd.read_sql(
        "SELECT video_id, channel_id, published_at, view_count, "
        "recent_channel_avg_views, performance_ratio FROM videos "
        "WHERE published_at IS NOT NULL",
        get_engine(),
    )
    if frame.empty:
        logger.warning("No rows to verify.")
        return True

    frame = frame.sort_values(["channel_id", "published_at", "video_id"]).reset_index(drop=True)
    expected_avg = shifted_rolling_mean(frame, window_size=window_size, min_history=min_history)
    frame["expected_avg"] = expected_avg
    frame["expected_ratio"] = frame["view_count"] / frame["expected_avg"].replace(0, pd.NA)

    stored_avg = frame["recent_channel_avg_views"]
    both_null = stored_avg.isna() & expected_avg.isna()
    close = (stored_avg - expected_avg).abs() <= tolerance * expected_avg.abs().clip(lower=1)
    mismatches = frame[~(both_null | close.fillna(False))]

    if mismatches.empty:
        logger.info("Verification passed: SQL and pandas agree on all %d row(s).", len(frame))
        return True

    logger.error("Verification FAILED on %d row(s):", len(mismatches))
    for _, row in mismatches.head(10).iterrows():
        logger.error(
            "  %s: stored=%s expected=%s",
            row["video_id"], row["recent_channel_avg_views"], row["expected_avg"],
        )
    return False


# --------------------------------------------------------------------------- #
# Orchestration                                                                #
# --------------------------------------------------------------------------- #
def run(
    *,
    window_size: int = BASELINE_WINDOW,
    min_history: int = MIN_HISTORY,
    dry_run: bool = False,
    verify: bool = False,
) -> RunStats:
    """Compute every target and report what landed."""
    stats = RunStats()
    try:
        with managed_session() as session:
            logger.info(
                "Computing targets: baseline = mean of up to %d preceding videos, "
                "minimum %d of history.",
                window_size, min_history,
            )
            updated = compute_targets(session, window_size=window_size, min_history=min_history)

            if dry_run:
                session.rollback()
                logger.warning("DRY RUN: %d row(s) would have been updated; rolled back.", updated)
                return gather_stats(session)

            session.commit()
            logger.info("Updated %d row(s).", updated)

            stats = gather_stats(session)
            log_distribution(session)
            warn_about_young_videos(session)

    except SQLAlchemyError as exc:
        logger.error("Target calculation failed: %s", exc)
        return stats

    if verify and not dry_run and not verify_with_pandas(
        window_size=window_size, min_history=min_history
    ):
        logger.error("Stored targets do not match an independent recomputation.")

    return stats


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Calculate engagement_rate, recent_channel_avg_views and performance_ratio.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--window", type=int, default=BASELINE_WINDOW, help="How many preceding videos form the baseline."
    )
    parser.add_argument(
        "--min-history", type=int, default=MIN_HISTORY,
        help="Minimum preceding videos required before a baseline is trusted.",
    )
    parser.add_argument(
        "--verify", action="store_true", help="Recompute in pandas and compare against the stored values."
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Run the UPDATE and roll it back, reporting the row count."
    )
    parser.add_argument("--log-level", default=settings.log_level, help="DEBUG, INFO, WARNING, ERROR.")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    if args.min_history > args.window:
        logger.error("--min-history cannot exceed --window.")
        return 2
    if not healthcheck():
        logger.error("Database unreachable. Check DB_CONN_STR in .env.")
        return 1

    logger.info("Target database: %s", settings.masked_summary())

    stats = run(
        window_size=args.window,
        min_history=args.min_history,
        dry_run=args.dry_run,
        verify=args.verify,
    )
    stats.log_summary()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
