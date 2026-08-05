"""Re-decide ``is_shorts`` for videos already in the database.

Rows collected before the 3-minute Shorts limit was accounted for carry a label
produced by a ``duration <= 60`` rule. On this dataset that rule mislabelled 17
of the 32 videos between 40s and 200s -- every 61-180s Short was filed as
long-form, which corrupts any per-format baseline built on top of it.

Usage (from the repo root, venv active)::

    python services/01_youtube_collector/repair_shorts.py --dry-run
    python services/01_youtube_collector/repair_shorts.py

Costs no YouTube API quota: the probe is a plain HEAD request to
``/shorts/<id>``, and only videos in the ambiguous 60-180s band need one.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from api_client import (  # noqa: E402
    CERTAIN_SHORTS_DURATION,
    MAX_SHORTS_DURATION,
    ShortsProbe,
    classify_shorts_by_duration,
)
from sqlalchemy import or_, select  # noqa: E402
from tqdm import tqdm  # noqa: E402

from core.config import settings  # noqa: E402
from core.database import healthcheck, session_scope  # noqa: E402
from core.models import Video  # noqa: E402

logger = logging.getLogger("repair_shorts")


def run(*, dry_run: bool = False, limit: Optional[int] = None) -> int:
    """Re-label every video whose Shorts status duration cannot settle."""
    probe = ShortsProbe()
    changed = confirmed = undecided = 0

    try:
        with session_scope() as session:
            stmt = select(Video).where(
                or_(
                    # The ambiguous band: only a probe can decide these.
                    Video.duration_seconds.between(
                        CERTAIN_SHORTS_DURATION + 1, MAX_SHORTS_DURATION
                    ),
                    Video.is_shorts.is_(None),
                )
            ).order_by(Video.duration_seconds)
            if limit:
                stmt = stmt.limit(limit)

            videos = list(session.execute(stmt).scalars())
            if not videos:
                logger.info("No ambiguous videos -- nothing to repair.")
                return 0

            logger.info("Probing %d video(s) in the 60-180s band.", len(videos))

            for video in tqdm(videos, desc="Probing", unit="video", ncols=90):
                truth = probe.is_shorts(video.video_id, duration_seconds=video.duration_seconds)
                if truth is None:
                    # Fall back to whatever duration can say, which may be NULL.
                    truth = classify_shorts_by_duration(video.duration_seconds)
                    if truth is None:
                        undecided += 1
                        continue

                if bool(video.is_shorts) != truth or video.is_shorts is None:
                    logger.info(
                        "%s (%ds): %s -> %s | %s",
                        video.video_id, video.duration_seconds or 0,
                        video.is_shorts, truth, (video.title or "")[:44],
                    )
                    if not dry_run:
                        video.is_shorts = truth
                    changed += 1
                else:
                    confirmed += 1

            if dry_run:
                session.rollback()

    finally:
        probe.close()

    logger.info("=" * 60)
    logger.info("Relabelled : %d%s", changed, " (dry run, not saved)" if dry_run else "")
    logger.info("Confirmed  : %d already correct", confirmed)
    logger.info("Undecided  : %d (probe gave no answer)", undecided)
    logger.info("=" * 60)
    return changed


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Re-decide is_shorts for stored videos.")
    parser.add_argument("--dry-run", action="store_true", help="Report changes without saving.")
    parser.add_argument("--limit", type=int, default=None, help="Probe at most N videos.")
    parser.add_argument("--log-level", default=settings.log_level)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    if not healthcheck():
        logger.error("Database unreachable. Check DB_CONN_STR in .env.")
        return 1

    run(dry_run=args.dry_run, limit=args.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
