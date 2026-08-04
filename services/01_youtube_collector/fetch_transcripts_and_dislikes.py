"""Backfill transcripts and dislike counts for videos already in MSSQL.

Network-bound and cheap: no YouTube Data API quota is spent here. Run it as
often as you like; ``analyze_media_pacing.py`` is the heavy sibling.

Usage (from the repo root, venv active)::

    python services/01_youtube_collector/fetch_transcripts_and_dislikes.py
    python services/01_youtube_collector/fetch_transcripts_and_dislikes.py --limit 200
    python services/01_youtube_collector/fetch_transcripts_and_dislikes.py --only transcripts
    python services/01_youtube_collector/fetch_transcripts_and_dislikes.py --retry-failed

Which rows are picked up
------------------------
A video is queued when it is missing a value **and** has never been tried:

    (full_transcript IS NULL AND transcript_checked_at IS NULL)
     OR (dislike_count  IS NULL AND dislike_checked_at    IS NULL)

The ``*_checked_at`` markers are what make reruns cheap. Roughly a third of
YouTube videos have captions permanently disabled; keying the queue on
``full_transcript IS NULL`` alone would re-request those on every single run,
forever. Failures are recorded as "attempted" and skipped next time --
``--retry-failed`` ignores the markers when you want to sweep them again
(after an IP block has lifted, say).
"""

from __future__ import annotations

import argparse
import logging
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from api_client import BLOCK_ABORT_THRESHOLD, DislikeClient, TranscriptClient  # noqa: E402
from sqlalchemy import or_, select  # noqa: E402
from sqlalchemy.exc import SQLAlchemyError  # noqa: E402
from sqlalchemy.orm import Session  # noqa: E402
from tqdm import tqdm  # noqa: E402

from core.config import settings  # noqa: E402
from core.database import get_db, healthcheck  # noqa: E402
from core.models import Video  # noqa: E402

logger = logging.getLogger("enricher")


@dataclass
class RunStats:
    """Counters summarised at the end of the run."""

    processed: int = 0
    transcripts_ok: int = 0
    transcripts_missing: int = 0
    blocked: int = 0
    dislikes_ok: int = 0
    dislikes_missing: int = 0
    errors: int = 0

    def log_summary(self) -> None:
        logger.info("=" * 60)
        logger.info("Videos processed : %d", self.processed)
        logger.info("Transcripts      : %d fetched / %d unavailable",
                    self.transcripts_ok, self.transcripts_missing)
        if self.blocked:
            logger.warning("Transcripts blocked by YouTube : %d (still queued, not written off)",
                           self.blocked)
        logger.info("Dislikes         : %d fetched / %d unavailable",
                    self.dislikes_ok, self.dislikes_missing)
        logger.info("Row errors       : %d", self.errors)
        logger.info("=" * 60)


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
    """Naive UTC now, matching the DATETIME2 columns."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


# --------------------------------------------------------------------------- #
# Query                                                                        #
# --------------------------------------------------------------------------- #
def find_pending_videos(
    session: Session,
    *,
    limit: Optional[int] = None,
    want_transcripts: bool = True,
    want_dislikes: bool = True,
    retry_failed: bool = False,
    channel_id: Optional[str] = None,
) -> list[Video]:
    """Return videos still missing transcript and/or dislike data.

    Ordered newest-first so that an interrupted run has already covered the
    videos most likely to matter.
    """
    conditions = []
    if want_transcripts:
        needs_transcript = Video.full_transcript.is_(None)
        if not retry_failed:
            needs_transcript = needs_transcript & Video.transcript_checked_at.is_(None)
        conditions.append(needs_transcript)
    if want_dislikes:
        needs_dislikes = Video.dislike_count.is_(None)
        if not retry_failed:
            needs_dislikes = needs_dislikes & Video.dislike_checked_at.is_(None)
        conditions.append(needs_dislikes)

    if not conditions:
        return []

    stmt = select(Video).where(or_(*conditions))
    if channel_id:
        stmt = stmt.where(Video.channel_id == channel_id)
    stmt = stmt.order_by(Video.published_at.desc())
    if limit:
        stmt = stmt.limit(limit)

    return list(session.execute(stmt).scalars())


# --------------------------------------------------------------------------- #
# Per-video enrichment                                                         #
# --------------------------------------------------------------------------- #
def enrich_transcript(video: Video, client: TranscriptClient, stats: RunStats) -> bool:
    """Fetch and store the transcript for one video.

    On success writes ``full_transcript``, ``word_count`` and
    ``words_per_minute`` (derived from the ``duration_seconds`` already in the
    database). A genuine miss -- captions disabled, no track -- leaves the
    fields ``NULL`` and stamps the attempt marker so the video leaves the queue.

    Returns:
        True when YouTube blocked the request. A block says nothing about this
        video, so **no marker is written** and the row stays queued; stamping it
        would silently discard the transcript forever over a temporary block.
    """
    result = client.fetch(video.video_id)

    if result.blocked:
        stats.blocked += 1
        return True

    video.transcript_checked_at = _utcnow()

    if not result.text:
        # api_client already logged which failure it was.
        logger.warning("No transcript stored for %s; marked as checked.", video.video_id)
        video.full_transcript = None
        video.word_count = None
        video.words_per_minute = None
        stats.transcripts_missing += 1
        return False

    video.full_transcript = result.text
    video.word_count = result.word_count
    video.words_per_minute = result.words_per_minute(video.duration_seconds)

    if video.words_per_minute is None and video.word_count:
        logger.warning(
            "%s has a transcript but no usable duration; words_per_minute left NULL.",
            video.video_id,
        )

    stats.transcripts_ok += 1
    logger.debug(
        "%s: %d words, %.1f wpm (%s)",
        video.video_id, result.word_count or 0, video.words_per_minute or 0.0, result.language,
    )
    return False


def enrich_dislikes(video: Video, client: DislikeClient, stats: RunStats) -> None:
    """Fetch and store the dislike count and like/dislike ratio for one video.

    ``like_dislike_ratio`` uses the official ``like_count`` from our own row
    when present, falling back to the value the dislike API reports. A zero or
    unknown dislike count yields ``NULL`` rather than a division by zero.
    """
    result = client.fetch(video.video_id)
    video.dislike_checked_at = _utcnow()

    if result.dislikes is None:
        logger.warning("No dislike data for %s; marked as checked.", video.video_id)
        video.dislike_count = None
        video.like_dislike_ratio = None
        stats.dislikes_missing += 1
        return

    video.dislike_count = result.dislikes
    video.like_dislike_ratio = result.like_dislike_ratio(video.like_count)
    stats.dislikes_ok += 1
    logger.debug(
        "%s: %d dislikes, ratio %s",
        video.video_id, result.dislikes, video.like_dislike_ratio,
    )


# --------------------------------------------------------------------------- #
# Orchestration                                                                #
# --------------------------------------------------------------------------- #
def run(
    *,
    limit: Optional[int] = None,
    batch_size: int = 20,
    want_transcripts: bool = True,
    want_dislikes: bool = True,
    retry_failed: bool = False,
    channel_id: Optional[str] = None,
) -> RunStats:
    """Enrich every pending video, committing every ``batch_size`` rows."""
    stats = RunStats()
    transcript_client = TranscriptClient() if want_transcripts else None
    dislike_client = DislikeClient() if want_dislikes else None

    try:
        with managed_session() as session:
            videos = find_pending_videos(
                session,
                limit=limit,
                want_transcripts=want_transcripts,
                want_dislikes=want_dislikes,
                retry_failed=retry_failed,
                channel_id=channel_id,
            )
            if not videos:
                logger.info("Nothing pending -- every video already has what it needs.")
                return stats

            logger.info("%d video(s) queued for enrichment.", len(videos))

            pending = 0
            consecutive_blocks = 0
            progress = tqdm(videos, desc="Enriching", unit="video", ncols=90)
            for video in progress:
                progress.set_postfix_str(video.video_id)
                stats.processed += 1

                try:
                    if transcript_client is not None and (
                        video.full_transcript is None
                        and (retry_failed or video.transcript_checked_at is None)
                    ):
                        if enrich_transcript(video, transcript_client, stats):
                            consecutive_blocks += 1
                            if consecutive_blocks >= BLOCK_ABORT_THRESHOLD:
                                # Hammering a blocked endpoint only extends the
                                # block; dislikes are unaffected, so keep those.
                                logger.error(
                                    "YouTube has blocked this IP for transcripts (%d in a row). "
                                    "Stopping transcript fetches; the rows stay queued. "
                                    "Wait a few hours, use a VPN, or set TRANSCRIPT_PROXY_URL "
                                    "in .env, then rerun.",
                                    consecutive_blocks,
                                )
                                transcript_client = None
                        else:
                            consecutive_blocks = 0

                    if dislike_client is not None and (
                        video.dislike_count is None
                        and (retry_failed or video.dislike_checked_at is None)
                    ):
                        enrich_dislikes(video, dislike_client, stats)

                except SQLAlchemyError as exc:
                    logger.error("Database error on %s: %s", video.video_id, exc)
                    session.rollback()
                    stats.errors += 1
                    pending = 0
                    continue
                except Exception as exc:
                    logger.exception("Unexpected failure on %s: %s", video.video_id, exc)
                    stats.errors += 1
                    continue

                pending += 1
                if pending >= batch_size:
                    pending = _commit(session, stats, pending)

            progress.close()
            if pending:
                _commit(session, stats, pending)

    except KeyboardInterrupt:
        logger.warning("Interrupted; committed batches are safe.")
    finally:
        if dislike_client is not None:
            dislike_client.close()

    return stats


def _commit(session: Session, stats: RunStats, pending: int) -> int:
    """Commit the staged rows; roll back and count the loss on failure."""
    try:
        session.commit()
        logger.debug("Committed %d row(s).", pending)
    except SQLAlchemyError as exc:
        logger.error("Batch commit failed, rolling back %d row(s): %s", pending, exc)
        session.rollback()
        stats.errors += pending
    return 0


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Backfill transcripts and dislike counts for stored videos.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--limit", type=int, default=None, help="Process at most N videos.")
    parser.add_argument("--batch-size", type=int, default=20, help="Rows per database commit.")
    parser.add_argument(
        "--only",
        choices=["transcripts", "dislikes", "both"],
        default="both",
        help="Restrict the run to one data source.",
    )
    parser.add_argument("--channel", default=None, help="Limit to a single channel id.")
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Also retry videos whose previous attempt failed (ignores the *_checked_at markers).",
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

    if not healthcheck():
        logger.error("Database unreachable. Check DB_CONN_STR in .env.")
        return 1

    logger.info("Target database: %s", settings.masked_summary())

    stats = run(
        limit=args.limit,
        batch_size=args.batch_size,
        want_transcripts=args.only in ("transcripts", "both"),
        want_dislikes=args.only in ("dislikes", "both"),
        retry_failed=args.retry_failed,
        channel_id=args.channel,
    )
    stats.log_summary()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
