"""Collect a channel's recent uploads into MSSQL.

Usage (from the workspace root, with the venv active)::

    python services/01_youtube_collector/fetch_historical.py UCxxxxxxxxxxxxxxxxxxxxxx
    python services/01_youtube_collector/fetch_historical.py @veritabani --limit 100
    python services/01_youtube_collector/fetch_historical.py UC... --skip-transcripts --only-new

Flow
----
1. Resolve the channel id (accepts ``UC...``, ``@handle`` or a channel URL).
2. Upsert the channel row.
3. Read the newest ``--limit`` video ids from the uploads playlist.
4. For each video: details -> transcript -> dislikes -> upsert, committing
   every ``--batch-size`` rows.

The script is idempotent: rerunning it refreshes counters on existing rows
rather than duplicating them, and it never overwrites columns owned by the
later pipeline stages (AI scores, media metrics, targets) with ``NULL``.
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

# The service directory starts with a digit, so it is not an importable package.
# Put the workspace root on sys.path so `core` and the sibling module resolve.
WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from api_client import (  # noqa: E402
    ChannelData,
    CollectorClients,
    CollectorError,
    DislikeData,
    QuotaExceededError,
    ShortsProbe,
    TranscriptData,
    VideoData,
)
from sqlalchemy.exc import SQLAlchemyError  # noqa: E402
from sqlalchemy.orm import Session  # noqa: E402
from tqdm import tqdm  # noqa: E402

from core.config import settings  # noqa: E402
from core.database import get_db, healthcheck  # noqa: E402
from core.models import Channel, Video  # noqa: E402

logger = logging.getLogger("collector")


# --------------------------------------------------------------------------- #
# Bookkeeping                                                                  #
# --------------------------------------------------------------------------- #
@dataclass
class RunStats:
    """Counters summarised at the end of the run."""

    videos_seen: int = 0
    inserted: int = 0
    updated: int = 0
    skipped_existing: int = 0
    unavailable: int = 0
    transcripts_ok: int = 0
    transcripts_missing: int = 0
    dislikes_ok: int = 0
    dislikes_missing: int = 0
    errors: int = 0

    def log_summary(self) -> None:
        logger.info("=" * 62)
        logger.info("Videos processed : %d", self.videos_seen)
        logger.info("  inserted       : %d", self.inserted)
        logger.info("  updated        : %d", self.updated)
        logger.info("  skipped        : %d (already present, --only-new)", self.skipped_existing)
        logger.info("  unavailable    : %d (deleted/private)", self.unavailable)
        logger.info("Transcripts      : %d ok / %d missing", self.transcripts_ok, self.transcripts_missing)
        logger.info("Dislikes         : %d ok / %d missing", self.dislikes_ok, self.dislikes_missing)
        logger.info("Row errors       : %d", self.errors)
        logger.info("=" * 62)


@contextmanager
def managed_session() -> Iterator[Session]:
    """Wrap :func:`core.database.get_db` in a context manager.

    ``get_db`` is a generator dependency, so this drives it manually: the
    caller commits in batches, and the session is always closed. Any escaping
    exception rolls back whatever is still pending.
    """
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
# Upserts                                                                      #
# --------------------------------------------------------------------------- #
def upsert_channel(session: Session, data: ChannelData) -> Channel:
    """Insert or refresh the channel row and flush it.

    The channel must exist before any video row can satisfy the foreign key,
    so this flushes immediately.
    """
    channel = session.get(Channel, data.channel_id)
    if channel is None:
        channel = Channel(channel_id=data.channel_id)
        session.add(channel)
        logger.info("New channel: %s (%s)", data.title, data.channel_id)
    else:
        logger.info("Refreshing channel: %s (%s)", data.title, data.channel_id)

    channel.title = data.title
    channel.country = data.country
    channel.subscriber_count = data.subscriber_count
    channel.total_views = data.total_views
    channel.video_count = data.video_count
    channel.channel_creation_date = data.channel_creation_date
    channel.last_scraped_at = _utcnow()

    session.flush()
    return channel


def upsert_video(
    session: Session,
    details: VideoData,
    transcript: TranscriptData,
    dislikes: DislikeData,
    *,
    channel_id: str,
) -> bool:
    """Insert or refresh one video row. Returns True when it was an insert.

    Only columns this service owns are written. Fields belonging to
    ``02_ai_analyzer`` and ``03_predictor_engine`` are left untouched so that a
    re-collection does not wipe out expensive AI scores.
    """
    video = session.get(Video, details.video_id)
    is_new = video is None
    if is_new:
        video = Video(video_id=details.video_id)
        session.add(video)

    # -- YouTube Data API ------------------------------------------------
    video.channel_id = details.channel_id or channel_id
    video.published_at = details.published_at
    video.title = details.title
    video.description = details.description
    video.tags = details.tags
    video.category_id = details.category_id
    video.duration_seconds = details.duration_seconds
    video.view_count = details.view_count
    video.like_count = details.like_count
    video.comment_count = details.comment_count
    video.thumbnail_url_maxres = details.thumbnail_url_maxres
    if details.is_shorts is not None:
        video.is_shorts = details.is_shorts

    # -- Transcript ------------------------------------------------------
    # Keep an existing transcript if this run could not fetch one: a transient
    # block should not erase text we already paid to collect.
    if transcript.text is not None:
        video.full_transcript = transcript.text
        video.word_count = transcript.word_count
        video.words_per_minute = transcript.words_per_minute(details.duration_seconds)
    elif is_new:
        video.full_transcript = None
        video.word_count = None
        video.words_per_minute = None

    # -- Dislikes --------------------------------------------------------
    if dislikes.dislikes is not None:
        video.dislike_count = dislikes.dislikes
        video.like_dislike_ratio = dislikes.like_dislike_ratio(details.like_count)
    elif is_new:
        video.dislike_count = None
        video.like_dislike_ratio = None

    # Cheap and derived purely from this row; stage 3 recomputes it anyway.
    video.compute_engagement_rate()

    return is_new


# --------------------------------------------------------------------------- #
# Orchestration                                                                #
# --------------------------------------------------------------------------- #
def collect_channel(
    channel_input: str,
    *,
    limit: int = 50,
    batch_size: int = 10,
    skip_transcripts: bool = False,
    skip_dislikes: bool = False,
    verify_shorts: bool = False,
    only_new: bool = False,
) -> RunStats:
    """Run the full collection for one channel and return the run counters."""
    stats = RunStats()
    clients = CollectorClients(shorts_probe=ShortsProbe() if verify_shorts else None)

    try:
        channel_id = clients.youtube.resolve_channel_id(channel_input)
        if not channel_id:
            logger.error("Could not resolve %r to a channel id.", channel_input)
            return stats

        channel_data = clients.youtube.get_channel(channel_id)
        if channel_data is None:
            logger.error("Channel %s returned no data; aborting.", channel_id)
            return stats

        if not channel_data.uploads_playlist_id:
            logger.error("Channel %s exposes no uploads playlist; aborting.", channel_id)
            return stats

        with managed_session() as session:
            upsert_channel(session, channel_data)
            session.commit()

            video_ids = clients.youtube.get_video_ids(channel_data.uploads_playlist_id, limit=limit)
            if not video_ids:
                logger.warning("No videos found for channel %s.", channel_id)
                return stats

            if only_new:
                existing = {
                    vid
                    for (vid,) in session.query(Video.video_id).filter(Video.video_id.in_(video_ids)).all()
                }
                stats.skipped_existing = len(existing)
                video_ids = [v for v in video_ids if v not in existing]
                logger.info("--only-new: skipping %d already-stored video(s).", len(existing))
                if not video_ids:
                    logger.info("Nothing new to collect.")
                    return stats

            # One batched call for all details; transcripts/dislikes are per video.
            logger.info("Fetching details for %d video(s)...", len(video_ids))
            details_map = clients.youtube.get_videos(video_ids)

            pending = 0
            progress = tqdm(video_ids, desc="Collecting", unit="video", ncols=90)
            for video_id in progress:
                progress.set_postfix_str(video_id)
                stats.videos_seen += 1

                details = details_map.get(video_id)
                if details is None:
                    stats.unavailable += 1
                    continue

                if verify_shorts and clients.shorts_probe is not None:
                    probed = clients.shorts_probe.is_shorts(
                        video_id, duration_seconds=details.duration_seconds
                    )
                    if probed is not None:
                        details.is_shorts = probed

                transcript = TranscriptData()
                if not skip_transcripts:
                    transcript = clients.transcripts.fetch(video_id)
                    if transcript.text:
                        stats.transcripts_ok += 1
                    else:
                        stats.transcripts_missing += 1

                dislikes = DislikeData()
                if not skip_dislikes:
                    dislikes = clients.dislikes.fetch(video_id)
                    if dislikes.dislikes is not None:
                        stats.dislikes_ok += 1
                    else:
                        stats.dislikes_missing += 1

                try:
                    inserted = upsert_video(
                        session, details, transcript, dislikes, channel_id=channel_id
                    )
                except SQLAlchemyError as exc:
                    # One bad row must not cost us the whole batch.
                    logger.error("Failed to stage %s: %s", video_id, exc)
                    session.rollback()
                    stats.errors += 1
                    pending = 0
                    continue

                stats.inserted += int(inserted)
                stats.updated += int(not inserted)
                pending += 1

                if pending >= batch_size:
                    pending = _commit(session, stats, pending)

            progress.close()
            if pending:
                _commit(session, stats, pending)

    except QuotaExceededError as exc:
        logger.error("%s", exc)
        logger.error("Partial results were committed; rerun tomorrow to continue.")
    except CollectorError as exc:
        logger.error("%s", exc)
    finally:
        clients.close()

    return stats


def _commit(session: Session, stats: RunStats, pending: int) -> int:
    """Commit the staged rows; on failure roll back and count the loss."""
    try:
        session.commit()
        logger.debug("Committed %d row(s).", pending)
    except SQLAlchemyError as exc:
        logger.error("Batch commit failed, rolling back %d row(s): %s", pending, exc)
        session.rollback()
        stats.errors += pending
        stats.inserted = max(0, stats.inserted - pending)
    return 0


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect YouTube channel and video data into MSSQL.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("channel", help="Channel id (UC...), @handle, or channel URL.")
    parser.add_argument("--limit", type=int, default=50, help="How many recent videos to collect.")
    parser.add_argument("--batch-size", type=int, default=10, help="Rows per database commit.")
    parser.add_argument("--skip-transcripts", action="store_true", help="Do not fetch captions.")
    parser.add_argument("--skip-dislikes", action="store_true", help="Do not call the dislike API.")
    parser.add_argument(
        "--verify-shorts",
        action="store_true",
        help="Confirm Shorts via an HTTP probe instead of the duration<=60s proxy.",
    )
    parser.add_argument(
        "--only-new", action="store_true", help="Skip videos already stored instead of refreshing them."
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
    # googleapiclient is chatty about its discovery cache at INFO.
    logging.getLogger("googleapiclient.discovery_cache").setLevel(logging.ERROR)

    if args.limit < 1:
        logger.error("--limit must be at least 1.")
        return 2

    if not settings.youtube_api_key:
        logger.error("YOUTUBE_API_KEY is not set in .env; cannot call the Data API.")
        return 1

    if not healthcheck():
        logger.error("Database unreachable. Check DB_CONN_STR in .env.")
        return 1

    logger.info("Target database: %s", settings.masked_summary())

    stats = collect_channel(
        args.channel,
        limit=args.limit,
        batch_size=args.batch_size,
        skip_transcripts=args.skip_transcripts,
        skip_dislikes=args.skip_dislikes,
        verify_shorts=args.verify_shorts,
        only_new=args.only_new,
    )
    stats.log_summary()

    if stats.videos_seen == 0 and stats.skipped_existing == 0:
        return 1
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt as exc:
        logger.warning("Interrupted by user; committed batches are safe.")
        raise SystemExit(130) from exc
