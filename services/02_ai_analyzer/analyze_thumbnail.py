"""Analyse video thumbnails with the OpenAI Vision API.

Fills ``thumbnail_has_face``, ``thumbnail_face_emotion``, ``thumbnail_text``
and ``title_thumbnail_synergy``.

Usage (from the repo root, venv active)::

    python services/02_ai_analyzer/analyze_thumbnail.py --limit 50
    python services/02_ai_analyzer/analyze_thumbnail.py --model gpt-4.1-mini
    python services/02_ai_analyzer/analyze_thumbnail.py --dry-run --limit 3

Unreachable images
------------------
YouTube serves ``maxresdefault.jpg`` for most uploads but not all; the URL 404s
for older or low-resolution videos, and OpenAI would then bill a failed call.
Each URL is HEAD-checked first, and on 404 the analyser automatically falls
back to ``hqdefault.jpg``, which YouTube generates for every video. Only if
both fail is the row marked and skipped.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Literal, Optional

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import requests  # noqa: E402
from ai_client import (  # noqa: E402
    AnalysisError,
    FatalAIError,
    UsageTracker,
    clamp_score,
    get_client,
    parse_structured,
)
from pydantic import BaseModel, Field  # noqa: E402
from sqlalchemy import select  # noqa: E402
from sqlalchemy.exc import SQLAlchemyError  # noqa: E402
from sqlalchemy.orm import Session  # noqa: E402
from tqdm import tqdm  # noqa: E402

from core.config import settings  # noqa: E402
from core.database import get_db, healthcheck  # noqa: E402
from core.models import Video  # noqa: E402

logger = logging.getLogger("thumbnail_analyzer")

#: Closed vocabulary; "None" is the answer when there is no face at all.
FaceEmotion = Literal[
    "Happy", "Surprised", "Angry", "Fearful", "Sad", "Disgusted", "Neutral", "None"
]

#: Fallback thumbnail sizes, best first. hqdefault always exists.
THUMBNAIL_FALLBACKS = ("maxresdefault", "sddefault", "hqdefault", "mqdefault")

#: NVARCHAR(512) on the column; keep a margin for multi-byte characters.
MAX_THUMBNAIL_TEXT = 500


class ThumbnailAnalysis(BaseModel):
    """Structured description of one thumbnail.

    ``title_thumbnail_synergy`` has no ``ge``/``le``: strict Structured Outputs
    rejects numeric bounds, so the range lives in the prompt and is clamped by
    :func:`ai_client.clamp_score`.
    """

    thumbnail_has_face: bool = Field(
        description="True if at least one human face is clearly visible."
    )
    thumbnail_face_emotion: FaceEmotion = Field(
        description="Dominant expression of the most prominent face, or 'None' if there is no face."
    )
    thumbnail_text: str = Field(
        description="Text overlaid on the thumbnail, transcribed verbatim. Empty string if none."
    )
    title_thumbnail_synergy: float = Field(
        description="1-10. How well the thumbnail and title work together rather than repeat."
    )
    reasoning: str = Field(
        description="One sentence justifying the synergy score. Max 200 characters."
    )


SYSTEM_PROMPT = """\
You are a YouTube thumbnail expert who has A/B tested click-through rates on \
thousands of videos. You describe what is actually in the image and score \
objectively across the full range of the scale.

You receive a video TITLE and its THUMBNAIL image. Report:

1. thumbnail_has_face - true only if a human face is clearly visible and \
recognisable as a face. Cartoon or illustrated faces count; a tiny face in a \
crowd or a back-of-head shot does not.

2. thumbnail_face_emotion - the dominant expression of the largest/most \
prominent face. Use exactly "None" when there is no face. Exaggerated \
open-mouth "shock" faces are Surprised, not Happy.

3. thumbnail_text - transcribe any text burned into the image exactly as it \
appears, preserving line order and separating lines with " | ". Include \
numbers and symbols. Return an empty string if the image has no text. Ignore \
watermarks, channel logos, and platform UI such as the duration badge.

4. title_thumbnail_synergy - 1.0 to 10.0. How well do the title and the \
thumbnail work TOGETHER?
   9-10: the thumbnail adds information the title does not carry -- a visual \
stake, a before/after, a reaction, or text that sharpens the title's promise \
without repeating its words.
   5-6:  relevant but passive; illustrates the topic and adds nothing.
   3-4:  the thumbnail text simply repeats the title's words. Redundancy \
wastes the click, so score it low even when the design is attractive.
   1-2:  the thumbnail contradicts, misleads about, or has nothing to do with \
the title.
   Judge the pairing, not the visual polish: a plain thumbnail that adds a real \
question outscores a beautiful one that echoes the title.

Keep reasoning under 200 characters.
"""


@dataclass
class RunStats:
    processed: int = 0
    analyzed: int = 0
    unreachable: int = 0
    failures: int = 0
    errors: int = 0
    fallbacks_used: int = 0

    def log_summary(self) -> None:
        logger.info("=" * 60)
        logger.info("Thumbnails processed : %d", self.processed)
        logger.info("Analysed             : %d", self.analyzed)
        logger.info("  via fallback URL   : %d", self.fallbacks_used)
        logger.info("Unreachable images   : %d", self.unreachable)
        logger.info("API failures         : %d", self.failures)
        logger.info("Row errors           : %d", self.errors)
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
    return datetime.now(timezone.utc).replace(tzinfo=None)


# --------------------------------------------------------------------------- #
# Image reachability                                                           #
# --------------------------------------------------------------------------- #
def candidate_urls(video: Video) -> list[str]:
    """Return the stored thumbnail URL plus progressively smaller fallbacks."""
    urls: list[str] = []
    if video.thumbnail_url_maxres:
        urls.append(video.thumbnail_url_maxres)

    # Rebuild the canonical i.ytimg.com paths, which exist even when the stored
    # URL has rotted.
    for size in THUMBNAIL_FALLBACKS:
        candidate = f"https://i.ytimg.com/vi/{video.video_id}/{size}.jpg"
        if candidate not in urls:
            urls.append(candidate)
    return urls


def first_reachable_url(
    urls: list[str], session: requests.Session, *, timeout: float = 10.0
) -> Optional[str]:
    """Return the first URL that answers 200, or ``None`` if all fail.

    OpenAI charges for a vision call even when the image fetch fails on their
    side, so this cheap HEAD check pays for itself.
    """
    for url in urls:
        try:
            response = session.head(url, timeout=timeout, allow_redirects=True)
        except requests.RequestException as exc:
            logger.debug("HEAD failed for %s: %s", url, exc)
            continue
        if response.status_code == 200:
            return url
        logger.debug("HTTP %s for %s", response.status_code, url)
    return None


# --------------------------------------------------------------------------- #
# Query                                                                        #
# --------------------------------------------------------------------------- #
def find_pending_videos(
    session: Session,
    *,
    limit: Optional[int] = None,
    channel_id: Optional[str] = None,
    reanalyze: bool = False,
) -> list[Video]:
    """Return videos with a thumbnail URL but no vision results yet."""
    stmt = select(Video).where(Video.thumbnail_url_maxres.is_not(None))
    if not reanalyze:
        stmt = stmt.where(Video.thumbnail_has_face.is_(None))
    if channel_id:
        stmt = stmt.where(Video.channel_id == channel_id)
    stmt = stmt.order_by(Video.published_at.desc())
    if limit:
        stmt = stmt.limit(limit)
    return list(session.execute(stmt).scalars())


def build_messages(video: Video, image_url: str, *, detail: str = "low") -> list[dict]:
    """Assemble the multimodal payload for one thumbnail.

    ``detail="low"`` downsamples the image to 512x512 for a flat ~85 tokens.
    Thumbnails are read at a glance anyway -- faces, big text and composition
    all survive the downsample, and it cuts the per-image cost by roughly an
    order of magnitude versus "high".
    """
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"VIDEO TITLE: {video.title or '(untitled)'}"},
                {"type": "image_url", "image_url": {"url": image_url, "detail": detail}},
            ],
        },
    ]


def apply_analysis(video: Video, analysis: ThumbnailAnalysis, model: str) -> None:
    """Write the vision results onto the ORM object.

    Writes only the vision columns. ``ai_model_version``/``ai_analyzed_at``
    belong to the text analyser -- touching them here would let whichever
    service ran last claim credit for both sets of scores.
    """
    video.thumbnail_has_face = analysis.thumbnail_has_face

    # Normalise the "no face" case to NULL rather than the string "None", so
    # the column never mixes a literal with an absence.
    emotion = analysis.thumbnail_face_emotion
    video.thumbnail_face_emotion = None if emotion == "None" else emotion

    text = re.sub(r"\s+", " ", analysis.thumbnail_text or "").strip()
    video.thumbnail_text = text[:MAX_THUMBNAIL_TEXT] or None

    video.title_thumbnail_synergy = clamp_score(analysis.title_thumbnail_synergy)
    video.vision_model_version = model
    video.vision_analyzed_at = _utcnow()


# --------------------------------------------------------------------------- #
# Orchestration                                                                #
# --------------------------------------------------------------------------- #
def run(
    *,
    model: str,
    limit: Optional[int] = None,
    batch_size: int = 10,
    detail: str = "low",
    channel_id: Optional[str] = None,
    reanalyze: bool = False,
    dry_run: bool = False,
) -> RunStats:
    """Analyse every pending thumbnail, committing every ``batch_size`` rows."""
    stats = RunStats()
    usage = UsageTracker()
    client = get_client()
    http = requests.Session()
    http.headers.update({"User-Agent": "youtube-prediction-analyzer/0.1"})

    try:
        with managed_session() as session:
            videos = find_pending_videos(
                session, limit=limit, channel_id=channel_id, reanalyze=reanalyze
            )
            if not videos:
                logger.info("Nothing pending -- every thumbnail already has vision results.")
                return stats

            logger.info("%d thumbnail(s) queued for analysis with %s.", len(videos), model)
            if dry_run:
                logger.warning("DRY RUN: results will be printed, not saved.")

            pending = 0
            progress = tqdm(videos, desc="Vision", unit="image", ncols=90)
            for video in progress:
                progress.set_postfix_str(video.video_id)
                stats.processed += 1

                urls = candidate_urls(video)
                image_url = first_reachable_url(urls, http)
                if image_url is None:
                    logger.warning(
                        "No reachable thumbnail for %s (tried %d URL(s)); skipping.",
                        video.video_id, len(urls),
                    )
                    stats.unreachable += 1
                    continue
                if image_url != video.thumbnail_url_maxres:
                    logger.info("Using fallback thumbnail for %s: %s", video.video_id, image_url)
                    stats.fallbacks_used += 1

                try:
                    result = parse_structured(
                        client,
                        model=model,
                        schema=ThumbnailAnalysis,
                        messages=build_messages(video, image_url, detail=detail),
                        usage=usage,
                    )
                    analysis = result.value
                except AnalysisError as exc:
                    logger.warning("Skipping %s: %s", video.video_id, exc)
                    stats.failures += 1
                    continue
                except FatalAIError:
                    raise
                except Exception:
                    logger.exception("Unexpected failure on %s", video.video_id)
                    stats.failures += 1
                    continue

                logger.info(
                    "%s face=%s emotion=%s synergy=%.1f text=%r",
                    video.video_id, analysis.thumbnail_has_face, analysis.thumbnail_face_emotion,
                    analysis.title_thumbnail_synergy, (analysis.thumbnail_text or "")[:60],
                )
                stats.analyzed += 1

                if dry_run:
                    continue

                try:
                    apply_analysis(video, analysis, result.model)
                except SQLAlchemyError as exc:
                    logger.error("Database error on %s: %s", video.video_id, exc)
                    session.rollback()
                    stats.errors += 1
                    pending = 0
                    continue

                pending += 1
                if pending >= batch_size:
                    pending = _commit(session, stats, pending)

            progress.close()
            if pending and not dry_run:
                _commit(session, stats, pending)

    except FatalAIError as exc:
        logger.error("%s", exc)
    except KeyboardInterrupt:
        logger.warning("Interrupted; committed batches are safe.")
    finally:
        http.close()
        usage.log_summary()

    return stats


def _commit(session: Session, stats: RunStats, pending: int) -> int:
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
        description="Analyse video thumbnails with the OpenAI Vision API.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", default=settings.openai_vision_model, help="Vision-capable model id.")
    parser.add_argument("--limit", type=int, default=None, help="Process at most N thumbnails.")
    parser.add_argument("--batch-size", type=int, default=10, help="Rows per database commit.")
    parser.add_argument(
        "--detail", choices=["low", "high", "auto"], default="low",
        help="Image fidelity sent to the model. 'low' is ~85 tokens per image.",
    )
    parser.add_argument("--channel", default=None, help="Limit to a single channel id.")
    parser.add_argument(
        "--reanalyze", action="store_true", help="Re-analyse thumbnails that already have results."
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Call the API and print results without saving."
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
    logging.getLogger("httpx").setLevel(logging.WARNING)

    if not settings.openai_api_key:
        logger.error("OPENAI_API_KEY is not set in .env.")
        return 1
    if not healthcheck():
        logger.error("Database unreachable. Check DB_CONN_STR in .env.")
        return 1

    logger.info("Target database: %s", settings.masked_summary())

    stats = run(
        model=args.model,
        limit=args.limit,
        batch_size=args.batch_size,
        detail=args.detail,
        channel_id=args.channel,
        reanalyze=args.reanalyze,
        dry_run=args.dry_run,
    )
    stats.log_summary()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
