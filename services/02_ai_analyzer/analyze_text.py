"""Score the title and opening transcript of each video with an LLM.

Fills four columns: ``hook_score``, ``curiosity_gap_score``, ``emotion_tone``
and ``niche_relevance``, then stamps ``ai_analyzed_at`` and
``ai_model_version``.

Usage (from the repo root, venv active)::

    python services/02_ai_analyzer/analyze_text.py --limit 50
    python services/02_ai_analyzer/analyze_text.py --model gpt-4.1-mini
    python services/02_ai_analyzer/analyze_text.py --dry-run --limit 3

Cost control
------------
Only the first ``--max-chars`` (default 3000) characters of the transcript are
sent. The hook lives in the first 30 seconds, so the tail adds cost without
adding signal: a 40-minute talk with a 42k-character transcript is billed as
~800 tokens instead of ~11k.
"""

from __future__ import annotations

import argparse
import logging
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

logger = logging.getLogger("text_analyzer")

#: Closed vocabulary for `emotion_tone`. A free-text string would fragment into
#: dozens of near-synonyms and be useless as a model feature.
EmotionTone = Literal[
    "Educational",
    "Entertainment",
    "Motivational",
    "Drama",
    "News",
    "Tech",
    "Comedy",
    "Review",
    "Tutorial",
    "Other",
]


class VideoTextAnalysis(BaseModel):
    """Structured scores returned by the model.

    No ``ge``/``le`` constraints: strict Structured Outputs rejects
    ``minimum``/``maximum``. The 1-10 range is stated in the prompt and
    enforced by :func:`ai_client.clamp_score` on the way in.
    """

    hook_score: float = Field(
        description="1-10. How strongly the opening lines earn the next 30 seconds of attention."
    )
    curiosity_gap_score: float = Field(
        description="1-10. How well the title opens a question the content genuinely answers."
    )
    emotion_tone: EmotionTone = Field(
        description="The single dominant tone of the content."
    )
    niche_relevance: float = Field(
        description="1-10. How coherent and specific the topic is for a defined audience."
    )
    reasoning: str = Field(
        description="One sentence justifying the hook and curiosity scores. Max 200 characters."
    )


SYSTEM_PROMPT = """\
You are a senior YouTube content strategist who has audited thousands of videos \
for retention and click-through performance. You score objectively and use the \
full range of the scale.

You will receive a video TITLE and the OPENING of its transcript (roughly the \
first few minutes, as spoken). Score four dimensions from 1.0 to 10.0.

1. hook_score - Does the opening earn the viewer's next 30 seconds?
   9-10: states a concrete stake, question or surprising claim within the first \
sentences; zero throat-clearing.
   5-6:  competent but generic ("in this video we'll look at...").
   1-3:  long intros, channel branding, greetings, sponsor reads or waffle \
before any substance.
   Judge only what is actually said at the start; do not credit what the video \
might do later.

2. curiosity_gap_score - Does the title open a question the opening promises to \
answer?
   9-10: a specific, resolvable question that the content clearly addresses.
   5-6:  descriptive title, mild interest, no real tension.
   1-3:  either flat and purely descriptive, OR clickbait -- a gap the content \
does not close. Penalise unresolved clickbait as hard as blandness; both break \
the viewer's trust.

3. emotion_tone - The single dominant register of the content.

4. niche_relevance - How specific and coherent the topic is for a well-defined \
audience.
   9-10: unmistakably aimed at one audience with consistent terminology.
   5-6:  broad general-interest treatment.
   1-3:  scattered, off-topic or incoherent.

Rules:
- Judge the writing and delivery, not the subject's popularity.
- A transcript may be auto-generated, so ignore punctuation and spelling noise.
- If the transcript is too short or empty to judge, score conservatively \
around 3-4 and say so in your reasoning.
- Keep reasoning under 200 characters.
"""


@dataclass
class RunStats:
    processed: int = 0
    analyzed: int = 0
    failures: int = 0
    errors: int = 0

    def log_summary(self) -> None:
        logger.info("=" * 60)
        logger.info("Videos processed : %d", self.processed)
        logger.info("Scored           : %d", self.analyzed)
        logger.info("API failures     : %d", self.failures)
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
    return datetime.now(timezone.utc).replace(tzinfo=None)


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
    """Return videos that have a transcript but no text scores yet."""
    stmt = select(Video).where(Video.full_transcript.is_not(None))
    if not reanalyze:
        stmt = stmt.where(Video.hook_score.is_(None))
    if channel_id:
        stmt = stmt.where(Video.channel_id == channel_id)
    stmt = stmt.order_by(Video.published_at.desc())
    if limit:
        stmt = stmt.limit(limit)
    return list(session.execute(stmt).scalars())


def build_messages(video: Video, *, max_chars: int) -> list[dict[str, str]]:
    """Assemble the two-message payload for one video."""
    transcript = (video.full_transcript or "")[:max_chars].strip()
    if not transcript:
        transcript = "(no transcript text available)"

    user_content = (
        f"TITLE: {video.title or '(untitled)'}\n"
        f"DURATION: {video.duration_seconds or 'unknown'} seconds\n\n"
        f"TRANSCRIPT OPENING (first {len(transcript)} characters):\n{transcript}"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def apply_analysis(video: Video, analysis: VideoTextAnalysis, model: str) -> None:
    """Write the scores onto the ORM object, clamping to the allowed range."""
    video.hook_score = clamp_score(analysis.hook_score)
    video.curiosity_gap_score = clamp_score(analysis.curiosity_gap_score)
    video.niche_relevance = clamp_score(analysis.niche_relevance)
    video.emotion_tone = analysis.emotion_tone
    video.ai_analyzed_at = _utcnow()
    video.ai_model_version = model


# --------------------------------------------------------------------------- #
# Orchestration                                                                #
# --------------------------------------------------------------------------- #
def run(
    *,
    model: str,
    limit: Optional[int] = None,
    batch_size: int = 10,
    max_chars: int = 3000,
    channel_id: Optional[str] = None,
    reanalyze: bool = False,
    dry_run: bool = False,
) -> RunStats:
    """Score every pending video, committing every ``batch_size`` rows."""
    stats = RunStats()
    usage = UsageTracker()
    client = get_client()

    try:
        with managed_session() as session:
            videos = find_pending_videos(
                session, limit=limit, channel_id=channel_id, reanalyze=reanalyze
            )
            if not videos:
                logger.info("Nothing pending -- every transcript already has text scores.")
                return stats

            logger.info("%d video(s) queued for text analysis with %s.", len(videos), model)
            if dry_run:
                logger.warning("DRY RUN: results will be printed, not saved.")

            pending = 0
            progress = tqdm(videos, desc="Scoring", unit="video", ncols=90)
            for video in progress:
                progress.set_postfix_str(video.video_id)
                stats.processed += 1

                try:
                    result = parse_structured(
                        client,
                        model=model,
                        schema=VideoTextAnalysis,
                        messages=build_messages(video, max_chars=max_chars),
                        usage=usage,
                    )
                    analysis = result.value
                except AnalysisError as exc:
                    # Row-specific: leave the columns NULL so a later run retries.
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
                    "%s hook=%.1f curiosity=%.1f niche=%.1f tone=%s | %s",
                    video.video_id, analysis.hook_score, analysis.curiosity_gap_score,
                    analysis.niche_relevance, analysis.emotion_tone,
                    analysis.reasoning[:120],
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
        description="Score video titles and transcript openings with an LLM.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", default=settings.openai_text_model, help="OpenAI model id.")
    parser.add_argument("--limit", type=int, default=None, help="Process at most N videos.")
    parser.add_argument("--batch-size", type=int, default=10, help="Rows per database commit.")
    parser.add_argument(
        "--max-chars", type=int, default=3000, help="Transcript characters sent to the API."
    )
    parser.add_argument("--channel", default=None, help="Limit to a single channel id.")
    parser.add_argument(
        "--reanalyze", action="store_true", help="Re-score videos that already have scores."
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
        max_chars=args.max_chars,
        channel_id=args.channel,
        reanalyze=args.reanalyze,
        dry_run=args.dry_run,
    )
    stats.log_summary()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
