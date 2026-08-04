"""Extract audio and visual pacing metrics from the videos themselves.

This is the CPU-heavy sibling of ``fetch_transcripts_and_dislikes.py``: it
downloads a low-quality copy of each video, measures it locally, writes three
columns, and deletes the file. Expect ~20-60s per video, dominated by pitch
tracking.

    silence_ratio          fraction of the audio that is dead air (0-1)
    pitch_variance         std-dev of the fundamental frequency, in Hz
    scene_cuts_per_minute  detected shot changes per minute

Usage (from the repo root, venv active)::

    python services/01_youtube_collector/analyze_media_pacing.py --limit 10
    python services/01_youtube_collector/analyze_media_pacing.py --max-seconds 180
    python services/01_youtube_collector/analyze_media_pacing.py --retry-failed

Requires the **ffmpeg** binary on PATH (used to demux audio to WAV).

Sampling
--------
Long videos are analysed from a bounded window (``--max-seconds``, default 300)
rather than end to end. Editing rhythm and vocal energy are stable within a
video, so a five-minute sample of a two-hour upload gives essentially the same
numbers for a fraction of the CPU. The sample starts at ``--skip-intro``
seconds so channel intros do not dominate short videos.
"""

from __future__ import annotations

import argparse
import logging
import math
import shutil
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from sqlalchemy import or_, select  # noqa: E402
from sqlalchemy.exc import SQLAlchemyError  # noqa: E402
from sqlalchemy.orm import Session  # noqa: E402
from tqdm import tqdm  # noqa: E402

from core.config import settings  # noqa: E402
from core.database import get_db, healthcheck  # noqa: E402
from core.models import Video  # noqa: E402

logger = logging.getLogger("media")

#: Audio is resampled to this rate; plenty for speech F0 (80-400 Hz).
AUDIO_SAMPLE_RATE = 22050

#: librosa.effects.split threshold. 30 dB below peak counts as silence --
#: forgiving enough to ignore room tone, strict enough to catch real pauses.
SILENCE_TOP_DB = 30

#: Mean absolute frame difference (0-255) above which we call it a cut.
SCENE_CUT_THRESHOLD = 28.0

#: Analyse every Nth frame. At 30fps this samples ~3 times a second.
FRAME_SAMPLE_STEP = 10


class MediaError(RuntimeError):
    """Raised when a video cannot be downloaded or decoded."""


@dataclass
class MediaMetrics:
    """The three pacing measurements for one video."""

    silence_ratio: Optional[float] = None
    pitch_variance: Optional[float] = None
    scene_cuts_per_minute: Optional[float] = None

    def is_empty(self) -> bool:
        return all(
            v is None
            for v in (self.silence_ratio, self.pitch_variance, self.scene_cuts_per_minute)
        )


@dataclass
class RunStats:
    processed: int = 0
    analyzed: int = 0
    download_failures: int = 0
    analysis_failures: int = 0
    errors: int = 0

    def log_summary(self) -> None:
        logger.info("=" * 60)
        logger.info("Videos processed  : %d", self.processed)
        logger.info("Fully analysed    : %d", self.analyzed)
        logger.info("Download failures : %d", self.download_failures)
        logger.info("Analysis failures : %d", self.analysis_failures)
        logger.info("Row errors        : %d", self.errors)
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
# 1. Download                                                                  #
# --------------------------------------------------------------------------- #
@contextmanager
def download_media(
    video_id: str,
    workdir: Path,
    *,
    window: Optional[tuple[int, int]] = None,
) -> Iterator[Path]:
    """Download the smallest usable copy of a video into ``workdir``.

    ``window`` is an optional ``(start, end)`` second range. Supplying it makes
    yt-dlp fetch only that slice instead of the whole file, which is the single
    biggest speed win here: analysing a 2-minute window of a 40-minute upload
    otherwise spends minutes downloading footage that is immediately discarded.

    Yields the downloaded path and **always** deletes it on exit, including
    when the analysis raises. Disk usage therefore stays at one video at a
    time regardless of how the body fails.

    Raises:
        MediaError: download failed, was blocked, or produced no file.
    """
    import yt_dlp
    from yt_dlp.utils import download_range_func

    outtmpl = str(workdir / f"{video_id}.%(ext)s")
    options = {
        # Smallest progressive stream that still carries audio *and* video.
        "format": "worst[acodec!=none][vcodec!=none]/worst",
        "outtmpl": outtmpl,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "noplaylist": True,
        "socket_timeout": 30,
        "retries": 2,
        "logger": logging.getLogger("yt_dlp"),
    }

    if window is not None:
        start, end = window
        # Cuts at the nearest keyframe rather than re-encoding: a second or two
        # of slack at the edges is irrelevant for pacing statistics.
        options["download_ranges"] = download_range_func(None, [(start, end)])
        options["force_keyframes_at_cuts"] = False

    downloaded: Optional[Path] = None
    try:
        try:
            with yt_dlp.YoutubeDL(options) as ydl:
                ydl.download([f"https://www.youtube.com/watch?v={video_id}"])
        except Exception as exc:
            raise MediaError(f"download failed: {type(exc).__name__}: {exc}") from exc

        matches = sorted(workdir.glob(f"{video_id}.*"))
        if not matches:
            raise MediaError("download produced no file")

        downloaded = matches[0]
        size_mb = downloaded.stat().st_size / 1_048_576
        if size_mb == 0:
            raise MediaError("downloaded file is empty")
        logger.debug("Downloaded %s (%.1f MB)", downloaded.name, size_mb)

        yield downloaded

    finally:
        # CRITICAL: never leave media behind -- a few hundred videos would
        # otherwise fill the disk.
        for leftover in workdir.glob(f"{video_id}.*"):
            try:
                leftover.unlink()
                logger.debug("Deleted %s", leftover.name)
            except OSError as exc:
                logger.error("Could not delete %s: %s", leftover, exc)


def extract_audio(media_path: Path, wav_path: Path, *, max_seconds: int, skip_seconds: int) -> Path:
    """Demux a mono WAV window from the media file using ffmpeg.

    librosa cannot read MP4/WebM directly, and going through ffmpeg once is
    faster and far more predictable than audioread's fallbacks.

    Raises:
        MediaError: ffmpeg is missing, failed, or produced nothing.
    """
    if shutil.which("ffmpeg") is None:
        raise MediaError("ffmpeg is not on PATH; audio metrics cannot be computed")

    command = [
        "ffmpeg", "-nostdin", "-loglevel", "error", "-y",
        "-ss", str(skip_seconds),
        "-t", str(max_seconds),
        "-i", str(media_path),
        "-vn",                      # drop the video stream
        "-ac", "1",                 # mono
        "-ar", str(AUDIO_SAMPLE_RATE),
        str(wav_path),
    ]
    try:
        result = subprocess.run(command, capture_output=True, timeout=300, check=False)
    except subprocess.TimeoutExpired as exc:
        raise MediaError("ffmpeg timed out while extracting audio") from exc

    if result.returncode != 0 or not wav_path.exists() or wav_path.stat().st_size == 0:
        detail = result.stderr.decode("utf-8", "replace").strip()[:200]
        raise MediaError(f"ffmpeg could not extract audio: {detail or 'no output'}")

    return wav_path


# --------------------------------------------------------------------------- #
# 2. Audio analysis                                                            #
# --------------------------------------------------------------------------- #
def analyze_audio(wav_path: Path) -> tuple[Optional[float], Optional[float]]:
    """Return ``(silence_ratio, pitch_variance)`` for a WAV file.

    * ``silence_ratio`` -- 1 minus the share of the timeline that
      ``librosa.effects.split`` marks as non-silent. 0.0 means wall-to-wall
      sound, 0.4 means almost half dead air.
    * ``pitch_variance`` -- standard deviation in Hz of the voiced F0 track
      from ``librosa.pyin``. A monotone reader lands near 15-25 Hz; an
      animated presenter is well above that.

    Either element is ``None`` when it could not be measured; the function
    itself does not raise for musical or silent audio.
    """
    import librosa

    try:
        y, sr = librosa.load(str(wav_path), sr=AUDIO_SAMPLE_RATE, mono=True)
    except Exception as exc:
        raise MediaError(f"could not load audio: {exc}") from exc

    if y.size == 0:
        raise MediaError("audio track is empty")

    total_duration = len(y) / sr
    if total_duration < 1.0:
        raise MediaError("audio track is shorter than one second")

    # --- silence ratio -------------------------------------------------
    silence_ratio: Optional[float] = None
    try:
        intervals = librosa.effects.split(y, top_db=SILENCE_TOP_DB)
        voiced_samples = int(sum(end - start for start, end in intervals))
        silence_ratio = float(np.clip(1.0 - voiced_samples / len(y), 0.0, 1.0))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Silence detection failed: %s", exc)

    # --- pitch variance ------------------------------------------------
    pitch_variance: Optional[float] = None
    try:
        f0, voiced_flag, _ = librosa.pyin(
            y,
            sr=sr,
            fmin=float(librosa.note_to_hz("C2")),   # ~65 Hz, low male speech
            fmax=float(librosa.note_to_hz("C7")),   # ~2093 Hz, high/child voice
            frame_length=2048,
        )
        voiced = f0[voiced_flag & ~np.isnan(f0)] if voiced_flag is not None else f0[~np.isnan(f0)]
        if voiced.size >= 10:
            value = float(np.std(voiced))
            pitch_variance = value if math.isfinite(value) else None
        else:
            logger.warning("Too few voiced frames for a pitch estimate (%d).", voiced.size)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Pitch tracking failed: %s", exc)

    return silence_ratio, pitch_variance


# --------------------------------------------------------------------------- #
# 3. Visual analysis                                                           #
# --------------------------------------------------------------------------- #
def analyze_scene_cuts(
    media_path: Path,
    *,
    max_seconds: int,
    skip_seconds: int,
    threshold: float = SCENE_CUT_THRESHOLD,
    step: int = FRAME_SAMPLE_STEP,
) -> Optional[float]:
    """Count shot changes per minute by comparing sampled frames.

    Every ``step``-th frame is downscaled to 160x90 greyscale and compared to
    the previous sample; a mean absolute difference above ``threshold`` counts
    as a cut. Downscaling makes the measure robust to compression noise and
    camera shake, which would otherwise register as cuts.

    Returns ``None`` when the file has no usable video stream.
    """
    capture = cv2.VideoCapture(str(media_path))
    if not capture.isOpened():
        raise MediaError("OpenCV could not open the video stream")

    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not fps or math.isnan(fps) or fps <= 0:
            fps = 25.0  # sane default for a broken header
            logger.debug("Unreadable FPS; assuming %.0f.", fps)

        if skip_seconds > 0:
            capture.set(cv2.CAP_PROP_POS_MSEC, skip_seconds * 1000)

        max_frames = int(max_seconds * fps)
        previous: Optional[np.ndarray] = None
        cuts = 0
        frames_read = 0
        frames_sampled = 0

        while frames_read < max_frames:
            ok, frame = capture.read()
            if not ok:
                break
            frames_read += 1
            if frames_read % step:
                continue

            small = cv2.cvtColor(cv2.resize(frame, (160, 90)), cv2.COLOR_BGR2GRAY)
            if previous is not None:
                difference = float(np.mean(cv2.absdiff(small, previous)))
                if difference > threshold:
                    cuts += 1
            previous = small
            frames_sampled += 1

        if frames_sampled < 2:
            raise MediaError("video stream yielded too few frames to compare")

        analysed_minutes = frames_read / fps / 60.0
        if analysed_minutes <= 0:
            return None

        cuts_per_minute = cuts / analysed_minutes
        logger.debug(
            "%d cut(s) over %.2f min (%d frames sampled) -> %.2f cuts/min",
            cuts, analysed_minutes, frames_sampled, cuts_per_minute,
        )
        return float(cuts_per_minute)

    finally:
        capture.release()


# --------------------------------------------------------------------------- #
# 4. Per-video pipeline                                                        #
# --------------------------------------------------------------------------- #
def analyze_video(
    video_id: str,
    workdir: Path,
    *,
    max_seconds: int,
    skip_seconds: int,
    duration_seconds: Optional[int] = None,
) -> MediaMetrics:
    """Download, measure and clean up one video.

    When ``duration_seconds`` says the upload is longer than the analysis
    window, only that window is downloaded. The resulting clip then starts at
    zero, so the audio and frame passes read it from the beginning.

    Raises:
        MediaError: the video could not be downloaded or decoded at all.
    """
    metrics = MediaMetrics()

    window: Optional[tuple[int, int]] = None
    offset = skip_seconds
    if duration_seconds and duration_seconds > skip_seconds + max_seconds:
        window = (skip_seconds, skip_seconds + max_seconds)
        offset = 0  # the downloaded slice already starts at skip_seconds
        logger.debug("Fetching only %ds-%ds of %s (%ds long).", *window, video_id, duration_seconds)

    with download_media(video_id, workdir, window=window) as media_path:
        wav_path = workdir / f"{video_id}.analysis.wav"
        try:
            extract_audio(media_path, wav_path, max_seconds=max_seconds, skip_seconds=offset)
            metrics.silence_ratio, metrics.pitch_variance = analyze_audio(wav_path)
        except MediaError as exc:
            # A silent or audio-less video is still worth its visual metrics.
            logger.warning("Audio analysis skipped for %s: %s", video_id, exc)
        finally:
            wav_path.unlink(missing_ok=True)

        metrics.scene_cuts_per_minute = analyze_scene_cuts(
            media_path, max_seconds=max_seconds, skip_seconds=offset
        )

    return metrics


# --------------------------------------------------------------------------- #
# 5. Orchestration                                                             #
# --------------------------------------------------------------------------- #
def find_pending_videos(
    session: Session,
    *,
    limit: Optional[int] = None,
    retry_failed: bool = False,
    channel_id: Optional[str] = None,
    skip_shorts: bool = False,
) -> list[Video]:
    """Return videos whose pacing metrics are still missing."""
    needs_metrics = or_(
        Video.scene_cuts_per_minute.is_(None),
        Video.silence_ratio.is_(None),
        Video.pitch_variance.is_(None),
    )
    stmt = select(Video).where(needs_metrics)
    if not retry_failed:
        stmt = stmt.where(Video.media_checked_at.is_(None))
    if channel_id:
        stmt = stmt.where(Video.channel_id == channel_id)
    if skip_shorts:
        stmt = stmt.where(or_(Video.is_shorts.is_(False), Video.is_shorts.is_(None)))
    stmt = stmt.order_by(Video.published_at.desc())
    if limit:
        stmt = stmt.limit(limit)
    return list(session.execute(stmt).scalars())


def run(
    *,
    limit: Optional[int] = None,
    max_seconds: int = 300,
    skip_seconds: int = 0,
    retry_failed: bool = False,
    channel_id: Optional[str] = None,
    skip_shorts: bool = False,
    workdir: Optional[Path] = None,
) -> RunStats:
    """Analyse every pending video, committing after each one.

    Commits per video rather than in batches: each row costs tens of seconds
    of CPU, so losing a batch to a crash is far more expensive here than the
    extra round-trips.
    """
    stats = RunStats()
    temp_root = workdir or Path(settings.media_cache_dir)
    temp_root.mkdir(parents=True, exist_ok=True)

    try:
        with managed_session() as session:
            videos = find_pending_videos(
                session,
                limit=limit,
                retry_failed=retry_failed,
                channel_id=channel_id,
                skip_shorts=skip_shorts,
            )
            if not videos:
                logger.info("Nothing pending -- every video already has pacing metrics.")
                return stats

            logger.info("%d video(s) queued for media analysis.", len(videos))
            logger.info("Working directory: %s", temp_root)

            progress = tqdm(videos, desc="Analysing", unit="video", ncols=90)
            for video in progress:
                progress.set_postfix_str(video.video_id)
                stats.processed += 1

                # Each video gets its own scratch dir so parallel runs and
                # crashed leftovers cannot collide.
                with tempfile.TemporaryDirectory(dir=temp_root, prefix=f"{video.video_id}_") as scratch:
                    try:
                        metrics = analyze_video(
                            video.video_id,
                            Path(scratch),
                            max_seconds=max_seconds,
                            skip_seconds=skip_seconds,
                            duration_seconds=video.duration_seconds,
                        )
                    except MediaError as exc:
                        message = str(exc)[:300]
                        logger.warning("Skipping %s: %s", video.video_id, message)
                        video.media_checked_at = _utcnow()
                        video.media_error = message
                        if "download" in message:
                            stats.download_failures += 1
                        else:
                            stats.analysis_failures += 1
                        _commit(session, stats)
                        continue
                    except Exception as exc:
                        logger.exception("Unexpected failure on %s", video.video_id)
                        video.media_checked_at = _utcnow()
                        video.media_error = f"{type(exc).__name__}: {exc}"[:300]
                        stats.analysis_failures += 1
                        _commit(session, stats)
                        continue

                if metrics.silence_ratio is not None:
                    video.silence_ratio = metrics.silence_ratio
                if metrics.pitch_variance is not None:
                    video.pitch_variance = metrics.pitch_variance
                if metrics.scene_cuts_per_minute is not None:
                    video.scene_cuts_per_minute = metrics.scene_cuts_per_minute

                video.media_checked_at = _utcnow()
                if metrics.is_empty():
                    video.media_error = "no metric could be computed"
                    stats.analysis_failures += 1
                else:
                    video.media_analyzed_at = _utcnow()
                    video.media_error = None
                    stats.analyzed += 1
                    logger.info(
                        "%s: silence=%s pitch_sd=%s cuts/min=%s",
                        video.video_id,
                        _fmt(metrics.silence_ratio),
                        _fmt(metrics.pitch_variance),
                        _fmt(metrics.scene_cuts_per_minute),
                    )

                _commit(session, stats)

            progress.close()

    except KeyboardInterrupt:
        logger.warning("Interrupted; analysed videos are already committed.")

    return stats


def _fmt(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def _commit(session: Session, stats: RunStats) -> None:
    """Commit one row's worth of changes."""
    try:
        session.commit()
    except SQLAlchemyError as exc:
        logger.error("Commit failed: %s", exc)
        session.rollback()
        stats.errors += 1


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute audio and visual pacing metrics for stored videos.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--limit", type=int, default=None, help="Process at most N videos.")
    parser.add_argument(
        "--max-seconds", type=int, default=300, help="Length of the analysed window per video."
    )
    parser.add_argument(
        "--skip-intro", type=int, default=0, dest="skip_seconds",
        help="Start the window this many seconds in, to skip channel intros.",
    )
    parser.add_argument("--channel", default=None, help="Limit to a single channel id.")
    parser.add_argument("--skip-shorts", action="store_true", help="Ignore Shorts.")
    parser.add_argument(
        "--retry-failed", action="store_true", help="Also retry videos whose previous attempt failed."
    )
    parser.add_argument(
        "--workdir", default=None, help="Scratch directory for downloads (default: MEDIA_CACHE_DIR)."
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
    logging.getLogger("yt_dlp").setLevel(logging.ERROR)

    if shutil.which("ffmpeg") is None:
        logger.warning("ffmpeg is not on PATH -- audio metrics will be skipped for every video.")

    if not healthcheck():
        logger.error("Database unreachable. Check DB_CONN_STR in .env.")
        return 1

    logger.info("Target database: %s", settings.masked_summary())

    stats = run(
        limit=args.limit,
        max_seconds=args.max_seconds,
        skip_seconds=args.skip_seconds,
        retry_failed=args.retry_failed,
        channel_id=args.channel,
        skip_shorts=args.skip_shorts,
        workdir=Path(args.workdir) if args.workdir else None,
    )
    stats.log_summary()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
