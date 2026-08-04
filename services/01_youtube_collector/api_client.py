"""API clients for the collector service.

Wraps three very different data sources behind one small, predictable surface:

===========================  ==========================================
Source                       Reliability
===========================  ==========================================
YouTube Data API v3          Official, quota-limited (10k units/day)
youtube-transcript-api       Unofficial, scrapes the player; often 404s
Return YouTube Dislike API   Third-party estimate; rate-limited
===========================  ==========================================

Contract for every public method here: **never raise for missing data.**
A video without captions, without dislike data, or that has been deleted
returns ``None`` (or a dataclass with ``None`` fields) and logs a warning. The
only exceptions that escape are the ones the caller genuinely must act on:

* :class:`QuotaExceededError` -- the daily API quota is gone, stop the run.
* :class:`InvalidApiKeyError`  -- misconfiguration, stop the run.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Iterator, Optional, Sequence

import isodate
import requests
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from youtube_transcript_api import (
    CouldNotRetrieveTranscript,
    IpBlocked,
    PoTokenRequired,
    RequestBlocked,
    YouTubeRequestFailed,
    YouTubeTranscriptApi,
    YouTubeTranscriptApiException,
)

from core.config import settings

logger = logging.getLogger(__name__)

#: The Data API accepts at most 50 ids per videos.list call.
MAX_IDS_PER_REQUEST = 50

#: Preference order when several caption tracks exist.
DEFAULT_TRANSCRIPT_LANGUAGES: tuple[str, ...] = ("tr", "en", "en-US", "en-GB")

#: Transcript failures caused by YouTube refusing *this IP*, not by the video.
#: These must never be recorded as "we checked and there is nothing there".
BLOCKING_ERRORS = (IpBlocked, RequestBlocked, PoTokenRequired, YouTubeRequestFailed)

#: Consecutive blocks after which a sweep should stop asking. Hammering a
#: blocked endpoint only extends the block.
BLOCK_ABORT_THRESHOLD = 5


class CollectorError(RuntimeError):
    """Base class for errors that should abort the collection run."""


class QuotaExceededError(CollectorError):
    """The YouTube Data API daily quota has been exhausted."""


class InvalidApiKeyError(CollectorError):
    """The configured API key was rejected by Google."""


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #
def parse_duration_to_seconds(iso_duration: Optional[str]) -> Optional[int]:
    """Convert an ISO-8601 duration such as ``PT15M33S`` into whole seconds.

    Returns ``None`` for missing or unparsable input, and for the ``P0D`` that
    the API reports for live streams that never finished.

    >>> parse_duration_to_seconds("PT15M33S")
    933
    >>> parse_duration_to_seconds("PT1H2M3S")
    3723
    >>> parse_duration_to_seconds("P0D") is None
    True
    """
    if not iso_duration:
        return None
    try:
        seconds = int(isodate.parse_duration(iso_duration).total_seconds())
    except (isodate.ISO8601Error, ValueError, TypeError):
        logger.warning("Could not parse duration %r", iso_duration)
        return None
    return seconds or None


def parse_youtube_datetime(value: Optional[str]) -> Optional[datetime]:
    """Parse an RFC-3339 timestamp into a naive UTC ``datetime``.

    The ORM columns are ``DATETIME2`` without offset, so everything is
    normalised to UTC and stripped of tzinfo before it reaches the database.
    """
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        logger.warning("Could not parse timestamp %r", value)
        return None
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
    return parsed


def _to_int(value: Any) -> Optional[int]:
    """Coerce an API string counter to int; ``None`` when hidden or absent.

    Creators can hide like counts, in which case the key is simply missing --
    that is genuinely unknown, not zero, so it must stay ``None``.
    """
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def count_words(text: Optional[str]) -> Optional[int]:
    """Count word-ish tokens in a transcript. ``None`` when there is no text."""
    if not text:
        return None
    return len(re.findall(r"\b[\w']+\b", text, flags=re.UNICODE))


def chunked(items: Sequence[str], size: int) -> Iterator[Sequence[str]]:
    """Yield consecutive slices of ``items`` of at most ``size`` elements."""
    for start in range(0, len(items), size):
        yield items[start : start + size]


class _RateLimiter:
    """Thread-safe minimum-interval throttle for the unofficial endpoints.

    Return YouTube Dislike starts returning 429 fairly quickly; spacing calls
    out is cheaper than retrying after a ban.
    """

    def __init__(self, min_interval_seconds: float) -> None:
        self._min_interval = min_interval_seconds
        self._lock = threading.Lock()
        self._last_call = 0.0

    def wait(self) -> None:
        with self._lock:
            elapsed = time.monotonic() - self._last_call
            if elapsed < self._min_interval:
                time.sleep(self._min_interval - elapsed)
            self._last_call = time.monotonic()


# --------------------------------------------------------------------------- #
# Data transfer objects                                                        #
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class ChannelData:
    """Normalised channel payload, ready to map onto :class:`core.models.Channel`."""

    channel_id: str
    title: Optional[str] = None
    country: Optional[str] = None
    subscriber_count: Optional[int] = None
    total_views: Optional[int] = None
    video_count: Optional[int] = None
    channel_creation_date: Optional[datetime] = None
    uploads_playlist_id: Optional[str] = None


@dataclass(slots=True)
class VideoData:
    """Normalised video payload, ready to map onto :class:`core.models.Video`."""

    video_id: str
    channel_id: str
    published_at: Optional[datetime] = None
    title: Optional[str] = None
    description: Optional[str] = None
    tags: Optional[list[str]] = None
    category_id: Optional[int] = None
    duration_seconds: Optional[int] = None
    view_count: Optional[int] = None
    like_count: Optional[int] = None
    comment_count: Optional[int] = None
    thumbnail_url_maxres: Optional[str] = None
    is_shorts: Optional[bool] = None


@dataclass(slots=True)
class TranscriptData:
    """Transcript text plus the derived word metrics.

    ``blocked`` separates the two very different reasons for an empty result:

    * ``blocked=False`` -- this video has no captions and never will. Record
      the attempt and move on.
    * ``blocked=True``  -- YouTube is refusing *our IP*, so the answer says
      nothing about the video. Callers must NOT record an attempt, or a
      temporary block would permanently write off every video in the queue.
    """

    text: Optional[str] = None
    word_count: Optional[int] = None
    language: Optional[str] = None
    is_generated: Optional[bool] = None
    blocked: bool = False

    def words_per_minute(self, duration_seconds: Optional[int]) -> Optional[float]:
        """Speaking rate, or ``None`` when either input is missing/zero."""
        if not self.word_count or not duration_seconds:
            return None
        return self.word_count / (duration_seconds / 60.0)


@dataclass(slots=True)
class DislikeData:
    """Return YouTube Dislike response, trimmed to what we persist."""

    dislikes: Optional[int] = None
    likes: Optional[int] = None
    view_count: Optional[int] = None

    def like_dislike_ratio(self, like_count: Optional[int]) -> Optional[float]:
        """likes / dislikes, preferring the official like count when available."""
        likes = like_count if like_count is not None else self.likes
        if likes is None or not self.dislikes:
            return None
        return likes / self.dislikes


# --------------------------------------------------------------------------- #
# YouTube Data API v3                                                          #
# --------------------------------------------------------------------------- #
class YouTubeDataClient:
    """Thin wrapper over ``googleapiclient`` with quota-aware error handling."""

    def __init__(self, api_key: Optional[str] = None) -> None:
        self._api_key = api_key or settings.require_youtube_api_key()
        # cache_discovery=False silences the oauth2client file-cache warning.
        self._service = build("youtube", "v3", developerKey=self._api_key, cache_discovery=False)
        logger.debug("YouTube Data API client ready.")

    # -- error translation -------------------------------------------------
    @staticmethod
    def _classify(error: HttpError) -> Exception:
        """Turn an ``HttpError`` into either a fatal CollectorError or itself."""
        status = getattr(error.resp, "status", None)
        detail = str(error)
        if status == 403:
            if "quotaExceeded" in detail or "dailyLimitExceeded" in detail:
                return QuotaExceededError(
                    "YouTube Data API daily quota exhausted. Collection cannot continue today."
                )
            if "keyInvalid" in detail or "API key not valid" in detail:
                return InvalidApiKeyError("The configured YOUTUBE_API_KEY was rejected by Google.")
        return error

    @retry(
        retry=retry_if_exception_type(HttpError),
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    def _execute(self, request: Any) -> dict[str, Any]:
        """Execute a Google API request, retrying transient 5xx/429 failures.

        Fatal errors (quota, bad key) and 4xx client errors are re-raised at
        once rather than burning the retry budget.
        """
        try:
            return request.execute()
        except HttpError as exc:
            fatal = self._classify(exc)
            if isinstance(fatal, CollectorError):
                raise fatal from exc
            status = getattr(exc.resp, "status", None)
            if status is not None and 400 <= status < 500 and status != 429:
                raise  # not retryable; let the caller decide
            logger.warning("Transient YouTube API error (HTTP %s), retrying: %s", status, exc)
            raise

    # -- channels ----------------------------------------------------------
    def get_channel(self, channel_id: str) -> Optional[ChannelData]:
        """Fetch channel statistics. Returns ``None`` if the channel is gone."""
        try:
            response = self._execute(
                self._service.channels().list(
                    part="snippet,statistics,contentDetails", id=channel_id, maxResults=1
                )
            )
        except CollectorError:
            raise
        except Exception as exc:  # noqa: BLE001 - degrade gracefully
            logger.error("Failed to fetch channel %s: %s", channel_id, exc)
            return None

        items = response.get("items") or []
        if not items:
            logger.error("Channel %s not found (private, deleted, or bad id).", channel_id)
            return None

        item = items[0]
        snippet = item.get("snippet", {})
        stats = item.get("statistics", {})
        uploads = (
            item.get("contentDetails", {}).get("relatedPlaylists", {}).get("uploads")
        )

        # hiddenSubscriberCount means the number is genuinely unknown, not 0.
        subscribers = None if stats.get("hiddenSubscriberCount") else _to_int(stats.get("subscriberCount"))

        return ChannelData(
            channel_id=item.get("id", channel_id),
            title=snippet.get("title"),
            country=snippet.get("country"),
            subscriber_count=subscribers,
            total_views=_to_int(stats.get("viewCount")),
            video_count=_to_int(stats.get("videoCount")),
            channel_creation_date=parse_youtube_datetime(snippet.get("publishedAt")),
            uploads_playlist_id=uploads,
        )

    def resolve_channel_id(self, handle_or_id: str) -> Optional[str]:
        """Resolve a channel id, ``@handle``, bare handle or URL to ``UC...``.

        Plain ``UC...`` ids are returned untouched without spending quota. A
        bare word (``ThePrimeTimeagen``) is treated as a handle: PowerShell
        eats a leading ``@`` as its splatting operator unless the argument is
        quoted, so requiring one would break the obvious command line.
        """
        value = handle_or_id.strip()
        if value.startswith("UC") and len(value) == 24:
            return value

        match = re.search(r"(?:youtube\.com/)?(?:channel/)?(UC[\w-]{22})", value)
        if match:
            return match.group(1)

        # @handle, youtube.com/@handle, or a bare handle.
        handle_match = re.search(r"@([\w.\-]+)", value)
        if handle_match:
            handle = handle_match.group(1)
        elif re.fullmatch(r"[\w.\-]+", value):
            handle = value
            logger.debug("Treating %r as a bare handle.", value)
        else:
            logger.error("Could not interpret %r as a channel id, handle or URL.", handle_or_id)
            return None

        try:
            response = self._execute(
                self._service.channels().list(part="id", forHandle=f"@{handle}", maxResults=1)
            )
        except CollectorError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.error("Handle lookup failed for %s: %s", handle_or_id, exc)
            return None

        items = response.get("items") or []
        if not items:
            logger.error("No channel found for handle @%s.", handle)
            return None
        return items[0]["id"]

    # -- video ids ---------------------------------------------------------
    def get_video_ids(
        self,
        uploads_playlist_id: str,
        limit: int = 50,
    ) -> list[str]:
        """Walk the uploads playlist and return up to ``limit`` video ids.

        The playlist is ordered newest-first, so this yields the most recent
        uploads. Paginates automatically when ``limit`` exceeds 50.
        """
        video_ids: list[str] = []
        page_token: Optional[str] = None

        while len(video_ids) < limit:
            remaining = limit - len(video_ids)
            try:
                response = self._execute(
                    self._service.playlistItems().list(
                        part="contentDetails",
                        playlistId=uploads_playlist_id,
                        maxResults=min(MAX_IDS_PER_REQUEST, remaining),
                        pageToken=page_token,
                    )
                )
            except CollectorError:
                raise
            except Exception as exc:  # noqa: BLE001
                logger.error("Failed to list playlist %s: %s", uploads_playlist_id, exc)
                break

            for item in response.get("items", []):
                vid = item.get("contentDetails", {}).get("videoId")
                if vid:
                    video_ids.append(vid)

            page_token = response.get("nextPageToken")
            if not page_token:
                break

        logger.info("Collected %d video ids from playlist %s.", len(video_ids), uploads_playlist_id)
        return video_ids[:limit]

    # -- video details -----------------------------------------------------
    def get_videos(self, video_ids: Sequence[str]) -> dict[str, VideoData]:
        """Fetch details for many videos, 50 per request.

        Returns a ``{video_id: VideoData}`` map. Ids that the API does not
        return (deleted/private) are simply absent -- the caller decides how
        to treat them.
        """
        results: dict[str, VideoData] = {}
        if not video_ids:
            return results

        for batch in chunked(list(video_ids), MAX_IDS_PER_REQUEST):
            try:
                response = self._execute(
                    self._service.videos().list(
                        part="snippet,statistics,contentDetails", id=",".join(batch), maxResults=len(batch)
                    )
                )
            except CollectorError:
                raise
            except Exception as exc:  # noqa: BLE001
                logger.error("Failed to fetch video batch (%d ids): %s", len(batch), exc)
                continue

            for item in response.get("items", []):
                video = self._parse_video(item)
                if video:
                    results[video.video_id] = video

            missing = set(batch) - results.keys()
            if missing:
                logger.warning("YouTube returned no data for %d id(s): %s", len(missing), ", ".join(sorted(missing)))

        return results

    def get_video(self, video_id: str) -> Optional[VideoData]:
        """Fetch a single video's details, or ``None`` when unavailable."""
        return self.get_videos([video_id]).get(video_id)

    @staticmethod
    def _parse_video(item: dict[str, Any]) -> Optional[VideoData]:
        """Map one ``videos.list`` item onto :class:`VideoData`."""
        video_id = item.get("id")
        if not video_id:
            return None

        snippet = item.get("snippet", {})
        stats = item.get("statistics", {})
        content = item.get("contentDetails", {})
        duration = parse_duration_to_seconds(content.get("duration"))

        # Prefer maxres, but it does not exist for every upload -- walk down.
        thumbnails = snippet.get("thumbnails", {})
        thumbnail_url = None
        for size in ("maxres", "standard", "high", "medium", "default"):
            if thumbnails.get(size, {}).get("url"):
                thumbnail_url = thumbnails[size]["url"]
                break

        return VideoData(
            video_id=video_id,
            channel_id=snippet.get("channelId", ""),
            published_at=parse_youtube_datetime(snippet.get("publishedAt")),
            title=snippet.get("title"),
            description=snippet.get("description"),
            tags=snippet.get("tags") or None,
            category_id=_to_int(snippet.get("categoryId")),
            duration_seconds=duration,
            view_count=_to_int(stats.get("viewCount")),
            like_count=_to_int(stats.get("likeCount")),
            comment_count=_to_int(stats.get("commentCount")),
            thumbnail_url_maxres=thumbnail_url,
            # Duration proxy; refine with YouTubeShortsProbe when accuracy matters.
            is_shorts=(duration is not None and duration <= 60),
        )


# --------------------------------------------------------------------------- #
# Shorts probe (optional, one cheap HTTP call per video)                       #
# --------------------------------------------------------------------------- #
class ShortsProbe:
    """Confirms Shorts status via the ``/shorts/<id>`` URL.

    The duration proxy misclassifies short *landscape* videos. YouTube redirects
    ``/shorts/<id>`` to ``/watch?v=<id>`` when a video is not a Short, so the
    redirect target is an authoritative answer without any quota cost.
    """

    def __init__(self, timeout: float = 10.0, min_interval: float = 0.2) -> None:
        self._timeout = timeout
        self._limiter = _RateLimiter(min_interval)
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "youtube-prediction-analyzer/0.1"})

    def is_shorts(self, video_id: str, *, duration_seconds: Optional[int] = None) -> Optional[bool]:
        """Return True/False, or ``None`` if the probe could not decide."""
        if duration_seconds is not None and duration_seconds > 60:
            return False  # a Short can never exceed 60s; skip the request
        self._limiter.wait()
        try:
            response = self._session.head(
                f"https://www.youtube.com/shorts/{video_id}",
                allow_redirects=False,
                timeout=self._timeout,
            )
        except requests.RequestException as exc:
            logger.debug("Shorts probe failed for %s: %s", video_id, exc)
            return None

        if response.status_code == 200:
            return True
        if response.status_code in (301, 302, 303, 307, 308):
            return "/watch" not in response.headers.get("Location", "")
        return None

    def close(self) -> None:
        self._session.close()


# --------------------------------------------------------------------------- #
# Transcripts                                                                  #
# --------------------------------------------------------------------------- #
class TranscriptClient:
    """Fetches caption tracks, preferring manually written ones.

    Every failure mode is logged and turned into an empty
    :class:`TranscriptData`, but they are not all equal: exceptions in
    :data:`BLOCKING_ERRORS` mean YouTube is refusing this IP, and the result
    is flagged ``blocked=True`` so callers can stop rather than mistake a
    network-wide block for 50 caption-less videos.

    An optional ``HTTP_PROXY_URL`` in ``.env`` routes requests through a proxy,
    which is the practical way out of a block without waiting it out.
    """

    def __init__(
        self,
        languages: Iterable[str] = DEFAULT_TRANSCRIPT_LANGUAGES,
        *,
        min_interval: Optional[float] = None,
        proxy_url: Optional[str] = None,
    ) -> None:
        self._languages = tuple(languages)
        # Unthrottled scraping is what earns the IP block in the first place.
        # Tune with TRANSCRIPT_MIN_INTERVAL in .env: raise it after a block,
        # never lower it to "catch up".
        interval = settings.transcript_min_interval if min_interval is None else min_interval
        self._limiter = _RateLimiter(interval)

        proxy = proxy_url or settings.transcript_proxy_url
        proxy_config = None
        if proxy:
            from youtube_transcript_api.proxies import GenericProxyConfig

            proxy_config = GenericProxyConfig(http_url=proxy, https_url=proxy)
            logger.info("Transcript requests will go through a proxy.")

        self._api = YouTubeTranscriptApi(proxy_config=proxy_config)

    def fetch(self, video_id: str) -> TranscriptData:
        """Return the transcript for ``video_id``; never raises."""
        self._limiter.wait()
        try:
            fetched = self._api.fetch(video_id, languages=self._languages)
        except BLOCKING_ERRORS as exc:
            logger.warning("YouTube is blocking this IP (%s) on %s.", type(exc).__name__, video_id)
            return TranscriptData(blocked=True)
        except CouldNotRetrieveTranscript as exc:
            # TranscriptsDisabled, NoTranscriptFound, VideoUnavailable,
            # AgeRestricted: genuine properties of the video.
            logger.warning("No transcript for %s: %s", video_id, type(exc).__name__)
            return self._fallback_any_language(video_id)
        except YouTubeTranscriptApiException as exc:
            logger.warning("Transcript API error for %s: %s", video_id, exc)
            return TranscriptData()
        except Exception as exc:  # noqa: BLE001 - unofficial API, expect surprises
            logger.warning("Unexpected transcript failure for %s: %s", video_id, exc)
            return TranscriptData()

        return self._build(fetched)

    def _fallback_any_language(self, video_id: str) -> TranscriptData:
        """Retry with whatever track exists before giving up entirely.

        A Turkish channel with only Spanish auto-captions still yields usable
        word-rate signal, so preferred-language misses are worth a second look.
        """
        try:
            transcript_list = self._api.list(video_id)
            transcript = next(iter(transcript_list), None)
            if transcript is None:
                return TranscriptData()
            return self._build(transcript.fetch())
        except BLOCKING_ERRORS:
            return TranscriptData(blocked=True)
        except (CouldNotRetrieveTranscript, YouTubeTranscriptApiException):
            return TranscriptData()
        except Exception as exc:  # noqa: BLE001
            logger.debug("Transcript fallback failed for %s: %s", video_id, exc)
            return TranscriptData()

    @staticmethod
    def _build(fetched: Any) -> TranscriptData:
        """Join snippets into one text blob and compute the word count."""
        try:
            text = " ".join(snippet.text.strip() for snippet in fetched if snippet.text)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not assemble transcript text: %s", exc)
            return TranscriptData()

        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return TranscriptData()

        return TranscriptData(
            text=text,
            word_count=count_words(text),
            language=getattr(fetched, "language_code", None),
            is_generated=getattr(fetched, "is_generated", None),
        )


# --------------------------------------------------------------------------- #
# Return YouTube Dislike                                                       #
# --------------------------------------------------------------------------- #
class DislikeClient:
    """Client for the Return YouTube Dislike estimate API."""

    def __init__(
        self,
        endpoint: Optional[str] = None,
        timeout: float = 15.0,
        min_interval: float = 0.35,
        max_attempts: int = 3,
    ) -> None:
        self._endpoint = endpoint or settings.returnyoutubedislike_api_url
        self._timeout = timeout
        self._max_attempts = max_attempts
        self._limiter = _RateLimiter(min_interval)
        self._session = requests.Session()
        self._session.headers.update(
            {"Accept": "application/json", "User-Agent": "youtube-prediction-analyzer/0.1"}
        )

    def fetch(self, video_id: str) -> DislikeData:
        """Return dislike data for ``video_id``; never raises.

        Retries on 429/5xx with a linear backoff and honours ``Retry-After``.
        """
        for attempt in range(1, self._max_attempts + 1):
            self._limiter.wait()
            try:
                response = self._session.get(
                    self._endpoint, params={"videoId": video_id}, timeout=self._timeout
                )
            except requests.RequestException as exc:
                logger.warning("Dislike request failed for %s (attempt %d): %s", video_id, attempt, exc)
                time.sleep(min(2 ** attempt, 10))
                continue

            if response.status_code == 200:
                return self._parse(response, video_id)

            if response.status_code == 404:
                logger.warning("Dislike API has no record for %s.", video_id)
                return DislikeData()

            if response.status_code == 429 or response.status_code >= 500:
                delay = self._retry_after(response, attempt)
                logger.warning(
                    "Dislike API returned HTTP %s for %s; sleeping %.1fs (attempt %d/%d).",
                    response.status_code, video_id, delay, attempt, self._max_attempts,
                )
                time.sleep(delay)
                continue

            logger.warning("Dislike API returned HTTP %s for %s; giving up.", response.status_code, video_id)
            return DislikeData()

        logger.warning("Dislike lookup exhausted retries for %s.", video_id)
        return DislikeData()

    @staticmethod
    def _retry_after(response: requests.Response, attempt: int) -> float:
        """Seconds to wait, from the ``Retry-After`` header when present."""
        header = response.headers.get("Retry-After")
        if header:
            try:
                return min(float(header), 60.0)
            except ValueError:
                pass
        return float(min(2 ** attempt, 30))

    @staticmethod
    def _parse(response: requests.Response, video_id: str) -> DislikeData:
        """Extract the counters from a 200 response.

        The API answers 200 with every counter at zero for videos it has never
        seen, so an all-zero payload is treated as *unknown* rather than as a
        genuine zero. A real video with views but no dislikes still records 0.
        """
        try:
            payload = response.json()
        except ValueError:
            logger.warning("Dislike API returned non-JSON for %s.", video_id)
            return DislikeData()

        if not isinstance(payload, dict):
            return DislikeData()

        data = DislikeData(
            dislikes=_to_int(payload.get("dislikes")),
            likes=_to_int(payload.get("likes")),
            view_count=_to_int(payload.get("viewCount")),
        )

        if not any((data.dislikes, data.likes, data.view_count)):
            logger.warning("Dislike API has no record for %s (all-zero payload).", video_id)
            return DislikeData()

        return data

    def close(self) -> None:
        self._session.close()


# --------------------------------------------------------------------------- #
# Facade                                                                       #
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class CollectorClients:
    """Bundle of the three clients, so the orchestrator wires up once."""

    youtube: YouTubeDataClient = field(default_factory=YouTubeDataClient)
    transcripts: TranscriptClient = field(default_factory=TranscriptClient)
    dislikes: DislikeClient = field(default_factory=DislikeClient)
    shorts_probe: Optional[ShortsProbe] = None

    def close(self) -> None:
        """Release the pooled HTTP sessions."""
        self.dislikes.close()
        if self.shorts_probe is not None:
            self.shorts_probe.close()


__all__ = [
    "ChannelData",
    "CollectorClients",
    "CollectorError",
    "DislikeClient",
    "DislikeData",
    "InvalidApiKeyError",
    "QuotaExceededError",
    "ShortsProbe",
    "TranscriptClient",
    "TranscriptData",
    "VideoData",
    "YouTubeDataClient",
    "count_words",
    "parse_duration_to_seconds",
    "parse_youtube_datetime",
]
