"""SQLAlchemy ORM models for the YouTube performance prediction warehouse.

Schema overview
---------------
``channels`` 1 --- N ``videos``

The ``videos`` table is deliberately wide (a "one big table" analytics layout):
every stage of the pipeline enriches the same row.

    stage 1 (01_youtube_collector)  -> API metadata, transcript, dislikes,
                                       audio/visual tempo metrics
    stage 2 (02_ai_analyzer)        -> hook/curiosity/thumbnail scores
    stage 3 (03_predictor_engine)   -> target variables

All AI and target columns are nullable on purpose: a freshly collected video is
a valid row long before it has been scored. Use the ``*_enriched_at`` timestamps
to find work that is still pending.

Dialect notes (MSSQL)
---------------------
* Text columns use ``NVARCHAR`` so non-Latin titles/transcripts survive intact.
* Free-form long text uses ``NVARCHAR(MAX)``.
* Datetimes use ``DATETIME2`` (wider range and better precision than DATETIME).
* Booleans map to ``BIT``.
"""

from __future__ import annotations

import datetime as dt
from typing import Any, Optional

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    Float,
    ForeignKey,
    Index,
    Integer,
    Unicode,
    func,
)
from sqlalchemy.dialects.mssql import DATETIME2, NVARCHAR
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
    validates,
)
from sqlalchemy.types import JSON

# ``NVARCHAR(MAX)`` on SQL Server; falls back to plain NVARCHAR elsewhere.
NVARCHAR_MAX = NVARCHAR(None)

#: Datetime with timezone stripped -- YouTube timestamps are normalised to UTC
#: before insert, so we store naive UTC values consistently.
UTCDateTime = DATETIME2


class Base(DeclarativeBase):
    """Declarative base shared by every model in the monorepo."""

    type_annotation_map = {
        dict[str, Any]: JSON,
        list[str]: JSON,
    }

    def to_dict(self) -> dict[str, Any]:
        """Return a plain-dict view of the mapped columns (handy for pandas)."""
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        pk = self.__mapper__.primary_key[0].name
        return f"<{type(self).__name__} {pk}={getattr(self, pk)!r}>"


class TimestampMixin:
    """Row-level bookkeeping columns applied to every table."""

    created_at: Mapped[dt.datetime] = mapped_column(
        UTCDateTime,
        server_default=func.sysutcdatetime(),
        nullable=False,
        doc="UTC timestamp of first insert.",
    )
    updated_at: Mapped[dt.datetime] = mapped_column(
        UTCDateTime,
        server_default=func.sysutcdatetime(),
        onupdate=func.sysutcdatetime(),
        nullable=False,
        doc="UTC timestamp of the most recent ORM update.",
    )


class Channel(TimestampMixin, Base):
    """A YouTube channel; the denominator for every relative performance metric."""

    __tablename__ = "channels"

    channel_id: Mapped[str] = mapped_column(
        Unicode(32), primary_key=True, doc="YouTube channel id, e.g. 'UC_x5XG1OV2P6uZZ5FSM9Ttw'."
    )
    title: Mapped[Optional[str]] = mapped_column(Unicode(255), doc="Channel display name.")
    country: Mapped[Optional[str]] = mapped_column(Unicode(8), doc="ISO 3166-1 alpha-2 country code.")

    subscriber_count: Mapped[Optional[int]] = mapped_column(
        Integer, doc="Subscribers at collection time (rounded by the YouTube API)."
    )
    total_views: Mapped[Optional[int]] = mapped_column(
        BigInteger, doc="Lifetime channel view count."
    )
    video_count: Mapped[Optional[int]] = mapped_column(Integer, doc="Public videos on the channel.")
    channel_creation_date: Mapped[Optional[dt.datetime]] = mapped_column(
        UTCDateTime, doc="Channel 'publishedAt' (UTC)."
    )

    last_scraped_at: Mapped[Optional[dt.datetime]] = mapped_column(
        UTCDateTime, doc="Last successful refresh of this channel's statistics."
    )

    videos: Mapped[list["Video"]] = relationship(
        back_populates="channel",
        cascade="all, delete-orphan",
        passive_deletes=True,
        lazy="selectin",
    )

    __table_args__ = (
        CheckConstraint("subscriber_count IS NULL OR subscriber_count >= 0", name="ck_channels_subs_non_negative"),
        CheckConstraint("total_views IS NULL OR total_views >= 0", name="ck_channels_views_non_negative"),
        Index("ix_channels_subscriber_count", "subscriber_count"),
    )


class Video(TimestampMixin, Base):
    """A single video and every feature the pipeline derives from it."""

    __tablename__ = "videos"

    # ------------------------------------------------------------------ #
    # Identity / YouTube Data API                                         #
    # ------------------------------------------------------------------ #
    video_id: Mapped[str] = mapped_column(
        Unicode(24), primary_key=True, doc="YouTube video id, e.g. 'dQw4w9WgXcQ'."
    )
    channel_id: Mapped[str] = mapped_column(
        Unicode(32),
        ForeignKey("channels.channel_id", ondelete="CASCADE", onupdate="CASCADE"),
        nullable=False,
        index=True,
    )

    published_at: Mapped[Optional[dt.datetime]] = mapped_column(
        UTCDateTime, doc="Publication timestamp (UTC)."
    )
    title: Mapped[Optional[str]] = mapped_column(Unicode(512))
    description: Mapped[Optional[str]] = mapped_column(NVARCHAR_MAX)
    tags: Mapped[Optional[list[str]]] = mapped_column(
        JSON, doc="Creator-supplied tags, stored as a JSON array."
    )
    category_id: Mapped[Optional[int]] = mapped_column(Integer, doc="YouTube category id.")
    duration_seconds: Mapped[Optional[int]] = mapped_column(
        Integer, doc="Video length in seconds (parsed from the ISO-8601 duration)."
    )

    view_count: Mapped[Optional[int]] = mapped_column(BigInteger)
    like_count: Mapped[Optional[int]] = mapped_column(Integer)
    comment_count: Mapped[Optional[int]] = mapped_column(Integer)

    thumbnail_url_maxres: Mapped[Optional[str]] = mapped_column(Unicode(1024))
    is_shorts: Mapped[Optional[bool]] = mapped_column(
        Boolean, doc="True when duration < 60s and the frame is vertical."
    )

    # ------------------------------------------------------------------ #
    # Unofficial sources: transcripts + dislikes                          #
    # ------------------------------------------------------------------ #
    full_transcript: Mapped[Optional[str]] = mapped_column(
        NVARCHAR_MAX, doc="Concatenated transcript text (youtube-transcript-api)."
    )
    word_count: Mapped[Optional[int]] = mapped_column(Integer)
    words_per_minute: Mapped[Optional[float]] = mapped_column(
        Float, doc="word_count / (duration_seconds / 60)."
    )
    dislike_count: Mapped[Optional[int]] = mapped_column(
        Integer, doc="Estimate from the Return YouTube Dislike API."
    )
    like_dislike_ratio: Mapped[Optional[float]] = mapped_column(
        Float, doc="like_count / dislike_count."
    )

    # ------------------------------------------------------------------ #
    # Audio & visual tempo (Librosa / OpenCV / FFmpeg)                    #
    # ------------------------------------------------------------------ #
    silence_ratio: Mapped[Optional[float]] = mapped_column(
        Float, doc="Fraction of the audio track that is dead air (0-1)."
    )
    pitch_variance: Mapped[Optional[float]] = mapped_column(
        Float, doc="Variance of the fundamental frequency; monotone voices score low."
    )
    scene_cuts_per_minute: Mapped[Optional[float]] = mapped_column(
        Float, doc="Detected shot changes per minute; a proxy for edit pacing."
    )
    media_analyzed_at: Mapped[Optional[dt.datetime]] = mapped_column(
        UTCDateTime, doc="When audio/visual metrics were last computed."
    )

    # ------------------------------------------------------------------ #
    # AI-generated features (02_ai_analyzer)                              #
    # ------------------------------------------------------------------ #
    hook_score: Mapped[Optional[float]] = mapped_column(
        Float, doc="1-10 quality of the first 30 seconds of transcript."
    )
    curiosity_gap_score: Mapped[Optional[float]] = mapped_column(
        Float, doc="1-10 tension between the title's promise and the content."
    )
    emotion_tone: Mapped[Optional[str]] = mapped_column(
        Unicode(64), doc="Dominant tone, e.g. Educational / Drama / Motivational."
    )
    niche_relevance: Mapped[Optional[float]] = mapped_column(
        Float, doc="1-10 fit between the video and the channel's usual niche."
    )
    thumbnail_has_face: Mapped[Optional[bool]] = mapped_column(Boolean)
    thumbnail_face_emotion: Mapped[Optional[str]] = mapped_column(Unicode(64))
    thumbnail_text: Mapped[Optional[str]] = mapped_column(
        Unicode(512), doc="Text read off the thumbnail (OCR / Vision API)."
    )
    title_thumbnail_synergy: Mapped[Optional[float]] = mapped_column(
        Float, doc="1-10 how well the title and thumbnail reinforce each other."
    )
    ai_analyzed_at: Mapped[Optional[dt.datetime]] = mapped_column(
        UTCDateTime, doc="When the AI enrichment last succeeded."
    )
    ai_model_version: Mapped[Optional[str]] = mapped_column(
        Unicode(64), doc="Model id used for the scores above, for reproducibility."
    )

    # ------------------------------------------------------------------ #
    # Target variables (03_predictor_engine)                              #
    # ------------------------------------------------------------------ #
    recent_channel_avg_views: Mapped[Optional[float]] = mapped_column(
        Float, doc="Rolling mean views of the 10 videos preceding this one."
    )
    performance_ratio: Mapped[Optional[float]] = mapped_column(
        Float, doc="view_count / recent_channel_avg_views. The primary target."
    )
    engagement_rate: Mapped[Optional[float]] = mapped_column(
        Float, doc="(like_count + comment_count) / view_count."
    )
    targets_computed_at: Mapped[Optional[dt.datetime]] = mapped_column(UTCDateTime)

    channel: Mapped["Channel"] = relationship(back_populates="videos", lazy="joined")

    __table_args__ = (
        CheckConstraint("duration_seconds IS NULL OR duration_seconds >= 0", name="ck_videos_duration_non_negative"),
        CheckConstraint("view_count IS NULL OR view_count >= 0", name="ck_videos_views_non_negative"),
        CheckConstraint(
            "hook_score IS NULL OR (hook_score >= 0 AND hook_score <= 10)",
            name="ck_videos_hook_score_range",
        ),
        CheckConstraint(
            "curiosity_gap_score IS NULL OR (curiosity_gap_score >= 0 AND curiosity_gap_score <= 10)",
            name="ck_videos_curiosity_score_range",
        ),
        CheckConstraint(
            "niche_relevance IS NULL OR (niche_relevance >= 0 AND niche_relevance <= 10)",
            name="ck_videos_niche_relevance_range",
        ),
        CheckConstraint(
            "title_thumbnail_synergy IS NULL OR (title_thumbnail_synergy >= 0 AND title_thumbnail_synergy <= 10)",
            name="ck_videos_synergy_range",
        ),
        # Channel + chronology: the access path for rolling-window target maths.
        Index("ix_videos_channel_published", "channel_id", "published_at"),
        Index("ix_videos_published_at", "published_at"),
        Index("ix_videos_performance_ratio", "performance_ratio"),
        # Worklist lookups for the enrichment services.
        Index("ix_videos_ai_analyzed_at", "ai_analyzed_at"),
        Index("ix_videos_media_analyzed_at", "media_analyzed_at"),
    )

    # ------------------------------------------------------------------ #
    # Light-weight derived helpers                                        #
    # ------------------------------------------------------------------ #
    @validates("emotion_tone", "thumbnail_face_emotion")
    def _normalize_label(self, _key: str, value: Optional[str]) -> Optional[str]:
        """Trim whitespace on free-text label columns so grouping stays clean."""
        return value.strip() if isinstance(value, str) else value

    def compute_words_per_minute(self) -> Optional[float]:
        """Derive and store ``words_per_minute``; returns ``None`` if not derivable."""
        if not self.word_count or not self.duration_seconds:
            return None
        self.words_per_minute = self.word_count / (self.duration_seconds / 60.0)
        return self.words_per_minute

    def compute_like_dislike_ratio(self) -> Optional[float]:
        """Derive and store ``like_dislike_ratio``; ``None`` when dislikes are 0/unknown."""
        if self.like_count is None or not self.dislike_count:
            return None
        self.like_dislike_ratio = self.like_count / self.dislike_count
        return self.like_dislike_ratio

    def compute_engagement_rate(self) -> Optional[float]:
        """Derive and store ``engagement_rate``; ``None`` when views are 0/unknown."""
        if not self.view_count:
            return None
        likes = self.like_count or 0
        comments = self.comment_count or 0
        self.engagement_rate = (likes + comments) / self.view_count
        return self.engagement_rate

    def compute_performance_ratio(self) -> Optional[float]:
        """Derive and store ``performance_ratio`` from the rolling channel average."""
        if not self.view_count or not self.recent_channel_avg_views:
            return None
        self.performance_ratio = self.view_count / self.recent_channel_avg_views
        return self.performance_ratio


__all__ = ["Base", "Channel", "TimestampMixin", "Video"]
