"""Shared OpenAI plumbing for the AI analyzer service.

Both ``analyze_text.py`` and ``analyze_thumbnail.py`` need the same three
things -- a client, a retrying structured-output call, and token accounting --
so they live here rather than being copy-pasted twice.

Notes on Structured Outputs
---------------------------
Strict JSON-schema mode rejects several JSON-Schema keywords, ``minimum`` and
``maximum`` among them. That means a Pydantic ``Field(ge=1, le=10)`` makes the
request fail outright. Ranges are therefore enforced in two places that *are*
allowed: the prompt (which states the scale) and a client-side validator that
clamps whatever comes back. Categorical fields use ``Literal``, which does
compile to a permitted ``enum``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional, TypeVar

from openai import (
    APIConnectionError,
    APITimeoutError,
    BadRequestError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)
from pydantic import BaseModel
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from core.config import settings

logger = logging.getLogger(__name__)

TModel = TypeVar("TModel", bound=BaseModel)

#: Errors worth retrying: transient server/network/rate-limit conditions.
RETRYABLE = (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError)

#: Model families that reject the ``temperature`` parameter.
_NO_TEMPERATURE_PREFIXES = ("gpt-5", "o1", "o3", "o4")


class AnalysisError(RuntimeError):
    """A request failed in a way that is specific to one row, not the run."""


class FatalAIError(RuntimeError):
    """Configuration is wrong -- bad key, unknown model. Stop the run."""


@dataclass
class UsageTracker:
    """Running token totals, so a run can report what it cost."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0
    _seen: set[str] = field(default_factory=set)

    def add(self, response: Any) -> None:
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        self.calls += 1
        self.prompt_tokens += getattr(usage, "prompt_tokens", 0) or 0
        self.completion_tokens += getattr(usage, "completion_tokens", 0) or 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def log_summary(self) -> None:
        if not self.calls:
            return
        logger.info(
            "OpenAI usage: %d call(s), %d prompt + %d completion = %d tokens",
            self.calls, self.prompt_tokens, self.completion_tokens, self.total_tokens,
        )


def get_client(api_key: Optional[str] = None, *, timeout: float = 60.0) -> OpenAI:
    """Build an OpenAI client, failing loudly if the key is missing."""
    key = api_key or settings.openai_api_key
    if not key:
        raise FatalAIError("OPENAI_API_KEY is not set in .env; the AI analyzer cannot run.")
    return OpenAI(api_key=key, timeout=timeout, max_retries=0)  # tenacity owns retries


def supports_temperature(model: str) -> bool:
    """Whether ``model`` accepts a ``temperature`` argument."""
    return not model.startswith(_NO_TEMPERATURE_PREFIXES)


@retry(
    retry=retry_if_exception_type(RETRYABLE),
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=4, max=60),
    reraise=True,
)
def _parse_with_retry(client: OpenAI, **kwargs: Any) -> Any:
    """Call the structured-output endpoint, retrying transient failures.

    Backs off 4s, 8s, 16s, 32s -- long enough for a rate-limit window to
    reopen without stalling a large run.
    """
    try:
        return client.chat.completions.parse(**kwargs)
    except RETRYABLE as exc:
        logger.warning("Transient OpenAI error (%s); backing off.", type(exc).__name__)
        raise


@dataclass
class ParsedResult[T: BaseModel]:
    """A parsed response plus the snapshot that actually produced it."""

    value: T
    model: str
    """Resolved model id, e.g. ``gpt-4o-mini-2024-07-18`` when ``gpt-4o-mini``
    was requested. Storing the alias would make old rows unreproducible the
    moment OpenAI repoints it."""


def parse_structured(
    client: OpenAI,
    *,
    model: str,
    schema: type[TModel],
    messages: list[dict[str, Any]],
    usage: Optional[UsageTracker] = None,
    max_tokens: int = 500,
) -> ParsedResult[TModel]:
    """Run one structured-output completion and return the parsed result.

    Raises:
        AnalysisError: the model refused, hit the length limit, or the request
            was rejected for something row-specific (an unreachable image, say).
        FatalAIError: the key or model name is wrong -- no point continuing.
    """
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "response_format": schema,
        "max_completion_tokens": max_tokens,
    }
    if supports_temperature(model):
        # Deterministic scoring: the same video should not drift between runs.
        kwargs["temperature"] = 0.0

    try:
        response = _parse_with_retry(client, **kwargs)
    except BadRequestError as exc:
        detail = str(exc)
        if "model" in detail and "does not exist" in detail:
            raise FatalAIError(f"Model {model!r} is not available on this account.") from exc
        # Invalid/unreachable image URLs land here -- a per-row problem.
        raise AnalysisError(f"request rejected: {detail[:300]}") from exc
    except RETRYABLE as exc:
        raise AnalysisError(f"{type(exc).__name__} after retries: {exc}") from exc

    if usage is not None:
        usage.add(response)

    choice = response.choices[0]
    if getattr(choice.message, "refusal", None):
        raise AnalysisError(f"model refused: {choice.message.refusal}")
    if choice.finish_reason == "length":
        raise AnalysisError("response hit the token limit before completing")

    parsed = choice.message.parsed
    if parsed is None:
        raise AnalysisError("model returned no parsable content")
    return ParsedResult(value=parsed, model=getattr(response, "model", model) or model)


def clamp_score(value: Optional[float], low: float = 1.0, high: float = 10.0) -> Optional[float]:
    """Clip a model-supplied score into the documented range.

    Strict schema mode cannot express ``minimum``/``maximum``, so the range is
    enforced here instead. Models very rarely stray, but a stray 0 or 11 would
    otherwise trip the CHECK constraints on the videos table.
    """
    if value is None:
        return None
    return float(min(max(value, low), high))


__all__ = [
    "AnalysisError",
    "FatalAIError",
    "ParsedResult",
    "UsageTracker",
    "clamp_score",
    "get_client",
    "parse_structured",
    "supports_temperature",
]
