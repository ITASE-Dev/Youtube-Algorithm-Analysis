# 02_ai_analyzer

Stage 2. Turns raw text and images into semantic features using the OpenAI API.

| File | Fills |
|---|---|
| `ai_client.py` | shared client, retrying structured-output call, token accounting |
| `analyze_text.py` | `hook_score`, `curiosity_gap_score`, `emotion_tone`, `niche_relevance` |
| `analyze_thumbnail.py` | `thumbnail_has_face`, `thumbnail_face_emotion`, `thumbnail_text`, `title_thumbnail_synergy` |

```bash
python services/02_ai_analyzer/analyze_text.py --limit 50
```

```bash
python services/02_ai_analyzer/analyze_thumbnail.py --limit 50
```

Both accept `--model`, `--limit`, `--batch-size`, `--channel`, `--reanalyze`,
`--dry-run`, `--log-level`. Defaults come from `OPENAI_TEXT_MODEL` /
`OPENAI_VISION_MODEL` in `.env`.

Start with `--dry-run --limit 3`: it calls the API and prints the scores with the
model's one-line reasoning, but writes nothing. It is the cheapest way to see
whether a prompt change moved the numbers in the direction you wanted.

## Design notes

**Strict Structured Outputs forbids numeric bounds.** A Pydantic
`Field(ge=1, le=10)` emits `minimum`/`maximum` into the JSON schema, which the
strict parser rejects outright — every call would 400. The 1–10 range is instead
stated in the prompt and clamped by `clamp_score()` on the way into the
database, where it also protects the `CHECK` constraints on `videos`. A test
walks both schemas and fails if a bounded field is ever reintroduced.

Categorical fields (`emotion_tone`, `thumbnail_face_emotion`) use `Literal`,
which *is* allowed and compiles to an `enum`. Free-text categories would
fragment into dozens of near-synonyms and be worthless as model features.

**Cost.** Text sends only the first `--max-chars` (3000) of the transcript —
the hook lives in the opening, so a 42k-character transcript is billed at ~800
tokens instead of ~11k. Vision sends images at `detail="low"` (512×512, flat
~85 tokens); thumbnails are read at a glance, and faces, big text and
composition all survive the downsample. Measured: ~1250 tokens per video for
text, ~820 for vision.

**Model versions are stored per service.** `ai_model_version` belongs to the
text analyser and `vision_model_version` to the vision one. Sharing a single
column meant whichever script ran last silently claimed both sets of scores.
Both record the *resolved snapshot* the API reports (`gpt-4o-mini-2024-07-18`),
not the alias that was requested — aliases get repointed, and old rows would
otherwise become unreproducible.

**Unreachable thumbnails.** `maxresdefault.jpg` does not exist for every video
and older URLs rot. Each URL is HEAD-checked before the call — OpenAI bills for
a vision request even when their fetch fails — and on 404 the analyser walks
down to `sddefault`, `hqdefault`, `mqdefault`, the last of which YouTube
generates for everything. Only if all fail is the row logged and skipped, with
its columns left `NULL` so a later run retries.

**Retries.** `tenacity` retries `RateLimitError`, `APIConnectionError`,
`APITimeoutError` and `InternalServerError` five times with exponential backoff
(4s → 60s). The SDK's own retries are disabled so the two do not compound. A
refusal, a length cut-off or a bad image is *not* retried: it is logged, the row
is skipped, and the sweep continues.

## Determinism

`temperature=0` is set for models that accept it, but OpenAI does not guarantee
identical output. Re-scoring the same video moved one hook score from 4.0 to 3.0
between runs. Treat these as ±1 measurements, not exact values — if you need
stable numbers for a feature, score once and keep the result rather than
re-running with `--reanalyze`.

Models in the `gpt-5` and `o*` families reject `temperature` entirely;
`supports_temperature()` omits the parameter for them so those models still work.
