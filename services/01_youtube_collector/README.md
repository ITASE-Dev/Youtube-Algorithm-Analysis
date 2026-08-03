# 01_youtube_collector

Stage 1 of the pipeline. Populates `channels` and the raw/derived columns of `videos`.

## Files

| File | Role |
|---|---|
| `api_client.py` | `YouTubeDataClient`, `TranscriptClient`, `DislikeClient`, `ShortsProbe` + parsing helpers |
| `fetch_historical.py` | CLI orchestrator: resolve channel → upsert → loop videos → batch commit |

## Usage

Run from the workspace root with the venv active:

```bash
python services/01_youtube_collector/fetch_historical.py UC_x5XG1OV2P6uZZ5FSM9Ttw --limit 50
```

Accepts a channel id, an `@handle`, or a channel URL.

| Flag | Effect |
|---|---|
| `--limit N` | How many recent uploads to collect (default 50, paginates past 50) |
| `--batch-size N` | Rows per commit (default 10) |
| `--only-new` | Skip videos already in the database instead of refreshing them |
| `--skip-transcripts` | Do not call the transcript API |
| `--skip-dislikes` | Do not call the Return YouTube Dislike API |
| `--verify-shorts` | Confirm Shorts via HTTP probe instead of the `duration<=60s` proxy |
| `--log-level` | DEBUG / INFO / WARNING / ERROR |

## Behaviour worth knowing

**Idempotent.** Rerunning refreshes counters on existing rows; it never duplicates.
Columns owned by stages 2 and 3 (`hook_score`, `silence_ratio`, `performance_ratio`, …)
are never overwritten, so re-collection does not destroy paid-for AI results.

**Missing data is `None`, not zero.** A hidden like count, absent captions, or a
video with no Return YouTube Dislike record all store `NULL`. The dislike API
answers `200` with all-zero counters for videos it has never seen — that payload
is treated as unknown, while a real video with views and zero dislikes stores `0`.

**A dead transcript never kills the run.** `TranscriptsDisabled`, `NoTranscriptFound`,
`VideoUnavailable`, `IpBlocked` and friends are logged as warnings; the row is
saved without transcript fields and the loop continues. If the preferred
languages (`tr`, `en`) are missing, it retries with whatever track exists.

**Quota.** Roughly `2 + ceil(N/50) * 2` units per run (channels.list 1,
playlistItems.list 1/page, videos.list 1/page). The daily default is 10,000, so
quota is rarely the constraint — the unofficial APIs are. Hitting the quota
raises `QuotaExceededError`, which stops the run cleanly with committed batches intact.

**Rate limits.** The dislike client spaces calls ~0.35s apart, honours
`Retry-After`, and backs off on 429/5xx. Expect ~2.5s per video end to end.
