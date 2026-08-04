# 01_youtube_collector

Stage 1 of the pipeline. Populates `channels` and the raw/derived columns of `videos`.

## Files

| File | Role |
|---|---|
| `api_client.py` | `YouTubeDataClient`, `TranscriptClient`, `DislikeClient`, `ShortsProbe` + parsing helpers |
| `fetch_historical.py` | CLI orchestrator: resolve channel → upsert → loop videos → batch commit |
| `fetch_transcripts_and_dislikes.py` | Backfill transcripts + dislikes for rows already stored (network-bound, fast) |
| `analyze_media_pacing.py` | Download → measure audio/visual pacing → delete (CPU-bound, slow) |

Run them in that order. The two enrichers are split on purpose: the first is
pure network and finishes hundreds of videos in minutes, the second pins a CPU
core for ~50s per video, so you can run it overnight while the light one runs
whenever.

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

---

## fetch_transcripts_and_dislikes.py

Backfills rows that are missing transcript or dislike data. No YouTube Data API
quota is spent.

```bash
python services/01_youtube_collector/fetch_transcripts_and_dislikes.py --limit 200
```

`--only transcripts|dislikes|both`, `--channel UC...`, `--batch-size`, `--retry-failed`.

**IP blocks are not "no captions".** `youtube-transcript-api` scrapes the player
page, and YouTube blocks IPs that pull captions in bulk — roughly 100 in 20
minutes was enough to trigger it here. `IpBlocked`, `RequestBlocked`,
`PoTokenRequired` and `YouTubeRequestFailed` are therefore classified as
environmental: the row keeps `NULL` **and no marker**, so it stays queued. After
5 consecutive blocks the sweep stops asking (hammering only extends the block)
while dislikes keep collecting. Requests are throttled to 1/second to avoid
earning a block in the first place. Ways out: wait a few hours, use a VPN, or
set `TRANSCRIPT_PROXY_URL` in `.env`.

**Attempt markers.** The queue is
`(full_transcript IS NULL AND transcript_checked_at IS NULL) OR (dislike_count IS NULL AND dislike_checked_at IS NULL)`.
Storing only `NULL` on failure would re-request permanently caption-less videos
on every run forever, so each attempt stamps `*_checked_at` whether it succeeded
or not. `--retry-failed` ignores those markers for a fresh sweep — use it after
an IP block lifts.

## analyze_media_pacing.py

Downloads a low-quality copy of each video, measures it, writes
`silence_ratio`, `pitch_variance`, `scene_cuts_per_minute`, then deletes the file.

```bash
python services/01_youtube_collector/analyze_media_pacing.py --limit 20 --max-seconds 120
```

`--max-seconds` (analysis window, default 300), `--skip-intro`, `--channel`,
`--skip-shorts`, `--retry-failed`, `--workdir`.

Requires **ffmpeg** on PATH.

| Metric | How | Reads as |
|---|---|---|
| `silence_ratio` | `librosa.effects.split`, 30 dB floor | 0.07 = wall-to-wall talking, 0.30 = lots of pauses |
| `pitch_variance` | std-dev of `librosa.pyin` F0, in Hz | under ~25 = monotone, 100+ = very animated |
| `scene_cuts_per_minute` | mean abs. diff of every 10th frame at 160×90 | 3 = static talking head, 12+ = fast-cut edit |

`pitch_variance` holds a **standard deviation**, not a variance — same units as
the pitch itself (Hz), which is far easier to reason about. The column name is
kept for compatibility with the data dictionary.

**Windowed download.** Only the analysed window is downloaded, not the whole
video (`yt-dlp download_ranges`). On a 39-minute upload this took a run from
9 minutes to ~50 seconds, and produced identical metrics. Pacing and vocal
energy are stable within a video, so a bounded sample measures the same thing.

**Disk safety.** Each video gets its own scratch directory under
`MEDIA_CACHE_DIR`, deleted in a `finally` block — so a crash mid-analysis still
cleans up. Verified: the cache holds 0 files before and after a run.

**Failures are recorded, not retried blindly.** A download block or corrupt file
writes the reason to `media_error` and stamps `media_checked_at`; the sweep
continues. `--retry-failed` picks those rows up again.
