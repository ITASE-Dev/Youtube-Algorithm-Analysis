# 03_predictor_engine

Stage 3. Computes the target variables, then (next step) trains the model.

| File | Fills |
|---|---|
| `calculate_targets.py` | `engagement_rate`, `recent_channel_avg_views`, `performance_ratio`, `targets_computed_at` |

```bash
python services/03_predictor_engine/calculate_targets.py --verify
```

`--window` (default 10), `--min-history` (default 3), `--dry-run`, `--verify`,
`--log-level`. Idempotent — rerun it whenever new videos are collected.

## The baseline window

```sql
AVG(view_count) OVER (
    PARTITION BY channel_id, is_shorts
    ORDER BY published_at, video_id
    ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING
)
```

`is_shorts` is in the partition because the two formats are published
interleaved and differ by roughly 2x in median views on the same channel. A
shared baseline would measure format mix rather than performance: a long video
following a run of Shorts would look like a hit purely because the baseline
collapsed. `--no-split-shorts` restores the single channel-wide baseline.
Correct `is_shorts` labels are a prerequisite — see `repair_shorts.py`.

`AND 1 PRECEDING` — never `CURRENT ROW` — is the most important clause in the
service. Including the current row would leak the answer into its own baseline:
a video's views would inflate the average it is measured against, pulling every
ratio toward 1.0 and teaching the model a relationship that cannot exist at
prediction time. A test asserts `CURRENT ROW` never appears in the SQL, and a
second implementation in pandas (`--verify`) recomputes the whole table and
compares, because an off-by-one here produces plausible numbers and a worthless
model.

Videos with fewer than `--min-history` earlier videos keep a `NULL` baseline
rather than a noisy one. Baselines only see videos **present in this database**,
so the oldest videos of every channel are always `NULL`.

`ORDER BY published_at, video_id` — the id breaks ties so two videos posted in
the same second sort deterministically across runs.

The window bound is interpolated into the SQL string rather than bound as a
parameter: T-SQL requires a literal in `ROWS BETWEEN n PRECEDING`. It is forced
through `int()` first, so nothing but a number can reach the query.

## Two properties of the target you must handle before training

Measured on the current 103-video sample:

**1. Recent videos read artificially low.** `view_count` is today's number for
every row, but the baseline videos have had months to accumulate while a video
posted last week has had days.

| Age | Videos | Median ratio |
|---|---|---|
| 0–7 days | 15 | 0.69 |
| 7–30 days | 38 | **0.40** |
| 30–90 days | 22 | 0.78 |

The script warns when recent videos have ratios. Exclude videos younger than
~30 days from the training set, or the model will learn "new = bad".

**2. The raw ratio is severely right-skewed** (skew ≈ 6.5, max 15.1). Views are
heavy-tailed, so an arithmetic-mean baseline sits above the typical video by
construction: the median ratio lands at 0.61 with only 30% above 1.0. A median
baseline would give 0.94 and 46% — the mean is not wrong, it is just not centred.

Train on `log(performance_ratio)` rather than the raw value: on this sample that
takes skew from **6.5 to −0.10**, near-symmetric, which is what squared-error
learners assume. Convert back with `exp()` for interpretation.

## engagement_rate

`(like_count + comment_count) / view_count`, with `like_count`/`comment_count`
treated as 0 when the creator hides them. `view_count = 0` yields `0.0`;
`view_count IS NULL` yields `NULL`, because unknown is not zero.
