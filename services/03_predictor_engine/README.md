# 03_predictor_engine

Stage 3. Computes the target variables and trains the model.

Target calculation (per channel, ordered by `published_at`):

1. `recent_channel_avg_views` — rolling mean of `view_count` over the **10
   preceding** videos (shifted, so the video itself never leaks into its own
   baseline).
2. `performance_ratio` = `view_count / recent_channel_avg_views`.
3. `engagement_rate` = `(like_count + comment_count) / view_count`.

Then trains a regressor on the metadata + audio/visual + AI feature blocks to
predict `performance_ratio`.
