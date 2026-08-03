# 02_ai_analyzer

Stage 2. Reads rows where `ai_analyzed_at IS NULL` and fills the AI feature block:

- Text (transcript + title): `hook_score`, `curiosity_gap_score`, `emotion_tone`, `niche_relevance`
- Vision (thumbnail): `thumbnail_has_face`, `thumbnail_face_emotion`, `thumbnail_text`, `title_thumbnail_synergy`

Stamps `ai_analyzed_at` and `ai_model_version` on success so runs are resumable
and results are reproducible.
