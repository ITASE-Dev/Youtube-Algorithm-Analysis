"""Is YouTube still refusing transcripts from this IP?

One request against a video we already have, so the answer costs nothing and
tells you whether it is worth rerunning the enricher::

    python services/01_youtube_collector/check_block.py

Exit code 0 means clear, 1 means still blocked -- handy in a wait loop.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from api_client import TranscriptClient  # noqa: E402
from sqlalchemy import select  # noqa: E402

from core.config import settings  # noqa: E402
from core.database import session_scope  # noqa: E402
from core.models import Video  # noqa: E402

#: Known-good video with public captions, used when the database is empty.
FALLBACK_VIDEO_ID = "dQw4w9WgXcQ"


def pick_probe_video() -> str:
    """Prefer a video we already know has captions, so a miss means a block."""
    try:
        with session_scope() as session:
            video_id = session.execute(
                select(Video.video_id).where(Video.full_transcript.is_not(None)).limit(1)
            ).scalar()
            return video_id or FALLBACK_VIDEO_ID
    except Exception:  # noqa: BLE001 - the probe must work without a database
        return FALLBACK_VIDEO_ID


def main() -> int:
    logging.basicConfig(level=logging.CRITICAL)

    video_id = pick_probe_video()
    if settings.transcript_proxy_url:
        print("Proxy is configured; this tests the proxy's IP, not yours.")

    result = TranscriptClient(min_interval=0.0).fetch(video_id)

    if result.text:
        print(f"CLEAR - transcripts are working again ({result.word_count} words from {video_id}).")
        print("Run: python services/01_youtube_collector/fetch_transcripts_and_dislikes.py")
        return 0

    if result.blocked:
        print(f"BLOCKED - YouTube is still refusing captions from this IP (probe: {video_id}).")
        print("Wait a few hours, switch network (mobile hotspot / VPN), or set")
        print("TRANSCRIPT_PROXY_URL in .env. Nothing in the database is lost.")
        return 1

    print(f"UNCLEAR - {video_id} returned no captions but was not a block.")
    print("Try again, or check the video manually.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
