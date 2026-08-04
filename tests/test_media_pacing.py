"""Tests for the media pacing analyser.

The scene-cut test builds a synthetic video with a known number of hard cuts,
so the detector is measured against ground truth rather than a guess. No
network access and no downloads.
"""

from __future__ import annotations

import sys
import wave
from pathlib import Path

import numpy as np
import pytest

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE_ROOT))
sys.path.insert(0, str(WORKSPACE_ROOT / "services" / "01_youtube_collector"))

import cv2  # noqa: E402
from analyze_media_pacing import (  # noqa: E402
    MediaError,
    MediaMetrics,
    analyze_audio,
    analyze_scene_cuts,
)

FPS = 30
WIDTH, HEIGHT = 320, 180


def _write_video(path: Path, *, seconds: int, cut_every: int) -> int:
    """Write a video that hard-cuts between solid colours. Returns the cut count."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT))
    colours = [(20, 20, 20), (240, 240, 240), (10, 200, 10), (200, 10, 10)]
    cuts = 0
    previous_index = 0
    for frame_number in range(seconds * FPS):
        index = (frame_number // (cut_every * FPS)) % len(colours)
        if index != previous_index:
            cuts += 1
            previous_index = index
        frame = np.full((HEIGHT, WIDTH, 3), colours[index], dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return cuts


def test_scene_cuts_matches_known_ground_truth(tmp_path):
    """12s of video cutting every 2s contains 5 cuts -> 25 cuts/minute."""
    video = tmp_path / "cuts.mp4"
    expected_cuts = _write_video(video, seconds=12, cut_every=2)
    assert expected_cuts == 5

    result = analyze_scene_cuts(video, max_seconds=12, skip_seconds=0)
    assert result is not None
    # 5 cuts in 0.2 min = 25/min. Frame sampling can miss or double a cut at
    # the boundary, so allow one cut of slack either way.
    assert 20.0 <= result <= 30.0


def test_static_video_reports_no_cuts(tmp_path):
    """A single unchanging shot must not invent cuts from compression noise."""
    video = tmp_path / "static.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT))
    for _ in range(FPS * 6):
        writer.write(np.full((HEIGHT, WIDTH, 3), (128, 128, 128), dtype=np.uint8))
    writer.release()

    assert analyze_scene_cuts(video, max_seconds=6, skip_seconds=0) == 0.0


def test_unopenable_file_raises_mediaerror(tmp_path):
    broken = tmp_path / "broken.mp4"
    broken.write_bytes(b"this is not a video")
    with pytest.raises(MediaError):
        analyze_scene_cuts(broken, max_seconds=10, skip_seconds=0)


def _write_wav(path: Path, samples: np.ndarray, rate: int = 22050) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes((np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes())


def test_silence_ratio_on_half_silent_audio(tmp_path):
    """Two seconds of tone followed by two of silence is ~50% dead air."""
    rate = 22050
    t = np.linspace(0, 2, rate * 2, endpoint=False)
    tone = 0.5 * np.sin(2 * np.pi * 220 * t)
    audio = np.concatenate([tone, np.zeros(rate * 2)])

    wav = tmp_path / "half.wav"
    _write_wav(wav, audio, rate)

    silence_ratio, _pitch = analyze_audio(wav)
    assert silence_ratio is not None
    assert 0.4 <= silence_ratio <= 0.6


def test_pitch_variance_separates_monotone_from_varied(tmp_path):
    """A steady tone must score far lower than a sweeping one."""
    rate = 22050
    t = np.linspace(0, 3, rate * 3, endpoint=False)

    steady = tmp_path / "steady.wav"
    _write_wav(steady, 0.5 * np.sin(2 * np.pi * 220 * t), rate)

    # Sweep 150 Hz -> 400 Hz; phase is the integral of the frequency ramp.
    frequency = np.linspace(150, 400, t.size)
    phase = 2 * np.pi * np.cumsum(frequency) / rate
    varied = tmp_path / "varied.wav"
    _write_wav(varied, 0.5 * np.sin(phase), rate)

    _s1, steady_pitch = analyze_audio(steady)
    _s2, varied_pitch = analyze_audio(varied)

    assert steady_pitch is not None and varied_pitch is not None
    assert steady_pitch < 5.0           # near-zero spread for a constant tone
    assert varied_pitch > steady_pitch * 5


def test_empty_audio_raises(tmp_path):
    wav = tmp_path / "empty.wav"
    _write_wav(wav, np.zeros(100), 22050)
    with pytest.raises(MediaError):
        analyze_audio(wav)


def test_media_metrics_is_empty():
    assert MediaMetrics().is_empty()
    assert not MediaMetrics(silence_ratio=0.1).is_empty()
