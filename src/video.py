"""Small video-writing helper with a GIF fallback."""

from __future__ import annotations

from pathlib import Path


def save_video_or_gif(path: Path, frames: list, fps: int = 25) -> Path:
    """Save frames to MP4 when possible, otherwise fall back to GIF."""
    import imageio.v2 as imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        imageio.mimsave(path, frames, fps=fps)
        return path
    except ValueError as exc:
        if path.suffix.lower() != ".mp4":
            raise
        fallback = path.with_suffix(".gif")
        imageio.mimsave(fallback, frames, fps=min(fps, 20))
        print(f"MP4 export failed ({exc}). Saved GIF fallback to {fallback}")
        return fallback
