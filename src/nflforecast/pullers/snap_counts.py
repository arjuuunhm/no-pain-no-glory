"""Snap counts puller.

Source: nflreadpy.load_snap_counts. Only available from the 2012 season
onward (nflverse raises a ValueError for earlier seasons) -- pass seasons
>= 2012.
"""

from __future__ import annotations

from pathlib import Path

import nflreadpy as nfl

from nflforecast.config import RAW_DIR, get_logger

logger = get_logger(__name__)

OUTPUT_PATH = RAW_DIR / "snap_counts.parquet"
MIN_SEASON = 2012


def pull_snap_counts(seasons: list[int], output_path: Path = OUTPUT_PATH) -> Path:
    """Pull offensive/defensive/ST snap counts and snap % for the given seasons."""
    usable = [s for s in seasons if s >= MIN_SEASON]
    dropped = sorted(set(seasons) - set(usable))
    if dropped:
        logger.warning("snap_counts unavailable before %s; dropping seasons %s", MIN_SEASON, dropped)
    if not usable:
        raise ValueError(f"No requested seasons >= {MIN_SEASON} for snap_counts")

    logger.info("Pulling snap counts for seasons %s-%s", min(usable), max(usable))
    df = nfl.load_snap_counts(seasons=usable)
    df.write_parquet(output_path)
    logger.info(
        "Wrote %s rows x %s cols -> %s (seasons %s-%s)",
        df.height,
        df.width,
        output_path,
        min(usable),
        max(usable),
    )
    return output_path
