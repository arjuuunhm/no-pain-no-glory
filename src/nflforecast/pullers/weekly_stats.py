"""Weekly player stats puller.

Source: nflreadpy.load_player_stats(summary_level="week")
One row per player per game per season. This is the core table feature
engineering builds on (targets, carries, receiving/rushing/passing yards
and TDs, EPA, etc. -- ~150 columns as of nflreadpy 0.1.x).
"""

from __future__ import annotations

from pathlib import Path

import nflreadpy as nfl

from nflforecast.config import RAW_DIR, get_logger

logger = get_logger(__name__)

OUTPUT_PATH = RAW_DIR / "weekly_player_stats.parquet"


def pull_weekly_player_stats(seasons: list[int], output_path: Path = OUTPUT_PATH) -> Path:
    """Pull weekly (per-game) player stats for the given seasons and write to parquet.

    Idempotent: re-running overwrites the file with a fresh pull covering the
    requested seasons (nflverse weekly data for past seasons is stable/frozen;
    only the current in-progress season changes week to week).
    """
    logger.info("Pulling weekly player stats for seasons %s-%s", min(seasons), max(seasons))
    df = nfl.load_player_stats(seasons=seasons, summary_level="week")
    df.write_parquet(output_path)
    logger.info(
        "Wrote %s rows x %s cols -> %s (seasons %s-%s)",
        df.height,
        df.width,
        output_path,
        min(seasons),
        max(seasons),
    )
    return output_path
