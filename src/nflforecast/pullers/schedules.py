"""Schedules puller -- also the Vegas lines source.

Source: nflreadpy.load_schedules. One row per game. Contrary to the
assumption that Vegas data needs a separate source, nflverse bakes closing
lines directly into the schedule table: spread_line, total_line, plus
away/home moneyline and spread odds (juice). No extra pull needed for
opening/closing spread & total -- see docs/data_pipeline.md for what's
verified present.
"""

from __future__ import annotations

from pathlib import Path

import nflreadpy as nfl

from nflforecast.config import RAW_DIR, get_logger

logger = get_logger(__name__)

OUTPUT_PATH = RAW_DIR / "schedules.parquet"

VEGAS_COLUMNS = (
    "spread_line",
    "total_line",
    "away_moneyline",
    "home_moneyline",
    "away_spread_odds",
    "home_spread_odds",
)


def pull_schedules(seasons: list[int], output_path: Path = OUTPUT_PATH) -> Path:
    logger.info("Pulling schedules for seasons %s-%s", min(seasons), max(seasons))
    df = nfl.load_schedules(seasons=seasons)
    df.write_parquet(output_path)

    missing_vegas = [c for c in VEGAS_COLUMNS if c not in df.columns]
    if missing_vegas:
        logger.warning("Expected Vegas columns missing from schedules: %s", missing_vegas)
    else:
        null_spread = df["spread_line"].is_null().sum()
        logger.info(
            "Vegas columns present: %s (spread_line null in %s/%s rows, e.g. future/unplayed games)",
            VEGAS_COLUMNS,
            null_spread,
            df.height,
        )

    logger.info(
        "Wrote %s rows x %s cols -> %s (seasons %s-%s)",
        df.height,
        df.width,
        output_path,
        min(seasons),
        max(seasons),
    )
    return output_path
