"""Participation (personnel/formation) puller.

Source: nflreadpy.load_participation. Play-by-play-level offense/defense
personnel, formation, box count, pass rush count, coverage type, etc.

Quirks discovered while building this pipeline (see docs/data_pipeline.md):
- The returned table has NO `season` column -- it's keyed by
  `nflverse_game_id` (format "{season}_{week}_{away}_{home}"), so we derive
  a `season` column by parsing the game_id prefix for downstream filtering.
- Coverage is genuinely uneven across seasons, confirming resources.md's
  warning: `defenders_in_box` is ~26% null in 2016-2022 but ~0% null in
  2023-2024; `offense_formation` is ~27-28% null pre-2023 vs ~20% in
  2023-2024. Don't build features assuming these columns are populated for
  older seasons without checking null rates per season first.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
import nflreadpy as nfl

from nflforecast.config import RAW_DIR, get_logger

logger = get_logger(__name__)

OUTPUT_PATH = RAW_DIR / "participation.parquet"


def pull_participation(seasons: list[int], output_path: Path = OUTPUT_PATH) -> Path:
    logger.info("Pulling participation for seasons %s-%s", min(seasons), max(seasons))
    df = nfl.load_participation(seasons=seasons)
    df = df.with_columns(
        pl.col("nflverse_game_id").str.slice(0, 4).cast(pl.Int32).alias("season")
    )
    df.write_parquet(output_path)

    null_rates = df.group_by("season").agg(
        pl.col("offense_formation").is_null().mean().round(3).alias("null_rate_offense_formation"),
        pl.col("defenders_in_box").is_null().mean().round(3).alias("null_rate_defenders_in_box"),
    ).sort("season")
    logger.info("Participation null rates by season (coverage varies -- see docstring):\n%s", null_rates)

    logger.info(
        "Wrote %s rows x %s cols -> %s (seasons %s-%s)",
        df.height,
        df.width,
        output_path,
        min(seasons),
        max(seasons),
    )
    return output_path
