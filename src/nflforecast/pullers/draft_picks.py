"""Draft picks puller.

Source: nflreadpy.load_draft_picks. Historical NFL draft data (round, pick,
team, college, career approximate value) -- goes back decades, well beyond
our modeling window; we still scope it to the requested seasons to keep the
crosswalk aligned to the same window as the other pullers, since draft
capital is a modeling feature (resources.md §8) primarily for players with
<2 seasons of data.
"""

from __future__ import annotations

from pathlib import Path

import nflreadpy as nfl

from nflforecast.config import RAW_DIR, get_logger

logger = get_logger(__name__)

OUTPUT_PATH = RAW_DIR / "draft_picks.parquet"


def pull_draft_picks(seasons: list[int], output_path: Path = OUTPUT_PATH) -> Path:
    logger.info("Pulling draft picks for seasons %s-%s", min(seasons), max(seasons))
    df = nfl.load_draft_picks(seasons=seasons)
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
