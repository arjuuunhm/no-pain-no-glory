"""Weekly roster status puller.

Source: nflreadpy.load_rosters_weekly. Week-by-week active/inactive status
per player -- this is the *realized* availability signal (games actually
played), distinct from load_injuries()'s self-reported, pre-game status.
Use this to build the availability model's label. Also carries birth_date,
which combined with the schedule week gives an as-of-that-week age (cleaner
than season-level age from load_players() for in-season features).
"""

from __future__ import annotations

from pathlib import Path

import nflreadpy as nfl

from nflforecast.config import RAW_DIR, get_logger

logger = get_logger(__name__)

OUTPUT_PATH = RAW_DIR / "rosters_weekly.parquet"


def pull_rosters_weekly(seasons: list[int], output_path: Path = OUTPUT_PATH) -> Path:
    logger.info("Pulling weekly rosters for seasons %s-%s", min(seasons), max(seasons))
    df = nfl.load_rosters_weekly(seasons=seasons)
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
