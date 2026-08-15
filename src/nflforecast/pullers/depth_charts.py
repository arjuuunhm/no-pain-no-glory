"""Weekly official depth chart puller.

Source: nflreadpy.load_depth_charts. depth_team (1 = starter, 2 = backup,
...) per player per week -- used as a same-week opportunity-risk feature and,
combined with snap counts, to detect committee backfields.
"""

from __future__ import annotations

from pathlib import Path

import nflreadpy as nfl

from nflforecast.config import RAW_DIR, get_logger

logger = get_logger(__name__)

OUTPUT_PATH = RAW_DIR / "depth_charts.parquet"


def pull_depth_charts(seasons: list[int], output_path: Path = OUTPUT_PATH) -> Path:
    logger.info("Pulling depth charts for seasons %s-%s", min(seasons), max(seasons))
    df = nfl.load_depth_charts(seasons=seasons)
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
