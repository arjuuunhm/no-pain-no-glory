"""Master player bio/draft table puller.

Source: nflreadpy.load_players. Not season-scoped (one row per player,
full history) -- carries birth_date, position, draft_year/round/pick, which
is the single cleanest source for age and draft-capital prior features
(docs/features.md §5), instead of joining draft_picks + rosters by name.
"""

from __future__ import annotations

from pathlib import Path

import nflreadpy as nfl

from nflforecast.config import RAW_DIR, get_logger

logger = get_logger(__name__)

OUTPUT_PATH = RAW_DIR / "players.parquet"


def pull_players(output_path: Path = OUTPUT_PATH) -> Path:
    logger.info("Pulling master player bio/draft table")
    df = nfl.load_players()
    df.write_parquet(output_path)
    logger.info("Wrote %s rows x %s cols -> %s", df.height, df.width, output_path)
    return output_path
