"""Roster and player-ID crosswalk pullers.

- pull_rosters: seasonal rosters (nflreadpy.load_rosters) -- team, position,
  jersey number, birthdate, draft info, per player per season.
- pull_player_ids: cross-source ID crosswalk (nflreadpy.load_ff_playerids),
  mapping gsis_id/pfr_id/espn_id/sleeper_id/etc so weekly stats, ADP sources,
  and injury feeds can be joined on a common key. Not season-partitioned --
  it's a single current snapshot.
"""

from __future__ import annotations

from pathlib import Path

import nflreadpy as nfl

from nflforecast.config import RAW_DIR, get_logger

logger = get_logger(__name__)

ROSTERS_OUTPUT_PATH = RAW_DIR / "rosters.parquet"
PLAYER_IDS_OUTPUT_PATH = RAW_DIR / "player_ids.parquet"


def pull_rosters(seasons: list[int], output_path: Path = ROSTERS_OUTPUT_PATH) -> Path:
    logger.info("Pulling rosters for seasons %s-%s", min(seasons), max(seasons))
    df = nfl.load_rosters(seasons=seasons)
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


def pull_player_ids(output_path: Path = PLAYER_IDS_OUTPUT_PATH) -> Path:
    """Pull the cross-source player ID crosswalk (not season-scoped)."""
    logger.info("Pulling player ID crosswalk (ff_playerids)")
    df = nfl.load_ff_playerids()
    df.write_parquet(output_path)
    logger.info("Wrote %s rows x %s cols -> %s", df.height, df.width, output_path)
    return output_path
