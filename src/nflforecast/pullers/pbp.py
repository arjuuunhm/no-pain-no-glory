"""Play-by-play puller (curated column subset).

Source: nflreadpy.load_pbp. Full pbp is ~370 columns and large; we select a
curated subset needed for v1 feature engineering (game-script buckets,
red-zone/inside-10 opportunity, goal-line carries) rather than storing
everything, since none of the other pullers expose play-level data.
"""

from __future__ import annotations

from pathlib import Path

import nflreadpy as nfl

from nflforecast.config import RAW_DIR, get_logger

logger = get_logger(__name__)

OUTPUT_PATH = RAW_DIR / "pbp.parquet"

# Columns needed for: score-differential/game-script buckets, red-zone and
# inside-10 opportunity shares, scramble-vs-designed-run splits, O-line
# run-block/pass-protection proxies, and team scheme/pace tendencies. See
# docs/features.md §1d/§4/§7 for what each feeds.
KEEP_COLUMNS = (
    # --- identity / grain
    "season",
    "week",
    "season_type",
    "game_id",
    "posteam",
    "defteam",
    "play_type",
    # --- play classification
    "rush_attempt",
    "pass_attempt",
    "pass",
    "qb_scramble",
    "qb_dropback",
    "rusher_player_id",
    "receiver_player_id",
    "passer_player_id",
    # --- situation
    "yardline_100",
    "score_differential",
    "half_seconds_remaining",
    "game_seconds_remaining",
    "qtr",
    "down",
    "ydstogo",
    "goal_to_go",
    # --- outcome / efficiency
    "yards_gained",
    "epa",
    "success",
    "first_down",
    "touchdown",
    # --- O-line proxies (features/oline.py): sacks and hits allowed on
    # dropbacks, and stuffed/negative runs, are the only line-quality signals
    # available without paid charting.
    "sack",
    "qb_hit",
    "tackled_for_loss",
    # --- scheme / pace tendencies (features/scheme.py): xpass is nflverse's
    # expected-pass model, so pass - xpass gives pass rate over expected.
    "xpass",
    "pass_oe",
    "shotgun",
    "no_huddle",
    "fixed_drive",
)


def pull_play_by_play(seasons: list[int], output_path: Path = OUTPUT_PATH) -> Path:
    logger.info("Pulling play-by-play for seasons %s-%s", min(seasons), max(seasons))
    df = nfl.load_pbp(seasons=seasons)

    missing = [c for c in KEEP_COLUMNS if c not in df.columns]
    if missing:
        logger.warning("Expected pbp columns missing, dropping from selection: %s", missing)
    keep = [c for c in KEEP_COLUMNS if c in df.columns]
    df = df.select(keep)

    df.write_parquet(output_path)
    logger.info(
        "Wrote %s rows x %s cols -> %s (seasons %s-%s, %s of %s source columns kept)",
        df.height,
        df.width,
        output_path,
        min(seasons),
        max(seasons),
        len(keep),
        len(KEEP_COLUMNS),
    )
    return output_path
