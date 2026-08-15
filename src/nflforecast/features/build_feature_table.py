"""Assemble the player-week feature table and its label tables.

Grain: one row per rostered RB/WR/TE per regular-season week, built off
`spine.build_spine` rather than off whoever recorded a stat line. Three files
land in data/processed/:

- `player_week_features.parquet` -- features only, nothing measured at or
  after the row's own week.
- `player_week_labels.parquet` -- weekly targets, same key.
- `player_season_labels.parquet` -- season targets, player-season grain.

The split is the leakage control. Feature modules each drop their raw
current-week columns; keeping the targets in separate files means a training
script has to join them in deliberately.

Join discipline, by block:

- **Player-grain rolling blocks** (opportunity, red zone, game script) attach
  through `attach_asof`, which binds each spine week to the most recent
  player row *strictly before* it. This is what lets a player who missed a
  game still carry features into that week.
- **Team-grain blocks** (O-line, scheme) join on `(team, season, week)` and
  are already shift(1)-lagged internally.
- **Vegas** joins on `(team, season, week)` unlagged -- a closing line is
  known pre-kickoff.
- **Priors** join on `(player_id, season)`; age and draft capital are fixed
  within a season and carry no outcome information.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from nflforecast.config import PROCESSED_DIR, RAW_DIR, get_logger
from nflforecast.features.availability import (
    build_injury_report_features,
    build_participation_history,
    build_prior_season_availability,
    fill_injury_report_defaults,
)
from nflforecast.features.game_script import build_game_script_features
from nflforecast.features.labels import build_season_labels, build_weekly_labels
from nflforecast.features.oline import build_oline_features
from nflforecast.features.opportunity import build_opportunity_features
from nflforecast.features.priors import build_prior_features
from nflforecast.features.redzone import build_redzone_features
from nflforecast.features.scheme import build_scheme_features
from nflforecast.features.spine import build_spine
from nflforecast.features.utils import attach_asof
from nflforecast.features.vegas import build_vegas_features

logger = get_logger(__name__)

FEATURES_PATH = PROCESSED_DIR / "player_week_features.parquet"
WEEKLY_LABELS_PATH = PROCESSED_DIR / "player_week_labels.parquet"
SEASON_LABELS_PATH = PROCESSED_DIR / "player_season_labels.parquet"


def build_feature_table(
    raw_dir: Path = RAW_DIR,
    features_path: Path = FEATURES_PATH,
    weekly_labels_path: Path = WEEKLY_LABELS_PATH,
    season_labels_path: Path = SEASON_LABELS_PATH,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    logger.info("Loading raw tables from %s", raw_dir)
    weekly_stats = pl.read_parquet(raw_dir / "weekly_player_stats.parquet")
    snap_counts = pl.read_parquet(raw_dir / "snap_counts.parquet")
    players = pl.read_parquet(raw_dir / "players.parquet")
    schedules = pl.read_parquet(raw_dir / "schedules.parquet")
    pbp = pl.read_parquet(raw_dir / "pbp.parquet")
    rosters_weekly = pl.read_parquet(raw_dir / "rosters_weekly.parquet")
    injuries = pl.read_parquet(raw_dir / "injuries.parquet")
    depth_charts = pl.read_parquet(raw_dir / "depth_charts.parquet")

    logger.info("Building RB/WR/TE roster-week spine")
    spine = build_spine(rosters_weekly, snap_counts, players)
    logger.info("Spine: %s player-weeks, %s played", spine.height, spine["played"].sum())

    logger.info("Building opportunity features")
    opportunity = build_opportunity_features(weekly_stats, snap_counts, players)

    logger.info("Building red-zone features")
    redzone = build_redzone_features(pbp)

    logger.info("Building game-script features")
    game_script = build_game_script_features(pbp)

    logger.info("Building O-line features")
    oline = build_oline_features(pbp)

    logger.info("Building scheme / play-caller features")
    scheme = build_scheme_features(pbp, schedules)

    logger.info("Building Vegas context")
    vegas = build_vegas_features(schedules)

    logger.info("Building priors, availability history, injury reports")
    priors = build_prior_features(rosters_weekly, players).drop("position")
    prior_avail = build_prior_season_availability(spine)
    participation = build_participation_history(spine)
    injury_report = build_injury_report_features(injuries, depth_charts)

    logger.info("Joining feature blocks onto the spine")
    df = spine.drop("offense_snaps")

    # Player-grain rolling blocks: strictly-prior as-of attach.
    for name, block in (("opportunity", opportunity), ("redzone", redzone), ("game_script", game_script)):
        before = df.height
        df = attach_asof(df, block, by="player_id")
        assert df.height == before, f"{name} as-of attach changed row count"

    # Team-grain blocks (already lagged internally) and pre-kickoff context.
    df = df.join(oline, on=["team", "season", "week"], how="left")
    df = df.join(scheme, on=["team", "season", "week"], how="left")
    df = df.join(vegas, on=["team", "season", "week"], how="left")

    # Player-season blocks.
    df = df.join(priors, on=["player_id", "season"], how="left")
    df = df.join(prior_avail, on=["player_id", "season"], how="left")
    df = df.join(participation, on=["player_id", "season", "week"], how="left")
    df = df.join(injury_report, on=["player_id", "season", "week"], how="left")
    df = fill_injury_report_defaults(df)

    # `played` is the availability *label*; it belongs in the label table only.
    df = df.drop("played")

    logger.info("Building label tables")
    weekly_labels = build_weekly_labels(spine, weekly_stats)
    season_labels = build_season_labels(spine, weekly_labels)

    df.write_parquet(features_path)
    weekly_labels.write_parquet(weekly_labels_path)
    season_labels.write_parquet(season_labels_path)

    logger.info("features      %6s rows x %3s cols -> %s", df.height, df.width, features_path)
    logger.info("weekly labels %6s rows x %3s cols -> %s", weekly_labels.height, weekly_labels.width, weekly_labels_path)
    logger.info("season labels %6s rows x %3s cols -> %s", season_labels.height, season_labels.width, season_labels_path)
    return df, weekly_labels, season_labels
