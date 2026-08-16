"""Assemble the player-week feature table and its label tables.

Grain: one row per rostered RB/WR/TE per regular-season week, built off
`spine.build_spine` rather than off whoever recorded a stat line. Three files
land in data/processed/:

- `player_week_features.parquet` -- features only, nothing measured at or
  after the row's own week.
- `player_week_labels.parquet` -- weekly targets, same key.
- `player_season_labels.parquet` -- season targets, player-season grain.
- `season_fixtures.parquet` / `opponent_season_profile.parquet` -- byproducts
  of the `vegas` and `matchup` blocks below, written separately so `model/`
  can expand a season projection into its scheduled games (and each game's
  opponent's prior-season profile) without ever reading `data/raw/`. See
  `model/gbm.py`'s per-game mechanism.

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
- **Matchup** joins on `(opponent, season, week)`, after Vegas, which is the
  block that attaches `opponent` in the first place.
- **Priors** join on `(player_id, season)`; age and draft capital are fixed
  within a season and carry no outcome information.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from nflforecast.config import (
    HISTORY_END_SEASON,
    HISTORY_START_SEASON,
    PROCESSED_DIR,
    RAW_DIR,
    get_logger,
)
from nflforecast.features.availability import (
    build_injury_report_features,
    build_participation_history,
    build_prior_season_availability,
    fill_injury_report_defaults,
)
from nflforecast.features.game_script import build_game_script_features
from nflforecast.features.labels import build_season_labels, build_weekly_labels
from nflforecast.features.matchup import build_matchup_features, build_opponent_season_profile
from nflforecast.features.oline import build_oline_features
from nflforecast.features.opportunity import build_opportunity_features
from nflforecast.features.priors import build_prior_features
from nflforecast.features.redzone import build_redzone_features
from nflforecast.features.scheme import build_scheme_features
from nflforecast.features.spine import build_preseason_spine, build_spine
from nflforecast.features.utils import attach_asof
from nflforecast.features.vegas import build_vegas_features

logger = get_logger(__name__)

FEATURES_PATH = PROCESSED_DIR / "player_week_features.parquet"
WEEKLY_LABELS_PATH = PROCESSED_DIR / "player_week_labels.parquet"
SEASON_LABELS_PATH = PROCESSED_DIR / "player_season_labels.parquet"
PRESEASON_FEATURES_PATH = PROCESSED_DIR / "preseason_features.parquet"
FIXTURES_PATH = PROCESSED_DIR / "season_fixtures.parquet"
OPPONENT_PROFILE_PATH = PROCESSED_DIR / "opponent_season_profile.parquet"


def build_feature_table(
    raw_dir: Path = RAW_DIR,
    features_path: Path = FEATURES_PATH,
    weekly_labels_path: Path = WEEKLY_LABELS_PATH,
    season_labels_path: Path = SEASON_LABELS_PATH,
    upcoming_season: int | None = None,
    preseason_features_path: Path = PRESEASON_FEATURES_PATH,
    fixtures_path: Path = FIXTURES_PATH,
    opponent_profile_path: Path = OPPONENT_PROFILE_PATH,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Build the feature table, optionally including an unplayed season.

    `upcoming_season` appends `spine.build_preseason_spine` rows for a season
    that has not started, runs them through the identical block joins, and
    writes them to their **own** file. Separate rather than appended, for one
    reason: the label tables are built from the real spine only, and a row
    with no season yet played would otherwise land in the panel as a genuine
    zero-point season (`panel.build_panel` fills missing actuals with 0) and
    be trained on. The split is what keeps that impossible rather than
    merely unlikely.
    """
    logger.info("Loading raw tables from %s", raw_dir)
    weekly_stats = pl.read_parquet(raw_dir / "weekly_player_stats.parquet")
    snap_counts = pl.read_parquet(raw_dir / "snap_counts.parquet")
    players = pl.read_parquet(raw_dir / "players.parquet")
    schedules = pl.read_parquet(raw_dir / "schedules.parquet")
    pbp = pl.read_parquet(raw_dir / "pbp.parquet")
    rosters_weekly = pl.read_parquet(raw_dir / "rosters_weekly.parquet")
    injuries = pl.read_parquet(raw_dir / "injuries.parquet")
    depth_charts = pl.read_parquet(raw_dir / "depth_charts.parquet")

    # The raw directory deliberately keeps its older nflverse pulls as a
    # cache, but this project currently trains only on 2020-25.  Filter at
    # ingestion rather than relying on every downstream block to remember the
    # boundary.  Schedules also retain the requested target season so a
    # preseason projection can fan out over its real fixtures.
    max_feature_season = upcoming_season if upcoming_season is not None else HISTORY_END_SEASON

    def in_window(frame: pl.DataFrame, end: int = HISTORY_END_SEASON) -> pl.DataFrame:
        return frame.filter(pl.col("season").is_between(HISTORY_START_SEASON, end))

    weekly_stats = in_window(weekly_stats)
    snap_counts = in_window(snap_counts)
    schedules = in_window(schedules, max_feature_season)
    pbp = in_window(pbp)
    rosters_weekly = in_window(rosters_weekly)
    injuries = in_window(injuries)
    depth_charts = in_window(depth_charts)

    logger.info("Building RB/WR/TE roster-week spine")
    spine = build_spine(rosters_weekly, snap_counts, players)
    logger.info("Spine: %s player-weeks, %s played", spine.height, spine["played"].sum())

    # Labels are built from `spine` alone, below. `feature_spine` is what the
    # feature blocks attach to, and is the only one the upcoming season joins.
    feature_spine = spine
    if upcoming_season is not None:
        rosters = pl.read_parquet(raw_dir / "rosters.parquet")
        preseason = build_preseason_spine(rosters, upcoming_season)
        if preseason.height == 0:
            raise ValueError(
                f"no {upcoming_season} rosters in {raw_dir / 'rosters.parquet'} -- "
                f"pull it first (build_dataset.py --end {upcoming_season})"
            )
        logger.info("Preseason spine: %s players for %s", preseason.height, upcoming_season)
        feature_spine = pl.concat([spine, preseason], how="vertical_relaxed")

    logger.info("Building opportunity features")
    opportunity = build_opportunity_features(weekly_stats, snap_counts, players)

    logger.info("Building red-zone features")
    redzone = build_redzone_features(pbp)

    logger.info("Building game-script features")
    game_script = build_game_script_features(pbp)

    # Team-grain blocks emit rows only where games exist, so the upcoming
    # season needs a placeholder team-week to carry the prior season's
    # trailing windows across the boundary. See `utils.append_upcoming_week`.
    logger.info("Building O-line features")
    oline = build_oline_features(pbp, upcoming_season=upcoming_season)

    logger.info("Building scheme / play-caller features")
    scheme = build_scheme_features(pbp, schedules, upcoming_season=upcoming_season)

    logger.info("Building Vegas context")
    vegas = build_vegas_features(schedules)

    logger.info("Building opponent matchup features")
    matchup = build_matchup_features(pbp, weekly_stats, upcoming_season=upcoming_season)

    logger.info("Building priors, availability history, injury reports")
    # `build_prior_features` reads only (gsis_id, season, years_exp) from this
    # frame and takes age/draft capital from `players`. The upcoming season has
    # no weekly rosters, so its experience counts come from the seasonal file --
    # without this, age and draft capital are null for every preseason row, and
    # they are precisely what a projection leans on for young players.
    prior_source = rosters_weekly.select(["gsis_id", "season", "years_exp"])
    if upcoming_season is not None:
        prior_source = pl.concat(
            [
                prior_source,
                rosters.filter(pl.col("season") == upcoming_season).select(
                    ["gsis_id", "season", "years_exp"]
                ),
            ],
            how="vertical_relaxed",
        )
    priors = build_prior_features(prior_source, players).drop("position")
    # Availability history is derived from the spine's own play record, so it
    # is built on `feature_spine`: the upcoming season's rows carry no plays
    # of their own but must still see the seasons behind them.
    prior_avail = build_prior_season_availability(feature_spine)
    participation = build_participation_history(feature_spine)
    injury_report = build_injury_report_features(injuries, depth_charts)

    logger.info("Joining feature blocks onto the spine")
    df = feature_spine.drop("offense_snaps")

    # Player-grain rolling blocks: strictly-prior as-of attach.
    for name, block in (("opportunity", opportunity), ("redzone", redzone), ("game_script", game_script)):
        before = df.height
        df = attach_asof(df, block, by="player_id")
        assert df.height == before, f"{name} as-of attach changed row count"

    # Team-grain blocks (already lagged internally) and pre-kickoff context.
    df = df.join(oline, on=["team", "season", "week"], how="left")
    df = df.join(scheme, on=["team", "season", "week"], how="left")
    df = df.join(vegas, on=["team", "season", "week"], how="left")
    # Matchup keys off `opponent`, which vegas.py attaches above -- must join
    # after it. See features/matchup.py's module docstring.
    df = df.join(matchup, on=["opponent", "season", "week"], how="left")

    # Player-season blocks.
    df = df.join(priors, on=["player_id", "season"], how="left")
    df = df.join(prior_avail, on=["player_id", "season"], how="left")
    df = df.join(participation, on=["player_id", "season", "week"], how="left")
    df = df.join(injury_report, on=["player_id", "season", "week"], how="left")
    df = fill_injury_report_defaults(df)

    # `played` is the availability *label*; it belongs in the label table only.
    df = df.drop("played")

    # Labels from the real spine only: the upcoming season has no outcomes,
    # and a zero-filled row for it would be indistinguishable from a rostered
    # player who genuinely never scored.
    logger.info("Building label tables")
    weekly_labels = build_weekly_labels(spine, weekly_stats)
    season_labels = build_season_labels(spine, weekly_labels)

    if upcoming_season is not None:
        preseason_features = df.filter(pl.col("season") == upcoming_season)
        df = df.filter(pl.col("season") != upcoming_season)
        preseason_features.write_parquet(preseason_features_path)
        logger.info(
            "preseason    %6s rows x %3s cols -> %s",
            preseason_features.height,
            preseason_features.width,
            preseason_features_path,
        )

    df.write_parquet(features_path)
    weekly_labels.write_parquet(weekly_labels_path)
    season_labels.write_parquet(season_labels_path)

    logger.info("features      %6s rows x %3s cols -> %s", df.height, df.width, features_path)
    logger.info("weekly labels %6s rows x %3s cols -> %s", weekly_labels.height, weekly_labels.width, weekly_labels_path)
    logger.info("season labels %6s rows x %3s cols -> %s", season_labels.height, season_labels.width, season_labels_path)

    # Fixtures are `vegas`'s (team, season, week, opponent) as-is -- no new
    # computation, and no dependency on `upcoming_season` (the raw schedule
    # already carries every pulled season, 2026 included). The opponent
    # profile is recomputed directly from `build_opponent_season_profile`
    # rather than read off the joined `matchup` block above: `matchup` only
    # carries a next-season row when `upcoming_season` was passed, and only
    # at the single placeholder week `append_upcoming_week` adds -- the
    # direct call always has every team's next-season row, which is what a
    # per-game model needs for every week of a projected season, not just
    # week 1.
    fixtures = vegas.select(["team", "season", "week", "opponent"])
    opponent_profile = build_opponent_season_profile(pbp, weekly_stats)
    fixtures.write_parquet(fixtures_path)
    opponent_profile.write_parquet(opponent_profile_path)
    logger.info("fixtures      %6s rows x %3s cols -> %s", fixtures.height, fixtures.width, fixtures_path)
    logger.info(
        "opp profile   %6s rows x %3s cols -> %s",
        opponent_profile.height,
        opponent_profile.width,
        opponent_profile_path,
    )

    return df, weekly_labels, season_labels
