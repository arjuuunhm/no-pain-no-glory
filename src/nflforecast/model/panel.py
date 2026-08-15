"""The projection panel: one row per player per season, as known on draft day.

The feature table is player-*week*; a draft projection is player-*season*. This
module is the bridge, and it is the only place that decides what "known before
season N" means:

- **Pre-season state** comes from the week-1 row of the feature table. Every
  feature block is already lagged (see `features/utils.py`), so a week-1 row
  contains prior-season information and nothing else -- exactly the draft-day
  information set. Reading week 1 rather than re-deriving is what keeps this
  package honest: it inherits the leakage guarantees `validate_features.py`
  checks rather than restating them.
- **Prior production** comes from `player_season_labels.parquet`, joined at
  lags 1/2/3. Labels are targets for the season they belong to and features
  for the seasons after it; the lag shift is what separates the two roles.

A season a player was not rostered for is left null rather than zeroed. Marcel
treats a missing season as no plate appearances -- it contributes nothing to
either side of the weighted average -- and zeroing it would instead assert the
player was in the league and produced nothing, which is a different and
usually false claim.

Rookies are kept in the panel with null priors and `has_history = False`. They
are excluded from benchmark scoring by default (rung 1 is undefined without a
prior season), but dropping them at build time would hide how much of a real
draft board the benchmarks simply cannot speak to.
"""

from __future__ import annotations

import polars as pl

from nflforecast.config import PROCESSED_DIR, SKILL_POSITIONS, get_logger

logger = get_logger("panel")

PRIOR_LAGS = (1, 2, 3)

# Carried from the week-1 feature snapshot. Deliberately narrow: these are the
# columns the benchmark ladder needs (position for the regression mean, age for
# Marcel's age adjustment, draft capital for reading results on young players).
# A boosted model takes the full snapshot instead -- see `snapshot_features`.
_SNAPSHOT_COLS = [
    "team",
    "position",
    "age_years",
    "years_exp",
    "draft_round",
    "draft_pick",
    "is_undrafted",
]

_ACTUALS = {
    "season_fantasy_points_ppr": "actual_points",
    "season_games_played": "actual_games",
    "season_ppr_per_game": "actual_ppg",
}


def build_panel(
    features: pl.DataFrame | None = None,
    season_labels: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """One row per (player_id, season) rostered at week 1, with priors and actuals.

    Columns:
      keys        player_id, season, team, position
      pre-season  age_years, years_exp, draft_round, draft_pick, is_undrafted
      priors      prev{1,2,3}_{points,games,ppg}, n_prior_seasons, has_history
      actuals     actual_points, actual_games, actual_ppg
    """
    if features is None:
        features = pl.read_parquet(PROCESSED_DIR / "player_week_features.parquet")
    if season_labels is None:
        season_labels = pl.read_parquet(PROCESSED_DIR / "player_season_labels.parquet")

    # Week 1 is the draft-day snapshot. `unique` guards against a player
    # appearing on two rosters in the same week-1 (rare, but rosters_weekly
    # does carry same-week team changes).
    snapshot = (
        features.filter(pl.col("week") == 1)
        .filter(pl.col("position").is_in(SKILL_POSITIONS))
        .select(["player_id", "season", *_SNAPSHOT_COLS])
        .unique(subset=["player_id", "season"], keep="first")
    )

    actuals = season_labels.select(
        ["player_id", "season", *_ACTUALS.keys()]
    ).rename(_ACTUALS)

    panel = snapshot.join(actuals, on=["player_id", "season"], how="left")

    # A rostered player who never appeared scored zero points across zero
    # games. Points are a real zero; per-game is genuinely undefined and stays
    # null so it cannot be averaged into a positional mean.
    panel = panel.with_columns(
        pl.col("actual_points").fill_null(0.0),
        pl.col("actual_games").fill_null(0).cast(pl.Int64),
    )

    for lag in PRIOR_LAGS:
        prior = actuals.rename(
            {
                "actual_points": f"prev{lag}_points",
                "actual_games": f"prev{lag}_games",
                "actual_ppg": f"prev{lag}_ppg",
            }
        ).with_columns((pl.col("season") + lag).alias("season"))
        panel = panel.join(prior, on=["player_id", "season"], how="left")

    panel = panel.with_columns(
        sum(pl.col(f"prev{lag}_games").is_not_null().cast(pl.Int32) for lag in PRIOR_LAGS).alias(
            "n_prior_seasons"
        ),
    ).with_columns(
        pl.col("prev1_games").is_not_null().alias("has_history"),
    )

    panel = panel.sort(["season", "player_id"])
    logger.info(
        "panel: %s player-seasons, %s seasons, %.1f%% with a prior season",
        panel.height,
        panel["season"].n_unique(),
        100 * panel["has_history"].mean(),
    )
    return panel


def snapshot_features(features: pl.DataFrame | None = None) -> pl.DataFrame:
    """The full week-1 feature row per (player_id, season), for model stages.

    The benchmark ladder does not need this -- it runs off the panel's prior
    columns alone -- but the opportunity model will, and the week-1 filter is
    the piece that must not be re-invented per model.
    """
    if features is None:
        features = pl.read_parquet(PROCESSED_DIR / "player_week_features.parquet")
    return (
        features.filter(pl.col("week") == 1)
        .filter(pl.col("position").is_in(SKILL_POSITIONS))
        .unique(subset=["player_id", "season"], keep="first")
        .drop("week")
    )
