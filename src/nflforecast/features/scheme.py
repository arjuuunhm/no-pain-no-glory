"""Offensive scheme and play-caller features, at team-week and team-season grain.

Two questions this block answers about the offense a player sits in: *how
much opportunity does this system generate*, and *is the person generating
it the same one as last year*.

**Scheme tendencies** (team-week, then rolled). These are the levers that set
the size of the opportunity pool before any player-level share is applied:

- `pass_rate_over_expected` -- nflverse's `pass_oe`, i.e. actual pass rate
  minus its expected-pass model, in percentage points. The single best
  one-number description of a play-caller's aggression, because it already
  controls for down/distance/score.
- `neutral_pass_rate` -- raw pass rate on 1st/2nd down, quarters 1-3, within
  one score. The pre-model version of the same idea; kept alongside PROE
  because it is interpretable and does not inherit nflverse's model error.
- `plays_per_game`, `sec_per_play` -- pace. `sec_per_play` is measured
  *within a drive* (consecutive snaps of the same possession), so it is real
  tempo rather than an artifact of how many possessions the game had.
- `shotgun_rate`, `no_huddle_rate` -- formation/tempo identity.
- `rz_trips_per_game` -- how often the system reaches scoring range, which
  is what converts opportunity into touchdown equity.
- `team_pass_epa`, `team_rush_epa` -- how well the system executes, split by
  phase.

**Play-caller identity and continuity.** This is where the data runs out and
the module has to be honest about it: *nflverse has no offensive-coordinator
field*. `load_schedules()` carries `home_coach`/`away_coach`, head coach
only. So the default play-caller identity here is the head coach, which is
right for the many teams where the HC calls plays and wrong for the rest.

`load_coordinator_map()` reads an optional, hand-maintained
`data/manual/coordinators.csv` (`team,season,offensive_coordinator`) and
prefers it when present, falling back to head coach per row where it is not.
Absent that file the features still build; `play_caller` is simply the head
coach everywhere.

`play_caller_is_new` is the feature that actually earns its keep: a team
whose caller changed carries far less of its prior-season tendency into the
current one, and it lets a model discount the lagged `prev_season_*` columns
instead of trusting them uniformly.

**Grain and leakage.** Both halves ship two views:
- `_last{3,5,8}` -- in-season form, shift(1)-lagged like every other rolling
  block, for "what has this offense been doing lately".
- `prev_season_*` -- the team's *completed prior season*, which is the
  play-caller track record proper. It is constant within a season and known
  before week 1, which is exactly what a preseason draft-value model needs
  and what the in-season rolling columns cannot supply for early weeks.
"""

from __future__ import annotations

import polars as pl

from nflforecast.config import MANUAL_DIR, ROLLING_WINDOWS, get_logger, normalize_team
from nflforecast.features.utils import add_trailing_rolling

logger = get_logger(__name__)

COORDINATOR_CSV = MANUAL_DIR / "coordinators.csv"

TENDENCY_COLS = [
    "pass_rate_over_expected",
    "neutral_pass_rate",
    "plays_per_game",
    "sec_per_play",
    "shotgun_rate",
    "no_huddle_rate",
    "rz_trips_per_game",
    "team_pass_epa",
    "team_rush_epa",
]

# "Neutral" script: early down, first three quarters, within one score. Filters
# out the garbage-time and two-minute snaps that otherwise dominate raw
# pass-rate and pace numbers.
_NEUTRAL = (
    pl.col("down").is_in([1, 2])
    & (pl.col("qtr") <= 3)
    & (pl.col("score_differential").abs() <= 7)
)


def _team_week_tendencies(pbp: pl.DataFrame) -> pl.DataFrame:
    reg = pbp.filter(
        (pl.col("season_type") == "REG")
        & pl.col("posteam").is_not_null()
        & pl.col("play_type").is_in(["pass", "run"])
    ).with_columns(normalize_team("posteam"))

    # Within-drive seconds between consecutive snaps = tempo. Ordering by
    # descending clock inside a drive gives the true snap order; the first
    # snap of each drive has no predecessor and drops out via the null diff.
    reg = reg.sort(["game_id", "posteam", "fixed_drive"], descending=[False, False, False]).sort(
        ["game_id", "posteam", "fixed_drive", "game_seconds_remaining"], descending=[False, False, False, True]
    )
    reg = reg.with_columns(
        (
            pl.col("game_seconds_remaining").shift(1) - pl.col("game_seconds_remaining")
        ).over(["game_id", "posteam", "fixed_drive"]).alias("_sec_elapsed")
    )

    return reg.group_by(["posteam", "season", "week"]).agg(
        pl.col("pass_oe").mean().alias("pass_rate_over_expected"),
        pl.col("pass").filter(_NEUTRAL).mean().alias("neutral_pass_rate"),
        pl.len().alias("plays_per_game"),
        # Clamp to plausible play-clock range: negative values are ordering
        # artifacts at drive seams and huge ones span timeouts/reviews.
        pl.col("_sec_elapsed").filter(_NEUTRAL & pl.col("_sec_elapsed").is_between(1, 60))
        .mean()
        .alias("sec_per_play"),
        pl.col("shotgun").mean().alias("shotgun_rate"),
        pl.col("no_huddle").mean().alias("no_huddle_rate"),
        pl.col("fixed_drive").filter(pl.col("yardline_100") <= 20).n_unique().alias("rz_trips_per_game"),
        pl.col("epa").filter(pl.col("pass") == 1).mean().alias("team_pass_epa"),
        pl.col("epa").filter(pl.col("rush_attempt") == 1).mean().alias("team_rush_epa"),
    ).rename({"posteam": "team"})


def load_coordinator_map() -> pl.DataFrame | None:
    """Optional hand-maintained team-season -> offensive coordinator crosswalk."""
    if not COORDINATOR_CSV.exists():
        logger.info(
            "No %s -- play_caller falls back to head coach (nflverse has no OC field)",
            COORDINATOR_CSV.name,
        )
        return None
    df = pl.read_csv(COORDINATOR_CSV).with_columns(normalize_team("team"))
    logger.info("Loaded coordinator map: %s team-seasons", df.height)
    return df.select(["team", "season", "offensive_coordinator"])


def build_play_caller_features(schedules: pl.DataFrame) -> pl.DataFrame:
    """Team-season play-caller identity plus a changed-since-last-year flag."""
    reg = schedules.filter(pl.col("game_type") == "REG")
    long = pl.concat(
        [
            reg.select(
                "season",
                pl.col("home_team").alias("team"),
                pl.col("home_coach").alias("head_coach"),
            ),
            reg.select(
                "season",
                pl.col("away_team").alias("team"),
                pl.col("away_coach").alias("head_coach"),
            ),
        ]
    ).with_columns(normalize_team("team"))

    # Modal coach for the season, so a late-season interim does not rewrite
    # the team's whole-season identity.
    coach = (
        long.drop_nulls("head_coach")
        .group_by(["team", "season", "head_coach"])
        .agg(pl.len().alias("n"))
        .sort(["team", "season", "n"], descending=[False, False, True])
        .unique(subset=["team", "season"], keep="first", maintain_order=True)
        .drop("n")
    )

    coordinators = load_coordinator_map()
    if coordinators is not None:
        coach = coach.join(coordinators, on=["team", "season"], how="left")
    else:
        coach = coach.with_columns(pl.lit(None, dtype=pl.Utf8).alias("offensive_coordinator"))

    coach = coach.with_columns(
        pl.coalesce("offensive_coordinator", "head_coach").alias("play_caller")
    )

    return coach.sort(["team", "season"]).with_columns(
        (pl.col("play_caller") != pl.col("play_caller").shift(1).over("team"))
        .fill_null(True)  # first observed season: treat as new, we have no prior
        .alias("play_caller_is_new"),
        (pl.col("head_coach") != pl.col("head_coach").shift(1).over("team"))
        .fill_null(True)
        .alias("head_coach_is_new"),
    ).select(
        ["team", "season", "head_coach", "play_caller", "play_caller_is_new", "head_coach_is_new"]
    )


def build_scheme_features(pbp: pl.DataFrame, schedules: pl.DataFrame) -> pl.DataFrame:
    """Team-week scheme features: rolled in-season form + prior-season track record."""
    weekly = _team_week_tendencies(pbp).sort(["team", "season", "week"])

    rolled = add_trailing_rolling(
        weekly, group_col="team", order_cols=["season", "week"],
        value_cols=TENDENCY_COLS, windows=list(ROLLING_WINDOWS),
    )

    # Prior-season track record: aggregate the team's completed season, then
    # attach it to the *following* season. Uses a season+1 join key rather than
    # shift() so a team missing from a season cannot silently misalign.
    season_level = (
        weekly.group_by(["team", "season"])
        .agg([pl.col(c).mean().alias(f"prev_season_{c}") for c in TENDENCY_COLS])
        .with_columns((pl.col("season") + 1).alias("season"))
    )

    df = rolled.drop(TENDENCY_COLS).join(season_level, on=["team", "season"], how="left")
    return df.join(build_play_caller_features(schedules), on=["team", "season"], how="left")
