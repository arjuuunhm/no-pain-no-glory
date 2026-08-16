"""Opponent-defence profile -- the block that makes a per-game model worth building.

Every other feature block describes the player or the offence he sits in, and
those are constant across a season on draft day. This one is the only thing
that *varies week to week* in a draft-day information set, because the schedule
is published in May: for a 2026 projection we know all 272 regular-season
games, so we know exactly which 17 defences each player faces.

That is the whole argument for the per-game grain. A season projection built as
a sum over scheduled games can say "this back faces the four worst run defences
in the league in weeks 3-6"; a season-grain model cannot say anything of the
kind, because the schedule has nowhere to enter.

**Opportunity allowed, not points allowed.** Conventional "defence vs position"
tables rank defences by fantasy points surrendered, which is mostly touchdown
noise and mostly an efficiency statement. This block deliberately measures the
*volume* a defence concedes -- plays faced, dropbacks faced, carries and targets
by position -- because opportunity is what the model predicts and what actually
carries year to year. Efficiency allowed (EPA, success rate) is here too, but as
context for the volume rather than as the headline.

The mechanism the volume columns are picking up is game script and funnelling:

- A defence that cannot stop the run sees *fewer* pass attempts, because
  offences that are ahead run out the clock. `opp_funnel_index` states this
  directly as pass EPA allowed minus rush EPA allowed -- positive means the
  defence is worse against the pass than the run, so offences throw on it.
- A defence whose offence scores constantly forces its opponents to throw,
  independent of how good the secondary is. `opp_score_diff_faced` carries that.
- Pace compounds: a fast defence-facing environment means more total snaps for
  every skill player on the field, before any share is applied.

**Two views, and only one of them is legal on draft day.** Like `scheme.py`,
this ships rolling in-season form (`_last{3,5,8}`) and a completed prior season
(`prev_season_opp_*`). The rolling columns are shift(1)-lagged and therefore
safe *for a week-N model*, but they are not safe for a draft-day season
projection: the week-10 window reads weeks 1-9 of the season being projected.
`model/panel.py` therefore reads `build_opponent_season_profile` -- the
prior-season view, which is a constant known before week 1 -- and never touches
the rolling columns. Keeping both in the feature table is deliberate; the
in-season model is the obvious next thing to build and it should not have to
re-derive this block.

**The known weakness**, stated up front: a defence's prior season is a mediocre
forecast of its current one. Units turn over, and the correlation is real but
loose. That is an argument for shrinking the matchup effect, not for dropping
it -- and shrinkage is what a GBDT does natively when a feature is weak, which
is why these are handed over raw rather than pre-adjusted.
"""

from __future__ import annotations

import polars as pl

from nflforecast.config import ROLLING_WINDOWS, SKILL_POSITIONS, normalize_team
from nflforecast.features.utils import add_trailing_rolling, append_upcoming_week

# Every metric this block produces, at defence-week grain, before rolling. The
# `opp_` prefix is applied here rather than at attach time because these are
# only ever read as "the defence on the other side of the ball" -- there is no
# view of the pipeline in which a team consumes its own row from this block.
PROFILE_COLS = [
    # -- volume conceded: the pool a player's share is taken from -------
    "opp_plays_faced",
    "opp_dropbacks_faced",
    "opp_rush_att_faced",
    "opp_neutral_pass_rate_faced",
    "opp_rz_plays_faced",
    # -- efficiency conceded: context for the volume --------------------
    "opp_pass_epa_allowed",
    "opp_rush_epa_allowed",
    "opp_pass_sr_allowed",
    "opp_rush_sr_allowed",
    "opp_sack_rate",
    "opp_funnel_index",
    # -- game script: why the volume was what it was --------------------
    "opp_score_diff_faced",
    # -- opportunity conceded by position: DvP in volume terms ----------
    "opp_tgt_allowed_rb",
    "opp_tgt_allowed_wr",
    "opp_tgt_allowed_te",
    "opp_car_allowed_rb",
    "opp_air_yards_allowed",
]

# Early down, first three quarters, within one score. Same definition as
# scheme.py's, and deliberately identical: `opp_neutral_pass_rate_faced` is
# meant to be read against the offence's own `neutral_pass_rate`, and two
# different neutral-script filters would make that comparison meaningless.
_NEUTRAL = (
    pl.col("down").is_in([1, 2])
    & (pl.col("qtr") <= 3)
    & (pl.col("score_differential").abs() <= 7)
)


def _defence_week_from_pbp(pbp: pl.DataFrame) -> pl.DataFrame:
    """Play-by-play half of the profile, one row per defence per week."""
    reg = pbp.filter(
        (pl.col("season_type") == "REG")
        & pl.col("defteam").is_not_null()
        & pl.col("play_type").is_in(["pass", "run"])
    ).with_columns(normalize_team("defteam"))

    return (
        reg.group_by(["defteam", "season", "week"])
        .agg(
            pl.len().alias("opp_plays_faced"),
            pl.col("qb_dropback").sum().alias("opp_dropbacks_faced"),
            pl.col("rush_attempt").sum().alias("opp_rush_att_faced"),
            pl.col("pass").filter(_NEUTRAL).mean().alias("opp_neutral_pass_rate_faced"),
            pl.col("yardline_100").le(20).sum().alias("opp_rz_plays_faced"),
            pl.col("epa").filter(pl.col("pass") == 1).mean().alias("opp_pass_epa_allowed"),
            pl.col("epa").filter(pl.col("rush_attempt") == 1).mean().alias("opp_rush_epa_allowed"),
            pl.col("success").filter(pl.col("pass") == 1).mean().alias("opp_pass_sr_allowed"),
            pl.col("success").filter(pl.col("rush_attempt") == 1).mean().alias("opp_rush_sr_allowed"),
            pl.col("sack").filter(pl.col("qb_dropback") == 1).mean().alias("opp_sack_rate"),
            # From the *offence's* perspective, so a positive mean means this
            # defence spent the game being led against -- the state that makes
            # opponents run rather than throw.
            pl.col("score_differential").mean().alias("opp_score_diff_faced"),
        )
        .with_columns(
            (pl.col("opp_pass_epa_allowed") - pl.col("opp_rush_epa_allowed")).alias(
                "opp_funnel_index"
            )
        )
        .rename({"defteam": "opponent"})
    )


def _defence_week_from_stats(weekly_stats: pl.DataFrame) -> pl.DataFrame:
    """Position-split opportunity conceded, from the weekly stat lines.

    Play-by-play carries the receiver's id but not his position, so the split
    by RB/WR/TE comes from `weekly_player_stats` instead, which carries both
    `position` and `opponent_team`. Summing there rather than joining a
    position onto pbp keeps this a single group-by and avoids a crosswalk that
    would silently drop players whose position changed mid-career.
    """
    stats = weekly_stats.filter(
        (pl.col("season_type") == "REG")
        & pl.col("opponent_team").is_not_null()
        & pl.col("position").is_in(SKILL_POSITIONS)
    ).with_columns(normalize_team("opponent_team"))

    per_position = [
        pl.col("targets").filter(pl.col("position") == pos).sum().alias(f"opp_tgt_allowed_{pos.lower()}")
        for pos in SKILL_POSITIONS
    ]

    return (
        stats.group_by(["opponent_team", "season", "week"])
        .agg(
            *per_position,
            pl.col("carries").filter(pl.col("position") == "RB").sum().alias("opp_car_allowed_rb"),
            pl.col("receiving_air_yards").sum().alias("opp_air_yards_allowed"),
        )
        .rename({"opponent_team": "opponent"})
    )


def _defence_week_profile(pbp: pl.DataFrame, weekly_stats: pl.DataFrame) -> pl.DataFrame:
    """One row per (opponent, season, week) with every `PROFILE_COLS` metric."""
    profile = _defence_week_from_pbp(pbp).join(
        _defence_week_from_stats(weekly_stats),
        on=["opponent", "season", "week"],
        how="left",
    )
    return profile.select(["opponent", "season", "week", *PROFILE_COLS]).sort(
        ["opponent", "season", "week"]
    )


def build_opponent_season_profile(
    pbp: pl.DataFrame, weekly_stats: pl.DataFrame
) -> pl.DataFrame:
    """The draft-day view: a defence's completed prior season, keyed to the next one.

    One row per `(opponent, season)`, carrying `prev_season_opp_*` -- the
    per-game average of everything the defence conceded in season N-1, attached
    to season N. This is the only matchup view `model/panel.py` is allowed to
    read, because it is the only one whose value is fixed before week 1.

    Keyed by `season + 1` rather than by `shift()`, for the same reason every
    other block in this package is: a franchise missing from a season would
    otherwise pick up a stale row two years old and nothing would flag it.
    """
    weekly = _defence_week_profile(pbp, weekly_stats)
    return (
        weekly.group_by(["opponent", "season"])
        .agg([pl.col(c).mean().alias(f"prev_season_{c}") for c in PROFILE_COLS])
        .with_columns((pl.col("season") + 1).alias("season"))
        .sort(["opponent", "season"])
    )


def build_matchup_features(
    pbp: pl.DataFrame,
    weekly_stats: pl.DataFrame,
    upcoming_season: int | None = None,
) -> pl.DataFrame:
    """Team-week matchup block: rolled in-season form plus the prior-season view.

    Joined onto the player-week spine on `(opponent, season, week)`, which
    requires `vegas.py` to have attached `opponent` first -- the schedule is the
    only place the fixture list lives.

    The rolling columns are shift(1)-lagged and safe at week N. They are *not*
    safe for a draft-day season projection and the panel does not read them; see
    the module docstring.
    """
    weekly = _defence_week_profile(pbp, weekly_stats)
    season_level = build_opponent_season_profile(pbp, weekly_stats)

    if upcoming_season is not None:
        weekly = append_upcoming_week(weekly, "opponent", upcoming_season)

    rolled = add_trailing_rolling(
        weekly,
        group_col="opponent",
        order_cols=["season", "week"],
        value_cols=PROFILE_COLS,
        windows=list(ROLLING_WINDOWS),
    )

    return rolled.drop(PROFILE_COLS).join(season_level, on=["opponent", "season"], how="left")
