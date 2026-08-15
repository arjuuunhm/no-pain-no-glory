"""Shared helpers for lag-safe rolling / as-of feature construction.

docs/features.md §7 flags leakage from un-lagged season-to-date shares as the
single most important pitfall to avoid: a feature for week n must be computed
from weeks < n only. Everything here enforces that, by one of two routes
depending on the grain:

- **Team grain** (`add_trailing_rolling`): teams play every week, so
  shift-by-one-then-roll is exact and cheap.
- **Player grain** (`add_inclusive_rolling` + `attach_asof`): players miss
  games, so their rows are not a dense weekly series. Rolling is computed
  *inclusive* of each played game, and the strictly-prior row is then
  attached to the target week by an as-of join. This is the piece the
  previous shift-then-roll version got wrong for injured players: it could
  only produce a feature for weeks the player actually played, so an
  availability model had nothing to score on the weeks that mattered most.
"""

from __future__ import annotations

import polars as pl

# Monotonic within- and across-season week key. Weeks never exceed 99, so
# season*100 + week orders correctly and increments by 1 within a season.
WEEK_INDEX = "wk_idx"


def add_week_index(df: pl.DataFrame, season_col: str = "season", week_col: str = "week") -> pl.DataFrame:
    return df.with_columns(
        (pl.col(season_col).cast(pl.Int64) * 100 + pl.col(week_col).cast(pl.Int64)).alias(WEEK_INDEX)
    )


def add_trailing_rolling(
    df: pl.DataFrame,
    group_col: str,
    order_cols: list[str],
    value_cols: list[str],
    windows: list[int],
) -> pl.DataFrame:
    """Add shift(1)-then-rolling-mean columns, named f"{col}_last{w}".

    For dense series only -- one row per period per group, no gaps. Use it for
    team-week features, where that holds; use add_inclusive_rolling +
    attach_asof for player features, where it does not.

    Windows are not reset at season boundaries: a team's "last 3 games" can
    span the prior season, which is the intended Marcel-style continuity
    (docs/features.md §7).
    """
    df = df.sort([group_col, *order_cols])
    exprs = []
    for col in value_cols:
        shifted = pl.col(col).shift(1).over(group_col)
        for w in windows:
            exprs.append(
                shifted.rolling_mean(window_size=w, min_samples=1).over(group_col).alias(f"{col}_last{w}")
            )
    return df.with_columns(exprs)


def add_inclusive_rolling(
    df: pl.DataFrame,
    group_col: str,
    order_cols: list[str],
    value_cols: list[str],
    windows: list[int],
) -> pl.DataFrame:
    """Add rolling means *including* the current row, named f"{col}_last{w}".

    Inclusive is deliberate and is only safe in combination with attach_asof,
    which is what enforces the strictly-before-this-week cutoff. Rolling here
    is over the player's last N *games played*, so a player returning from a
    four-week absence carries the form he had before it rather than a gap.
    """
    df = df.sort([group_col, *order_cols])
    exprs = [
        pl.col(col).rolling_mean(window_size=w, min_samples=1).over(group_col).alias(f"{col}_last{w}")
        for col in value_cols
        for w in windows
    ]
    return df.with_columns(exprs)


def add_season_to_date(
    df: pl.DataFrame,
    group_col: str,
    season_col: str,
    order_cols: list[str],
    value_cols: list[str],
) -> pl.DataFrame:
    """Add cumulative-mean-through-current-row columns, named f"{col}_std", reset each season.

    Inclusive for the same reason as add_inclusive_rolling: the cutoff is
    applied by attach_asof, not here.
    """
    df = df.sort([group_col, season_col, *order_cols])
    group_keys = [group_col, season_col]
    exprs = [
        pl.col(col).cum_sum().over(group_keys).alias(f"__{col}_cumsum") for col in value_cols
    ]
    exprs.append(pl.int_range(1, pl.len() + 1).over(group_keys).alias("__n"))
    df = df.with_columns(exprs)
    df = df.with_columns(
        [(pl.col(f"__{col}_cumsum") / pl.col("__n")).alias(f"{col}_std") for col in value_cols]
    )
    return df.drop([c for c in df.columns if c.startswith("__")])


def attach_asof(
    spine: pl.DataFrame,
    features: pl.DataFrame,
    by: str = "player_id",
    feature_cols: list[str] | None = None,
) -> pl.DataFrame:
    """Attach each spine row the most recent feature row *strictly before* its week.

    Both frames must carry `by`, `season`, and `week`. The strictness is what
    makes this leakage-safe: looking up at (this week - 1) means a row can
    never see a feature computed from its own week's outcome, and a player who
    did not play still receives his last known form rather than a null.
    """
    feature_cols = feature_cols or [
        c for c in features.columns if c not in (by, "season", "week", WEEK_INDEX)
    ]

    left = add_week_index(spine).with_columns((pl.col(WEEK_INDEX) - 1).alias("__lookup"))
    right = add_week_index(features).select([by, WEEK_INDEX, *feature_cols])

    left = left.sort("__lookup")
    right = right.sort(WEEK_INDEX)

    joined = left.join_asof(
        right, left_on="__lookup", right_on=WEEK_INDEX, by=by, strategy="backward"
    )
    # join_asof echoes the right-hand key back as `<key>_right`; drop both it
    # and our scratch columns so repeated attaches stay idempotent.
    scratch = ("__lookup", WEEK_INDEX, f"{WEEK_INDEX}_right")
    return joined.drop([c for c in scratch if c in joined.columns])
