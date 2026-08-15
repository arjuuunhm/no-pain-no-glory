"""Vegas-implied team context, at team-week grain.

From `load_schedules()`'s closing `spread_line` / `total_line`. The market's
implied team total is the single best available forecast of how many points
an offense will score, which sets the ceiling on the scoring opportunity any
individual player can claim a share of.

`spread_line` convention (verified against 2023 data): positive = home team
favored by that many points. So team_implied_total = total/2 + spread/2 for
the home side and total/2 - spread/2 for the away side.

**Team abbreviations are normalised here, and that fixes a silent bug.**
`load_schedules()` is the one loader that keeps historic franchise codes: it
carries OAK and SD where player stats and pbp carry LV and LAC. The previous
version joined on the raw column, so every Raiders week from 2016-2019 and
every Chargers week in 2016 failed to match and came through as null implied
totals -- about 1.7% of skill-position player-weeks, concentrated entirely in
two franchises, which is exactly the shape of bias that a temporal validation
split will not catch.

Unlike the rolling blocks, these need no lag: a closing line is known before
kickoff, so the week-N line is legitimately a week-N feature.
"""

from __future__ import annotations

import polars as pl

from nflforecast.config import normalize_team


def build_vegas_features(schedules: pl.DataFrame) -> pl.DataFrame:
    reg = schedules.filter(pl.col("game_type") == "REG") if "game_type" in schedules.columns else schedules

    def _side(team_col: str, opp_col: str, sign: int) -> pl.DataFrame:
        return reg.select(
            pl.col("season"),
            pl.col("week"),
            pl.col(team_col).alias("team"),
            pl.col(opp_col).alias("opponent"),
            (pl.col("total_line") / 2 + sign * pl.col("spread_line") / 2).alias("team_implied_total"),
            (pl.col("total_line") / 2 - sign * pl.col("spread_line") / 2).alias("opponent_implied_total"),
            (sign * pl.col("spread_line")).alias("spread_line"),
            pl.col("total_line"),
            pl.col("div_game"),
            pl.col("roof"),
        )

    home = _side("home_team", "away_team", 1)
    away = _side("away_team", "home_team", -1)

    return pl.concat([home, away]).with_columns(
        normalize_team("team"), normalize_team("opponent")
    )
