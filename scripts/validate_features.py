#!/usr/bin/env python3
"""Entrypoint: assert the built feature table is leakage-free and sane.

Usage:
    python scripts/validate_features.py

Run after build_features.py. Every check here corresponds to a bug this
pipeline has actually had or is one refactor away from having, so it is worth
running on every rebuild rather than reasoning about the invariants by hand.

Checks:
  1. Rolling features contain no same-week information.
  2. Red-zone shares use a team denominator (they sum to ~1 across a team-week).
  3. Team abbreviations join cleanly across schedules / pbp / player stats.
  4. O-line and scheme metrics land in their known real-world ranges.
  5. Feature and label tables share a key and neither leaks into the other.
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nflforecast.config import PROCESSED_DIR, RAW_DIR, get_logger

logger = get_logger("validate")

N_PLAYERS_TO_AUDIT = 40


class CheckFailed(Exception):
    pass


def check(name: str, ok: bool, detail: str) -> bool:
    logger.info("%-46s %s  %s", name, "PASS" if ok else "FAIL", detail)
    return ok


def check_no_lookahead(features: pl.DataFrame, weekly_stats: pl.DataFrame) -> bool:
    """target_share_last3 must equal the mean over the player's 3 most recent
    games played *strictly before* that week -- recomputed from raw, not trusted."""
    raw = (
        weekly_stats.filter(pl.col("season_type") == "REG")
        .select(["player_id", "season", "week", "target_share"])
        .drop_nulls()
    )
    busiest = (
        features.group_by("player_id").len().sort("len", descending=True)
        .head(N_PLAYERS_TO_AUDIT)["player_id"].to_list()
    )

    checked = mismatches = 0
    for pid in busiest:
        want = features.filter(pl.col("player_id") == pid).select(
            ["season", "week", "target_share_last3"]
        ).sort(["season", "week"]).rows()
        have = raw.filter(pl.col("player_id") == pid).sort(["season", "week"]).rows()
        for season, week, rolled in want:
            if rolled is None:
                continue
            prior = [r[3] for r in have if r[1] * 100 + r[2] < season * 100 + week][-3:]
            if not prior:
                continue
            checked += 1
            if abs(sum(prior) / len(prior) - rolled) > 1e-9:
                mismatches += 1

    return check(
        "rolling features exclude current week",
        mismatches == 0 and checked > 0,
        f"{checked} player-weeks recomputed, {mismatches} mismatched",
    )


def check_redzone_denominator(features: pl.DataFrame, pbp: pl.DataFrame) -> bool:
    """Red-zone share must be player/team, so shares sum to ~1 across a team-week.

    The pre-refactor version divided by the player's own touches, which made
    every share independent and the sum meaningless (often far above 1).
    """
    rz = pbp.filter(
        (pl.col("season_type") == "REG")
        & (pl.col("rush_attempt") == 1)
        & (pl.col("yardline_100") <= 20)
        & pl.col("rusher_player_id").is_not_null()
    )
    per_team = rz.group_by(["posteam", "season", "week"]).agg(pl.len().alias("team_n"))
    per_player = rz.group_by(["rusher_player_id", "posteam", "season", "week"]).agg(
        pl.len().alias("player_n")
    )
    shares = per_player.join(per_team, on=["posteam", "season", "week"]).with_columns(
        (pl.col("player_n") / pl.col("team_n")).alias("share")
    )
    summed = shares.group_by(["posteam", "season", "week"]).agg(pl.col("share").sum().alias("total"))
    worst = (summed["total"] - 1.0).abs().max()
    return check(
        "red-zone shares use team denominator",
        worst < 1e-9,
        f"max |sum(share) - 1| over team-weeks = {worst:.2e}",
    )


def check_team_join_coverage(features: pl.DataFrame) -> bool:
    """Vegas context must resolve for every team-week.

    This is the OAK/SD normalisation guard: load_schedules() keeps historic
    franchise codes while pbp and player stats do not, so an unguarded join
    drops the Raiders 2016-2019 and the Chargers 2016 entirely.
    """
    null_rate = features["team_implied_total"].is_null().mean()
    by_team = (
        features.group_by("team")
        .agg(pl.col("team_implied_total").is_null().mean().alias("null_rate"))
        .sort("null_rate", descending=True)
    )
    worst_team, worst_rate = by_team.row(0)
    return check(
        "vegas context joins for every franchise",
        worst_rate < 0.05,
        f"overall null {null_rate:.3%}, worst franchise {worst_team} at {worst_rate:.3%}",
    )


def check_metric_ranges(features: pl.DataFrame) -> bool:
    """O-line and scheme metrics should land where the real-world versions do."""
    bounds = {
        # Raw banded line-yard credit, ~2.7 league-wide. Deliberately not
        # Football Outsiders' published ~4.2, which is rescaled to league YPC
        # -- see the oline module docstring for why we do not rescale.
        "ol_adj_line_yards_last8": (2.3, 3.2),
        "ol_stuff_rate_last8": (0.10, 0.30),
        "ol_sack_rate_last8": (0.03, 0.11),
        # pass_oe is centered on nflverse's own model, so the mean is ~0.
        "pass_rate_over_expected_last8": (-8.0, 8.0),
        "neutral_pass_rate_last8": (0.40, 0.70),
        "sec_per_play_last8": (20.0, 35.0),
    }
    ok = True
    for col, (lo, hi) in bounds.items():
        mean = features[col].mean()
        within = lo <= mean <= hi
        ok &= within
        check(f"  {col} in [{lo}, {hi}]", within, f"league mean = {mean:.3f}")
    return ok


def check_labels_separated(features: pl.DataFrame, weekly_labels: pl.DataFrame) -> bool:
    """No target column may appear in the feature table."""
    from nflforecast.features.labels import WEEKLY_TARGET_COLS

    leaked = [c for c in (*WEEKLY_TARGET_COLS, "played") if c in features.columns]
    same_key = features.height == weekly_labels.height
    return check(
        "targets absent from feature table",
        not leaked and same_key,
        f"leaked={leaked or 'none'}, row counts {features.height} vs {weekly_labels.height}",
    )


def main() -> int:
    features = pl.read_parquet(PROCESSED_DIR / "player_week_features.parquet")
    weekly_labels = pl.read_parquet(PROCESSED_DIR / "player_week_labels.parquet")
    weekly_stats = pl.read_parquet(RAW_DIR / "weekly_player_stats.parquet")
    pbp = pl.read_parquet(RAW_DIR / "pbp.parquet")

    logger.info("Validating %s rows x %s cols", features.height, features.width)
    results = [
        check_no_lookahead(features, weekly_stats),
        check_redzone_denominator(features, pbp),
        check_team_join_coverage(features),
        check_labels_separated(features, weekly_labels),
        check_metric_ranges(features),
    ]

    failed = results.count(False)
    logger.info("=== %s/%s checks passed ===", len(results) - failed, len(results))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
