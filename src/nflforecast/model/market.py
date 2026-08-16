"""Timestamped preseason market-consensus inputs.

The checked-in source is FantasyPros expert consensus ranking (ECR), not ADP.
It is nevertheless useful market information, but is intentionally named ECR
throughout so reports never overstate what was measured.  For each season we
select the newest archived scrape no later than the calendar day before the
first regular-season game.  This is a strict, reproducible information cut:
later edits to an upstream ranking page cannot alter a historical fold.
"""

from __future__ import annotations

from datetime import timedelta
import polars as pl

from nflforecast.config import RAW_DIR, SKILL_POSITIONS, get_logger

logger = get_logger("market")

ECR_PATH = RAW_DIR / "ff_rankings.parquet"
SCHEDULES_PATH = RAW_DIR / "schedules.parquet"


def preseason_ecr(
    rankings: pl.DataFrame | None = None,
    schedules: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Return one ECR snapshot per player-season at the preseason cutoff.

    The result has one row per ``(player_id, season)``.  A missing source file
    returns an empty, typed frame so the football-only path remains runnable;
    callers can inspect ``market_ecr`` null coverage rather than accidentally
    substituting a current ranking for historical data.
    """
    schema = {
        "player_id": pl.String,
        "season": pl.Int32,
        "market_ecr": pl.Float64,
        "market_ecr_sd": pl.Float64,
        "market_snapshot_date": pl.Date,
        "market_cutoff_date": pl.Date,
    }
    if rankings is None:
        if not ECR_PATH.exists() or not SCHEDULES_PATH.exists():
            logger.info("no archived ECR/schedule input; market consensus unavailable")
            return pl.DataFrame(schema=schema)
        rankings = pl.read_parquet(ECR_PATH)
    if schedules is None:
        if not SCHEDULES_PATH.exists():
            return pl.DataFrame(schema=schema)
        schedules = pl.read_parquet(SCHEDULES_PATH)

    required = {"player_id", "season", "scrape_date", "position", "ecr"}
    missing = required - set(rankings.columns)
    if missing:
        raise ValueError(f"ECR input missing columns: {sorted(missing)}")

    starts = (
        schedules.filter((pl.col("game_type") == "REG") & (pl.col("week") == 1))
        .group_by("season")
        .agg(
            pl.col("gameday").str.to_date(strict=False).min().alias("first_game_date"),
        )
        .with_columns(pl.col("season").cast(pl.Int32))
        .drop_nulls("first_game_date")
        .with_columns((pl.col("first_game_date") - timedelta(days=1)).alias("market_cutoff_date"))
        .select("season", "market_cutoff_date")
    )
    candidates = (
        rankings.with_columns(
            pl.col("season").cast(pl.Int32),
            pl.col("scrape_date").cast(pl.Date, strict=False),
            pl.col("ecr").cast(pl.Float64, strict=False),
        )
        .filter(pl.col("position").is_in(SKILL_POSITIONS))
        .join(starts, on="season", how="inner")
        .filter((pl.col("scrape_date") <= pl.col("market_cutoff_date")) & (pl.col("ecr") > 0))
    )
    if candidates.height == 0:
        return pl.DataFrame(schema=schema)

    latest = candidates.group_by("season").agg(pl.col("scrape_date").max())
    return (
        candidates.join(latest, on=["season", "scrape_date"], how="inner")
        .select(
            "player_id",
            "season",
            pl.col("ecr").alias("market_ecr"),
            pl.col("sd").cast(pl.Float64, strict=False).alias("market_ecr_sd")
            if "sd" in candidates.columns
            else pl.lit(None, dtype=pl.Float64).alias("market_ecr_sd"),
            pl.col("scrape_date").alias("market_snapshot_date"),
            "market_cutoff_date",
        )
        .unique(subset=["player_id", "season"], keep="first")
        .cast(schema)
    )


def attach_preseason_ecr(frame: pl.DataFrame) -> pl.DataFrame:
    """Left-join selected ECR onto any player-season frame."""
    ecr = preseason_ecr()
    if ecr.height == 0:
        return frame.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("market_ecr"),
            pl.lit(None, dtype=pl.Float64).alias("market_ecr_sd"),
            pl.lit(None, dtype=pl.Date).alias("market_snapshot_date"),
            pl.lit(None, dtype=pl.Date).alias("market_cutoff_date"),
        )
    return frame.join(ecr, on=["player_id", "season"], how="left")
