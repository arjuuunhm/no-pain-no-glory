#!/usr/bin/env python3
"""Entrypoint: score the benchmark ladder with walk-forward validation.

Usage:
    python scripts/evaluate_benchmarks.py
    python scripts/evaluate_benchmarks.py --rookies --with-market

Run after build_features.py. Prints a per-fold table and a leaderboard, and
writes every projection to data/processed/projection_predictions.parquet.

These numbers are the bar. resources.md §5: if a model cannot beat rung 1
(last season regressed to the mean), it is not ready -- and per docs/evaluation.md
a couple of percent of RMSE at ~500 player-seasons a season is noise, so read
the per-fold spread, not just the mean.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nflforecast.config import get_logger
from nflforecast.model.benchmarks import ConsensusECR, PositionalMean, default_ladder
from nflforecast.model.evaluate import PREDICTIONS_PATH, run_walk_forward, summarise

logger = get_logger("benchmarks")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--min-train-seasons",
        type=int,
        default=3,
        help="minimum panel seasons before a fold is scored (default: 3)",
    )
    p.add_argument(
        "--with-market",
        action="store_true",
        help="include the archived FantasyPros ECR benchmark where a preseason snapshot exists",
    )
    p.add_argument(
        "--rookies",
        "--include-rookies",
        dest="rookies",
        action="store_true",
        help="score only true rookies; historical-prior rungs are replaced by a rookie positional mean",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if args.rookies:
        projectors = [PositionalMean(rookies_only=True)]
        if args.with_market:
            projectors.append(ConsensusECR(rookies_only=True))
    else:
        projectors = default_ladder(include_market=args.with_market)

    metrics, predictions = run_walk_forward(
        projectors=projectors,
        min_train_seasons=args.min_train_seasons,
        scoring_cohort="rookies" if args.rookies else "history",
        common_coverage=args.rookies and args.with_market,
    )

    with pl.Config(tbl_rows=-1, tbl_cols=-1, float_precision=3):
        for target in ("points", "ppg", "games"):
            logger.info("--- per fold: %s ---", target)
            print(
                metrics.filter(pl.col("target") == target)
                .drop("target", "n_train_seasons")
                .sort(["test_season", "rmse"])
            )
        logger.info("--- leaderboard: season PPR total, averaged over folds ---")
        print(summarise(metrics, "points"))

    predictions.write_parquet(PREDICTIONS_PATH)
    logger.info("wrote %s projections -> %s", predictions.height, PREDICTIONS_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
