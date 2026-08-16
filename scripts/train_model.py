#!/usr/bin/env python3
"""Entrypoint: train the learned projector and score it against the ladder.

Usage:
    python scripts/train_model.py
    python scripts/train_model.py --rookies --with-market
    python scripts/train_model.py --blend-weight 0.35
    python scripts/train_model.py --with-post-draft      # measure the held-back columns

Run after build_features.py. Fits `model/gbm.py` on every walk-forward fold
alongside the benchmark ladder, so the model and the bar it has to clear are
scored by the same harness on the same rows -- the comparison is not a table
copied between two runs.

Prints per-fold and averaged tables, and writes every projection to
data/processed/model_predictions.parquet. That file is a superset of what
evaluate_benchmarks.py writes (same schema, more projectors); the two paths
are kept separate so re-running either does not silently truncate the other.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nflforecast.config import PROCESSED_DIR, get_logger
from nflforecast.model.benchmarks import ConsensusECR, PositionalMean, default_ladder
from nflforecast.model.evaluate import run_walk_forward, summarise
from nflforecast.model.gbm import GBMProjector, RookieGBMProjector
from nflforecast.model.panel import snapshot_features

logger = get_logger("train")

MODEL_PREDICTIONS_PATH = PROCESSED_DIR / "model_predictions.parquet"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--min-train-seasons", type=int, default=3)
    p.add_argument(
        "--rookies",
        "--include-rookies",
        dest="rookies",
        action="store_true",
        help="run a true-rookie-only evaluation using models fitted on prior rookie classes",
    )
    p.add_argument(
        "--with-post-draft",
        action="store_true",
        help="let the model see the week-1 injury report and depth chart (published after drafts)",
    )
    p.add_argument(
        "--blend-weight",
        type=float,
        default=0.0,
        help="if > 0, also score a GBM/Marcel stage-level blend at this weight on Marcel",
    )
    p.add_argument("--importances", action="store_true", help="print gain by feature per stage")
    p.add_argument(
        "--with-market",
        action="store_true",
        help="score ECR and fit a separate market-informed GBM (the default GBM remains football-only)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    # One snapshot read, shared by every model instance -- it is the same
    # frame, and re-reading it per projector is 16MB of parquet per fit.
    snapshot = snapshot_features(include_market=args.with_market)
    projector_cls = RookieGBMProjector if args.rookies else GBMProjector
    model = projector_cls(snapshot=snapshot, include_post_draft=args.with_post_draft)
    if args.rookies:
        # Historical rungs have no player prior for a rookie. The honest floor
        # is the rookie positional mean, followed by football context, then
        # market consensus and a model that explicitly consumes it.
        projectors = [PositionalMean(rookies_only=True), model]
        if args.with_market:
            projectors.extend(
                [
                    ConsensusECR(rookies_only=True),
                    RookieGBMProjector(
                        name="rookie_gbm_market",
                        snapshot=snapshot,
                        include_post_draft=args.with_post_draft,
                        include_market=True,
                    ),
                ]
            )
    else:
        projectors = [*default_ladder(include_market=args.with_market), model]
        if args.with_market:
            projectors.append(
                GBMProjector(
                    name="gbm_market",
                    snapshot=snapshot,
                    include_post_draft=args.with_post_draft,
                    include_market=True,
                )
            )
    if args.blend_weight > 0 and not args.rookies:
        projectors.append(
            GBMProjector(
                name="gbm_marcel_blend",
                snapshot=snapshot,
                include_post_draft=args.with_post_draft,
                blend_with_marcel=args.blend_weight,
            )
        )

    metrics, predictions = run_walk_forward(
        projectors=projectors,
        min_train_seasons=args.min_train_seasons,
        scoring_cohort="rookies" if args.rookies else "history",
        common_coverage=args.rookies and args.with_market,
    )

    with pl.Config(tbl_rows=-1, tbl_cols=-1, float_precision=3):
        logger.info("--- per fold: season PPR total, RMSE ---")
        print(
            metrics.filter(pl.col("target") == "points")
            .select("projector", "test_season", "rmse")
            .pivot("projector", index="test_season", values="rmse")
            .sort("test_season")
        )
        logger.info("--- leaderboard: season PPR total, averaged over folds ---")
        print(summarise(metrics, "points"))
        for target in ("ppg", "games"):
            logger.info("--- stage: %s ---", target)
            print(
                metrics.filter(pl.col("target") == target)
                .group_by("projector")
                .agg(
                    pl.col("rmse").mean(),
                    pl.col("mae").mean(),
                    pl.col("spearman").mean(),
                )
                .sort("rmse")
            )

    if args.importances:
        for stage in ("games", "tgt_pg", "car_pg"):
            logger.info("--- gain: %s (last fold) ---", stage)
            print(model.importances(stage))

    predictions.write_parquet(MODEL_PREDICTIONS_PATH)
    logger.info("wrote %s projections -> %s", predictions.height, MODEL_PREDICTIONS_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
