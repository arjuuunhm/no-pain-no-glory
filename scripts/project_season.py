#!/usr/bin/env python3
"""Entrypoint: project a season that has not been played, and print a draft board.

Usage:
    python scripts/project_season.py --season 2026
    python scripts/project_season.py --season 2026 --top 60 --position RB

Requires the preseason feature table:

    python scripts/build_features.py --upcoming-season 2026

Unlike `train_model.py`, nothing here is scored -- there are no outcomes to
score against. The model is fitted on **every** completed season rather than
on a fold's training seasons, because there is no test season to hold out:
the walk-forward discipline exists to estimate how well this generalises, and
`train_model.py` has already done that. Using the same fitted-per-fold code
path with "all seasons" as the training set is what keeps the two honest.

What this cannot tell you is how good the projection is. Read
docs/modeling.md for that, and note two caveats it records:

- `roster_status` is used only to define the active-roster spine; it is not a
  feature of the learned projection.
- `--with-market` uses the archived FantasyPros ECR feature; it is a
  market-informed projection, not evidence of an edge over the market.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nflforecast.config import PROCESSED_DIR, SKILL_POSITIONS, get_logger
from nflforecast.model.benchmarks import ConsensusECR
from nflforecast.model.gbm import GBMProjector, RookieGBMProjector
from nflforecast.model.market import attach_preseason_ecr
from nflforecast.model.panel import (
    PRESEASON_FEATURES_PATH,
    build_panel,
    build_preseason_panel,
    snapshot_features,
)

logger = get_logger("project")

BOARD_PATH = PROCESSED_DIR / "draft_board.parquet"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--season", type=int, required=True, help="season to project, e.g. 2026")
    p.add_argument("--top", type=int, default=40, help="rows to print (default: 40)")
    p.add_argument("--position", choices=SKILL_POSITIONS, help="restrict the printed board")
    p.add_argument(
        "--with-post-draft",
        action="store_true",
        help="include week-1 injury/depth columns (absent for an unplayed season; see docs)",
    )
    p.add_argument(
        "--with-market",
        action="store_true",
        help="include the archived FantasyPros ECR feature in a separate market-informed projection",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    history = build_panel()
    train = history.filter(pl.col("season") < args.season)
    if train.height == 0:
        raise ValueError(f"no completed seasons before {args.season}")
    logger.info(
        "training on %s player-seasons, %s-%s",
        train.height,
        train["season"].min(),
        train["season"].max(),
    )

    target = build_preseason_panel(args.season)

    # The model joins its wide week-1 snapshot by (player_id, season), so the
    # frame it is given has to span both the training seasons and the one
    # being projected. They are the same 161 columns built by the same blocks;
    # only the spine underneath them differs.
    snapshot = pl.concat(
        [snapshot_features(include_market=args.with_market), pl.read_parquet(PRESEASON_FEATURES_PATH).drop("week")],
        how="diagonal_relaxed",
    )
    if args.with_market:
        market_cols = (
            "market_ecr",
            "market_ecr_sd",
            "market_snapshot_date",
            "market_cutoff_date",
        )
        snapshot = attach_preseason_ecr(snapshot.drop([c for c in market_cols if c in snapshot.columns]))

    model = GBMProjector(
        name="gbm_market" if args.with_market else "gbm",
        snapshot=snapshot,
        include_post_draft=args.with_post_draft,
        include_market=args.with_market,
    )
    board = model.fit(train).predict(target).with_columns(
        pl.lit("gbm_market" if args.with_market else "gbm").alias("projection_source")
    )
    rookies = target.filter(pl.col("is_rookie"))
    if rookies.height:
        rookie_model = RookieGBMProjector(
            name="rookie_gbm_market" if args.with_market else "rookie_gbm",
            snapshot=snapshot,
            include_post_draft=args.with_post_draft,
            include_market=args.with_market,
        )
        rookie_board = rookie_model.fit(train).predict(rookies).with_columns(
            pl.lit("rookie_gbm_market" if args.with_market else "rookie_gbm").alias(
                "projection_source"
            )
        )
        if args.with_market:
            # Walk-forward results show calibrated ECR beating both rookie
            # GBMs in every measured fold. Use it where available and retain
            # the football model for prospects the market source does not rank.
            rookie_ecr = ConsensusECR(rookies_only=True).fit(train).predict(rookies)
            if rookie_ecr.height:
                rookie_board = pl.concat(
                    [
                        rookie_board.join(
                            rookie_ecr.select("player_id", "season"),
                            on=["player_id", "season"],
                            how="anti",
                        ),
                        rookie_ecr.with_columns(
                            pl.lit("rookie_ecr").alias("projection_source")
                        ),
                    ],
                    how="diagonal_relaxed",
                )
        board = pl.concat(
            [
                board.join(
                    rookies.select("player_id", "season"),
                    on=["player_id", "season"],
                    how="anti",
                ),
                rookie_board,
            ],
            how="diagonal_relaxed",
        )
        logger.info(
            "used rookie projections for %s rookies (%s ECR-covered)",
            rookies.height,
            rookie_board.filter(pl.col("projection_source") == "rookie_ecr").height,
        )

    board = (
        board.join(
            target.select(
                [
                    "player_id", "player_name", "team", "age_years", "years_exp",
                    "has_history", "is_rookie", "market_ecr",
                ]
            ),
            on="player_id",
            how="left",
        )
        .sort("pred_points", descending=True)
        .with_columns(pl.int_range(1, pl.len() + 1).alias("rank"))
        .with_columns(
            pl.col("pred_points").rank(descending=True).over("position").cast(pl.Int32).alias("pos_rank")
        )
    )

    board.write_parquet(BOARD_PATH)
    logger.info("wrote %s projections -> %s", board.height, BOARD_PATH)

    shown = board.filter(pl.col("position") == args.position) if args.position else board
    with pl.Config(tbl_rows=-1, tbl_cols=-1, float_precision=1, fmt_str_lengths=22):
        print(
            shown.head(args.top).select(
                "rank",
                "pos_rank",
                "player_name",
                "team",
                "position",
                pl.col("age_years").alias("age"),
                pl.col("pred_points").alias("pts"),
                pl.col("market_ecr").alias("ecr"),
                pl.col("pred_games").alias("gms"),
                pl.col("pred_ppg").alias("ppg"),
                pl.col("pred_points_q10").alias("q10"),
                pl.col("pred_points_q90").alias("q90"),
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
