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

- Before the late-August cut deadline, `roster_status` is ~99% ACT and carries
  almost none of the availability signal it carries in the backtest. Rebuild
  and re-run after cuts.
- There is no ADP benchmark yet (resources.md §5 rung 3), so nothing here
  establishes an edge over the market -- only over Marcel.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nflforecast.config import PROCESSED_DIR, SKILL_POSITIONS, get_logger
from nflforecast.model.gbm import GBMProjector
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
        [snapshot_features(), pl.read_parquet(PRESEASON_FEATURES_PATH).drop("week")],
        how="diagonal_relaxed",
    )

    model = GBMProjector(snapshot=snapshot, include_post_draft=args.with_post_draft)
    board = model.fit(train).predict(target)

    board = (
        board.join(
            target.select(
                ["player_id", "player_name", "team", "age_years", "years_exp", "has_history"]
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
                pl.col("pred_games").alias("gms"),
                pl.col("pred_ppg").alias("ppg"),
                pl.col("pred_points_q10").alias("q10"),
                pl.col("pred_points_q90").alias("q90"),
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
