"""Walk-forward evaluation harness.

One loop, one contract: for each fold, fit every projector on the training
seasons and score it on the test season. Projectors never see the test season
during `fit`, which is the whole reason this loop exists as shared code rather
than as a cell in each model's script.

Two frames come out:

- **metrics** -- one row per (projector, fold, target), where target is one of
  `points` / `ppg` / `games`. Broken out per fold rather than pooled, because
  `docs/evaluation.md` is right that ~500 player-seasons per test season is not
  enough for a pooled 2% RMSE difference to mean anything, and a per-fold table
  at least makes the season-to-season spread visible without adopting the
  bootstrap that document parks.
- **predictions** -- every projection with its actual, written to parquet so
  slices (per position, top-k, quantile calibration by group) can be computed
  after the fact without re-running the folds.
"""

from __future__ import annotations

import polars as pl

from nflforecast.config import PROCESSED_DIR, get_logger
from nflforecast.model import metrics as M
from nflforecast.model.benchmarks import QUANTILES, Projector, _qcol, default_ladder
from nflforecast.model.panel import build_panel
from nflforecast.model.splits import Fold, walk_forward

logger = get_logger("evaluate")

PREDICTIONS_PATH = PROCESSED_DIR / "projection_predictions.parquet"

# (prediction column, actual column) for each stage of the §4 decomposition.
SCORED_TARGETS = {
    "points": ("pred_points", "actual_points"),
    "ppg": ("pred_ppg", "actual_ppg"),
    "games": ("pred_games", "actual_games"),
}


def run_walk_forward(
    panel: pl.DataFrame | None = None,
    projectors: list[Projector] | None = None,
    min_train_seasons: int = 3,
    require_history: bool = True,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Fit and score every projector on every fold. Returns (metrics, predictions).

    `require_history` restricts scoring to players with a prior season. Rung 1
    is undefined without one, so including rookies would score the benchmarks
    on rows where they can only emit the positional mean -- flattering the
    comparison in a way a real draft board would not. The excluded share is
    logged per fold, because that share is the part of the draft the benchmark
    ladder cannot speak to at all.
    """
    panel = build_panel() if panel is None else panel
    projectors = default_ladder() if projectors is None else projectors

    universe = panel.filter(pl.col("has_history")) if require_history else panel
    folds = walk_forward(universe["season"].unique().to_list(), min_train_seasons)
    if not folds:
        raise ValueError(
            f"no folds with >= {min_train_seasons} training seasons in "
            f"{sorted(panel['season'].unique().to_list())}"
        )

    metric_rows: list[dict] = []
    prediction_frames: list[pl.DataFrame] = []

    for fold in folds:
        train = universe.filter(pl.col("season").is_in(fold.train_seasons))
        test = universe.filter(pl.col("season") == fold.test_season)
        excluded = panel.filter(
            (pl.col("season") == fold.test_season) & ~pl.col("has_history")
        ).height
        logger.info(
            "%s | train %s, test %s (+%s rookies not scored)",
            fold,
            train.height,
            test.height,
            excluded,
        )

        for projector in projectors:
            pred = projector.fit(train).predict(test)
            scored = pred.join(
                test.select(
                    "player_id", "season", "actual_points", "actual_ppg", "actual_games"
                ),
                on=["player_id", "season"],
            ).with_columns(pl.lit(projector.name).alias("projector"))

            metric_rows.extend(_score_fold(scored, projector.name, fold))
            prediction_frames.append(scored)

    return pl.DataFrame(metric_rows), pl.concat(prediction_frames, how="diagonal")


def _score_fold(scored: pl.DataFrame, name: str, fold: Fold) -> list[dict]:
    rows = []
    for target, (pred_col, actual_col) in SCORED_TARGETS.items():
        row = {
            "projector": name,
            "test_season": fold.test_season,
            "n_train_seasons": len(fold.train_seasons),
            "target": target,
            **M.score(scored, pred_col, actual_col),
        }
        rows.append(row)

    # Calibration is only defined on the points total, which is the only
    # quantity the benchmarks emit a distribution for.
    coverage = M.quantile_coverage(
        scored, {a: _qcol(a) for a in QUANTILES if _qcol(a) in scored.columns}, "actual_points"
    )
    for row in rows:
        if row["target"] == "points":
            row.update({f"coverage_q{int(a * 100):02d}": v for a, v in coverage.items()})
    return rows


def summarise(metrics: pl.DataFrame, target: str = "points") -> pl.DataFrame:
    """Mean of each per-fold metric for one target -- a leaderboard, not a verdict.

    Averaging over folds hides the season-to-season spread that the per-fold
    table shows, so read this to rank and the full table to believe.
    """
    return (
        metrics.filter(pl.col("target") == target)
        .group_by("projector")
        .agg(
            pl.col("n").sum().alias("n"),
            pl.col("rmse").mean().alias("rmse"),
            pl.col("mae").mean().alias("mae"),
            pl.col("spearman").mean().alias("spearman"),
            *[
                pl.col(c).mean().alias(c)
                for c in metrics.columns
                if c.startswith("coverage_")
            ],
        )
        .sort("rmse")
    )
