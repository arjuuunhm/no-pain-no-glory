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
    scoring_cohort: str | None = None,
    common_coverage: bool = False,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Fit and score every projector on every fold. Returns (metrics, predictions).

    ``scoring_cohort`` is one of ``history`` (the default), ``rookies``, or
    ``all``. ``require_history`` remains as a compatibility alias when no
    explicit cohort is supplied. ``common_coverage`` intersects player-season
    keys emitted by every projector within a fold; use it for a fair market
    comparison because ECR does not rank every rostered rookie.
    """
    panel = build_panel() if panel is None else panel
    projectors = default_ladder() if projectors is None else projectors

    # Cohorts restrict only the test rows. All completed player-seasons remain
    # legitimate training outcomes; rookie-specific projectors apply their own
    # training filter, while general models can learn from the whole panel.
    cohort = scoring_cohort or ("history" if require_history else "all")
    scoring_universe = select_scoring_universe(panel, cohort)
    folds = walk_forward(panel["season"].unique().to_list(), min_train_seasons)
    if not folds:
        raise ValueError(
            f"no folds with >= {min_train_seasons} training seasons in "
            f"{sorted(panel['season'].unique().to_list())}"
        )

    metric_rows: list[dict] = []
    prediction_frames: list[pl.DataFrame] = []

    for fold in folds:
        train = panel.filter(pl.col("season").is_in(fold.train_seasons))
        test = scoring_universe.filter(pl.col("season") == fold.test_season)
        full_test = panel.filter(pl.col("season") == fold.test_season)
        excluded = full_test.height - test.height
        logger.info(
            "%s | train %s, test %s in %s cohort (%s outside cohort)",
            fold,
            train.height,
            test.height,
            cohort,
            excluded,
        )

        fold_predictions: list[pl.DataFrame] = []
        missing_common_projector = False
        for projector in projectors:
            pred = projector.fit(train).predict(test)
            if pred.height == 0:
                logger.info("%s | no eligible rows in %s; skipped", fold, projector.name)
                missing_common_projector = missing_common_projector or common_coverage
                continue
            metadata = [
                c
                for c in (
                    "actual_points", "actual_ppg", "actual_games", "is_rookie",
                    "has_history", "market_ecr",
                )
                if c in test.columns and c not in pred.columns
            ]
            scored = pred.join(
                test.select("player_id", "season", *metadata),
                on=["player_id", "season"],
            ).with_columns(pl.lit(projector.name).alias("projector"))
            fold_predictions.append(scored)

        if common_coverage and missing_common_projector:
            logger.info("%s | no common coverage across every projector; fold skipped", fold)
            continue
        if common_coverage and fold_predictions:
            keys = fold_predictions[0].select("player_id", "season")
            for frame in fold_predictions[1:]:
                keys = keys.join(
                    frame.select("player_id", "season"),
                    on=["player_id", "season"],
                    how="inner",
                )
            fold_predictions = [
                frame.join(keys, on=["player_id", "season"], how="inner")
                for frame in fold_predictions
            ]
            logger.info("%s | common projector coverage: %s players", fold, keys.height)

        for scored in fold_predictions:
            if scored.height == 0:
                continue
            name = scored["projector"][0]
            metric_rows.extend(_score_fold(scored, name, fold))
            prediction_frames.append(scored)

    return pl.DataFrame(metric_rows), pl.concat(prediction_frames, how="diagonal")


def select_scoring_universe(panel: pl.DataFrame, cohort: str) -> pl.DataFrame:
    """Select a declared evaluation cohort without changing training rows."""
    if cohort == "history":
        return panel.filter(pl.col("has_history"))
    if cohort == "rookies":
        if "is_rookie" not in panel.columns:
            raise ValueError("rookie scoring requires panel.is_rookie")
        return panel.filter(pl.col("is_rookie"))
    if cohort == "all":
        return panel
    raise ValueError(f"unknown scoring cohort {cohort!r}; expected history, rookies, or all")


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
