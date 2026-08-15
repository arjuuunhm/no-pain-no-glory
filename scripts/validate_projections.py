#!/usr/bin/env python3
"""Entrypoint: assert the walk-forward harness is leakage-free and the ladder sane.

Usage:
    python scripts/validate_projections.py

The sibling of validate_features.py, one layer up. The feature pipeline's
leakage risk was in how a column was computed; this layer's is in *when a
constant was fitted*, which no column inspection can catch.

Checks:
  1. Panel priors are the previous season's labels, recomputed from raw.
  2. Corrupting the test season's actuals changes no prediction for that fold.
  3. Regression to the mean actually regresses (predictions are narrower than priors).
  4. Rungs 1 and 2 beat the positional-mean floor in every fold.
  5. Predictions stay inside the physically possible range.
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nflforecast.config import PROCESSED_DIR, get_logger
from nflforecast.model.benchmarks import default_ladder, last_season_regressed
from nflforecast.model.evaluate import run_walk_forward
from nflforecast.model.panel import build_panel
from nflforecast.model.splits import walk_forward

logger = get_logger("validate-model")

N_PLAYERS_TO_AUDIT = 40


def check(name: str, ok: bool, detail: str) -> bool:
    logger.info("%-46s %s  %s", name, "PASS" if ok else "FAIL", detail)
    return ok


def check_priors_are_lagged(panel: pl.DataFrame, season_labels: pl.DataFrame) -> bool:
    """prev1_ppg must be the player's season-1 label -- recomputed, not trusted."""
    busiest = (
        # player_id breaks ties so the audited sample is the same every run.
        panel.group_by("player_id").len().sort(["len", "player_id"], descending=[True, False])
        .head(N_PLAYERS_TO_AUDIT)["player_id"].to_list()
    )
    truth = {
        (pid, season): ppg
        for pid, season, ppg in season_labels.filter(pl.col("player_id").is_in(busiest))
        .select(["player_id", "season", "season_ppr_per_game"])
        .rows()
    }

    checked = mismatches = 0
    for pid, season, prev1 in (
        panel.filter(pl.col("player_id").is_in(busiest))
        .select(["player_id", "season", "prev1_ppg"])
        .rows()
    ):
        want = truth.get((pid, season - 1))
        if want is None and prev1 is None:
            continue
        checked += 1
        if want is None or prev1 is None or abs(want - prev1) > 1e-9:
            mismatches += 1

    return check(
        "panel priors are the prior season's labels",
        mismatches == 0 and checked > 0,
        f"{checked} player-seasons recomputed, {mismatches} mismatched",
    )


def check_no_test_season_leakage(panel: pl.DataFrame) -> bool:
    """Corrupting the test season's outcomes must not move a single prediction.

    The direct test of the claim the harness makes: constants are fitted on
    training seasons only. Only the `actual_*` columns are corrupted -- the
    `prev*` columns of the test season are the *prior* seasons' outcomes, which
    a draft-day projection is entitled to see.
    """
    folds = walk_forward(panel["season"].unique().to_list())
    fold = folds[-1]
    projectors = default_ladder()

    baseline, _ = _predictions_for(panel, fold.test_season, projectors)

    corrupted = panel.with_columns(
        [
            pl.when(pl.col("season") == fold.test_season)
            .then(pl.col(c).shuffle(seed=0) * 3.0)
            .otherwise(pl.col(c))
            .alias(c)
            for c in ("actual_points", "actual_games", "actual_ppg")
        ]
    )
    after, _ = _predictions_for(corrupted, fold.test_season, default_ladder())

    # Tolerance, not equality: the positional means are parallel sums whose
    # association order is not stable across runs, so an unleaked prediction
    # still moves by ~1e-15. A leak of any consequence is orders of magnitude
    # above that -- shuffling the actuals tripled them.
    pred_cols = [c for c in baseline.columns if c.startswith("pred_")]
    drift = max(
        (baseline[c] - after[c]).abs().max() or 0.0 for c in pred_cols
    )
    return check(
        "test-season outcomes cannot reach predictions",
        drift < 1e-9,
        f"{fold}, {baseline.height} predictions, max drift {drift:.2e} under corrupted actuals",
    )


def _predictions_for(panel: pl.DataFrame, test_season: int, projectors) -> tuple[pl.DataFrame, int]:
    universe = panel.filter(pl.col("has_history"))
    train = universe.filter(pl.col("season") < test_season)
    test = universe.filter(pl.col("season") == test_season)
    frames = [
        p.fit(train).predict(test).with_columns(pl.lit(p.name).alias("projector"))
        for p in projectors
    ]
    out = pl.concat(frames, how="diagonal").sort(["projector", "player_id"])
    return out, test.height


def check_regression_to_mean(panel: pl.DataFrame) -> bool:
    """Predictions must be narrower than the priors they came from.

    A shrinkage constant fitted to zero would silently turn rung 1 into "last
    season, verbatim" -- still a benchmark, but not the one §5 names, and one
    that flatters any model compared against it.
    """
    universe = panel.filter(pl.col("has_history"))
    train = universe.filter(pl.col("season") < 2023)
    test = universe.filter(pl.col("season") == 2023)
    pred = last_season_regressed().fit(train).predict(test)

    joined = pred.join(test.select(["player_id", "season", "prev1_ppg"]), on=["player_id", "season"])
    prior_sd = joined["prev1_ppg"].std()
    pred_sd = joined["pred_ppg"].std()
    corr = joined.select(pl.corr("pred_ppg", "prev1_ppg")).item()
    return check(
        "shrinkage narrows and preserves order",
        pred_sd < prior_sd and corr > 0.9,
        f"sd {pred_sd:.2f} vs prior {prior_sd:.2f}, corr {corr:.3f}",
    )


def check_ladder_ordering(metrics: pl.DataFrame) -> bool:
    """Both real rungs must beat the positional-mean floor in every fold."""
    points = metrics.filter(pl.col("target") == "points")
    floor = points.filter(pl.col("projector") == "positional_mean").select(
        ["test_season", "rmse"]
    ).rename({"rmse": "floor_rmse"})
    rungs = points.filter(pl.col("projector") != "positional_mean").join(
        floor, on="test_season"
    )
    losses = rungs.filter(pl.col("rmse") >= pl.col("floor_rmse"))
    return check(
        "rungs 1 and 2 beat the floor every fold",
        losses.height == 0,
        f"{rungs.height} (projector, fold) pairs, {losses.height} at or below the floor",
    )


def check_prediction_ranges(predictions: pl.DataFrame) -> bool:
    """Games in [0, 17], per-game and totals non-negative, points = games x rate."""
    bad_games = predictions.filter(
        (pl.col("pred_games") < 0) | (pl.col("pred_games") > 17)
    ).height
    bad_rate = predictions.filter(pl.col("pred_ppg") < 0).height
    inconsistent = predictions.filter(
        (pl.col("pred_points") - pl.col("pred_ppg") * pl.col("pred_games")).abs() > 1e-9
    ).height
    return check(
        "predictions physically possible",
        bad_games == 0 and bad_rate == 0 and inconsistent == 0,
        f"{bad_games} bad games, {bad_rate} negative rates, {inconsistent} inconsistent totals",
    )


def main() -> int:
    panel = build_panel()
    season_labels = pl.read_parquet(PROCESSED_DIR / "player_season_labels.parquet")
    metrics, predictions = run_walk_forward(panel)

    results = [
        check_priors_are_lagged(panel, season_labels),
        check_no_test_season_leakage(panel),
        check_regression_to_mean(panel),
        check_ladder_ordering(metrics),
        check_prediction_ranges(predictions),
    ]

    failed = results.count(False)
    logger.info("=== %s/%s checks passed ===", len(results) - failed, len(results))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
