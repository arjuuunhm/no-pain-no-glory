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
  6. No outcome column reaches the learned model's design matrix.
  7. The learned model clears rung 1 in every fold -- resources.md §5's bar.
  8. Rookie status is experience-based, not inferred from missing history.
  9. Market consensus reaches only explicitly market-informed models.
 10. The preseason table, if built, is projectable and carries no outcomes.

The learned model is included in the walk-forward, so checks 2 and 5 cover it
too. That costs about a minute; the alternative is a test suite that validates
the benchmarks and takes the model on faith.
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nflforecast.config import PROCESSED_DIR, get_logger
from nflforecast.model.benchmarks import default_ladder, last_season_regressed
from nflforecast.model.evaluate import run_walk_forward
from nflforecast.model.gbm import GBMProjector, MARKET_COLS
from nflforecast.model.panel import (
    PRESEASON_FEATURES_PATH,
    build_panel,
    build_preseason_panel,
    snapshot_features,
)
from nflforecast.model.splits import walk_forward

logger = get_logger("validate-model")

N_PLAYERS_TO_AUDIT = 40

# One snapshot read, shared by every model instance built below.
_SNAPSHOT = None


def all_projectors() -> list:
    """The ladder plus the learned model, freshly constructed.

    Fresh instances every call: the leakage check fits the same projectors
    twice on different data, and a booster left over from the first fit would
    make the second one's predictions look reassuringly identical for the
    wrong reason.
    """
    global _SNAPSHOT
    if _SNAPSHOT is None:
        _SNAPSHOT = snapshot_features()
    return [*default_ladder(), GBMProjector(snapshot=_SNAPSHOT)]


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

    baseline, _ = _predictions_for(panel, fold.test_season, all_projectors())

    corrupted = panel.with_columns(
        [
            pl.when(pl.col("season") == fold.test_season)
            .then(pl.col(c).shuffle(seed=0) * 3.0)
            .otherwise(pl.col(c))
            .alias(c)
            for c in ("actual_points", "actual_games", "actual_ppg")
        ]
    )
    after, _ = _predictions_for(corrupted, fold.test_season, all_projectors())

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


def _fit_game_design(panel: pl.DataFrame) -> GBMProjector:
    """A model with `_game_design` fitted the way `fit()` does, boosters untrained.

    Shared by the two checks below so the per-game frame -- `_expand_to_games`
    joined to the real weekly outcome, exactly stage 2's training input -- is
    only built once.
    """
    model = GBMProjector(snapshot=_SNAPSHOT if _SNAPSHOT is not None else snapshot_features())
    train = panel.filter(pl.col("season") < 2024)
    weekly_y = model._weekly_labels.filter(pl.col("played")).select(
        "player_id", "season", "week", "targets", "carries"
    )
    game_frame = model._expand_to_games(train).join(
        weekly_y, on=["player_id", "season", "week"], how="inner"
    )
    model._game_design.fit(game_frame)
    return model


def check_no_outcome_features(panel: pl.DataFrame) -> bool:
    """No `actual_*` column may appear in the learned model's design matrix.

    The panel deliberately carries features and labels in one frame, so the
    design matrix is built by exclusion -- and the first version of this model
    scored beautifully because `actual_targets` was its best predictor of
    targets per game. A column-name assertion is a blunt check, but it is the
    one that would have caught it, and check 2 alone would not have: the
    labels reach the model through the *training* rows, where they are not
    corrupted and not leakage in the temporal sense, merely the answer.

    Covers both designs: the season-grain `_design` and stage 2's per-game
    `_game_design`, which joins the real weekly outcome onto its training
    frame and so has its own way to leak (`targets`/`carries` becoming
    features of themselves) that `_design` never sees.
    """
    model = GBMProjector(snapshot=_SNAPSHOT if _SNAPSHOT is not None else snapshot_features())
    frame = model._design_frame(panel.filter(pl.col("season") < 2024))
    model._design.fit(frame)
    game_model = _fit_game_design(panel)

    outcomes = [c for c in model._design.columns if c.startswith("actual_")]
    outcomes += [
        c
        for c in game_model._game_design.columns
        if c.startswith("actual_") or c in ("targets", "carries")
    ]
    roster_status = [
        c
        for c in (*model._design.columns, *game_model._game_design.columns)
        if c == "roster_status"
    ]
    n_features = len(model._design.columns) + len(game_model._game_design.columns)
    return check(
        "no outcome or roster-status column is a model feature",
        not outcomes and not roster_status,
        f"{n_features} features, {len(outcomes)} outcome columns, {len(roster_status)} roster-status columns"
        f"{': ' + ', '.join(outcomes + roster_status) if outcomes or roster_status else ''}",
    )


def check_rookie_cohort(panel: pl.DataFrame) -> bool:
    """A missing prior season must not be used as a synonym for rookie."""
    expected = (
        pl.when(pl.col("years_exp").is_not_null())
        .then(pl.col("years_exp") == 0)
        .otherwise(pl.col("draft_year") == pl.col("season"))
        .fill_null(False)
    )
    mismatches = panel.filter(pl.col("is_rookie") != expected).height
    rookies_with_history = panel.filter(pl.col("is_rookie") & pl.col("has_history")).height
    returning = panel.filter(
        (pl.col("season") > panel["season"].min())
        & ~pl.col("has_history")
        & ~pl.col("is_rookie")
    ).height
    return check(
        "rookie cohort is explicit, not missing-history",
        mismatches == 0 and rookies_with_history == 0 and returning > 0,
        f"{panel.filter(pl.col('is_rookie')).height} rookies, {mismatches} flag mismatches, "
        f"{returning} returning/no-prior players kept separate",
    )


def check_market_is_opt_in(panel: pl.DataFrame) -> bool:
    """The football-only design must not silently consume panel ECR columns."""
    train = panel.filter(pl.col("season") < panel["season"].max())
    football = GBMProjector(snapshot=_SNAPSHOT if _SNAPSHOT is not None else snapshot_features())
    football._design.fit(football._design_frame(train))
    market = GBMProjector(
        snapshot=_SNAPSHOT if _SNAPSHOT is not None else snapshot_features(),
        include_market=True,
    )
    market._design.fit(market._design_frame(train))
    leaked = [c for c in MARKET_COLS if c in football._design.columns]
    return check(
        "market consensus is an explicit model opt-in",
        not leaked and "market_ecr" in market._design.columns,
        f"football market columns {leaked}; market model has ECR: "
        f"{'market_ecr' in market._design.columns}",
    )


def check_game_design_is_draft_day_legal(panel: pl.DataFrame) -> bool:
    """Stage 2's per-game design must never see in-season opponent form or a
    specific game's Vegas line.

    This is the invariant the whole per-game mechanism rests on: every week
    of a projected season reads `prev_season_opp_*` only (the defense's
    *completed* prior year), never the rolling in-season form or a market
    line, both of which are illegal for any week beyond the first on an
    unplayed season. `_expand_to_games` is supposed to strip these before the
    design ever sees them -- this asserts it actually does.
    """
    game_model = _fit_game_design(panel)
    illegal = [
        c
        for c in game_model._game_design.columns
        if (c.startswith("opp_") and c.endswith(("_last3", "_last5", "_last8")))
        or c in ("spread_line", "total_line", "team_implied_total", "opponent_implied_total", "div_game")
    ]
    return check(
        "per-game design carries no rolling opponent form or per-game Vegas line",
        not illegal,
        f"{len(game_model._game_design.columns)} features, {len(illegal)} illegal"
        f"{': ' + ', '.join(illegal) if illegal else ''}",
    )


def check_model_beats_rung1(metrics: pl.DataFrame) -> bool:
    """resources.md §5: "if you can't beat #1, the model isn't ready".

    Every fold, not on average -- five folds of ~390 players cannot separate a
    two-percent difference in the mean, so a model that wins overall by losing
    three folds and blowing out two has not shown anything.
    """
    points = metrics.filter(pl.col("target") == "points")
    if "gbm" not in points["projector"].unique().to_list():
        return check("learned model beats rung 1 every fold", False, "no gbm rows in metrics")
    rung1 = points.filter(pl.col("projector") == "last_season_regressed").select(
        ["test_season", "rmse"]
    ).rename({"rmse": "rung1_rmse"})
    model = points.filter(pl.col("projector") == "gbm").join(rung1, on="test_season")
    losses = model.filter(pl.col("rmse") >= pl.col("rung1_rmse"))
    margin = (
        model.select(
            (1 - pl.col("rmse").mean() / pl.col("rung1_rmse").mean()) * 100
        ).item()
    )
    return check(
        "learned model beats rung 1 every fold",
        losses.height == 0,
        f"{model.height} folds, {losses.height} lost, {margin:.1f}% mean RMSE improvement",
    )


def check_preseason_panel(panel: pl.DataFrame) -> bool:
    """If a preseason table exists, it must be projectable and outcome-free.

    Skipped when `preseason_features.parquet` is absent, since it is an
    optional artifact. Its failure mode is not an exception but silence: a
    team abbreviation that does not normalise, or a block that emits no row
    for an unplayed season, leaves a column entirely null and the projection
    still "works". Both of those happened while this was being built, so the
    check compares coverage against a real week-1 row rather than merely
    asserting the frame is non-empty.
    """
    if not PRESEASON_FEATURES_PATH.exists():
        return check("preseason panel projectable", True, "no preseason table built (skipped)")

    preseason = pl.read_parquet(PRESEASON_FEATURES_PATH)
    season = int(preseason["season"].max())
    target = build_preseason_panel(season, preseason_features=preseason)

    leaked = [c for c in target.columns if c.startswith("actual_")]

    # Team-grain columns are the ones that go silently null when a season has
    # no games; they must be as populated here as in a real week-1 row.
    team_cols = ["head_coach", "team_pass_epa_last8", "ol_adj_line_yards_last8"]
    real_week1 = pl.read_parquet(PROCESSED_DIR / "player_week_features.parquet").filter(
        (pl.col("week") == 1) & (pl.col("season") == panel["season"].max())
    )
    worst = max(preseason[c].null_count() / preseason.height for c in team_cols)
    baseline = max(real_week1[c].null_count() / real_week1.height for c in team_cols)

    ok = not leaked and target.height > 0 and worst <= baseline + 0.01
    return check(
        "preseason panel projectable",
        ok,
        f"{season}: {target.height} players, {len(leaked)} outcome columns, "
        f"team-feature nulls {worst:.1%} vs {baseline:.1%} in a real week 1",
    )


def main() -> int:
    panel = build_panel()
    season_labels = pl.read_parquet(PROCESSED_DIR / "player_season_labels.parquet")
    metrics, predictions = run_walk_forward(panel, projectors=all_projectors())

    results = [
        check_priors_are_lagged(panel, season_labels),
        check_no_test_season_leakage(panel),
        check_regression_to_mean(panel),
        check_ladder_ordering(metrics),
        check_prediction_ranges(predictions),
        check_no_outcome_features(panel),
        check_rookie_cohort(panel),
        check_market_is_opt_in(panel),
        check_game_design_is_draft_day_legal(panel),
        check_model_beats_rung1(metrics),
        check_preseason_panel(panel),
    ]

    failed = results.count(False)
    logger.info("=== %s/%s checks passed ===", len(results) - failed, len(results))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
