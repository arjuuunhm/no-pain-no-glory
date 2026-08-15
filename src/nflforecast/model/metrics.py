"""Scoring functions -- exactly the set resources.md §5 adopts, and no more.

RMSE/MAE for magnitude, Spearman for ordering (a draft is a ranking problem
before it is a regression problem), and quantile calibration for the
distributional claim §4 makes.

`docs/evaluation.md` argues for CRPS, for within-position/top-k rank metrics,
and for paired-bootstrap confidence intervals instead of pooled point
estimates. None of those are implemented here on purpose: that document parks
them until a model produces numbers to decide with, and building them now
would quietly adopt them. Per-fold predictions are written to parquet by the
harness, so any of the three can be computed after the fact without touching
this module.

Pure polars, no scipy -- the project's dependency set is three packages and a
rank correlation does not justify a fourth.
"""

from __future__ import annotations

import math

import polars as pl


def _paired(df: pl.DataFrame, pred_col: str, actual_col: str) -> pl.DataFrame:
    """Rows where both sides are present. Nulls are dropped, never imputed.

    `actual_ppg` is null for a player who never appeared -- a real state, not a
    missing measurement -- so silently filling it would score projections
    against a fabricated zero.
    """
    return df.select([pred_col, actual_col]).drop_nulls()


def rmse(df: pl.DataFrame, pred_col: str, actual_col: str) -> float | None:
    paired = _paired(df, pred_col, actual_col)
    if paired.height == 0:
        return None
    return math.sqrt(
        paired.select(((pl.col(pred_col) - pl.col(actual_col)) ** 2).mean()).item()
    )


def mae(df: pl.DataFrame, pred_col: str, actual_col: str) -> float | None:
    paired = _paired(df, pred_col, actual_col)
    if paired.height == 0:
        return None
    return paired.select((pl.col(pred_col) - pl.col(actual_col)).abs().mean()).item()


def spearman(df: pl.DataFrame, pred_col: str, actual_col: str) -> float | None:
    """Rank correlation, ties averaged. Needs >= 3 pairs to mean anything."""
    paired = _paired(df, pred_col, actual_col)
    if paired.height < 3:
        return None
    ranked = paired.select(
        pl.col(pred_col).rank(method="average").alias("r_pred"),
        pl.col(actual_col).rank(method="average").alias("r_actual"),
    )
    return ranked.select(pl.corr("r_pred", "r_actual")).item()


def quantile_coverage(
    df: pl.DataFrame, quantile_cols: dict[float, str], actual_col: str
) -> dict[float, float | None]:
    """Empirical P(actual <= predicted q_alpha) for each alpha.

    The §5 question in one number per alpha: "do 20% of players actually exceed
    their 80th-percentile projection?" A well-calibrated q80 returns ~0.80.
    Under 0.80 means the interval is too tight (the usual failure -- GBDTs are
    overconfident out of distribution, resources.md §4).
    """
    out: dict[float, float | None] = {}
    for alpha, col in sorted(quantile_cols.items()):
        paired = _paired(df, col, actual_col)
        out[alpha] = (
            None
            if paired.height == 0
            else paired.select((pl.col(actual_col) <= pl.col(col)).mean()).item()
        )
    return out


def score(df: pl.DataFrame, pred_col: str, actual_col: str) -> dict[str, float | None]:
    """The §5 triple for one prediction column, plus the n it was computed on."""
    return {
        "n": float(_paired(df, pred_col, actual_col).height),
        "rmse": rmse(df, pred_col, actual_col),
        "mae": mae(df, pred_col, actual_col),
        "spearman": spearman(df, pred_col, actual_col),
    }
