"""The benchmark ladder -- resources.md §5, rungs 0-2.

    0. positional mean          (not in §5; the floor a model must clear to exist)
    1. last season's per-game points, regressed to the mean
    2. Marcel: 3-year weighted average, regressed, with an age adjustment

Rungs 3 and 4 (consensus ADP/ECR, Vegas-informed projections) need data no
puller lands yet.

**Rungs 1 and 2 are the same estimator with different weights.** Marcel is a
weighted average over the last three seasons regressed toward the positional
mean by sample size; "last season regressed to the mean" is that with weights
(1, 0, 0). Writing them as one class is not code golf -- it means the only
difference the walk-forward numbers can attribute to Marcel is the extra two
seasons of history and the age curve, not an accidental difference in how the
regression was implemented.

Every projector emits the §4 decomposition rather than a points total alone:

    pred_points = pred_games x pred_ppg

so a benchmark is directly comparable to a decomposed model stage by stage,
and `docs/evaluation.md`'s open question -- whether a bad season projection
came from the availability half or the per-game half -- is answerable from the
output without re-running anything.

Shrinkage constants are **fitted on the training seasons of each fold**, never
tuned against the test season. This is the point in the pipeline where leakage
would be easiest to introduce and hardest to notice.
"""

from __future__ import annotations

from typing import Protocol

import polars as pl

from nflforecast.model.panel import PRIOR_LAGS

# Quantiles emitted alongside every point projection. §5 asks whether 20% of
# players beat their 80th-percentile projection; these are the columns that
# question is asked of. For a benchmark they come from the empirical spread of
# training residuals, which is the honest floor -- any real distributional
# model (LightGBM `objective="quantile"`, NGBoost) has to beat a model that
# just knows how wrong it usually is.
QUANTILES = (0.10, 0.50, 0.90)

# Age-adjustment shrinkage, in player-seasons. A (position, age) cell with 20
# observations gets half the raw residual it earned. Fixed rather than fitted:
# the panel is ~500 rows per season and every additional fitted constant buys
# more variance than signal at that n (resources.md §4, small-n caveat).
AGE_PRIOR_STRENGTH = 20.0
AGE_BUCKETS = (22, 31)

# Grid multipliers for the fitted shrinkage constants, expressed in units of
# the projector's own weight sum so the same grid covers both rungs.
_PPG_GRID = tuple(i * 0.5 for i in range(81))
_GAMES_GRID = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)


class Projector(Protocol):
    """Fit on a fold's training seasons, predict its test season."""

    name: str

    def fit(self, train: pl.DataFrame) -> "Projector": ...

    def predict(self, target: pl.DataFrame) -> pl.DataFrame: ...


def _mu_expr(mapping: dict[str, float], fallback: float) -> pl.Expr:
    """Per-position constant as an expression, with a league-wide fallback.

    The fallback matters for TE-thin folds and for any position code that
    slipped past the SKILL_POSITIONS filter -- a null mean would propagate
    silently into every prediction for that group.
    """
    return pl.col("position").replace_strict(
        mapping, default=fallback, return_dtype=pl.Float64
    )


def _shrunk(numerator: pl.Expr, denominator: pl.Expr, mu: pl.Expr, k: float) -> pl.Expr:
    """(observed_total + k*mu) / (observed_weight + k), falling back to mu.

    The standard regression-to-the-mean form: k is the number of
    league-average observations the prior is worth, so a player with a lot of
    history moves little and a player with none is the positional mean.
    """
    denom = denominator + k
    return pl.when(denom > 0).then((numerator + k * mu) / denom).otherwise(mu)


class PositionalMean:
    """Rung 0: every player at his position's mean. The floor.

    Not in resources.md §5, but §5 starts one rung too high to catch the
    failure that actually happens first -- a model whose apparent skill is
    entirely "TEs score less than WRs". Any Spearman this scores is coming
    from position labels alone.
    """

    name = "positional_mean"

    def __init__(self) -> None:
        self._mu_ppg: dict[str, float] = {}
        self._mu_games: dict[str, float] = {}
        self._fallback_ppg = 0.0
        self._fallback_games = 0.0
        self._resid_q: dict[float, float] = {}

    def fit(self, train: pl.DataFrame) -> "PositionalMean":
        self._mu_ppg, self._fallback_ppg = _positional_ppg(train)
        self._mu_games, self._fallback_games = _positional_games(train)
        self._resid_q = _residual_quantiles(self.predict(train), train)
        return self

    def predict(self, target: pl.DataFrame) -> pl.DataFrame:
        out = target.select(
            "player_id",
            "season",
            "position",
            _mu_expr(self._mu_ppg, self._fallback_ppg).alias("pred_ppg"),
            _mu_expr(self._mu_games, self._fallback_games).alias("pred_games"),
        )
        return _finish(out, self._resid_q)


class WeightedPrior:
    """Rungs 1 and 2: weighted prior seasons, regressed to the positional mean.

    `weights` are per prior season, most recent first. Marcel's canonical
    5/4/3 weights the last three; (1, 0, 0) is rung 1.

    Per-game rate and games played are shrunk separately, because they have
    different reliabilities -- rate stabilises with games played, availability
    barely stabilises at all -- and because §4 wants the two halves separable
    after the fact.
    """

    def __init__(
        self,
        name: str,
        weights: tuple[float, ...],
        age_adjust: bool = False,
    ) -> None:
        if len(weights) > len(PRIOR_LAGS):
            raise ValueError(f"panel carries {len(PRIOR_LAGS)} prior seasons, got {len(weights)} weights")
        self.name = name
        self.weights = weights
        self.age_adjust = age_adjust
        self._mu_ppg: dict[str, float] = {}
        self._mu_games: dict[str, float] = {}
        self._fallback_ppg = 0.0
        self._fallback_games = 0.0
        self._k_ppg = 0.0
        self._k_games = 0.0
        self._age_adj: dict[tuple[str, int], float] = {}
        self._resid_q: dict[float, float] = {}

    # -- fitted pieces -------------------------------------------------

    def fit(self, train: pl.DataFrame) -> "WeightedPrior":
        self._mu_ppg, self._fallback_ppg = _positional_ppg(train)
        self._mu_games, self._fallback_games = _positional_games(train)

        scale = sum(self.weights)
        self._k_ppg = self._fit_k(
            train,
            num=self._weighted_points(),
            den=self._weighted_games(),
            mu=_mu_expr(self._mu_ppg, self._fallback_ppg),
            actual="actual_ppg",
            grid=[g * scale for g in _PPG_GRID],
        )
        self._k_games = self._fit_k(
            train,
            num=self._weighted_games(),
            den=self._weight_mass(),
            mu=_mu_expr(self._mu_games, self._fallback_games),
            actual="actual_games",
            grid=[g * scale for g in _GAMES_GRID],
        )
        if self.age_adjust:
            self._age_adj = self._fit_age(train)
        self._resid_q = _residual_quantiles(self.predict(train), train)
        return self

    def _fit_k(
        self,
        train: pl.DataFrame,
        num: pl.Expr,
        den: pl.Expr,
        mu: pl.Expr,
        actual: str,
        grid: list[float],
    ) -> float:
        """Pick the shrinkage constant minimising in-train RMSE.

        One scalar chosen by grid search over the training seasons only. The
        alternative -- a fixed textbook constant -- would be defensible too,
        but the right amount of regression depends on the scoring format and
        the era, and both change.
        """
        scored = train.drop_nulls(actual)
        if scored.height == 0:
            return grid[0]
        errors = scored.select(
            [
                ((_shrunk(num, den, mu, k) - pl.col(actual)) ** 2).mean().alias(f"k{i}")
                for i, k in enumerate(grid)
            ]
        ).row(0)
        return grid[min(range(len(grid)), key=lambda i: errors[i])]

    def _fit_age(self, train: pl.DataFrame) -> dict[tuple[str, int], float]:
        """Mean residual by (position, age), shrunk toward no adjustment.

        Marcel's age factor is a fixed formula built for baseball. Football
        aging curves are position-specific and sharper (RBs fall off a cliff,
        WRs peak later -- resources.md §4), so this estimates the curve from
        the training seasons instead of importing a constant from another
        sport. Additive rather than multiplicative: at the bottom of the panel
        per-game points approach zero, where a ratio is unstable.
        """
        base = self._base_ppg(train).drop_nulls("actual_ppg")
        cells = (
            base.with_columns(_age_bucket().alias("age_bucket"))
            .drop_nulls("age_bucket")
            .group_by(["position", "age_bucket"])
            .agg(
                (pl.col("actual_ppg") - pl.col("pred_ppg")).sum().alias("resid_sum"),
                pl.len().alias("n"),
            )
            .with_columns(
                (pl.col("resid_sum") / (pl.col("n") + AGE_PRIOR_STRENGTH)).alias("adj")
            )
        )
        return {
            (pos, bucket): adj
            for pos, bucket, adj in cells.select(["position", "age_bucket", "adj"]).rows()
        }

    # -- prior aggregation ---------------------------------------------

    def _weighted_points(self) -> pl.Expr:
        """Sum of w_i * points_i -- the numerator of a games-weighted rate."""
        return sum(
            w * pl.col(f"prev{lag}_points").fill_null(0.0)
            for lag, w in zip(PRIOR_LAGS, self.weights)
        )

    def _weighted_games(self) -> pl.Expr:
        return sum(
            w * pl.col(f"prev{lag}_games").fill_null(0).cast(pl.Float64)
            for lag, w in zip(PRIOR_LAGS, self.weights)
        )

    def _weight_mass(self) -> pl.Expr:
        """Weight of the prior seasons that exist at all.

        A season the player was not in the league contributes nothing here, so
        two seasons of history shrink less than one and a first-year starter
        shrinks nearly all the way to the positional mean.
        """
        return sum(
            w * pl.col(f"prev{lag}_games").is_not_null().cast(pl.Float64)
            for lag, w in zip(PRIOR_LAGS, self.weights)
        )

    # -- prediction ----------------------------------------------------

    def _base_ppg(self, target: pl.DataFrame) -> pl.DataFrame:
        """Shrunk per-game rate, before any age adjustment."""
        return target.with_columns(
            _shrunk(
                self._weighted_points(),
                self._weighted_games(),
                _mu_expr(self._mu_ppg, self._fallback_ppg),
                self._k_ppg,
            ).alias("pred_ppg")
        )

    def predict(self, target: pl.DataFrame) -> pl.DataFrame:
        out = self._base_ppg(target).with_columns(
            _shrunk(
                self._weighted_games(),
                self._weight_mass(),
                _mu_expr(self._mu_games, self._fallback_games),
                self._k_games,
            ).alias("pred_games")
        )

        if self.age_adjust and self._age_adj:
            adj = pl.DataFrame(
                {
                    "position": [p for p, _ in self._age_adj],
                    "age_bucket": [b for _, b in self._age_adj],
                    "age_adj": list(self._age_adj.values()),
                },
                schema={"position": pl.String, "age_bucket": pl.Int32, "age_adj": pl.Float64},
            )
            out = (
                out.with_columns(_age_bucket().alias("age_bucket"))
                .join(adj, on=["position", "age_bucket"], how="left")
                .with_columns(
                    (pl.col("pred_ppg") + pl.col("age_adj").fill_null(0.0)).alias("pred_ppg")
                )
                .drop(["age_bucket", "age_adj"])
            )

        return _finish(
            out.select("player_id", "season", "position", "pred_ppg", "pred_games"),
            self._resid_q,
        )


def last_season_regressed() -> WeightedPrior:
    """Rung 1: only last season, regressed to the positional mean."""
    return WeightedPrior("last_season_regressed", weights=(1.0, 0.0, 0.0))


def marcel() -> WeightedPrior:
    """Rung 2: Tango's 5/4/3 weighting, regressed, with a fitted age curve."""
    return WeightedPrior("marcel", weights=(5.0, 4.0, 3.0), age_adjust=True)


def default_ladder() -> list[Projector]:
    return [PositionalMean(), last_season_regressed(), marcel()]


# -- shared helpers ----------------------------------------------------


def _age_bucket() -> pl.Expr:
    lo, hi = AGE_BUCKETS
    return pl.col("age_years").floor().clip(lo, hi).cast(pl.Int32)


def _positional_ppg(train: pl.DataFrame) -> tuple[dict[str, float], float]:
    """Games-weighted mean per-game points by position, from actuals.

    Games-weighted, not row-weighted: a player with two games and 30 points is
    a 15.0 per-game season that says almost nothing, and letting it count as
    much as a 17-game season would drag the positional mean toward the noise
    of small samples.
    """
    played = train.filter(pl.col("actual_games") > 0)
    agg = played.group_by("position").agg(
        (pl.col("actual_points").sum() / pl.col("actual_games").sum()).alias("mu")
    )
    fallback = played.select(
        pl.col("actual_points").sum() / pl.col("actual_games").sum()
    ).item()
    return dict(agg.rows()), fallback or 0.0


def _positional_games(train: pl.DataFrame) -> tuple[dict[str, float], float]:
    agg = train.group_by("position").agg(
        pl.col("actual_games").mean().cast(pl.Float64).alias("mu")
    )
    fallback = float(train["actual_games"].mean() or 0.0)
    return dict(agg.rows()), fallback


def _finish(pred: pl.DataFrame, resid_q: dict[float, float]) -> pl.DataFrame:
    """Clip to the possible, multiply the two stages, attach quantiles."""
    out = pred.with_columns(
        pl.col("pred_ppg").clip(0.0, None),
        pl.col("pred_games").clip(0.0, 17.0),
    ).with_columns((pl.col("pred_ppg") * pl.col("pred_games")).alias("pred_points"))

    for alpha, offset in resid_q.items():
        out = out.with_columns(
            (pl.col("pred_points") + offset).clip(0.0, None).alias(_qcol(alpha))
        )
    return out


def _qcol(alpha: float) -> str:
    return f"pred_points_q{int(round(alpha * 100)):02d}"


def _residual_quantiles(pred: pl.DataFrame, train: pl.DataFrame) -> dict[float, float]:
    """Empirical quantiles of the training residual, as offsets on pred_points.

    A constant-width interval is a crude distribution -- it cannot widen for
    the volatile young RB -- and that is the point: it is the bar a real
    quantile model has to clear. Slightly optimistic, since the shrinkage
    constants were fitted on these same rows; with two fitted scalars over
    ~2,000 player-seasons the in-sample tightening is small, but the
    calibration numbers for a benchmark should be read as a ceiling.
    """
    joined = pred.join(
        train.select("player_id", "season", "actual_points"), on=["player_id", "season"]
    ).with_columns((pl.col("actual_points") - pl.col("pred_points")).alias("resid"))
    if joined.height == 0:
        return {}
    return {alpha: float(joined["resid"].quantile(alpha)) for alpha in QUANTILES}
