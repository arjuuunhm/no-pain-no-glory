"""The learned projector -- resources.md §4, as three stages that multiply.

`benchmarks.py` answers "what did this player do lately, regressed". This
module answers "what does everything we knew on draft day say", and it is
built to be scored by the same harness against the same ladder: it satisfies
the `Projector` protocol, emits the same decomposition, and is fitted strictly
inside a fold's training seasons.

§4 is explicit that fantasy points should not be predicted directly, and this
is the shape it asks for:

    pred_points = pred_games x (tgt_per_game x pts_per_target
                                + car_per_game x pts_per_carry)

    stage 1  availability     pred_games            LightGBM, all rows
    stage 2  opportunity      tgt_pg, car_pg        LightGBM, played rows
    stage 3  efficiency       pts_per_tgt/carry     shrunk to positional mean

Only stages 1 and 2 are learned. Stage 3 is deliberately *not* a model: §4
says to regress efficiency toward the positional mean and "resist the urge to
model this hard", and the panel agrees -- year-over-year correlation of
points-per-target is weak enough that a GBDT fitted on it would be fitting
touchdown variance. So efficiency is the same shrinkage estimator the
benchmark ladder uses, with its constant grid-fitted per fold.

Two things follow from `docs/modeling.md`'s reading of the benchmark numbers:

- **Availability is where the error lives** (RMSE 4.95 vs a 5.50 floor,
  Spearman 0.39). It gets its own stage and the durability features rather
  than being folded into a points regression, but nobody should expect much:
  a model that only improves the per-game half moves the season total very
  little, and the reverse is where the headroom is.
- **The benchmark's intervals are a constant offset**, so they cannot widen
  for a volatile young back. The quantile heads here are separate LightGBM
  models with `objective="quantile"`, which can -- that is the specific
  failure they exist to beat, not an incidental extra.

**What the model may see** is the week-1 feature row (already lagged, see
`panel.snapshot_features`) plus the panel's prior-season columns, minus three
columns held back by default. `injury_report_status` and
`injury_practice_status` come off the mid-week practice report, and
`depth_chart_rank` off the week-1 depth chart; all three are published in the
week of game 1, which is *after* a draft. A season projection that leans on
them scores itself with information the drafter did not have, and would beat
the ladder partly by cheating. `include_post_draft=True` puts them back, which
is worth doing once to measure what they are worth -- if the answer is "a lot",
that is an argument for pulling a genuine preseason depth chart, not for
keeping these.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl

from nflforecast.config import get_logger
from nflforecast.model.benchmarks import (
    QUANTILES,
    _finish,
    _qcol,
    _shrunk,
    _mu_expr,
    marcel,
)
from nflforecast.model.panel import PRIOR_LAGS, snapshot_features

logger = get_logger("gbm")

# Marcel's weights, reused here to *engineer* priors rather than to project.
# A GBDT will not invent "weight the last three seasons 5/4/3 and divide by
# games" from three pairs of raw columns, and resources.md §4's list of things
# boosting will not invent for you is exactly this kind of transform.
_PRIOR_WEIGHTS = (5.0, 4.0, 3.0)

# Held out of the design matrix by default: all three are published during the
# week of game 1, which is after drafts happen. See the module docstring.
POST_DRAFT_COLS = ("injury_report_status", "injury_practice_status", "depth_chart_rank")

# Never features: identifiers and free text.
_DROP = ("player_id", "season", "player_name", "opponent", "head_coach", "play_caller")

# The panel carries its labels in the same frame as its features -- every
# `actual_*` column is an outcome of the season being projected. Excluding
# them by prefix rather than by name is the safe direction: a new label added
# to the panel is then dropped by default and has to be opted *in* to become a
# feature, instead of silently becoming one. Without this the design matrix
# happily takes `actual_targets` as the best predictor of targets per game.
_DROP_PREFIXES = ("actual_",)

# Low-cardinality strings that carry order or meaning; everything else string
# is dropped. Coach identity is 82 levels over ~2,000 training rows -- at that
# ratio a categorical split memorises the 2019 Ravens rather than learning
# anything, and the coach's *effect* is already in the team-grain features
# (pass rate over expected, plays per game, EPA) plus the two `_is_new` flags.
# `depth_chart_rank` is deliberately absent: it is ordinal, and on the opt-in
# path it is cast to a number rather than treated as a third unordered level.
_CATEGORICAL = (
    "position",
    "team",
    "roof",
    "roster_status",
    "injury_report_status",
    "injury_practice_status",
)

# Shrinkage grids, in "league-average observations the prior is worth". A
# target is worth ~1.3 PPR points and a carry ~0.55, so efficiency needs a lot
# of volume before it says anything -- the grid runs high on purpose.
_EFFICIENCY_GRID = tuple(
    float(k)
    for k in (0, 5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 500, 750, 1000, 2000, 5000)
)

# Small data, so: shallow trees, slow learning, heavy regularisation, and the
# column subsampling that keeps 150 correlated rolling windows from all
# splitting on the same thing. `deterministic` + `force_row_wise` make repeated
# fits bit-identical, which is what lets validate_projections.py assert that
# corrupting the test season moves no prediction.
BASE_PARAMS = {
    "objective": "l2",
    "learning_rate": 0.03,
    "num_leaves": 7,
    "min_data_in_leaf": 30,
    "feature_fraction": 0.5,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 5.0,
    "max_bin": 127,
    "verbosity": -1,
    "deterministic": True,
    "force_row_wise": True,
    "num_threads": 4,
    "seed": 0,
}
MAX_ROUNDS = 1500
EARLY_STOPPING_ROUNDS = 50
MIN_ROUNDS = 50
# Below these, the inner split is too small to stop on honestly and every
# stage takes a fixed budget instead. Reached only by the earliest folds.
MIN_TUNING_ROWS = 200
MIN_VALIDATION_ROWS = 50
DEFAULT_ROUNDS = 300


@dataclass
class _Design:
    """A fitted feature layout: column order, dtypes, and categorical codes.

    Built once on the training frame and replayed on the test frame, because
    LightGBM addresses features by position. A category that appears only in
    the test season maps to null, which is the honest encoding -- the model
    has never seen it and should fall back to whatever it does with missing.
    """

    numeric: list[str] = field(default_factory=list)
    categorical: list[str] = field(default_factory=list)
    codes: dict[str, dict[str, int]] = field(default_factory=dict)
    # Caller-supplied exclusions, on top of the always-dropped ones. Exists to
    # answer "how much of this model is one column?" without editing module
    # constants and forgetting to put them back.
    drop: tuple[str, ...] = ()

    @property
    def columns(self) -> list[str]:
        return self.numeric + self.categorical

    def fit(self, df: pl.DataFrame) -> "_Design":
        keep = lambda c: _usable(c) and c not in self.drop  # noqa: E731
        self.numeric = [
            c
            for c, dt in zip(df.columns, df.dtypes)
            if keep(c) and c not in _CATEGORICAL and (dt.is_numeric() or dt == pl.Boolean)
        ]
        self.categorical = [c for c in _CATEGORICAL if c in df.columns and keep(c)]
        self.codes = {
            c: {v: i for i, v in enumerate(sorted(df[c].drop_nulls().unique().to_list()))}
            for c in self.categorical
        }
        return self

    def matrix(self, df: pl.DataFrame) -> np.ndarray:
        cols = [pl.col(c).cast(pl.Float64) for c in self.numeric]
        cols += [
            pl.col(c).replace_strict(self.codes[c], default=None, return_dtype=pl.Float64)
            for c in self.categorical
        ]
        return df.select(cols).to_numpy()

    @property
    def categorical_indices(self) -> list[int]:
        return list(range(len(self.numeric), len(self.columns)))


class GBMProjector:
    """resources.md §4's three stages, fitted per fold. A `Projector`.

    `fit` sees only training seasons; `predict` may be handed any rows. The
    wide week-1 snapshot is joined in here rather than passed through the
    panel so that the harness contract stays "a projector takes the panel".
    """

    def __init__(
        self,
        name: str = "gbm",
        snapshot: pl.DataFrame | None = None,
        params: dict | None = None,
        include_post_draft: bool = False,
        blend_with_marcel: float = 0.0,
        drop_features: tuple[str, ...] = (),
    ) -> None:
        self.name = name
        self.params = {**BASE_PARAMS, **(params or {})}
        self.include_post_draft = include_post_draft
        self.drop_features = drop_features
        self.blend_with_marcel = blend_with_marcel
        self._snapshot = snapshot_features() if snapshot is None else snapshot
        self._design = _Design(drop=tuple(drop_features))
        self._boosters: dict[str, object] = {}
        self._rounds: dict[str, int] = {}
        self._mu_ppt: dict[str, float] = {}
        self._mu_ppc: dict[str, float] = {}
        self._fallback_ppt = 0.0
        self._fallback_ppc = 0.0
        self._k_ppt = 0.0
        self._k_ppc = 0.0
        self._marcel = marcel() if blend_with_marcel else None

    # -- feature assembly ----------------------------------------------

    def _design_frame(self, panel: pl.DataFrame) -> pl.DataFrame:
        """Panel rows joined to their week-1 snapshot, plus engineered priors.

        The join is exact on (player_id, season) and left: a panel row whose
        snapshot is missing keeps its priors and gets nulls for the rest,
        which LightGBM handles natively. Dropping it instead would silently
        change the scored universe between the ladder and the model.
        """
        wide = self._snapshot.drop(
            [c for c in ("team", "position", "age_years", "years_exp", "draft_round",
                         "draft_pick", "is_undrafted") if c in self._snapshot.columns]
        )
        if not self.include_post_draft:
            wide = wide.drop([c for c in POST_DRAFT_COLS if c in wide.columns])

        df = panel.join(wide, on=["player_id", "season"], how="left").with_columns(
            _engineered_priors()
        )
        if "depth_chart_rank" in df.columns:
            # Ordinal, not categorical: depth chart 1 < 2 < 3 is the whole
            # content of the column, and a null is "not on a depth chart",
            # which is worse than 3rd rather than a fourth unordered level.
            df = df.with_columns(pl.col("depth_chart_rank").cast(pl.Float64, strict=False))
        return df

    # -- fitting -------------------------------------------------------

    def fit(self, train: pl.DataFrame) -> "GBMProjector":
        frame = self._design_frame(train)
        self._design.fit(frame)
        x = self._design.matrix(frame)
        seasons = frame["season"].to_numpy()

        # Stage 1: availability, over every training row including the
        # never-played ones -- a zero-game season is the outcome this stage
        # exists to predict, so dropping it would remove the negative class.
        self._fit_stage("games", x, frame["actual_games"].cast(pl.Float64).to_numpy(), seasons)

        # Stage 2: opportunity, per game, on rows that produced a per-game
        # rate at all, weighted by the games behind it. Unweighted, a
        # two-game season with one big afternoon counts as much as a full
        # year -- `docs/modeling.md` flags exactly that as the first thing to
        # revisit about the benchmark's fitted constants.
        played = frame.filter(pl.col("actual_games") > 0)
        xp = self._design.matrix(played)
        played_seasons = played["season"].to_numpy()
        weight = played["actual_games"].cast(pl.Float64).to_numpy()
        for stage, col in (("tgt_pg", "actual_targets"), ("car_pg", "actual_carries")):
            y = (played[col] / played["actual_games"]).cast(pl.Float64).to_numpy()
            self._fit_stage(
                stage,
                xp,
                y,
                played_seasons,
                weight=weight,
                monotone=_monotone_for(stage, self._design),
            )

        # Stage 3: efficiency, not learned.
        self._fit_efficiency(frame)

        # Distribution: separate quantile heads on the season total. These are
        # not the point projection's own quantiles -- a quantile regression
        # has no reason to agree with a product of three conditional means --
        # and the ordering is enforced at predict time rather than assumed.
        y_points = frame["actual_points"].cast(pl.Float64).to_numpy()
        for alpha in QUANTILES:
            self._fit_stage(
                f"q{alpha}",
                x,
                y_points,
                seasons,
                params={"objective": "quantile", "alpha": alpha},
            )

        if self._marcel is not None:
            self._marcel.fit(train)
        return self

    def _fit_stage(
        self,
        stage: str,
        x: np.ndarray,
        y: np.ndarray,
        seasons: np.ndarray,
        weight: np.ndarray | None = None,
        monotone: list[int] | None = None,
        params: dict | None = None,
    ) -> None:
        """Early-stop on the last training season, then refit on all of it.

        The held-out season is the *latest* one available to the fold, never a
        random slice: resources.md §4 is explicit that same-season rows share
        team context, so a random validation split would stop on a season the
        model had already half-memorised. Refitting on the full training range
        afterwards is the standard trade -- the stopping point is chosen out of
        sample, and the final model still gets the most recent season, which
        is the one most like the season being projected.
        """
        import lightgbm as lgb

        cfg = {**self.params, **(params or {})}
        if monotone:
            cfg["monotone_constraints"] = monotone

        # Re-tuned every fold rather than cached on the instance: the harness
        # reuses one projector across folds, so a cached round count would let
        # fold 2020 set the budget for fold 2024 on a third of the history.
        rounds = self._tune_rounds(cfg, x, y, seasons, weight)
        self._rounds[stage] = rounds

        dataset = lgb.Dataset(
            x,
            label=y,
            weight=weight,
            feature_name=self._design.columns,
            categorical_feature=self._design.categorical_indices,
            free_raw_data=False,
        )
        self._boosters[stage] = lgb.train(cfg, dataset, num_boost_round=rounds)

    def _tune_rounds(
        self,
        cfg: dict,
        x: np.ndarray,
        y: np.ndarray,
        seasons: np.ndarray,
        weight: np.ndarray | None,
    ) -> int:
        """Number of boosting rounds, chosen on a held-out final season."""
        import lightgbm as lgb

        last = seasons.max()
        train_rows, valid_rows = seasons < last, seasons == last
        if train_rows.sum() < MIN_TUNING_ROWS or valid_rows.sum() < MIN_VALIDATION_ROWS:
            return DEFAULT_ROUNDS

        def subset(rows: np.ndarray, reference=None) -> "lgb.Dataset":
            return lgb.Dataset(
                x[rows],
                label=y[rows],
                weight=None if weight is None else weight[rows],
                feature_name=self._design.columns,
                categorical_feature=self._design.categorical_indices,
                reference=reference,
                free_raw_data=False,
            )

        inner = subset(train_rows)
        booster = lgb.train(
            cfg,
            inner,
            num_boost_round=MAX_ROUNDS,
            valid_sets=[subset(valid_rows, reference=inner)],
            callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
        )
        return max(booster.best_iteration, MIN_ROUNDS)

    def _fit_efficiency(self, frame: pl.DataFrame) -> None:
        """Positional means and shrinkage constants for points per opportunity.

        Volume-weighted on both sides. The mean is total points over total
        opportunities, not the mean of per-player rates, so a player with four
        targets and one touchdown does not get a vote worth 40 points per
        target; and `k` is chosen to minimise squared error in *points*, which
        is the quantity anyone cares about, rather than in a rate that is
        undefined for half the panel.
        """
        self._mu_ppt, self._fallback_ppt = _rate_means(frame, "actual_rec_points", "actual_targets")
        self._mu_ppc, self._fallback_ppc = _rate_means(frame, "actual_rush_points", "actual_carries")
        self._k_ppt = self._fit_k(
            frame, "ppt", _mu_expr(self._mu_ppt, self._fallback_ppt),
            "actual_rec_points", "actual_targets",
        )
        self._k_ppc = self._fit_k(
            frame, "ppc", _mu_expr(self._mu_ppc, self._fallback_ppc),
            "actual_rush_points", "actual_carries",
        )

    def _fit_k(
        self, frame: pl.DataFrame, kind: str, mu: pl.Expr, points_col: str, opp_col: str
    ) -> float:
        """Grid-search the shrinkage constant on the training seasons only."""
        scored = frame.filter(pl.col(opp_col) > 0)
        if scored.height == 0:
            return _EFFICIENCY_GRID[0]
        realized = pl.col(points_col) / pl.col(opp_col)
        errors = scored.select(
            [
                (((_efficiency_expr(kind, mu, k) - realized) ** 2) * pl.col(opp_col))
                .mean()
                .alias(f"k{i}")
                for i, k in enumerate(_EFFICIENCY_GRID)
            ]
        ).row(0)
        best = _EFFICIENCY_GRID[min(range(len(_EFFICIENCY_GRID)), key=lambda i: errors[i])]
        logger.debug("%s: fitted k_%s = %s", self.name, kind, best)
        return best

    # -- prediction ----------------------------------------------------

    def predict(self, target: pl.DataFrame) -> pl.DataFrame:
        frame = self._design_frame(target)
        x = self._design.matrix(frame)

        games = np.clip(self._boosters["games"].predict(x), 0.0, 17.0)
        tgt_pg = np.clip(self._boosters["tgt_pg"].predict(x), 0.0, None)
        car_pg = np.clip(self._boosters["car_pg"].predict(x), 0.0, None)

        efficiency = frame.select(
            _efficiency_expr("ppt", _mu_expr(self._mu_ppt, self._fallback_ppt), self._k_ppt).alias("ppt"),
            _efficiency_expr("ppc", _mu_expr(self._mu_ppc, self._fallback_ppc), self._k_ppc).alias("ppc"),
        )
        ppg = tgt_pg * efficiency["ppt"].to_numpy() + car_pg * efficiency["ppc"].to_numpy()

        out = target.select("player_id", "season", "position").with_columns(
            pl.Series("pred_ppg", ppg),
            pl.Series("pred_games", games),
            pl.Series("pred_tgt_per_game", tgt_pg),
            pl.Series("pred_car_per_game", car_pg),
            pl.Series("pred_pts_per_target", efficiency["ppt"]),
            pl.Series("pred_pts_per_carry", efficiency["ppc"]),
        )

        if self._marcel is not None:
            w = self.blend_with_marcel
            prior = self._marcel.predict(target).select(
                "player_id", "season", pl.col("pred_ppg").alias("m_ppg"),
                pl.col("pred_games").alias("m_games"),
            )
            out = out.join(prior, on=["player_id", "season"], how="left").with_columns(
                ((1 - w) * pl.col("pred_ppg") + w * pl.col("m_ppg").fill_null(pl.col("pred_ppg"))).alias("pred_ppg"),
                ((1 - w) * pl.col("pred_games") + w * pl.col("m_games").fill_null(pl.col("pred_games"))).alias("pred_games"),
            ).drop("m_ppg", "m_games")

        # `_finish` with no residual quantiles: it clips both stages and
        # enforces pred_points = pred_ppg x pred_games, the identity
        # validate_projections.py checks. The quantile columns are this
        # model's own, attached after.
        out = _finish(out, {})

        quantiles = np.sort(
            np.column_stack([self._boosters[f"q{a}"].predict(x) for a in sorted(QUANTILES)]),
            axis=1,
        ).clip(0.0)
        for i, alpha in enumerate(sorted(QUANTILES)):
            out = out.with_columns(pl.Series(_qcol(alpha), quantiles[:, i]))
        return out

    # -- introspection -------------------------------------------------

    def importances(self, stage: str = "tgt_pg", top: int = 20) -> pl.DataFrame:
        booster = self._boosters[stage]
        return (
            pl.DataFrame(
                {
                    "feature": self._design.columns,
                    "gain": booster.feature_importance("gain"),
                }
            )
            .sort("gain", descending=True)
            .head(top)
        )


def gbm(**kwargs) -> GBMProjector:
    """The learned projector, defaults as documented in docs/modeling.md."""
    return GBMProjector(**kwargs)


def gbm_marcel_blend(weight: float = 0.35) -> GBMProjector:
    """`gbm`, averaged with Marcel at the stage level.

    resources.md §4: "ensembling a boosted model with a Marcel-style baseline
    usually beats either alone". Blended per stage rather than on the points
    total so the decomposition survives -- averaging two totals would break
    pred_points = pred_games x pred_ppg.
    """
    return GBMProjector(name="gbm_marcel_blend", blend_with_marcel=weight)


# -- helpers -----------------------------------------------------------


def _usable(column: str) -> bool:
    """True if a column is allowed into the design matrix at all."""
    return column not in _DROP and not column.startswith(_DROP_PREFIXES)


def _monotone_for(stage: str, design: _Design) -> list[int]:
    """+1 on the matching prior, 0 elsewhere.

    resources.md §4: "use monotonic constraints where you have priors (more
    targets should never lower a projection)". Only the direct prior gets one
    -- the constraint is a statement about a causal direction, and there isn't
    one for most of 150 rolling windows.
    """
    prior = {"tgt_pg": "prior_tgt_per_game", "car_pg": "prior_car_per_game"}.get(stage)
    return [1 if c == prior else 0 for c in design.columns]


def _engineered_priors() -> list[pl.Expr]:
    """Marcel-shaped transforms of the raw prior columns.

    Weighted sums over the seasons that exist, divided by the weighted games
    behind them. A missing season contributes to neither side (panel.py's
    null-not-zero rule), so two years of history produce a genuinely different
    denominator from one.
    """
    weighted = lambda col: sum(  # noqa: E731
        w * pl.col(f"prev{lag}_{col}").fill_null(0.0)
        for lag, w in zip(PRIOR_LAGS, _PRIOR_WEIGHTS)
    )
    games = weighted("games")
    safe = pl.when(games > 0).then(games).otherwise(None)
    return [
        games.alias("prior_games_weighted"),
        (weighted("points") / safe).alias("prior_ppg"),
        (weighted("targets") / safe).alias("prior_tgt_per_game"),
        (weighted("carries") / safe).alias("prior_car_per_game"),
        (
            weighted("rec_points") / pl.when(weighted("targets") > 0).then(weighted("targets"))
        ).alias("prior_pts_per_target"),
        (
            weighted("rush_points") / pl.when(weighted("carries") > 0).then(weighted("carries"))
        ).alias("prior_pts_per_carry"),
        sum(
            pl.col(f"prev{lag}_games").fill_null(0.0).cast(pl.Float64) for lag in PRIOR_LAGS
        ).alias("prior_games_total"),
    ]


def _efficiency_expr(kind: str, mu: pl.Expr, k: float) -> pl.Expr:
    """Career-to-date points per opportunity, shrunk toward the positional mean."""
    points, opps = {
        "ppt": ("rec_points", "targets"),
        "ppc": ("rush_points", "carries"),
    }[kind]
    weighted = lambda col: sum(  # noqa: E731
        w * pl.col(f"prev{lag}_{col}").fill_null(0.0)
        for lag, w in zip(PRIOR_LAGS, _PRIOR_WEIGHTS)
    )
    return _shrunk(weighted(points), weighted(opps), mu, k)


def _rate_means(
    frame: pl.DataFrame, points_col: str, opp_col: str
) -> tuple[dict[str, float], float]:
    """Volume-weighted points per opportunity by position, plus a fallback."""
    used = frame.filter(pl.col(opp_col) > 0)
    if used.height == 0:
        return {}, 0.0
    agg = used.group_by("position").agg(
        (pl.col(points_col).sum() / pl.col(opp_col).sum()).alias("mu")
    )
    fallback = used.select(pl.col(points_col).sum() / pl.col(opp_col).sum()).item()
    return dict(agg.rows()), float(fallback or 0.0)
