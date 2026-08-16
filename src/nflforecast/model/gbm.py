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

from nflforecast.config import PROCESSED_DIR, get_logger
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

# Byproducts of `features/build_feature_table.py` -- see its module docstring.
# Read directly here rather than through a `features` import so `model/`
# keeps its existing rule of only ever touching `PROCESSED_DIR`.
FIXTURES_PATH = PROCESSED_DIR / "season_fixtures.parquet"
OPPONENT_PROFILE_PATH = PROCESSED_DIR / "opponent_season_profile.parquet"
WEEKLY_LABELS_PATH = PROCESSED_DIR / "player_week_labels.parquet"

# Marcel's weights, reused here to *engineer* priors rather than to project.
# A GBDT will not invent "weight the last three seasons 5/4/3 and divide by
# games" from three pairs of raw columns, and resources.md §4's list of things
# boosting will not invent for you is exactly this kind of transform.
_PRIOR_WEIGHTS = (5.0, 4.0, 3.0)

# Held out of the design matrix by default: all three are published during the
# week of game 1, which is after drafts happen. See the module docstring.
POST_DRAFT_COLS = ("injury_report_status", "injury_practice_status", "depth_chart_rank")

# The panel carries these so the ECR benchmark can use the same rows as every
# other projector. They are not football features: a GBM must opt in or the
# nominally football-only model silently becomes market-informed.
MARKET_COLS = ("market_ecr", "market_ecr_sd", "market_snapshot_date", "market_cutoff_date")

# Never features: identifiers, free text, and roster status. The latter is
# retained upstream to define the active-roster spine, but is deliberately
# excluded from the availability model.
_DROP = (
    "player_id",
    "season",
    "player_name",
    "opponent",
    "head_coach",
    "play_caller",
    "roster_status",
)

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
    "injury_report_status",
    "injury_practice_status",
)

# The per-game design (stage 2 only, see `GBMProjector._expand_to_games`):
# `opponent` is dropped from the season design above because a week-1
# opponent says nothing about the other 16 games, but it is exactly the
# feature a per-game design exists to use -- 32 levels over thousands of
# player-game rows, nothing like `head_coach`'s memorisation risk.
_GAME_CATEGORICAL = _CATEGORICAL + ("opponent",)
# `targets`/`carries` are here because `fit` joins the real weekly outcome
# onto the training frame before fitting the design -- the same column names
# as the stage's own labels, and without this they would be the best
# predictor of themselves. Mirrors `_DROP_PREFIXES` protecting the season
# design from `actual_*`.
_GAME_DROP = tuple(c for c in _DROP if c != "opponent") + ("targets", "carries")

# Week-1-snapshot columns that describe *that specific game* rather than the
# player or team all season, and so must not be reused unchanged for a
# different opponent every other week: the Vegas market (a future week's
# line does not exist in May) and whether week 1 happened to be divisional.
# `opponent`, `prev_season_opp_*`, and the rolling `opp_*_last{3,5,8}` form
# are handled separately in `_expand_to_games` since they are name-patterned
# rather than a fixed list.
_GAME_SPECIFIC_COLS = ("spread_line", "total_line", "team_implied_total", "opponent_implied_total", "div_game")

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
    # Which columns are treated as categorical, and which are always dropped
    # regardless of `drop` above. Default to the season-grain constants; the
    # per-game design overrides both to let `opponent` in -- see
    # `GBMProjector._expand_to_games`.
    categorical_cols: tuple[str, ...] = _CATEGORICAL
    always_drop: tuple[str, ...] = _DROP

    @property
    def columns(self) -> list[str]:
        return self.numeric + self.categorical

    def fit(self, df: pl.DataFrame) -> "_Design":
        usable = lambda c: c not in self.always_drop and not c.startswith(_DROP_PREFIXES)  # noqa: E731
        keep = lambda c: usable(c) and c not in self.drop  # noqa: E731
        self.numeric = [
            c
            for c, dt in zip(df.columns, df.dtypes)
            if keep(c) and c not in self.categorical_cols and (dt.is_numeric() or dt == pl.Boolean)
        ]
        self.categorical = [c for c in self.categorical_cols if c in df.columns and keep(c)]
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


def _load_fixtures() -> pl.DataFrame:
    """(team, season, week, opponent) for every REG game -- see build_feature_table.py."""
    return pl.read_parquet(FIXTURES_PATH)


def _load_opponent_profile() -> pl.DataFrame:
    """(opponent, season) -> prev_season_opp_* -- see build_feature_table.py."""
    return pl.read_parquet(OPPONENT_PROFILE_PATH)


def _load_weekly_labels() -> pl.DataFrame:
    """Real per-game outcomes, for stage 2's per-game training rows."""
    return pl.read_parquet(WEEKLY_LABELS_PATH)


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
        fixtures: pl.DataFrame | None = None,
        opponent_profile: pl.DataFrame | None = None,
        weekly_labels: pl.DataFrame | None = None,
        params: dict | None = None,
        include_post_draft: bool = False,
        include_market: bool = False,
        blend_with_marcel: float = 0.0,
        drop_features: tuple[str, ...] = (),
    ) -> None:
        self.name = name
        self.params = {**BASE_PARAMS, **(params or {})}
        self.include_post_draft = include_post_draft
        self.include_market = include_market
        self.drop_features = drop_features
        self.blend_with_marcel = blend_with_marcel
        self._snapshot = snapshot_features() if snapshot is None else snapshot
        # Both pure "known before any game of the target season" facts (a
        # public schedule; a defense's *completed prior* year, via the same
        # season+1 shift `panel.py`'s own prior joins rely on) -- read once
        # and reused across every fold, same as `self._snapshot`. See
        # `_expand_to_games`.
        self._fixtures = _load_fixtures() if fixtures is None else fixtures
        self._opponent_profile = _load_opponent_profile() if opponent_profile is None else opponent_profile
        self._weekly_labels = _load_weekly_labels() if weekly_labels is None else weekly_labels
        self._design = _Design(drop=tuple(drop_features))
        self._game_design = _Design(
            drop=tuple(drop_features), categorical_cols=_GAME_CATEGORICAL, always_drop=_GAME_DROP
        )
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
        duplicated = (
            "team", "position", "age_years", "years_exp", "draft_year",
            "draft_round", "draft_pick", "is_undrafted", *MARKET_COLS,
        )
        wide = self._snapshot.drop([c for c in duplicated if c in self._snapshot.columns])
        if not self.include_post_draft:
            wide = wide.drop([c for c in POST_DRAFT_COLS if c in wide.columns])

        df = panel.join(wide, on=["player_id", "season"], how="left")
        if not self.include_market:
            df = df.drop([c for c in MARKET_COLS if c in df.columns])
        df = df.with_columns(
            _engineered_priors()
        )
        if "depth_chart_rank" in df.columns:
            # Ordinal, not categorical: depth chart 1 < 2 < 3 is the whole
            # content of the column, and a null is "not on a depth chart",
            # which is worse than 3rd rather than a fourth unordered level.
            df = df.with_columns(pl.col("depth_chart_rank").cast(pl.Float64, strict=False))
        return df

    def _expand_to_games(self, panel: pl.DataFrame) -> pl.DataFrame:
        """One row per player-season per *scheduled game*, matchup-aware.

        The whole mechanism this class exists to add: a season projection is
        no longer one flat rate, it is a sum over the real 17-ish games on
        the schedule, each carrying its own opponent's prior-season profile.
        Used for both `fit` (joined to the real weekly labels below) and
        `predict` (fed straight to the stage-2 boosters) -- the *same*
        transform either way, so there is no train/serve skew to reason
        about: a training row and a serving row are built identically.

        Starts from the season-grain design frame (today's week-1 snapshot +
        engineered priors) and strips everything that describes *week 1's
        specific game* rather than the player or team all season -- its own
        `opponent`, that opponent's `prev_season_opp_*`, the in-season
        rolling opponent form (never legal beyond week 1 anyway), and the
        Vegas lines (a future week's line does not exist in May) -- then fans
        each row out over the real schedule and reattaches the *correct*
        opponent and `prev_season_opp_*` for every week on it.
        """
        static = self._design_frame(panel)
        strip = [
            c
            for c in static.columns
            if c == "opponent"
            or c.startswith("prev_season_opp_")
            or (c.startswith("opp_") and c.endswith(("_last3", "_last5", "_last8")))
            or c in _GAME_SPECIFIC_COLS
        ]
        games = static.drop(strip).join(self._fixtures, on=["team", "season"], how="inner")
        return games.join(self._opponent_profile, on=["opponent", "season"], how="left")

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

        # Stage 2: opportunity, per *game* and matchup-aware -- real games
        # only, one row each, no per-season averaging. Each row already is
        # one game, so unlike stage 1's season total there is nothing to
        # weight by games played: a two-game season contributes two rows,
        # exactly its share, which also retires the "unweighted per-game
        # RMSE" caveat `docs/modeling.md` used to flag about this stage's
        # fit. `_expand_to_games` is what makes the row-per-game a real
        # matchup rather than a repeated season average -- see its
        # docstring.
        weekly_y = self._weekly_labels.filter(pl.col("played")).select(
            "player_id", "season", "week", "targets", "carries"
        )
        game_frame = self._expand_to_games(train).join(
            weekly_y, on=["player_id", "season", "week"], how="inner"
        )
        self._game_design.fit(game_frame)
        xg = self._game_design.matrix(game_frame)
        game_seasons = game_frame["season"].to_numpy()
        for stage, col in (("tgt_pg", "targets"), ("car_pg", "carries")):
            y = game_frame[col].cast(pl.Float64).to_numpy()
            self._fit_stage(
                stage,
                xg,
                y,
                game_seasons,
                monotone=_monotone_for(stage, self._game_design),
                design=self._game_design,
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
        design: "_Design | None" = None,
    ) -> None:
        """Early-stop on the last training season, then refit on all of it.

        The held-out season is the *latest* one available to the fold, never a
        random slice: resources.md §4 is explicit that same-season rows share
        team context, so a random validation split would stop on a season the
        model had already half-memorised. Refitting on the full training range
        afterwards is the standard trade -- the stopping point is chosen out of
        sample, and the final model still gets the most recent season, which
        is the one most like the season being projected.

        `design` is whichever `_Design` built `x`'s columns -- the season
        design for every stage but stage 2, which is fit through the
        matchup-aware `_game_design` instead. LightGBM addresses features by
        position, so passing the wrong one here would silently mislabel every
        column without erroring.
        """
        import lightgbm as lgb

        design = design or self._design
        cfg = {**self.params, **(params or {})}
        if monotone:
            cfg["monotone_constraints"] = monotone

        # Re-tuned every fold rather than cached on the instance: the harness
        # reuses one projector across folds, so a cached round count would let
        # fold 2020 set the budget for fold 2024 on a third of the history.
        rounds = self._tune_rounds(cfg, x, y, seasons, weight, design)
        self._rounds[stage] = rounds

        dataset = lgb.Dataset(
            x,
            label=y,
            weight=weight,
            feature_name=design.columns,
            categorical_feature=design.categorical_indices,
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
        design: "_Design",
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
                feature_name=design.columns,
                categorical_feature=design.categorical_indices,
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

        # Stage 2: one prediction per scheduled game, matchup-aware -- the
        # mechanism this class exists for. `_expand_to_games` fans each
        # player-season out over its real schedule; efficiency stays
        # season-level and broadcasts across every game (it is deliberately
        # not matchup-aware -- see the module docstring). Summing the
        # per-game points and dividing back down by the games this stage
        # predicts turns the sum into a flat per-game rate, `pred_ppg`, which
        # is what keeps everything below -- the Marcel blend, `_finish`'s
        # pred_points = pred_ppg x pred_games identity -- unchanged from the
        # season-direct version this replaces.
        game_frame = self._expand_to_games(target)
        xg = self._game_design.matrix(game_frame)
        tgt_pg = np.clip(self._boosters["tgt_pg"].predict(xg), 0.0, None)
        car_pg = np.clip(self._boosters["car_pg"].predict(xg), 0.0, None)
        game_eff = game_frame.select(
            _efficiency_expr("ppt", _mu_expr(self._mu_ppt, self._fallback_ppt), self._k_ppt).alias("ppt"),
            _efficiency_expr("ppc", _mu_expr(self._mu_ppc, self._fallback_ppc), self._k_ppc).alias("ppc"),
        )
        game_points = tgt_pg * game_eff["ppt"].to_numpy() + car_pg * game_eff["ppc"].to_numpy()

        per_game = game_frame.select("player_id", "season").with_columns(
            pl.Series("tgt_pg", tgt_pg),
            pl.Series("car_pg", car_pg),
            pl.Series("game_points", game_points),
        )
        # `points_at_full_attendance`: the games stage is what accounts for
        # missed time, applied as a uniform share of the schedule rather than
        # picking which specific games are missed -- modelling *which* week a
        # player gets hurt is out of scope (resources.md §4's availability
        # stage is a season total, not a week-by-week hazard).
        season_summary = per_game.group_by(["player_id", "season"]).agg(
            pl.col("game_points").sum().alias("points_at_full_attendance"),
            pl.len().alias("n_scheduled"),
            pl.col("tgt_pg").mean().alias("pred_tgt_per_game"),
            pl.col("car_pg").mean().alias("pred_car_per_game"),
        )

        joined = (
            frame.select("player_id", "season", "position")
            .with_columns(pl.Series("pred_games", games))
            .join(season_summary, on=["player_id", "season"], how="left")
            .with_columns(
                # A team missing from the fixtures table should not happen
                # for a real season (`n_scheduled` would be null) -- fall
                # back to zero rather than crash, the same "unknown stays
                # unknown" posture nulls get everywhere else in this module.
                pl.col("points_at_full_attendance").fill_null(0.0),
                pl.col("n_scheduled").fill_null(1),
                pl.col("pred_tgt_per_game").fill_null(0.0),
                pl.col("pred_car_per_game").fill_null(0.0),
            )
            .with_columns(
                (
                    pl.col("points_at_full_attendance") * pl.col("pred_games") / pl.col("n_scheduled")
                ).alias("season_points")
            )
            .with_columns(
                pl.when(pl.col("pred_games") > 0)
                .then(pl.col("season_points") / pl.col("pred_games"))
                .otherwise(0.0)
                .alias("pred_ppg")
            )
        )

        # Joined by key, not by position: `joined` went through an extra
        # `season_summary` join above that `frame` never did, so its row
        # order is no longer guaranteed to match `frame`'s the way the rest
        # of this method assumes elsewhere.
        efficiency = frame.select(
            "player_id", "season",
            _efficiency_expr("ppt", _mu_expr(self._mu_ppt, self._fallback_ppt), self._k_ppt).alias(
                "pred_pts_per_target"
            ),
            _efficiency_expr("ppc", _mu_expr(self._mu_ppc, self._fallback_ppc), self._k_ppc).alias(
                "pred_pts_per_carry"
            ),
        )

        out = joined.select(
            "player_id", "season", "position", "pred_ppg", "pred_games",
            "pred_tgt_per_game", "pred_car_per_game",
        ).join(efficiency, on=["player_id", "season"], how="left")

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
        # tgt_pg/car_pg are fit through the matchup-aware `_game_design`,
        # which has a different column layout (and includes `opponent`) than
        # every other stage's `_design` -- see `_fit_stage`'s `design` param.
        design = self._game_design if stage in ("tgt_pg", "car_pg") else self._design
        return (
            pl.DataFrame(
                {
                    "feature": design.columns,
                    "gain": booster.feature_importance("gain"),
                }
            )
            .sort("gain", descending=True)
            .head(top)
        )


class RookieGBMProjector(GBMProjector):
    """The decomposed GBM fitted only on prior rookie classes.

    Veteran rows can teach a general model how NFL history maps to next-year
    production, but they vastly outnumber rookies and carry the very priors a
    rookie does not have. This projector makes the intended comparison
    explicit: draft capital and preseason context are learned from earlier
    rookie outcomes, then applied to the next class.
    """

    def __init__(self, name: str = "rookie_gbm", **kwargs) -> None:
        super().__init__(name=name, **kwargs)

    def fit(self, train: pl.DataFrame) -> "RookieGBMProjector":
        if "is_rookie" not in train.columns:
            raise ValueError("rookie projector requires panel.is_rookie")
        rookies = train.filter(pl.col("is_rookie"))
        if rookies.height == 0:
            raise ValueError("rookie projector has no prior rookie classes to fit")
        super().fit(rookies)
        return self

    def predict(self, target: pl.DataFrame) -> pl.DataFrame:
        if "is_rookie" not in target.columns:
            raise ValueError("rookie projector requires panel.is_rookie")
        return super().predict(target.filter(pl.col("is_rookie")))


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
