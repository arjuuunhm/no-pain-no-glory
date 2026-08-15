# Modelling — panel, walk-forward harness, benchmark ladder

Status: **benchmarks and a learned model implemented and scored.** The model
clears rung 1 in every fold; see [The learned model](#the-learned-model).

`resources.md` §5 says the ladder is what a model has to beat, and
`docs/evaluation.md` parks its own open questions "until there is a model
producing numbers". This layer produces the first numbers: the rungs, scored
the way the model will be scored, with the leakage guard that matters at this
grain.

```bash
python scripts/evaluate_benchmarks.py     # per-fold table + leaderboard -> stdout, predictions -> parquet
python scripts/train_model.py             # the same, with the learned model alongside the ladder
python scripts/validate_projections.py    # leakage + sanity checks, the sibling of validate_features.py
```

## The projection panel

`model/panel.py` turns the player-week feature table into one row per player
per season, as known on draft day:

- **Pre-season state** is the **week-1 row** of `player_week_features.parquet`.
  Every feature block is already lagged, so a week-1 row contains prior-season
  information and nothing else. Reading week 1 rather than re-deriving means
  this layer inherits the guarantees `validate_features.py` already checks.
- **Prior production** is `player_season_labels.parquet` joined at lags 1/2/3.
  The same table is a target at lag 0 and a feature at lag ≥ 1; the shift is
  the only thing separating those roles.
- **A season the player was not rostered for stays null**, never zero. Marcel
  treats a missing season as no plate appearances — contributing to neither
  side of the weighted average — and a zero would instead assert he was in the
  league and produced nothing.

Panel: 4,743 player-seasons over 2016–2024, 66% with a prior season.

## Folds

Train on seasons < N, score season N, walk forward (`model/splits.py`). With
`--min-train-seasons 3` (the default, so Marcel's three-year window is real
rather than a one-year average in disguise) that is **test seasons 2020–2024**,
~390 scored players each.

## What is scored

Every projector emits the §4 decomposition, not a points total alone:

```
pred_points = pred_games × pred_ppg
```

so all three stages are scored separately and a bad season projection can be
attributed to the availability half or the per-game half — which is
`docs/evaluation.md`'s open question about CRPS on season totals, answered by
construction instead of by metric choice.

Metrics are exactly §5: RMSE, MAE, Spearman, quantile coverage. CRPS,
within-position/top-k rank metrics, and paired bootstrap CIs are **not**
implemented — that document parks them, and per-fold predictions are written
to `data/processed/projection_predictions.parquet` so any of them can be
computed after the fact without touching the harness.

## The ladder

| Rung | Projector | What it is |
|---|---|---|
| 0 | `positional_mean` | Every player at his position's mean. Not in §5 — it catches the failure that comes first, a "model" whose whole signal is that TEs score less than WRs. |
| 1 | `last_season_regressed` | Last season's per-game points, regressed to the positional mean by games played. |
| 2 | `marcel` | Tango's 5/4/3 weighting over three seasons, regressed, with a fitted age curve. |

**Rungs 1 and 2 are one class with different weights** (`WeightedPrior`).
Marcel *is* a weighted average regressed by sample size; rung 1 is that with
weights (1, 0, 0). Sharing the implementation means the only thing the numbers
can attribute to Marcel is the extra history and the age curve, not an
accidental difference in how the regression was written.

Shrinkage constants are grid-fitted **on each fold's training seasons**, never
on the test season. The age adjustment is fitted the same way, as a mean
residual per (position, age) shrunk toward zero — Marcel's fixed age factor was
built for baseball, and football curves are position-specific and sharper. The
fitted curve recovers exactly that shape: RB per-game residuals go negative
from age 24 and reach −1.5 by 29; WRs hold flat through 27 before falling;
TEs barely move until 29.

## Results — 2020–2024 walk-forward, ~390 players/season

Season PPR total (mean over folds):

| Projector | RMSE | MAE | Spearman | q10 cov | q50 cov | q90 cov |
|---|---|---|---|---|---|---|
| `marcel` | **55.8** | **40.8** | **0.724** | 0.22 | 0.53 | 0.90 |
| `last_season_regressed` | 56.5 | 42.2 | 0.720 | 0.22 | 0.50 | 0.90 |
| `positional_mean` | 82.3 | 65.5 | 0.142 | 0.18 | 0.49 | 0.89 |

Per stage:

| Stage | `marcel` RMSE | rung 1 RMSE | floor RMSE | `marcel` Spearman |
|---|---|---|---|---|
| points per game | **3.30** | 3.46 | 5.36 | 0.77 |
| games played | 4.95 | 4.96 | 5.50 | 0.39 |

Per-fold RMSE on points — Marcel wins four of five, and loses 2021 by 0.3:

| Test season | `marcel` | rung 1 | floor |
|---|---|---|---|
| 2020 | 58.2 | 58.9 | 80.0 |
| 2021 | 57.8 | **57.5** | 81.1 |
| 2022 | 50.9 | 53.3 | 82.1 |
| 2023 | 54.4 | 54.9 | 83.4 |
| 2024 | 57.5 | 57.8 | 84.8 |

### What the numbers say

- **The bar is rung 1, and it is close to rung 2.** Marcel's edge is ~1% of
  RMSE, well inside the noise `docs/evaluation.md` warns about at n≈390.
  Treat rungs 1 and 2 as a single bar, not two.
- **Availability is where the error lives**, exactly as §4 predicts. The
  per-game stage separates cleanly from the floor (3.30 vs 5.36, Spearman
  0.77); the games stage barely does (4.95 vs 5.50, Spearman 0.39). A model
  that improves per-game projection and leaves availability alone will move
  the season total very little.
- **The benchmark's intervals are miscalibrated in the left tail** — 22% of
  players fall below their q10, not 10%. Expected: the residual offset is a
  constant, so it cannot widen for a volatile young RB. It is a floor for a
  real quantile model, not a target.

### Known limitations

- **Rookies are not scored.** ~100 per season are excluded (`--include-rookies`
  to see them), because rung 1 is undefined without a prior season and can only
  give them the positional mean. That is a real part of a draft board the
  ladder cannot speak to; draft capital in the panel is there for whatever
  eventually does.
- **Shrinkage is fitted on unweighted per-game RMSE**, so the fit is dominated
  by the bottom of the pool, where near-zero projections are easy. The fitted
  k is small (2.5 games for rung 1) partly for that reason. Fitting on a
  games-weighted or draftable-only slice would shrink the top of the board
  differently, and is the first thing to revisit.
- **Rungs 3 and 4 (consensus ADP/ECR, Vegas-informed) are missing**, because
  no puller lands ADP. Vegas lines are in the feature table but only per game,
  not as season win totals. Rung 3 is the one that matters — §5 calls ADP the
  benchmark you must beat, and rungs 1–2 are markedly easier.

## The learned model

`model/gbm.py`, scored by the same harness on the same rows as the ladder.
resources.md §4's decomposition, as three stages that multiply:

```
pred_points = pred_games × (tgt_per_game × pts_per_target
                            + car_per_game × pts_per_carry)
```

| Stage | How | Rows it fits on |
|---|---|---|
| availability | LightGBM, `pred_games` | all, including never-played |
| opportunity | LightGBM ×2, targets and carries per game | played, weighted by games |
| efficiency | **not learned** — shrunk to the positional mean | played |

Only the first two stages are learned. §4 says to regress efficiency toward
the positional mean and "resist the urge to model this hard", and the panel
agrees: at 50+ touches, year-over-year correlation of points per target is
**0.28** and points per carry **0.15**. The fitted shrinkage constant lands at
k = 1,000 targets and 5,000 carries — far beyond any career — which is the
grid's way of saying *use the positional mean*. A GBDT fitted here would be
fitting touchdown variance.

The panel carries features and labels in one frame, so the design matrix is
built by **exclusion**: everything not named in `_DROP` and not prefixed
`actual_`. The first version of this model omitted that prefix rule and scored
beautifully, because `actual_targets` was its best predictor of targets per
game. `validate_projections.py` now asserts the design matrix contains no
outcome column — the walk-forward leakage check does not catch this, since the
labels arrive through the *training* rows, where they are not leakage in the
temporal sense, merely the answer.

### Results — 2020–2024 walk-forward, ~390 players/season

Season PPR total (mean over folds):

| Projector | RMSE | MAE | Spearman | q10 cov | q50 cov | q90 cov |
|---|---|---|---|---|---|---|
| `gbm` | **51.4** | **37.4** | **0.805** | 0.20 | 0.55 | 0.87 |
| `marcel` | 55.8 | 40.8 | 0.724 | 0.22 | 0.53 | 0.90 |
| `last_season_regressed` | 56.5 | 42.2 | 0.720 | 0.22 | 0.50 | 0.90 |
| `positional_mean` | 82.3 | 65.5 | 0.142 | 0.18 | 0.49 | 0.89 |

Per fold, points RMSE — **the model wins all five**, by 4–6 points each:

| Test season | `gbm` | `marcel` | rung 1 | floor |
|---|---|---|---|---|
| 2020 | **53.1** | 58.2 | 58.9 | 80.0 |
| 2021 | **51.9** | 57.8 | 57.5 | 81.1 |
| 2022 | **48.8** | 50.9 | 53.3 | 82.1 |
| 2023 | **50.2** | 54.4 | 54.9 | 83.4 |
| 2024 | **52.8** | 57.5 | 57.8 | 84.8 |

That is 9.1% mean RMSE improvement over rung 1, won in every fold rather than
on average — which matters, because at n≈390 the ladder's own 1% spread is
noise. Per stage:

| Stage | `gbm` RMSE | `marcel` RMSE | `gbm` Spearman | `marcel` Spearman |
|---|---|---|---|---|
| points per game | **3.22** | 3.30 | **0.801** | 0.775 |
| games played | **4.12** | 4.95 | **0.538** | 0.387 |

### What the numbers say

- **The gain is mostly availability, which is the opposite of what the ladder
  predicted.** The benchmark section above argued a model improving only the
  per-game half would move the total very little — correct, and the model
  barely does improve it (3.30 → 3.22). The season total moved because *games
  played* did (4.95 → 4.12, Spearman 0.387 → 0.538).
- **And that gain is one column.** Ablating week-1 `roster_status` gives:

  | | points RMSE | games RMSE | games Spearman |
  |---|---|---|---|
  | `gbm` | 51.19 | 4.13 | 0.544 |
  | `gbm` without `roster_status` | 53.95 | 4.92 | 0.404 |
  | `marcel` | 55.74 | 4.95 | 0.387 |

  Without it the availability stage is **indistinguishable from Marcel**, and
  the model's overall edge falls from 8.2% to 3.2%. Everything the model knows
  about who will miss games, it learns from whether the player was on reserve
  or inactive in week 1 — not from injury history, age, or workload. Any claim
  about this model's value rests on getting that column at projection time;
  see [Projecting a season that has not been played](#projecting-a-season-that-has-not-been-played).
- **Opportunity behaves as §4 promised.** `prior_tgt_per_game` dominates the
  targets model, with the volatility columns (`targets_std`, `wopr_std`) next —
  the model is reading both the level and the reliability of a player's
  workload, which is the thing a weighted average of points cannot express.
- **The quantiles improved but are still too tight on the left.** 20% of
  players fall below their q10, against 22% for the benchmark's constant
  offset. Better, and now feature-dependent, but not calibrated — the honest
  reading is that this is the least finished part of the model.
- **Blending with Marcel did not help.** At weight 0.35 the blend scores 51.8
  against the model's 51.4, losing four folds of five. §4 expects the ensemble
  to win; it does not here, and the plausible reason is that the model already
  consumes Marcel-shaped priors as engineered features (`prior_ppg`,
  `prior_tgt_per_game`), so the blend re-weights information it already has.
  Kept behind `--blend-weight`, off by default.

### The draft-day information set

Three week-1 columns are **held out by default**: `injury_report_status`,
`injury_practice_status` (mid-week practice report) and `depth_chart_rank`
(week-1 depth chart). All three are published during the week of game 1 —
after drafts happen — so a model reading them scores itself with information
the drafter did not have.

They are worth **1.3% RMSE** (51.4 → 50.7) and +0.013 Spearman, measurable
with `--with-post-draft`. That is small enough that the honest default is
cheap, and the right response to wanting it back is to pull a genuine
preseason depth chart, not to relabel these as draft-day facts.

`roster_status` is the borderline case and is **kept**: week-1 reserve/PUP/
suspension designations mostly follow final roster cuts in late August, which
is at or before a typical draft. It is also the availability stage's largest
single feature, so if that call is wrong, the availability gain above is
overstated. Worth revisiting before trusting this on a live draft board.

### Known limitations

- **Rookies are still not scored** (`--include-rookies`). The model can at
  least form an opinion on them — draft capital and the team-grain features
  are populated — where rung 1 cannot, so this is now a real comparison the
  harness is declining to make rather than one it cannot.
- **Hyperparameters are fixed, not searched.** One shallow, heavily
  regularised configuration, with only the round count tuned per fold on the
  last training season. At ~2,000 training rows a real search would need a
  nested split the panel cannot spare.
- **The quantile heads are unconstrained.** They are three independent
  LightGBM fits, sorted at predict time to stop them crossing; nothing ties
  them to the point projection, which is a product of three conditional
  means. A q50 that disagrees with `pred_points` is expected, not a bug.
- **Efficiency is a positional constant.** Fully shrunk, it cannot express
  that a player is a genuinely better finisher — defensible from the
  year-over-year correlations, but it is a ceiling on the model, and the place
  a hierarchical partial-pooling model (§4's small-n suggestion) would go.

## Projecting a season that has not been played

```bash
python scripts/build_dataset.py  --start 2016 --end 2026   # rosters/schedules carry 2026; most loaders do not
python scripts/build_features.py --upcoming-season 2026
python scripts/project_season.py --season 2026
```

The backtest above is not a draft board, and the gap between them is
structural rather than a matter of re-running with newer data. The panel
defines draft-day state as the **week-1 row** of the feature table, and
`load_rosters_weekly()` refuses a season before it starts — 2026 raises
outright. So the projection path builds that row from a different spine:

- `spine.build_preseason_spine` reads the **seasonal** roster file, which does
  carry the upcoming season and uses the same `status` vocabulary, and emits
  it as week 1. Every feature block is keyed on (season, week) and already
  lagged, so those rows pick up the prior season and nothing else — the same
  guarantee the backtest inherits, by the same mechanism.
- `utils.append_upcoming_week` adds a placeholder team-week to the O-line and
  scheme blocks *before* rolling. The roll is shift(1)-then-rolling and does
  not reset at season boundaries, so the placeholder receives the trailing
  window over the end of the prior season — exactly what a real week-1 row
  receives, rather than an approximation of it.
- The result is written to `preseason_features.parquet`, **not** appended to
  the feature table, and label tables are built from the played spine only. A
  row for an unplayed season would otherwise reach `build_panel`, which fills
  a missing outcome with zero, and be trained on as a genuine zero-point
  season.

`project_season.py` then fits on every completed season — there is no test
season to hold out, and no scoring, because there are no outcomes yet.

### What a projection built today is worth

Less than the backtest suggests, for one specific reason. `roster_status`
carries the model's entire availability edge, and it is only informative once
final cuts and injured-reserve designations have happened. Measured on the
2026 roster in mid-August: **99% ACT**, against 81% in the training seasons'
week-1 rows. The column is present and populated, but nearly constant, so the
availability stage is currently operating closer to its ablated form — the
3.2%-over-Marcel row of the table above, not the 8.2% one.

**Rebuild after the late-August cut deadline.** That is not a refinement; it
is the difference between the two rows.

Two smaller gaps in a preseason build, both benign but worth knowing:

- **nflverse replaced the depth-chart feed.** From 2025 it is dated snapshots
  (`dt`, `pos_rank`) rather than week-keyed rows, so `pull_depth_charts` keeps
  only 2016–2024 while logging success. `depth_chart_rank` is already excluded
  as post-draft, so nothing in the default path depends on it — but that new
  feed is *dated*, which makes it the natural source for a genuine preseason
  depth chart, the thing that would justify putting the column back.
- **Rookies get a projection but a weak one.** 35% of the 2026 panel has no
  prior season, and for them the model is running on draft capital, age, and
  team context alone. They are not scored anywhere in the backtest, so their
  projections are unvalidated.
