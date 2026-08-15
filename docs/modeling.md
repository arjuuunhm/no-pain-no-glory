# Modelling — panel, walk-forward harness, benchmark ladder

Status: **benchmarks implemented and scored. No learned model yet.**

`resources.md` §5 says the ladder is what a model has to beat, and
`docs/evaluation.md` parks its own open questions "until there is a model
producing numbers". This layer produces the first numbers: the rungs, scored
the way the model will be scored, with the leakage guard that matters at this
grain.

```bash
python scripts/evaluate_benchmarks.py     # per-fold table + leaderboard -> stdout, predictions -> parquet
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
