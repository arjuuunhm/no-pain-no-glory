# Benchmarking — the canonical run and the results ledger

This document has one job: make "did that change help?" answerable with a
number that is comparable to every previous answer.

`docs/modeling.md` explains *why* the model is shaped the way it is, and
argues from results. This is the record of the results themselves — a fixed
run configuration, a procedure for measuring a change against it, and a
[ledger](#the-ledger) with one row per change. **Add a row before you commit a
change that touches `features/` or `model/`.**

## The canonical run

```bash
python scripts/train_model.py     # ~25s, 6 folds, 4 projectors
```

No flags. Every flag is a different experiment and belongs in its own ledger
row with the flag named. The defaults that make a number comparable:

| Setting | Value | Why it is fixed |
|---|---|---|
| test seasons | 2020–2025 | `--min-train-seasons 3`, so Marcel's window is real |
| scored players | ~390/season, 2,356 total | rookies excluded; rung 1 is undefined without a prior season |
| post-draft columns | held out | see [measurement hazards](#measurement-hazards) |
| blend weight | 0 | the GBM/Marcel blend is an experiment, not the model |

Anything that changes the fold set, the scored universe, or the panel changes
every number in the ledger at once. That is allowed — it is not a silent
event. Re-run the whole ledger and say so in the notes column.

## What is scored

Three targets, because every projector emits `pred_points = pred_games ×
pred_ppg` and the harness scores each separately:

- **`points`** — season PPR total. The headline.
- **`ppg`** — points per game. The opportunity + efficiency half.
- **`games`** — games played. The availability half.

Metrics are exactly `resources.md` §5: RMSE, MAE, Spearman, and q10/q50/q90
coverage. `model/metrics.py` implements them and nothing else — CRPS,
within-position rank metrics, and paired bootstrap CIs are parked in
`docs/evaluation.md` and are deliberately not built.

**Always record all three targets.** The single most useful fact this project
has produced — that the model's edge over Marcel is almost entirely
availability, not projection skill — is invisible in the points row and
obvious the moment the stages are split. A change that moves `points` by
nothing can move `ppg` by a lot; the competition block is exactly that case.

## How to read a result

Three floors, in increasing order of size. A result has to clear all three.

**1. Run-to-run jitter: ~0.005 RMSE.** The ladder and the baseline `gbm` are
bit-identical across repeat runs (`deterministic: True`, `force_row_wise`,
fixed seed). Ablation variants are not quite: two back-to-back runs of the
same six variants moved by ≤0.003 points RMSE. Report to 2 decimal places and
treat anything under 0.01 as zero.

**2. Sampling noise: ~1% of RMSE.** At n≈390 per season, `docs/evaluation.md`
warns that two projectors differing by 2% may be indistinguishable. The
ladder's own internal spread is the calibration: Marcel beats rung 1 by 1.2%
and that gap is *not* meaningful. **A change worth under ~0.5 RMSE on points
is not established by the mean alone.**

**3. So count per-fold wins.** Six folds, so "wins 6/6" is a real paired
result and "wins 4/6" is a coin flip with extra steps. This is the crude
paired test `docs/evaluation.md` candidate 3 asks for, and until a bootstrap
CI exists it is what we have. Record it in the ledger's `folds` column.

One more trap, learned the hard way and worth repeating here: **split gain and
RMSE measure different things.** The first version of the competition block
took 6.4% of split gain in the carry stage and bought nothing out of sample.
High gain with flat RMSE is what redundancy looks like. `--importances` tells
you what the model *used*, never whether it helped.

## Measuring a change

The three flags on `train_model.py` (`--with-post-draft`, `--blend-weight`,
`--include-rookies`) cover their own experiments. Everything else — ablating a
feature block, toggling a stage — goes through `GBMProjector`'s constructor,
which exists so you can answer "how much of this model is one column?" without
editing module constants and forgetting to put them back.

Save as `scripts/ablate.py` or run from the scratchpad:

```python
import polars as pl
from nflforecast.features.teammates import OUTPUT_COLS as COMP_COLS
from nflforecast.model.benchmarks import default_ladder
from nflforecast.model.evaluate import run_walk_forward, summarise
from nflforecast.model.gbm import GBMProjector
from nflforecast.model.panel import snapshot_features

snapshot = snapshot_features()          # read once; it is 16MB per fit otherwise

variants = [
    GBMProjector(name="gbm", snapshot=snapshot),                                   # always include
    GBMProjector(name="no_competition", snapshot=snapshot, drop_features=tuple(COMP_COLS)),
    GBMProjector(name="no_qb_cond_eff", snapshot=snapshot, qb_adjusted_efficiency=False),
    GBMProjector(name="qb_in_design",   snapshot=snapshot, qb_features_in_design=True),
    GBMProjector(name="no_roster_status", snapshot=snapshot, drop_features=("roster_status",)),
]
metrics, _ = run_walk_forward(projectors=[*default_ladder(), *variants])

with pl.Config(tbl_rows=-1, tbl_cols=-1, float_precision=4):
    print(summarise(metrics, "points"))
    for t in ("ppg", "games"):
        print(metrics.filter(pl.col("target") == t).group_by("projector")
              .agg(pl.col("rmse").mean(), pl.col("spearman").mean()).sort("rmse"))
    print(metrics.filter(pl.col("target") == "points")
          .select("projector", "test_season", "rmse")
          .pivot("projector", index="test_season", values="rmse").sort("test_season"))
```

**Score every variant in one process.** Two runs of two configurations are two
tables copied together; one run is a comparison. It also costs less — the
panel and snapshot are built once.

**Always include an unmodified `gbm`.** Not the ledger's last baseline row —
an actual baseline fitted in the same process on the same panel. If the panel
changed underneath you, that is the only thing that will tell you.

## The ledger

Every row is the mean over the six canonical folds. `Δ points` is against the
baseline row (positive = the variant is worse), and `folds` counts the seasons
in which **the shipped configuration beats that variant**, so 6/6 is a decisive
result for shipping and 1/6 is a decisive result against it. Bold is what ships
today.

### Baseline, as of 2026-08-15 (worktree at `30bb760` + unstaged QB/competition work)

| Projector | points RMSE | MAE | Spearman | ppg RMSE | games RMSE | q10 | q90 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **`gbm`** | **50.88** | **36.95** | **0.812** | **3.126** | **4.124** | 0.188 | 0.878 |
| `marcel` (rung 2) | 55.71 | 40.44 | 0.729 | 3.241 | 4.992 | 0.223 | 0.898 |
| `last_season_regressed` (rung 1) | 56.36 | 41.78 | 0.726 | 3.418 | 4.995 | 0.222 | 0.900 |
| `positional_mean` (rung 0) | 82.64 | 65.78 | 0.138 | 5.349 | 5.524 | 0.177 | 0.888 |

Model over rung 1: **9.7%** on points, won in all six folds. Over Marcel:
8.7%. The bar is rung 1 and rung 2 together — their 1.2% gap is noise.

Per fold, points RMSE:

| Test season | `gbm` | `marcel` | rung 1 | rung 0 |
|---|---:|---:|---:|---:|
| 2020 | **53.07** | 58.18 | 58.89 | 80.01 |
| 2021 | **51.32** | 57.84 | 57.52 | 81.11 |
| 2022 | **48.62** | 50.86 | 53.26 | 82.07 |
| 2023 | **49.73** | 54.38 | 54.91 | 83.36 |
| 2024 | **52.39** | 57.54 | 57.80 | 84.76 |
| 2025 | **50.16** | 55.44 | 55.79 | 84.53 |

### Change log

All rows re-measured together on 2026-08-15, in one process, on the folds
above. Earlier numbers scattered through `docs/modeling.md` were taken on
2020–2024 and are superseded — see [reconciliation](#reconciliation-with-docsmodelingmd).

| # | Change | points RMSE | Δ points | ppg RMSE | games RMSE | folds | Verdict |
|---|---|---:|---:|---:|---:|:---:|---|
| 1 | baseline `gbm` (all blocks) | 50.88 | — | 3.126 | 4.124 | — | ships |
| 2 | − teammate competition block | 51.15 | +0.27 | 3.155 | 4.132 | 5/6 | **keep the block** |
| 3 | − QB-conditional efficiency mean | 50.93 | +0.05 | 3.134 | 4.124 | 5/6 | keep, marginally |
| 4 | + QB columns in the design matrix | 51.06 | +0.18 | 3.130 | 4.138 | 5/6 | **rejected — hurts** |
| 5 | − `roster_status` | 53.19 | +2.31 | 3.119 | 4.857 | 6/6 | **the load-bearing column** |
| 6 | + post-draft columns (`--with-post-draft`) | 51.85 | +0.97 | 3.080 | 4.236 | 1/6 | held out; see hazards |

What each row is actually saying:

- **Row 2 — competition is a per-game effect, and the points row hides it.**
  0.53% on points looks marginal. On the `ppg` stage it is 0.029, against a
  total model-over-Marcel edge there of 3.241 → 3.126 = 0.115. The block is
  **25% of all the per-game edge the model has**, which is not marginal at
  all. Wins 5/6; the loss (2020) is 0.07.
- **Row 3 — small, and the games stage proves it is wired correctly.** `games`
  RMSE is identical to four decimals (4.1241 both ways), which is exactly
  right: the QB block feeds stage 3 only and must not touch availability. Use
  that as the regression test if stage 3 is ever refactored. The points gain is
  0.09% — real, 5/6 folds, and limited by the starter heuristic's 78.3%
  accuracy and the 25% of team-seasons with a null prior EPA.
- **Row 4 — the reason the QB columns are held out.** As plain features they
  cost 0.18 and lose 5 of 6 folds. Nine columns against ~2,000 training rows
  dilute, and QB quality correlates +0.045 with targets per game, so there is
  nothing there for the opportunity stages to learn.
- **Row 5 — one column carries 42% of the model's edge over rung 1.** Without
  it the games stage falls to 4.857, essentially Marcel's 4.992, and the
  overall edge drops from 9.7% to 5.6%. Note `ppg` gets *better* without it
  (3.119 vs 3.126) — the column is purely an availability signal. Any claim
  about this model rests on having it populated at projection time, which
  before the late-August cut deadline it is not.
- **Row 6 — see below. This row is currently unsafe to read as a mean.**

## Measurement hazards

**`--with-post-draft` is broken on the 2025 fold.** nflverse replaced the
depth-chart feed from 2025 with dated snapshots, so `pull_depth_charts` keeps
only 2016–2024 while still logging success. Result: `depth_chart_rank` is
**100% null in 2025** (489/489 week-1 rows) and ~15% null before it. The flag
therefore trains on a populated column through 2024 and tests on an empty one,
and the 2025 fold blows up:

| Fold | `gbm` | `--with-post-draft` |
|---|---:|---:|
| 2020–2024 (mean) | 51.03 | **50.55** |
| 2025 | **50.16** | 58.33 |
| 2020–2025 (mean) | **50.88** | 51.85 |

So the "post-draft columns are worth ~1% RMSE" finding is real but only
measurable on folds ≤ 2024, where it is 0.93%. It does not change the
decision — the columns stay held out because they are published after drafts,
not because of their size — but the ledger's row 6 mean is an artifact of a
data outage, not a modelling result. **Do not quote it.**

Two general lessons, both applicable to any future block:

- **A column that goes null in the test season is worse than a column you
  never had.** The model learns to lean on it in training and finds nothing.
  Check null rate *by season*, not pooled, before trusting an ablation.
- **A puller that logs success is not a puller that returned data.** This one
  did.

## Adding a row

1. Run the canonical run (or the ablation snippet, with a fresh `gbm`
   included) — one process, all variants.
2. Record all three targets. A points-only row is not enough to interpret.
3. Record per-fold wins against the in-process baseline, not the mean alone.
4. Note the date and what the tree was at. Cheap, and the alternative is a
   table nobody can reproduce.
5. If the panel, folds, or scored universe moved, **re-measure every row** and
   say so. Mixed-vintage numbers in one table is the failure this document
   exists to prevent.

## Reconciliation with `docs/modeling.md`

`docs/modeling.md` currently mixes two fold sets: its headline tables are
2020–2024 (`gbm` at 51.4) and its later ablation sections are 2020–2025
(`gbm` at 50.93, then 50.88). Both are correct as measured; neither is
comparable to the other, and there is no marker in the document saying so.

**This ledger is the canonical numbers going forward.** `modeling.md` keeps
the reasoning and the findings — the quartile tables, the year-over-year
correlations, why efficiency is not learned — and should cite this file rather
than restating totals. Its numbers have not been rewritten here; that is a
separate edit, worth doing next.

One substantive difference beyond the fold set: `modeling.md` reports the
`roster_status` ablation at 53.95 points / 4.92 games and the model's edge
falling "8.2% → 3.2%". Re-measured on 2020–2025 it is 53.19 / 4.857, and
9.7% → 5.6%. Same conclusion, different arithmetic; the 2025 fold is the
difference.
