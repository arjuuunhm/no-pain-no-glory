# Evaluation

## Status: still nothing adopted — but the trigger condition has fired

The metrics in `resources.md` §5 are what we use. This document records a
comparison against how Kaggle sports competitions score similar problems, and
parks the changes worth reconsidering **once there is a model producing
numbers**. Choosing a metric before anything scores against it is premature —
the deciding evidence is how the first walk-forward results actually behave.

**There is now a model producing numbers** (`model/gbm.py`, six walk-forward
folds, 2020–2025 — see `docs/modeling.md`). Two of the three conditions in
["When to revisit"](#when-to-revisit) below are met, and the third is
arguable. Nothing here has been implemented in response, deliberately: this
document is a set of options, and adopting one is a decision to take
explicitly rather than by drift. **Don't build against it without asking.**

What the first numbers say about each candidate is recorded inline below, so
the arguments can be read against evidence rather than in the abstract.

Nothing below is implemented. No code in the repo depends on it.

## What we score today (`resources.md` §5)

- **RMSE / MAE** against the benchmark ladder: last season's per-game points
  regressed to the mean → Marcel-style 3-year weighted average with age
  adjustment → consensus ADP/ECR → Vegas-informed projections.
- **Spearman rank correlation** — for a draft, ordering matters more than
  absolute accuracy.
- **Quantile calibration** — do 20% of players actually exceed their 80th
  percentile projection?
- **Temporal validation** — train on seasons ≤ N, test on N+1, walk forward.

## How Kaggle sports competitions score

| Competition | Target | Metric | Validation |
|---|---|---|---|
| [NFL Big Data Bowl 2020](https://github.com/Patman17/NFL-Big-Data-Bowl-2020) | Rushing yards from handoff | **CRPS** over a 199-bin cumulative distribution (−99 to +99 yds) | Held-out weeks |
| [March Machine Learning Mania](https://www.kaggle.com/competitions/march-machine-learning-mania-2023/overview/evaluation) | Game win probability | **Brier score** (switched from log loss in 2023) | Future tournament |
| [NFL Big Data Bowl 2026 — prediction](https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-prediction) | Player x/y location after the pass | Predicted vs actual location (NGS) | Train 2023–24, test 2025 wk 14–18 |
| [MLB Player Digital Engagement](https://www.kaggle.com/c/mlb-player-digital-engagement-forecasting/overview/evaluation) | Per-player, per-day engagement | MAE-based across four targets | Forward temporal split |

Three of the four are proper scoring rules on *distributions*. None uses rank
correlation.

## Candidate change 1 — CRPS as the headline metric

Big Data Bowl 2020 did not ask for a yardage number. It asked for a full CDF,
scored as:

```
C = 1/(199N) * Σ_{m=1..N} Σ_{n=-99..99} ( P(y ≤ n) − H(n − Y_m) )²
```

where `H(x) = 1 for x ≥ 0`, else 0. 199 columns, one per yard from −99 to +99,
each the cumulative probability of gaining at most that many yards.

Why it is attractive here:

- **It collapses two of our three metrics into one.** `resources.md` §4 says
  "predict distributions, not point estimates" and §5 separately says "check
  calibration of your quantiles". CRPS is the single proper scoring rule that
  does both jobs.
- **It reduces to MAE for a point forecast**, so a distributional model and a
  point-estimate benchmark (Marcel, ADP) are directly comparable on one
  number, with no second metric and no special-casing.
- **It is nearly free given the plan we already have.** CRPS is twice the
  integral of pinball loss over quantile levels. §4 already calls for
  LightGBM `objective="quantile"` at a grid of alphas — averaging those
  pinball losses gets us CRPS with no new machinery.

Open question, unresolved: CRPS on *season totals* conflates the availability
and per-game stages of the §4 decomposition. A player projected correctly
per-game but wrongly on games played scores badly for the wrong reason.
Scoring each stage separately would diagnose that, at the cost of no longer
having one headline number.

> **What the first numbers say.** The open question above is now answered by
> construction rather than by metric choice: every projector emits
> `pred_points = pred_games × pred_ppg`, and the harness scores all three
> separately. That is how we know the model's entire gain is in the
> availability stage (games RMSE 4.95 → 4.13) and almost none of it in the
> per-game stage (3.30 → 3.22) — a fact a single CRPS number would have
> hidden completely. The argument for CRPS as a *headline* is correspondingly
> weaker than it looked; the argument for it as a *calibration* measure is
> stronger, because the quantile heads exist now and are visibly
> miscalibrated (below).

## Candidate change 2 — global Spearman is the wrong ranking metric

No Kaggle sports competition uses rank correlation. That is not an oversight
on their part: they grade forecast accuracy, we grade a *draft decision*.
Keeping a rank metric is right for us.

The problem is the specific one we picked. Spearman over all ~600 players
weights the WR60-vs-WR61 ordering exactly as much as WR1-vs-WR2, and no
season turns on the former. The rank-aware family (NDCG and relatives) exists
to discount deep positions. The cheap fix, if this matters: compute Spearman
**within position** and restricted to the draftable top-k, rather than
globally.

> **What the first numbers say.** Global Spearman ran 0.81 for the model
> against 0.73 for Marcel — a gap wide enough that the choice of rank metric
> would not have changed which model won. The concern stands for finer
> comparisons, and note the draft board sorts by projected points, so the
> deep-position ordering this metric over-weights is not what anyone is
> actually reading.

## Candidate change 3 — report paired differences, not pooled point estimates

This is the one that matters most, and it is about our sample size rather
than our metric.

Big Data Bowl 2020 scored on tens of thousands of plays. We get ~600
fantasy-relevant player-seasons per test season and, realistically, three or
four walk-forward seasons. **Two models differing by 2% on RMSE will often be
indistinguishable from noise at that n.**

So whatever metric we land on, the comparison against a benchmark should be
reported as a *paired difference* with a bootstrap confidence interval, and
broken out per season rather than pooled into one number. Otherwise we will
read noise as progress — which in a small-n regime is the default outcome,
not the failure mode.

> **What the first numbers say.** This is the candidate that has most earned
> adoption. The model beat rung 1 by 9.2% on average — comfortably outside
> the noise floor this section warns about — but the *ladder's own* internal
> gaps are inside it: Marcel over rung 1 is ~1%, and a GBM/Marcel blend
> scored 51.8 against the model's 51.4 while losing four folds of five.
> Those two comparisons are exactly the "two models close enough that the
> ranking is in doubt" case, and they are currently resolved by counting
> per-fold wins (6/6, 1/5) rather than by a confidence interval. Counting
> wins is a crude paired test, but it is a paired one, which is the point
> this section is making.

## Cautionary note — metric choice drives modelling behaviour

March Machine Learning Mania moved from log loss to Brier score in 2023. The
community read is that Brier punishes confident-and-wrong far more gently,
which shifted optimal strategy toward 0/1 gambling on games considered
locks. Kaggle has not published an official rationale that I could find, so
treat the causal story as community inference rather than fact.

The transferable lesson holds regardless: in a small-n, high-variance sports
setting, a harshly-tailed metric turns evaluation into a variance contest
rather than a skill contest. Worth remembering if we ever consider log loss
for the availability stage, which is the natural place someone would reach
for it.

## What matches already

Temporal validation. Big Data Bowl 2026 trains on 2023–24 and evaluates on
2025 weeks 14–18. Our walk-forward split is standard practice, not an
idiosyncrasy — no change needed.

## When to revisit

Reconsider this document when any of the following is true. Status as of the
first learned model:

| Condition | Status |
|---|---|
| A first model exists and produces walk-forward results against benchmark #1 | **Met.** `gbm`, 6 folds, beats rung 1 in all six |
| Quantile outputs exist, making CRPS nearly free | **Met.** Three quantile-objective heads at α = 0.10/0.50/0.90 |
| Two candidate models are close enough on RMSE that the ranking is in doubt | **Arguable.** Marcel vs rung 1 differ by ~1%; the GBM/Marcel blend by ~0.8% |

The calibration evidence is the newest input and the one that most changes
the picture. Measured over 2020–2025:

| | q10 coverage | q90 coverage |
|---|---|---|
| target | 0.10 | 0.90 |
| `gbm` (quantile heads) | 0.19 | 0.88 |
| `marcel` (constant offset) | 0.22 | 0.90 |

Roughly twice as many players fall below their 10th-percentile projection as
should. Feature-dependent quantiles improved the left tail over a constant
offset, but not nearly enough, and this is the specific failure a proper
scoring rule would put a number on. Anyone picking this up should read that
as the strongest available argument for candidate 1 — and should still ask
before adopting it.
