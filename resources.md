# Resources — NFL Player Performance Prediction (2026)

Working premises: **gradient boosting on tabular data** (small-n regime), and **model opportunity, not points**.

---

## 1. Data

### nflverse — start here, everything else is secondary
The open-source backbone of public NFL analytics. Play-by-play back to 1999, weekly/seasonal player stats, snap counts, depth charts, rosters, draft picks, Next Gen Stats, ID crosswalks between sources.

- Docs: https://nflverse.nflverse.com/
- R: `nflreadr` / `nflfastR`
- Python: `nflreadpy` (current, polars-based) or `nfl_data_py` (older pandas API, still widely used in tutorials)
- Data repos are plain releases on GitHub, so you can pull parquet directly without the wrapper libs.

Key loaders to know: weekly player stats, snap counts, NGS (air yards, separation, rush yards over expected), FTN charting (2022+), participation/personnel (coverage varies by season — check what's actually populated before building features on it).

### Market data (your baseline, and a feature)
- **ADP**: FantasyPros, Underdog, Sleeper API. Consensus ADP is a strong ensemble of thousands of humans — it is the benchmark you must beat.
- **Vegas**: season win totals and game totals. The single best public proxy for team offensive context; team-implied points drive a huge share of fantasy scoring variance.

### Paid / charted (only if you hit a ceiling)
Route participation is the highest-value stat that isn't reliably free. PFF, Fantasy Points Data, Sports Info Solutions.

---

## 2. The opportunity framework

The core empirical fact: **volume metrics stabilize year-over-year; efficiency metrics mostly don't.** Build features from the former, treat the latter as noise to be regressed toward positional means.

Metrics worth computing:

| Position | Opportunity | Notes |
|---|---|---|
| WR/TE | target share, air yards share, **WOPR** (1.5×tgt share + 0.7×AY share), **TPRR** (targets per route run), aDOT, route participation % | TPRR is the strongest signal-per-snap metric; requires route data |
| RB | snap share, carry share, **targets** (RB receptions are worth ~2.5 carries in PPR), goal-line and inside-10 carries | receiving work is what separates RB1s |
| QB | dropbacks, designed rush attempts, rushing yards | rushing floor dominates QB scoring |

**Expected Fantasy Points (xFP)** is the pattern to build toward: a model that scores each *opportunity* (a target at 12 air yards on 2nd-and-8 from the 30, a carry from the 3) by its historical expected fantasy value. Sum over a season to get a volume-only projection. Then `actual − expected` isolates efficiency/luck, which you deliberately *don't* extrapolate. This is directly analogous to xG in soccer and xwOBA in baseball, and it's the cleanest way to operationalize your instinct.

Concept origins worth reading: Josh Hermsmeyer's air yards / WOPR work (RotoViz, FiveThirtyEight archives), and the RotoViz body of writing on opportunity stability.

---

## 3. Methodology

### Open Source Football — https://opensourcefootball.com/
Community blog of reproducible notebooks (mostly R, some Python) on exactly these problems: expected points, stabilization rates, aging curves, model validation. The closest thing to a public curriculum for this work.

### Book
**Eric Eager & Richard Erickson, *Football Analytics with Python & R* (O'Reilly, 2023)** — written by ex-PFF analysts, uses nflverse, covers regression → GLMs → tree ensembles on real NFL data. Most direct match to what you're doing.

### Borrow from baseball — the field is 20 years ahead
- **Marcel the Monkey** (Tom Tango): weight the last 3 seasons 5/4/3, regress to league mean by sample size, apply an age adjustment. That's the whole method. Build the football version as your **baseline** — it's shockingly hard to beat and it will tell you immediately whether your boosted model is adding anything.
- **Tango, Lichtman & Dolphin, *The Book*** — regression to the mean, aging curves, sample-size reasoning. Methodology transfers cleanly.
- Read the ZiPS / Steamer methodology writeups for how production systems handle playing-time projection separately from rate projection.

### Kaggle NFL Big Data Bowl
Multiple years of public competitions with winning solutions published. Best available source of high-quality NFL modeling code, especially for feature engineering and validation setup.

---

## 4. Modeling notes

**Decompose, don't predict fantasy points directly.** Three separate models multiply together:
1. **Availability** — games played. Injury risk + depth-chart risk. Survival analysis or a simple beta regression; historically the largest source of season-long error.
2. **Opportunity per game** — target share, carry share, snap share. This is where boosting earns its keep and where the signal actually lives.
3. **Efficiency** — heavily regressed toward positional mean. Resist the urge to model this hard.

**Predict distributions, not point estimates.** Draft value is about ceiling, not mean. LightGBM supports quantile objectives directly (`objective="quantile"`, one model per alpha); NGBoost gives full probabilistic output. A player projected for 200 points with a fat right tail is a different draft pick than one projected for 200 flat.

**Validation must be temporal.** Train on seasons ≤ N, test on N+1, walk forward. Random k-fold leaks catastrophically — same-season rows share team context, and same-player rows share talent.

**Boosting library**: LightGBM or CatBoost over XGBoost here. CatBoost handles the categorical explosion (team, position, coach, scheme) without encoding gymnastics, and is well-behaved on small data. Use monotonic constraints where you have priors (more targets should never lower a projection) and early stopping on a held-out season.

**Features boosting won't invent for you**: age (nonlinear, position-specific curves — RBs fall off a cliff, WRs peak ~26), draft capital (dominates for players with <2 seasons of data), offensive line quality, projected team pass rate, coaching/OC change, target competition added or lost in the offseason.

**Small-n caveat.** ~600 fantasy-relevant player-seasons per year is not a lot. Consider Bayesian hierarchical models with partial pooling by position as a complement — they handle uncertainty more honestly than GBDTs, which are overconfident out of distribution. Ensembling a boosted model with a Marcel-style baseline usually beats either alone.

---

## 5. Benchmarks to beat

In order of difficulty. If you can't beat #1, the model isn't ready.

1. Last season's per-game points, regressed to the mean
2. Marcel-style 3-year weighted average with age adjustment
3. Consensus ADP / FantasyPros ECR
4. Vegas-informed projections

Score with RMSE/MAE **and** rank correlation (Spearman) — for a draft, ordering matters more than absolute accuracy. Also check calibration of your quantiles: do 20% of players actually exceed their 80th-percentile projection?

---

## 6. Communities

- nflverse Discord — active, responsive, where the data maintainers are
- r/fantasyfootball (analytics threads) and r/dfsports
- Open Source Football's contributor list is a good follow list on its own

---

## 7. Kaggle precedent

Nothing on Kaggle matches this problem exactly (season-long fantasy projection isn't a competition
category), but several past competitions are close enough in structure to borrow validation setups,
feature patterns, and modeling tricks from.

| Competition | Why it's relevant |
|---|---|
| [NFL Big Data Bowl 2024](https://www.kaggle.com/competitions/nfl-big-data-bowl-2024) | Tackle prob/time/location from tracking data. Winning entries won on physics-informed features (closing speed, angle to ball carrier) over raw coordinates — same lesson applies to building opportunity features from context, not raw counts. |
| [NFL Big Data Bowl 2025](https://www.kaggle.com/competitions/nfl-big-data-bowl-2025) / [2026](https://www.kaggle.com/competitions/nfl-big-data-bowl-2026-prediction) | 2026 trains on 2023–24, tests strictly on 2025 weeks 14–18 — a clean forward-chaining split worth copying as-is for temporal validation (§4). |
| [MLB Player Digital Engagement Forecasting](https://www.kaggle.com/competitions/mlb-player-digital-engagement-forecasting) ([3rd-place solution](https://github.com/nyanp/mlb-player-digital-engagement)) | Structurally closest analog: forecast player-level metrics from lag features + rolling aggregates + team context, one LightGBM model per target, blended rather than stacked. Template for the availability/opportunity/efficiency decomposition. |
| [March Machine Learning Mania](https://www.kaggle.com/c/march-machine-learning-mania-2024) | Annual NCAA bracket competition. Community leaderboard culture is built entirely around log-loss/Brier calibration of probabilities — apply the same rigor to quantile calibration, not just RMSE/Spearman. |
| [Football Match Probability Prediction](https://www.kaggle.com/competitions/football-match-probability-prediction) | Soccer match outcome from each team's last-10-match sequence. A Kaggle-validated version of Marcel-style recency weighting — suggests letting a model *learn* recency weights from an explicit last-N-games feature block instead of hand-fixing them. |
| [Football Players Value Prediction](https://www.kaggle.com/competitions/1056lab-football-players-value-prediction) | Smaller/less notable; useful mainly as an example of handling high-cardinality categoricals (club, nationality) cleanly with CatBoost-style encoding — same issue you'll hit with team/position/coach/scheme. |

**Patterns worth adopting:**

1. **Forward-chaining split, not just train ≤ N / test N+1** — hold out the back half of season N as a
   second test slice, since NFL Kaggle competitions specifically test on late-season weeks to catch models
   that only work while rosters/roles are still stable.
2. **One LightGBM model per target in the decomposition** (availability / opportunity / efficiency), sharing
   a common lag/rolling feature set, blended rather than stacked — mirrors the MLB engagement competition's
   top solutions.
3. **Explicit last-N-games feature block** (last 3/5/8 games of targets, carries, snaps) as a learned
   alternative/complement to fixed Marcel weights.
4. **Game-script and context features over season aggregates**: score differential, Vegas-implied pass rate,
   red-zone snap share — the class of feature that separated top vs. median Big Data Bowl entries.
5. **Log-loss/Brier-style calibration checks** on quantile outputs ("did this player exceed their projected
   80th percentile?"), on top of the RMSE/MAE/Spearman already in §5.

---

## 8. Starter feature list

A concrete first pass at features for the opportunity model (§4, step 2), grouped by source. Not
exhaustive — meant to be enough to get a first LightGBM/CatBoost baseline running end-to-end before
iterating.

**Volume / opportunity (from nflverse weekly stats)**
- Target share, carry share, snap share (rolling last 3/5/8 games, and season-to-date)
- Air yards share, WOPR, aDOT
- Route participation %, TPRR (2022+ via FTN charting)
- RB: targets, goal-line/inside-10 carries
- QB: dropbacks, designed rush attempts

**Game context / team environment**
- Vegas implied team total, spread, over/under
- Team pass rate over expectation (PROE), score differential distribution (proxy for game script)
- Offensive line quality (e.g. PFF grades or pressure rate allowed, if available)
- Coaching/OC continuity flag (changed vs. same as prior season)

**Player-level priors**
- Age (position-specific nonlinear curve, e.g. spline or bucketed dummy)
- Draft capital (round/pick, or a capital score) — dominant signal for <2 seasons of data
- Years of experience
- Target competition added/lost in the offseason (net change in same-position teammates' prior-year
  target share)

**Availability model inputs**
- Injury history (games missed last 1–2 seasons, by type if available)
- Depth chart position / competition for snaps
- Age × position injury base rates

**Recency / trend (learned Marcel-style block)**
- Last 3/5/8 game values for each opportunity metric above, not just season-to-date
- Trend slope over trailing games (simple linear fit over last N)

**Labels / targets (per the three-model decomposition)**
- Availability: games played (0 to season length)
- Opportunity: per-game target/carry/snap share
- Efficiency: fantasy points per opportunity, heavily shrunk toward positional mean (used for xFP,
  not extrapolated directly)
