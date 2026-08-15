# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository state

Python 3.10+, polars-based, installed editable (`pip install -e .`). No test runner or linter is
configured yet — the two `validate_*.py` scripts are the closest thing to a test suite:
`validate_features.py` should pass after any change to feature construction, `validate_projections.py`
after any change to the modelling layer.

```bash
python3 -m venv .venv && source .venv/bin/activate && pip install -e .

python scripts/build_dataset.py --start 2016 --end 2026  # nflverse -> data/raw/*.parquet (only rosters/schedules carry 2026)
python scripts/build_features.py                         # data/raw -> data/processed/*.parquet
python scripts/build_features.py --upcoming-season 2026  # + preseason_features.parquet for an unplayed season
python scripts/validate_features.py                      # assert leakage-free + metrics sane

python scripts/evaluate_benchmarks.py                    # walk-forward benchmark ladder -> stdout + parquet
python scripts/train_model.py                            # ladder + learned model, same harness -> stdout + parquet
python scripts/validate_projections.py                   # assert the harness fits on train seasons only
python scripts/project_season.py --season 2026           # fit on all completed seasons -> draft board
```

### Architecture

- `src/nflforecast/pullers/` — one module per nflverse loader, each landing a parquet in `data/raw/`.
  Independent and re-runnable; `build_dataset.py` logs failures per-puller rather than aborting.
- `src/nflforecast/features/` — feature blocks, each at a declared grain, all joined onto the spine by
  `build_feature_table.py`.
- `src/nflforecast/model/` — player-**season** grain. `panel.py` builds the draft-day panel (week-1
  feature snapshot + season labels lagged 1/2/3), `splits.py` the walk-forward folds, `benchmarks.py`
  the ladder, `evaluate.py` the harness, `gbm.py` the learned projector. Everything implements one
  `Projector` protocol (`fit(train) -> predict(test)`), so the model and the ladder are scored by the
  same loop on the same rows.
- `data/processed/` — `player_week_features.parquet` plus `player_week_labels.parquet` and
  `player_season_labels.parquet`, features and targets kept in **separate files on a shared key**.
  `preseason_features.parquet` holds the same 161 columns for an unplayed season, deliberately apart.

Scoped to **RB/WR/TE** (`config.SKILL_POSITIONS`). See `docs/feature_table.md` for the full column map.

Evaluation follows `resources.md` §5 (RMSE/MAE + Spearman + quantile calibration). Implemented in
`model/metrics.py`; benchmark ladder rungs 0–2 are scored — see `docs/modeling.md` for the numbers a
learned model has to beat. `docs/evaluation.md` records deferred alternatives (CRPS, rank-aware
metrics, paired bootstrap) — explicitly not adopted; don't build against it without asking.

### Invariants worth not breaking

- **The spine is `rosters_weekly`, not `weekly_player_stats`.** One row per rostered RB/WR/TE per
  regular-season week, whether or not he played. Keying off stat lines makes availability unmodelable
  (no negative class) and drops team context on every inactive week. Restricted to active-roster
  statuses — practice-squad and released rows have a 0% play rate and would swamp the negative class.
- **No feature module emits a raw current-week column.** Team-grain blocks shift(1)-then-roll;
  player-grain blocks roll inclusive and bind via `utils.attach_asof`, an as-of join at `week − 1`.
  Vegas lines are the one intentional exception — a closing line is known pre-kickoff.
- **Nothing in `model/` is fitted outside a fold's training seasons.** Shrinkage constants, the age
  curve, and the GBM's boosting-round budget (early-stopped on the fold's *last training season*) are
  fitted per fold on seasons < N. The draft-day information set is the **week-1 row** of the feature
  table (already lagged) plus season labels at lag ≥ 1 — don't re-derive it per model.
- **The panel holds features and labels in one frame, so model inputs are chosen by exclusion.**
  `gbm.py` drops anything prefixed `actual_`; add a new label to the panel and it stays out of the
  design matrix by default. The walk-forward leakage check does *not* catch a label used as a feature
  — it arrives via the training rows — so `validate_projections.py` asserts this separately.
- **Week-1 columns published after drafts stay out of season projections** (`POST_DRAFT_COLS`: the two
  injury-report columns and `depth_chart_rank`). They are worth ~1.3% RMSE; opt in with
  `--with-post-draft` to measure, not to ship.
- **Normalize team abbreviations before any join on team** (`config.normalize_team`). `load_schedules()`
  keeps historic codes (`OAK`/`SD`) while every other loader does not; unguarded joins fail silently.
- **A season that has not started has no weekly rosters.** `load_rosters_weekly()` refuses it. The
  preseason path (`spine.build_preseason_spine` + `utils.append_upcoming_week`) rebuilds the week-1
  row from the *seasonal* roster file and lands it in `preseason_features.parquet` — never in the
  feature table, and never in the label tables, because `build_panel` fills a missing outcome with 0
  and would train on an unplayed season as a real zero.
- **`roster_status` is the availability model.** Ablating it puts the games stage back at Marcel's
  level (4.92 vs 4.95) and drops the model's overall edge from 8.2% to 3.2%. It is only informative
  after late-August cuts; before then it is ~99% ACT. Rebuild after the cut deadline.
- **nflverse has no offensive-coordinator field.** `scheme.py` falls back to head coach and reads an
  optional hand-maintained `data/manual/coordinators.csv` when present. Don't invent an OC source.

## Project premises

`resources.md` lays out the plan for an NFL player performance prediction project (2026 season), built on
two working premises:

- **Gradient boosting on tabular data**, in a small-n regime (~600 fantasy-relevant player-seasons/year).
  Prefer LightGBM or CatBoost over XGBoost — CatBoost handles categorical features (team, position, coach,
  scheme) without manual encoding and behaves well on small data.
- **Model opportunity, not points.** Volume metrics (target share, carry share, snap share, air yards
  share) stabilize year-over-year; efficiency metrics mostly don't and should be regressed toward
  positional means rather than modeled hard.

Core modeling approach, if implemented, should follow the decomposition in `resources.md` §4 rather than
predicting fantasy points directly:

1. **Availability** (games played) — injury/depth-chart risk.
2. **Opportunity per game** — target/carry/snap share; this is where boosting should earn its keep.
3. **Efficiency** — heavily regressed toward positional mean, not aggressively modeled.

Other constraints called out in `resources.md` that any implementation should respect:

- **Validation must be temporal** (train on seasons ≤ N, test on N+1). Random k-fold leaks badly here since
  same-season rows share team context and same-player rows share talent.
- **Predict distributions, not point estimates** — draft value is about ceiling, not mean (quantile
  objectives / NGBoost).
- Benchmark against, in order: last season's per-game points regressed to the mean → a Marcel-style 3-year
  weighted average with age adjustment → consensus ADP/ECR → Vegas-informed projections. Score with
  RMSE/MAE and Spearman rank correlation, and check quantile calibration.

## Data sources

Primary data source is **nflverse** (`nflreadpy` in Python, `nflreadr`/`nflfastR` in R) — see `resources.md`
§1 for the full list of loaders and secondary sources (ADP, Vegas lines, paid charting data).

See `resources.md` in full for the complete reasoning, feature tables, and reading list before starting
implementation work.
