# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository state

Python 3.10+, polars-based, installed editable (`pip install -e .`). No test runner or linter is
configured yet — `scripts/validate_features.py` is the closest thing to a test suite and should pass
after any change to feature construction.

```bash
python3 -m venv .venv && source .venv/bin/activate && pip install -e .

python scripts/build_dataset.py --start 2016 --end 2024  # nflverse -> data/raw/*.parquet
python scripts/build_features.py                         # data/raw -> data/processed/*.parquet
python scripts/validate_features.py                      # assert leakage-free + metrics sane
```

### Architecture

- `src/nflforecast/pullers/` — one module per nflverse loader, each landing a parquet in `data/raw/`.
  Independent and re-runnable; `build_dataset.py` logs failures per-puller rather than aborting.
- `src/nflforecast/features/` — feature blocks, each at a declared grain, all joined onto the spine by
  `build_feature_table.py`.
- `data/processed/` — `player_week_features.parquet` plus `player_week_labels.parquet` and
  `player_season_labels.parquet`, features and targets kept in **separate files on a shared key**.

Scoped to **RB/WR/TE** (`config.SKILL_POSITIONS`). See `docs/feature_table.md` for the full column map.

Evaluation follows `resources.md` §5 (RMSE/MAE + Spearman + quantile calibration) and is **not yet
implemented**. `docs/evaluation.md` records deferred alternatives (CRPS, rank-aware metrics, paired
bootstrap) — explicitly not adopted; don't build against it without asking.

### Invariants worth not breaking

- **The spine is `rosters_weekly`, not `weekly_player_stats`.** One row per rostered RB/WR/TE per
  regular-season week, whether or not he played. Keying off stat lines makes availability unmodelable
  (no negative class) and drops team context on every inactive week. Restricted to active-roster
  statuses — practice-squad and released rows have a 0% play rate and would swamp the negative class.
- **No feature module emits a raw current-week column.** Team-grain blocks shift(1)-then-roll;
  player-grain blocks roll inclusive and bind via `utils.attach_asof`, an as-of join at `week − 1`.
  Vegas lines are the one intentional exception — a closing line is known pre-kickoff.
- **Normalize team abbreviations before any join on team** (`config.normalize_team`). `load_schedules()`
  keeps historic codes (`OAK`/`SD`) while every other loader does not; unguarded joins fail silently.
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
