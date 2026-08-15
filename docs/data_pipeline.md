# Data pipeline

Pulls core nflverse datasets via `nflreadpy` (polars-based) and lands them as
parquet in `data/raw/`. Entry point: `scripts/build_dataset.py`. Individual
puller functions live in `src/nflforecast/pullers/`.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Run

```bash
python scripts/build_dataset.py --seasons 2022 2023 2024
# or a range:
python scripts/build_dataset.py --start 2016 --end 2024
# skip a puller:
python scripts/build_dataset.py --start 2016 --end 2024 --skip participation
```

Verified: ran successfully for seasons 2022-2024 and again for 2016-2024
(9 seasons, all 13 pullers). Row counts from the 2016-2024 run:

| File | Rows | Cols | Notes |
|---|---:|---:|---|
| `weekly_player_stats.parquet` | 162,833 | 150 | one row per player-game |
| `snap_counts.parquet` | 226,494 | 16 | 2012+ only |
| `rosters.parquet` | 27,868 | 36 | one row per player-season |
| `player_ids.parquet` | 12,472 | 35 | crosswalk, not season-scoped |
| `draft_picks.parquet` | 2,308 | 36 | |
| `ngs_passing.parquet` | 5,328 | 29 | 2016+ only |
| `ngs_receiving.parquet` | 13,329 | 23 | 2016+ only |
| `ngs_rushing.parquet` | 5,411 | 22 | 2016+ only |
| `schedules.parquet` | 2,476 | 46 | includes Vegas lines |
| `participation.parquet` | 433,805 | 27 | play-level; coverage varies (see below) |
| `pbp.parquet` | 435,483 | 36 | play-level, curated column subset (see below) |
| `rosters_weekly.parquet` | 419,434 | 36 | week-by-week active/inactive status |
| `injuries.parquet` | 49,488 | 16 | weekly injury report |
| `depth_charts.parquet` | 332,174 | 15 | weekly official depth chart |
| `players.parquet` | 25,035 | 39 | master bio/draft table, not season-scoped |

## What each puller fetches

- **`pull_weekly_player_stats(seasons)`** &rarr; `weekly_player_stats.parquet`
  — `nflreadpy.load_player_stats(summary_level="week")`. One row per
  player-per-game: targets, carries, receiving/rushing/passing yards & TDs,
  EPA, CPOE, PACR, etc. (150 columns). This is the primary table feature
  engineering builds on.
- **`pull_snap_counts(seasons)`** &rarr; `snap_counts.parquet` —
  `nflreadpy.load_snap_counts()`. Offense/defense/special-teams snap counts
  and snap %. **Only available from 2012 onward** — nflreadpy raises a
  `ValueError` outside that range; the puller filters and warns rather than
  failing the whole run.
- **`pull_rosters(seasons)`** &rarr; `rosters.parquet` — seasonal rosters:
  team, position, jersey number, birthdate, height/weight, draft info, one
  row per player-season.
- **`pull_player_ids()`** &rarr; `player_ids.parquet` —
  `nflreadpy.load_ff_playerids()`, the cross-source ID crosswalk
  (gsis_id/pfr_id/espn_id/sleeper_id/yahoo_id/etc). Not season-scoped —
  a single current snapshot, needed to join weekly stats against
  ADP/injury/other external sources that key off different IDs.
- **`pull_draft_picks(seasons)`** &rarr; `draft_picks.parquet` — round,
  pick, team, college, and career AV. Feeds the "draft capital" feature
  (resources.md §8), which dominates for players with <2 seasons of data.
- **`pull_nextgen_stats(seasons)`** &rarr; `ngs_passing.parquet`,
  `ngs_receiving.parquet`, `ngs_rushing.parquet` — one call per `stat_type`
  since column sets differ. Covers air yards/aDOT/CPOE (passing), average
  separation/cushion/share of team air yards (receiving), and rush yards
  over expected (rushing). **Only available from 2016 onward** — same
  clamp-and-warn behavior as snap counts.
- **`pull_schedules(seasons)`** &rarr; `schedules.parquet` — game-level
  schedule **and Vegas closing lines**. Contrary to the assumption in the
  original task scope that Vegas data needs a separate source, nflverse
  bakes it directly into `load_schedules()`: `spread_line`, `total_line`,
  `away_moneyline`, `home_moneyline`, `away_spread_odds`, `home_spread_odds`.
  No separate odds-provider pull needed for closing lines. (Opening lines /
  line movement over the week are *not* included — only the line nflverse
  captured, treat as close-to-closing.) The puller checks these columns
  exist and logs the null rate on `spread_line` (nulls = future/unplayed
  games in the requested season range, not a data quality issue).
- **`pull_participation(seasons)`** &rarr; `participation.parquet` — play-level
  offense/defense personnel, formation, box count, pass-rush count, coverage
  type. See data-quality notes below; this is the dataset resources.md
  specifically warns about.
- **`pull_play_by_play(seasons)`** &rarr; `pbp.parquet` —
  `nflreadpy.load_pbp()`, curated down from ~370 to 36 columns, grouped in
  `KEEP_COLUMNS` by what they feed:
  - *identity/situation*: season, week, posteam/defteam, play_type,
    yardline_100, score_differential, down, ydstogo, goal_to_go, qtr,
    half/game_seconds_remaining;
  - *play classification*: rush/pass attempt flags, qb_dropback,
    qb_scramble, rusher/receiver/passer IDs;
  - *outcome*: yards_gained, epa, success, first_down, touchdown;
  - *O-line proxies* (`features/oline.py`): sack, qb_hit,
    tackled_for_loss — the only line-quality signals available without paid
    charting;
  - *scheme/pace* (`features/scheme.py`): xpass, pass_oe, shotgun,
    no_huddle, fixed_drive.

  No other puller exposes play-level data, so this is the source for
  game-script buckets, red-zone share, O-line quality, and every team
  tendency feature.
- **`pull_rosters_weekly(seasons)`** &rarr; `rosters_weekly.parquet` —
  `nflreadpy.load_rosters_weekly()`. Week-by-week roster status (`ACT`,
  `INA`, `RES`, etc. — 21 distinct codes) — the *realized* participation
  signal, as opposed to `load_injuries()`'s pre-game stated risk. Also
  carries `years_exp`, used as the experience prior feature.
- **`pull_injuries(seasons)`** &rarr; `injuries.parquet` —
  `nflreadpy.load_injuries()`. Weekly report/practice status, team
  self-reported. Note `season`/`week` come back as `Float64` in this
  nflreadpy version, not `Int`; cast before joining on them.
- **`pull_depth_charts(seasons)`** &rarr; `depth_charts.parquet` —
  `nflreadpy.load_depth_charts()`. `depth_team` (1 = starter, 2 = backup,
  ...) per player per week.
- **`pull_players()`** &rarr; `players.parquet` —
  `nflreadpy.load_players()`. Not season-scoped; one row per player with
  `birth_date`, `position`, `draft_year`/`draft_round`/`draft_pick`, and
  the `gsis_id`&harr;`pfr_id` crosswalk needed to join `snap_counts`
  (keyed by `pfr_player_id`) onto everything else (keyed by `gsis_id`).

## nflverse quirks / data-quality findings

- **Vegas lines live in `load_schedules()`, not a separate table.**
  `spread_line` / `total_line` / moneylines are populated columns on the
  schedule row for every played game in the seasons we pulled (2016-2024,
  0 nulls on `spread_line` for completed games).
- **`load_schedules()` does *not* standardise team abbreviations, and every
  other loader does.** Measured on 2016-2024: schedules carries `OAK` and
  `SD` where `weekly_player_stats` and `pbp` carry `LV` and `LAC`. Joining
  on the raw column therefore silently drops every Raiders week 2016-2019
  and every Chargers week in 2016 — no error, just nulls, concentrated
  entirely in two franchises. Use `config.normalize_team()` before any join
  keyed on team; `scripts/validate_features.py` asserts the coverage.
- **Participation coverage genuinely varies by season**, confirming the
  warning in resources.md — measured directly on 2016-2024 data:
  - `defenders_in_box`: ~26% null for 2016-2022, then **0% null in
    2023-2024** — a real coverage/methodology change, not noise.
  - `offense_formation`: ~27-28% null for 2016-2022, drops to ~20% null in
    2023-2024.
  - **Action for feature engineering**: check per-season null rates before
    building any participation-derived feature (e.g. personnel-package
    share, box-count-adjusted rushing efficiency) and either drop pre-2023
    seasons for those specific features or impute/flag rather than assume
    uniform coverage across the training window.
- **`load_participation()` has no `season` column** — it's keyed by
  `nflverse_game_id` in the form `"{season}_{week}_{away}_{home}"`. The
  puller derives `season` by parsing the first 4 characters of the game ID;
  downstream joins should use this derived column, not assume one exists
  natively.
- **NGS and snap counts have hard season floors**: NGS starts 2016, snap
  counts start 2012. `nflreadpy` raises a `ValueError` (not an empty
  result) if you request seasons outside that range, so both pullers
  pre-filter the requested season list and log a warning for any dropped
  seasons rather than letting the whole run fail on an off-by-a-few-years
  request.
- **`player_ids` and `load_players()`-style crosswalks are not
  season-partitioned** — they're a current snapshot of "all known players,"
  so re-running for a new season list doesn't change this file's shape; it
  changes only from a fresh nflverse release.
- **Weekly player stats (150 columns)** already includes fantasy-relevant
  aggregates (EPA, CPOE, PACR, air yards) alongside raw counting stats — no
  need for a separate play-by-play aggregation pass to get most of the §8
  starter feature list; only route-level features (TPRR, route
  participation %) require FTN charting (2022+ only, not yet pulled here)
  or participation-derived route counts.

## Refresh cadence assumptions

- **In-season (current year)**: weekly stats, snap counts, schedules
  (Vegas lines), and rosters change week to week — re-run after each week's
  games settle (Tue/Wed morning is safe for nflverse's own refresh cycle).
- **Past/completed seasons**: effectively frozen once nflverse finalizes
  them post-season; re-running is safe (idempotent overwrite) but shouldn't
  change row counts for closed seasons.
- **NGS**: nflverse republishes weekly during the season, same cadence as
  weekly stats.
- **Draft picks / player IDs**: change rarely — draft picks once a year
  (April), player ID crosswalk on nflverse's own release cadence (roughly
  daily to weekly). Cheap to re-pull every run; not worth caching specially.
- **Participation**: same weekly cadence as weekly stats, but treat
  per-season coverage as a schema-stability risk, not just a freshness one
  (see quirks above).

## Not yet pulled (out of scope for this pass, noted for follow-up)

- FTN charting (`load_ftn_charting`, 2022+) — route participation / TPRR
  source recommended in resources.md §8; not in the initial puller set but
  the loader exists in `nflreadpy` and would follow the same pattern.
- ADP (FantasyPros/Underdog/Sleeper) — external/secondary source per
  resources.md §1, still not pulled.
- `load_ff_opportunity()` — nflverse's own pre-built xFP model output
  (docs/features.md §0); worth pulling as a baseline/ensemble candidate
  before investing in a from-scratch xFP model.

See `docs/feature_table.md` for what's built from these raw tables.
