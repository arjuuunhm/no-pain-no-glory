# Feature table

Builds the modelling tables from `data/raw/*.parquet`. Entry point:
`scripts/build_features.py`; modules live in `src/nflforecast/features/`.
Validate with `scripts/validate_features.py`.

Scoped to **RB/WR/TE** (`config.SKILL_POSITIONS`) — the positions whose value
is driven by opportunity share, which is what resources.md §4 step 2 says the
model should predict. QB volume is near-constant given availability, and
K/DST are a different problem.

## Setup / run

Requires `data/raw/*.parquet` to exist first (`scripts/build_dataset.py`).

```bash
python scripts/build_features.py                        # build
python scripts/build_features.py --upcoming-season 2026 # + an unplayed season
python scripts/validate_features.py                     # assert leakage-free + sane
```

Verified end-to-end against the 2016-2025 raw pull:

| File | Rows | Cols | Grain |
|---|---:|---:|---|
| `player_week_features.parquet` | 85,117 | 161 | `(player_id, season, week)` |
| `player_week_labels.parquet` | 85,117 | 18 | `(player_id, season, week)` |
| `player_season_labels.parquet` | 6,431 | 10 | `(player_id, season)` |
| `preseason_features.parquet` | 788 | 161 | `(player_id, season)`, week 1 only |

`preseason_features.parquet` appears only with `--upcoming-season`. It is the
same 161 columns for a season that has not been played, built off the seasonal
roster file because no weekly roster exists yet, and written **separately** —
it must never reach the label tables, because `model/panel.py` fills a missing
outcome with zero and would train on an unplayed season as a real zero-point
one. See `docs/features.md` §7, "Projecting a season that hasn't started".

Features and labels are **separate files on a shared key**. That split is the
enforcement mechanism for the leakage rules below: a raw current-week number
cannot be trained on by accident if it does not live in the file the model
reads.

## The spine

Everything hangs off `spine.py`: one row per rostered RB/WR/TE per
regular-season week, from `load_rosters_weekly()` — **not** from whoever
recorded a stat line. 85,117 player-weeks, 57,306 of them played (67%).

This is what makes availability modellable. Keyed off weekly stats, a player
who missed week 7 had no week-7 row at all, so the availability model had no
negative class and team-level joins silently dropped every inactive week.

Restricted to `ACTIVE_STATUSES` = `ACT`/`INA`/`RES`/`PUP`/`SUS`. Practice
squad (`DEV`) and released (`CUT`) rows are excluded: 30k rows at a measured
**0.0% play rate**, which would have made up ~40% of the spine and ~60% of
the negative class. A model would score beautifully learning "practice squad
⇒ did not play" and learn nothing about the actual question. A practice-squad
week is not a missed game.

## What's in it

| Module | Grain | Source | Produces |
|---|---|---|---|
| `spine.py` | player-week | `rosters_weekly`, `snap_counts`, `players` | the key itself: `team`, `position`, `roster_status`, `played` |
| `opportunity.py` | player-week | `weekly_player_stats`, `snap_counts`, `players` | `target_share`, `air_yards_share`, `wopr`, `carries`, `targets`, `receptions`, `offense_pct` — each `_last3/_last5/_last8` + `_std` |
| `redzone.py` | player-week | `pbp` | `rush`/`tgt` × `rz_share`/`i10_share`/`rz_n`/`i10_n`, rolling |
| `game_script.py` | player-week | `pbp` | `rush`/`tgt` × `share_trailing_big`/`share_close`/`share_leading_big`/`mean_score_diff`, rolling |
| `oline.py` | team-week | `pbp` | `ol_adj_line_yards`, `ol_stuff_rate`, `ol_power_success`, `ol_sack_rate`, `ol_qb_hit_rate`, rolling |
| `scheme.py` | team-week + team-season | `pbp`, `schedules` | `pass_rate_over_expected`, `neutral_pass_rate`, `plays_per_game`, `sec_per_play`, `shotgun_rate`, `no_huddle_rate`, `rz_trips_per_game`, `team_pass_epa`, `team_rush_epa` — rolling **and** `prev_season_*`; plus `head_coach`, `play_caller`, `play_caller_is_new` |
| `vegas.py` | team-week | `schedules` | `team_implied_total`, `opponent_implied_total`, `spread_line`, `total_line`, `div_game`, `roof` |
| `priors.py` | player-season | `rosters_weekly`, `players` | `age_years`, `years_exp`, `draft_year`, `draft_round`, `draft_pick`, `is_undrafted` |
| `availability.py` | player-week + player-season | `spine`, `injuries`, `depth_charts` | `prev_season_availability_rate`, `prev_season_games_played`, `played_rate_last{3,5,8}`, `games_missed_season_to_date`, `injury_report_status`, `injury_practice_status`, `depth_chart_rank` |
| `labels.py` | player-week + player-season | `spine`, `weekly_player_stats` | the two label tables |

## O-line features

No free line-charting exists in nflverse, so these are the box-score proxies
Football Outsiders' line metrics are built from — they isolate the line by
exploiting *where in a play's outcome distribution* the line's contribution
lives.

- **`ol_adj_line_yards`** — 120% of yards on stuffed runs, 100% of yards 0-4,
  50% of 5-10, 0% beyond. Capping the tail strips out most of the back's
  contribution. Two departures from FO's published metric, so **the number
  here is not comparable to theirs**: no opponent adjustment, and no
  rescaling. FO normalises the league mean to league YPC (~4.2); the raw
  banded credit is ~2.7 (verified: mean designed-run yardage 4.11 → mean
  banded credit 2.72). Rescaling was rejected deliberately — dividing by a
  league-season average means dividing by games that have not happened yet.
- **`ol_stuff_rate`** — designed runs gaining ≤ 0. League mean 21.9%.
- **`ol_power_success`** — conversion on ≤2 yards to go, 3rd/4th down or goal
  line. Null (not 0) when a team saw no such situation: "never tried" ≠
  "tried and failed".
- **`ol_sack_rate`, `ol_qb_hit_rate`** — per dropback. League sack mean 6.3%.
  Genuinely confounded by the QB, and exposed anyway: what a receiver's
  projection cares about is whether dropbacks survive long enough to become
  targets, and the QB's contribution to that is signal, not noise.

Scrambles are excluded from run-blocking reps — a scramble is a dropback that
broke down, and counting it would credit the line for the QB escaping bad
protection.

## Play-caller features, and what nflverse does not have

**nflverse has no offensive-coordinator field.** `load_schedules()` carries
`home_coach`/`away_coach` — head coach only. So `play_caller` defaults to the
head coach, which is right for the teams where the HC calls plays and wrong
for the rest.

`scheme.load_coordinator_map()` reads an optional hand-maintained
`data/manual/coordinators.csv` (`team,season,offensive_coordinator`) and
prefers it where present. Absent that file everything still builds; the
pipeline logs that it fell back.

`play_caller_is_new` is the column that earns its keep: a team whose caller
changed carries far less of its prior-season tendency forward, which lets a
model discount the `prev_season_*` columns rather than trust them uniformly.

Scheme tendencies ship in two views:
- **`_last{3,5,8}`** — in-season form, lagged like every other rolling block.
- **`prev_season_*`** — the completed prior season, constant within a season
  and known before week 1. This is the play-caller track record proper, and
  the only version available for an early-season or preseason projection.

Face validity, 2023: most pass-happy by PROE = KC, CIN; most run-committed =
ATL (−11.4), TEN, CHI; worst run blocking = NYJ, JAX.

## Leakage discipline (docs/features.md §7)

Enforced mechanically, and **verified on every build** by
`scripts/validate_features.py`, which recomputes rolling features from raw
play-by-play rather than trusting the construction.

Two mechanisms, by grain:

- **Team grain** (`oline`, `scheme`): teams play every week, so
  `add_trailing_rolling` shift(1)-then-rolls, which is exact.
- **Player grain** (`opportunity`, `redzone`, `game_script`): players miss
  games, so their rows are not a dense weekly series. Rolling is computed
  *inclusive* of each played game, then `attach_asof` binds each spine week
  to the most recent player row **strictly before** it (an as-of join at
  `week − 1`). A player who missed week 9 still gets week-9 features — his
  form as of week 8 — which is exactly the row an availability model needs
  and which the old shift-then-roll approach could not produce.

Also:
- Raw current-week columns are emitted by **no** feature module; they live in
  `labels.py`.
- Windows are games-based and do not reset at season boundaries (intentional
  Marcel-style continuity). `_std` columns do reset each season.
- Vegas lines are **not** lagged — a closing line is known pre-kickoff, so
  the week-*n* line is legitimately a week-*n* feature.

## Bugs this refactor fixed

- **Red-zone share used the wrong denominator.** It divided by the *player's
  own* touches, so `rush_rz_share` meant "what fraction of this player's
  carries came in the red zone" — a third-stringer handed the ball once from
  the four-yard line scored 1.000, a workhorse with 18 carries including 3
  inside the 20 scored 0.167. Close to the inverse of the intent. Denominator
  is now the team's red-zone touches; validated by asserting shares sum to 1
  across a team-week.
- **Vegas context silently dropped two franchises.** `load_schedules()` is
  the one loader that keeps historic franchise codes (`OAK`, `SD`) while
  player stats and pbp use `LV`/`LAC`, so the join failed for every Raiders
  week 2016-2019 and every Chargers week in 2016 — nulls concentrated
  entirely in two franchises, which is the shape of bias a temporal split
  will not catch. `config.normalize_team()` is now applied before every
  team join; null rate for LV/LAC is 0% across all nine seasons.
- **A season-level outcome was broadcast across weeks.** `games_played` was
  joined on `(player_id, season)`, so week 3 of 2023 carried how many games
  the player would go on to play through week 18. It was documented as a
  label, and it was — but nothing stopped it being read as a feature.
  Prior-season durability is now attached by a `season + 1` join key, and
  the label lives in `player_season_labels.parquet`.
- **Practice-squad rows would have swamped the availability negative class**
  (see spine section above).

## Known gaps / simplifications

- **`played` is offence-only** (`offense_snaps > 0`). Correct for RB/WR/TE;
  would need a `defense_snaps` fallback to generalise.
- **Injury-report nulls are filled `"Healthy"`** — ~95% of skill-position
  player-weeks have no report, so a null there reads as missing data rather
  than what it is. Filled after the spine join, not before.
- **`load_injuries()` is team-self-reported and known to be gamed**
  (docs/features.md §6) — exposed as a raw categorical, never as truth.
- **`snap_counts` is keyed by `pfr_player_id`**, everything else by
  `gsis_id` — joined through `players.parquet`'s crosswalk.
- **`injuries.parquet`'s `season`/`week` load as `Float64`** in this
  nflreadpy version — cast before joining.
- **Not built**: `load_ff_opportunity()` xFP baseline, FTN charting
  aggregates, opponent-adjusted matchups, weather. Route participation /
  TPRR from `participation.parquet` is deliberately skipped: `route` and
  `was_pressure` are ~62% null for 2016-2022 and only complete from 2023,
  so a feature built on them would be available for two seasons of a
  nine-season training window.
