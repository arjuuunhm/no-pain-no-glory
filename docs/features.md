# Feature Engineering Reference — NFL Player Performance Prediction

Companion to `resources.md` §8 (starter list). This is the deeper, prioritized version: exact
loader/column names, paid-vs-free flags, additional feature ideas with sourcing, known pitfalls, and a
v1/v2 priority split. Treat every column name below as "verify against the live data dictionary before
building" — nflverse column sets have shifted across versions (e.g. `nflreadr` → `nflreadpy` renamed some
fields to snake_case-consistent forms), so run `nflreadr::dictionary_player_stats` /
`nflreadpy.load_player_stats().columns` (or the R equivalents) at implementation time rather than trusting
this doc blindly. Where I could not confirm a column name against a rendered data dictionary (search/fetch
tooling only exposed partial tables for some pages), I've said so explicitly.

**Loader library note**: `nflreadpy` (Python, polars-based) is the current recommended Python interface per
[nflverse docs](https://nflreadpy.nflverse.com/); `nfl_data_py` is the older pandas API still common in
tutorials but less actively maintained. Function names below are given in `nflreadr` (R) form since that's
the canonical reference implementation and the one with the most complete public docs; `nflreadpy`
mirrors the same function names in `snake_case` (e.g. `load_player_stats()` in both).

---

## 0. Loader inventory (nflverse core + ffverse)

Confirmed from the [nflreadr reference index](https://nflreadr.nflverse.com/reference/index.html):

| Function | Loads | Free? |
|---|---|---|
| `load_pbp()` | Play-by-play, 1999– | Free |
| `load_player_stats()` | Weekly (and season) aggregated player stats, derived from pbp | Free |
| `load_team_stats()` | Weekly team-level stats | Free |
| `load_participation()` | Snap-level personnel/participation (offense/defense personnel groupings, men in box, pass rush) | Free, **coverage gaps** — see §2 |
| `load_players()` | Master player table (ids, bio, draft info) | Free |
| `load_rosters()` | Season rosters | Free |
| `load_rosters_weekly()` | Week-by-week roster status (active/inactive) | Free |
| `load_teams()` | Team metadata/colors/logos | Free (not a modeling feature) |
| `load_schedules()` | Game schedule incl. Vegas lines, weather, roof/surface | Free |
| `load_officials()` | Officiating crews per game | Free, niche |
| `load_trades()` | Trade transaction log | Free |
| `load_draft_picks()` | Draft picks (from Pro Football Reference) | Free |
| `load_combine()` | Combine measurables (from PFR) | Free |
| `load_nextgen_stats()` | NFL Next Gen Stats (tracking-derived, weekly) | Free (NFL publishes it) |
| `load_depth_charts()` | Weekly official depth charts | Free |
| `load_injuries()` | Weekly injury reports (practice + game status) | Free |
| `load_espn_qbr()` | ESPN's QBR | Free |
| `load_pfr_advstats()` | PFR advanced stats: pressure, drops, broken tackles, target depth, etc. | Free |
| `load_snap_counts()` | PFR snap counts (offense/defense/ST snaps and %) | Free |
| `load_contracts()` | Historical contracts (OverTheCap) | Free |
| `load_ftn_charting()` | FTN manual play charting (2022–) | Free, but **narrower than paid charting** — see §2 |
| `load_ff_playerids()` | Cross-source ID crosswalk (gsis, pfr, sleeper, etc.) | Free |
| `load_ff_rankings()` | FantasyPros consensus rankings | Free |
| `load_ff_opportunity()` | **Pre-built xFP model output** (ffopportunity package) | Free |

Sources: [nflreadr reference index](https://nflreadr.nflverse.com/reference/index.html),
[nflreadpy docs](https://nflreadpy.nflverse.com/).

**Important**: `load_ff_opportunity()` already ships a play-by-play xgboost expected-fantasy-points model
(`ep_pbp_pass`, `ep_pbp_rush`, and `ep_weekly` outputs) — [ffopportunity docs](https://ffopportunity.ffverse.com/),
[dictionary](https://nflreadr.nflverse.com/reference/dictionary_ff_opportunity.html). Before building a
custom xFP model from scratch (resources.md §2's stated goal), pull this and use `actual − expected` from
it as a baseline feature/diagnostic; only build a bespoke xFP model if you need finer control (e.g.
different scoring settings, additional context features) than the published one provides.

---

## 1. Opportunity / volume features

### 1a. From `load_player_stats()` (weekly, pbp-derived)
Per the [player stats data dictionary](https://nflreadr.nflverse.com/articles/dictionary_player_stats.html)
(~115 columns) and confirmed additions in [nflfastR PR #265](https://github.com/nflverse/nflfastR/pull/265),
the weekly table carries, among others:

- **Passing**: `attempts`, `completions`, `passing_yards`, `passing_tds`, `interceptions`,
  `passing_air_yards`, `passing_yards_after_catch`, `passing_first_downs`, `passing_epa`, `sacks`,
  `sack_yards`, `pacr` (Passing Air Conversion Ratio = `passing_yards / passing_air_yards`), `dakota`
  (EPA+CPOE composite, described as "the coefficients that best predict adjusted EPA/play in the following
  year" — i.e. explicitly built to be *stable*, worth pulling as a QB efficiency-persistence feature).
- **Rushing**: `carries`, `rushing_yards`, `rushing_tds`, `rushing_first_downs`, `rushing_epa`,
  `rushing_fumbles`.
- **Receiving**: `targets`, `receptions`, `receiving_yards`, `receiving_tds`, `receiving_air_yards`,
  `receiving_yards_after_catch`, `receiving_first_downs`, `receiving_epa`, `racr` (Receiver Air Conversion
  Ratio = `receiving_yards / receiving_air_yards`), `target_share`, `air_yards_share`, `wopr`
  (`1.5 × target_share + 0.7 × air_yards_share`, coefficients fit by Josh Hermsmeyer's original regression —
  [RotoViz origin](https://www.rotoviz.com/2018/08/introducing-a-better-metric-than-targets-air-yards/)).
- **Fantasy**: `fantasy_points`, `fantasy_points_ppr` (standard/PPR pre-computed — useful as a raw label or
  sanity check, but the whole point of the opportunity-first approach is *not* to regress on these
  directly for the opportunity stage).

These `target_share` / `air_yards_share` / `wopr` fields are computed at the **team-week** level
(share of *that team's* targets/air yards that week), so they're already opportunity-normalized — no need
to recompute share manually, though you should still sanity-check the denominator (e.g. a QB change
mid-game affecting whose air yards count).

### 1b. From `load_snap_counts()` (PFR)
Game-level snap counts and percentages. Expected fields per the [function reference](https://nflreadr.nflverse.com/reference/load_snap_counts.html)
and PFR's public snap-count tables: `game_id`, `pfr_player_id`, `position`, `team`, `opponent`,
`offense_snaps`, `offense_pct`, `defense_snaps`, `defense_pct`, `st_snaps`, `st_pct`. `offense_pct` is the
snap share feature; combine with `load_player_stats()` targets/carries to get **per-snap rates**
(targets/snap, carries/snap) which are a cleaner opportunity signal than raw counts for players whose role
is changing week to week.

### 1c. From `load_ftn_charting()` (2022–, free)
Manually charted by FTN, released via nflverse within 48h of each game. Confirmed columns from the
[FTN dictionary](https://nflreadr.nflverse.com/articles/dictionary_ftn_charting.html):
`is_no_huddle`, `is_motion`, `is_play_action`, `is_screen_pass`, `is_rpo`, `is_trick_play`,
`is_qb_out_of_pocket`, `is_interception_worthy`, `is_throw_away`, `read_thrown`, `is_catchable_ball`,
`is_contested_ball`, `is_created_reception`, `is_drop`, `is_qb_sneak`, `n_blitzers`, `n_pass_rushers`,
`n_offense_backfield`, `qb_location`, `starting_hash`. Note this is **play-level charting joined to pbp**,
not player-week aggregates — you'd aggregate `is_drop`, `is_contested_ball`, `is_created_reception` by
receiver-week yourself. **FTN does not include route participation as a column** in the public release —
that remains a paid-data gap (see §2).

### 1d. RB-specific
- Carries and targets already covered above (`carries`, `targets` in `load_player_stats()`).
- **Goal-line / inside-10 carries**: not a pre-built column anywhere in nflverse; derive from `load_pbp()`
  by filtering `yardline_100 <= 10` (or `<= 5` for "goal-line" strictly) and `rush_attempt == 1`, grouped by
  rusher-week. This is a real and cited signal (touchdown equity concentrates in short-yardage work), but
  it's a low-n stat per player-week — smooth with a rolling multi-week window, don't use single-game
  inside-10 counts raw.
- **Two-minute-drill / hurry-up work**: derivable from `load_pbp()`'s `half_seconds_remaining` and
  `no_huddle`/`play_type`; speculative feature, weak standalone evidence, mention only as a v2 idea.

### 1e. QB-specific
- `attempts`, `carries` (designed QB runs are mixed into `rushing_yards`/`carries` in `load_player_stats()`
  — you may need `load_pbp()`'s `qb_scramble` flag to separate scrambles from designed runs if that
  distinction matters for your model; scrambles are more sample-size-driven and less repeatable than
  designed run rate).
- `dakota` as above — an explicitly stability-optimized QB efficiency metric, worth using as a *prior*
  feature for the efficiency-stage model rather than reinventing EPA/CPOE blending yourself.

---

## 2. Paid / charted data vs. free

| Feature | Free (nflverse) | Paid needed |
|---|---|---|
| Target share, air yards share, WOPR, snap share | Yes | — |
| aDOT (`passing_air_yards / attempts`, or NGS's `avg_intended_air_yards`) | Yes | — |
| **Route participation %** | **No** — nflverse does not publish route-run counts | PFF, [Fantasy Points Data](https://www.fantasypoints.com/data), Sports Info Solutions |
| **TPRR (targets per route run)** | **No** — needs route participation as denominator | Same as above |
| Coverage scheme (man/zone), CB matchups | Partial (`load_participation()` has some personnel/box counts; FTN has some play-context flags) | PFF has the most complete coverage charting |
| O-line PFF grades / pressure allowed by lineman | No PFF grades free | PFF |
| Pressure rate **allowed by the offense as a whole** | Partially derivable: `load_pfr_advstats()` (pass) has PFR-charted pressure/hurry/hit/blitz counts against a QB, and `load_ftn_charting()` has `n_pass_rushers`/`n_blitzers` per play | PFF gives cleaner lineman-level attribution |
| Broken tackles, yards after contact | `load_pfr_advstats()` (rush: yards before/after contact, broken tackles; rec: broken tackles, drop rate) | PFF adds more granular "elusive rating" style composites |
| NGS separation, cushion, RYOE, CPOE | Yes — `load_nextgen_stats()` | — |

**Practical read**: nflverse + FTN + PFR-advanced covers volume, most efficiency proxies, and a
reasonable amount of context. **Route participation / TPRR is the single highest-value gap** — resources.md
§2 already flags this correctly. Everything else paid data buys you is refinement (better coverage
attribution, offensive line grading) rather than a fundamentally new signal category. Don't block a v1
baseline on acquiring paid data.

---

## 3. Efficiency features (regress hard, don't trust raw)

- `racr`, `pacr` — conversion ratios, noisy at the player-season level; league-average `racr` is the
  natural shrinkage target, and per Hermsmeyer's original research this is exactly why WOPR (a volume
  metric) was built to replace raw efficiency ratios as a *predictive* stat.
- **YPC, YPT, TD rate** — all classic high-variance, touchdown-rate-especially-so metrics. TD rate
  regression to position-average is well documented in the fantasy analytics community (RotoViz "touchdown
  regression" articles) as one of the most reliable single regression-to-mean plays in the field —
  practically, model TDs as `opportunities × position-average TD rate per opportunity type` (redzone
  target, redzone carry, non-RZ target, etc.) rather than letting a raw per-player TD rate persist.
- **CPOE (completion percentage over expected)** — in `load_nextgen_stats()` as
  `completion_percentage_above_expectation`, and also feeds into `dakota`. More stable than raw completion
  %, still noisy at <200 attempts.
- **RYOE (rushing yards over expected)** — `rush_yards_over_expected`,
  `rush_yards_over_expected_per_att` in `load_nextgen_stats()`. NFL's own tracking-based expected-yards
  model ([NFL.com explainer](https://www.nfl.com/news/next-gen-stats-intro-to-expected-rushing-yards)) —
  a legitimate efficiency signal, but RB efficiency in general has some of the weakest year-over-year
  correlation of any skill-position stat (well established in the opportunity-vs-efficiency stability
  literature resources.md §2 cites).
- **YAC over expected** (`avg_yac_above_expectation` in NGS) — receiver efficiency signal partially
  separable from QB/scheme, still shrink hard.

---

## 4. Game context / team environment / Vegas

- **Vegas lines**: `load_schedules()` carries betting-market columns — based on the
  [function reference](https://nflreadr.nflverse.com/reference/load_schedules.html) and general nflverse
  schedule structure, expect `spread_line`, `total_line`, `away_moneyline`, `home_moneyline` (verify exact
  names against `dictionary_schedules` — confirmed the dictionary exists with 45 documented fields, but I
  could not get the rendered column table through available tooling). Derive **team-implied point total**
  as `total_line/2 ± spread_line/2` — this is the single best free proxy for offensive context per
  resources.md §1, and is worth building before almost anything else.
- **Weather / venue**: same `load_schedules()` table carries `roof` (dome/outdoors/closed/open), `surface`,
  and (for outdoor/open games) `temp` and `wind` columns. Filter to `roof %in% c("outdoors","open")` before
  using temp/wind — indoor games will have garbage/NA weather values that should not be treated as "calm
  weather," they should be excluded or flagged separately. Wind is the more predictive of the two for
  passing volume/efficiency (extreme wind suppresses passing more than cold does — general meteorology +
  NFL analytics consensus, e.g. Warren Sharp's weather work), cold alone has a much weaker effect than
  commonly assumed.
- **PROE (pass rate over expectation)**: not a pre-aggregated `load_*()` table — derive from
  `load_pbp()`'s `xpass` column (win-probability/situation-conditioned expected dropback probability,
  computed by nflfastR's own model) as `mean(pass) - mean(xpass)` per team-week/season. Background:
  [Establish The Run explainer](https://establishtherun.com/pass-rate-over-expectation/),
  [FantasyLife explainer](https://www.fantasylife.com/articles/redraft/what-is-pass-rate-over-expected).
  Use **team-week PROE trailing average**, not a single game — single-game PROE is heavily driven by score
  script (see pitfalls, §7).
- **Score differential / game script**: from `load_pbp()`, bucket `score_differential` at snap time (e.g.
  garbage time flags, "trailing by 8+", "leading by 8+", "one-score game") and compute the *share of a
  player's snaps/targets/carries* that occurred in each bucket. This is explicitly called out in
  resources.md §6 as separating top vs. median Big Data Bowl entries — treat as a v1 feature for RBs
  especially (game script is the dominant driver of carry volume for a lead-back).
- **Offensive line quality**: no free PFF-grade equivalent; nearest free proxies are
  `load_pfr_advstats()` pressure/sack-allowed rates aggregated to team-week, or pass block win rate if you
  can source it from ESPN's public (non-nflverse) release. Treat as a partially-missing feature — team
  fixed effects or a coarse "top-10/bottom-10 pass block" bucket from public reporting is a reasonable v1
  substitute.
- **Coaching/OC continuity**: no dedicated loader; derive from `load_rosters()`/public coaching-change
  trackers (Pro Football Reference coaching history pages) — a manual join, not an automated pull. Flag as
  build-yourself.
- **Rest days / travel**: `load_schedules()` has `gameday`/`gametime`/`away_rest`/`home_rest` style fields
  in most nflverse schedule releases (short week / bye-week rest advantage is a well-established situational
  factor in betting-market research — Vegas already prices it into the spread, which is itself a reason to
  treat rest-day features as *largely redundant with the spread* rather than incremental signal; see
  pitfalls §7). Travel distance/time-zone-change is not in nflverse; would need to derive from stadium
  lat/long (available via `load_teams()` or a manual stadium table) — mark as speculative/v2, evidence for
  timezone/travel effects at the NFL level is much weaker and noisier than in e.g. MLB.

---

## 5. Player priors (age, draft capital, experience)

- **Age**: `load_players()` / `load_rosters()` carry birth date; compute age-at-season-start. Age curves
  are strongly position-specific and non-linear — RBs decline early and sharply (commonly cited inflection
  around age 27–28 in public research), WRs peak later (~26–27) with a gentler decline,
  [Open Source Football](https://opensourcefootball.com/) has multiple aging-curve posts worth pulling
  directly rather than re-deriving from scratch. Use a spline or position-specific bucketed dummy, not a
  single linear age term across all positions — resources.md's own CLAUDE.md flags this explicitly.
- **Draft capital**: `load_draft_picks()` (PFR-sourced) gives `round`, `pick`, `age` at draft, and college
  production context in some releases. Draft capital is the dominant prior for players with <2 NFL seasons
  of data (CLAUDE.md explicitly calls this out) — a simple `1/pick` or `265 - pick` transform, or bucketed
  round dummies, both work; there's no strong evidence one functional form dominates, so this is a place to
  let the GBM find the shape rather than hand-engineer it, as long as raw pick number is exposed as a
  feature.
- **Combine measurables**: `load_combine()` — 40-time, vertical, broad jump, etc. Weak/mixed predictive
  power once draft capital is already in the model (draft capital already prices most of what teams
  learned from the combine); include as supplementary features but expect draft round/pick to dominate
  importance.
- **Years of experience**: derivable from `load_rosters()`'s season history per `gsis_id`, or directly from
  `load_players()` if it carries a rookie-season field. Correlated with age but not identical (late-round
  rookies who redshirt on IR, UDFAs, etc.) — worth keeping both.
- **Target competition added/lost**: build by joining `load_rosters()` team rosters year-over-year against
  prior-season `target_share`/`carries` from `load_player_stats()` — sum the departed/arriving teammates'
  prior-year opportunity share at the same position. This is a real and important offseason-context signal
  (a WR2's target-share ceiling changes materially if the WR1 leaves in free agency) with no ready-made
  loader; expect to build the join yourself.

---

## 6. Availability / injury features

- **`load_injuries()`**: weekly injury report data. Expected fields based on the standard NFL injury report
  format that this loader mirrors: `report_status` (Questionable/Doubtful/Out), `practice_status`
  (Full/Limited/DNP participation each practice day), `report_primary_injury`/`report_secondary_injury`
  (body part / type). I could not confirm the exact column names via a rendered dictionary table in this
  session — verify against `nflreadr::dictionary_injuries()` before building. Note the NFL injury report is
  **self-reported by teams and known to be gamed** (teams list players as "questionable" for competitive
  reasons, not just health) — treat `report_status` as a noisy, somewhat adversarial signal, not ground
  truth severity.
- **`load_rosters_weekly()`**: week-by-week active/inactive status — the cleanest **realized availability**
  label (games actually played), as opposed to the injury report's stated *risk*. Use this to construct the
  availability model's label (games played out of scheduled games).
- **`load_depth_charts()`**: weekly official depth chart position/rank. Use depth-chart rank (starter vs.
  RB2/RB3, etc.) as a same-week opportunity-risk feature and, combined with `load_snap_counts()`, to detect
  committee backfields.
- **Age × position injury base rates**: not a loader; compute from historical
  `load_injuries()`/`load_rosters_weekly()` joins yourself — games-missed rate by position and age bucket.
  RBs have the highest in-season injury/missed-game rate of the skill positions in most public injury-rate
  writeups (general fantasy-analytics consensus, not a single canonical citation) — worth encoding as a
  positional prior in the availability model.
- **Injury history (prior 1–2 seasons)**: games missed in `t-1`/`t-2` is one of the stronger predictors of
  future missed time (injury recurrence is real, especially soft-tissue and joint injuries) — build from
  `load_rosters_weekly()` inactive counts, not from the injury *report*, to avoid the self-report noise
  above.

---

## 7. Known pitfalls

**Leakage**
- **Season-to-date shares computed without a lag** — if `target_share` for week 8 is computed using target
  counts *through* week 8 inclusive, you're leaking that week's own outcome into its own feature. Always
  lag: features for week *n* must be computed from weeks `< n` only (or `<= n-1`).
- **`load_ff_opportunity()`'s xFP model itself is trained on historical pbp** — if you use it as a feature,
  confirm it isn't implicitly trained on data that overlaps your test season; treat any pre-built model
  output the same as any other feature for temporal-validation purposes.
- **Depth chart / injury report snapshots**: `load_depth_charts()` and `load_injuries()` should be pulled
  **as of the Thursday/Friday before a given week's games**, not with hindsight from later in the week
  (status can change Wed→Fri) or from after the game (final inactive list). Pin the snapshot timing
  explicitly in the pipeline.
- **Vegas lines**: use the **closing line** only if you're doing a retrospective backtest that mimics
  betting on the closing number; for a projection meant to be usable *before* kickoff, use the line as of
  your projection's cutoff time, not the closing line, or you'll overstate real-world usable accuracy.
- **End-of-season roster/depth-chart fields** in `load_rosters()` (non-weekly) reflect final-season status,
  not the roster as it stood at each week — use `load_rosters_weekly()` for anything used as a per-week
  feature. The one deliberate exception is a season that has not been played, where no weekly roster
  exists at all (see "Projecting a season that hasn't started" below).

**Sample-size / small-n bias**
- **Efficiency ratios (RACR, PACR, YPC, TD rate) are mechanically noisier for low-volume players** — a
  4-target game with 1 TD produces an absurd per-target TD rate. Any ratio feature needs either a minimum
  attempt/target floor before being trusted, or a shrinkage/Bayesian-prior blend (e.g. empirical Bayes
  toward the position mean, weighted by attempts) rather than being fed raw into the model.
- **Rolling last-N-game windows** (resources.md §8's "learned Marcel block") have the same problem for
  players who missed games — a "last 3 games" window spanning 6 calendar weeks for an injury-interrupted
  player mixes stale and fresh information in a way a window spanning 3 consecutive weeks doesn't. Consider
  encoding recency in *calendar weeks since* rather than *games since*, or including games-since-window-start
  as its own feature.
- **Route participation / TPRR-style rate stats, if acquired via paid data, are extremely small-n early in
  a rookie's career** — same shrinkage logic applies, doubly so given rookies are exactly the case where you
  most want the stat to be informative.
- **Combine/athletic-testing outliers**: small numbers of combine reps per position per class mean extreme
  percentile claims (e.g. "99th percentile burst score") are themselves noisy; don't treat combine
  percentiles as more precise than they are.

**Projecting a season that hasn't started**

Everything above assumes the season being predicted has rows. A *draft-day*
projection for an upcoming season does not, and the gap is structural rather
than a matter of waiting for data:

- **`load_rosters_weekly()` has no upcoming season and will not fake one** — it
  raises `ValueError: Season must be between 2002 and 2025` when asked for 2026.
  The seasonal `load_rosters()` *does* carry it, uses the same `status`
  vocabulary, and is therefore the only available spine. This is the exception
  to the leakage bullet above: the objection to seasonal rosters is that they
  describe end-of-season status, which is irrelevant for a season with no
  status to be end-of yet.
- **Team-grain blocks emit no row for a season with no games**, so every team
  feature joins to null — silently, since a left join on a missing key is not
  an error. `utils.append_upcoming_week` adds a placeholder team-week *before*
  the rolling step; because the roll is shift(1)-then-rolling and deliberately
  does **not** reset at season boundaries (§7's Marcel-style continuity), the
  placeholder receives the trailing window over the end of the prior season —
  which is exactly what a real week-1 row receives, not an approximation.
- **Age and draft capital come from a different source than experience.**
  `build_prior_features` reads `years_exp` from whichever roster frame it is
  handed and the rest from `load_players()`; hand it the weekly file alone and
  every upcoming-season row loses age and draft capital, which are precisely
  the features a projection leans on for young players.
- **The output must not reach the label tables.** A row for an unplayed season
  has no outcome, and `model/panel.py` fills a missing outcome with zero — so
  an unplayed season quietly becomes a real zero-point season to train on.
  `preseason_features.parquet` is a separate file for that reason.
- **The failure mode throughout is silence, not exceptions.** Two of these bit
  during implementation (an unnormalised `AZ` abbreviation, and experience read
  from the wrong frame) and neither raised anything — the projection ran and
  produced plausible-looking numbers with an all-null column behind them.
  `scripts/validate_projections.py` compares preseason feature coverage against
  a real week-1 row for this reason.

See `docs/modeling.md`, "Projecting a season that has not been played", for the
end-to-end command sequence and what a projection built before the late-August
cut deadline is actually worth.

**Redundancy**
- **WOPR is a linear combination of `target_share` and `air_yards_share`** — including all three as
  separate features is not wrong for a GBM (trees handle collinearity fine for split-finding) but adds no
  information and can muddy feature-importance interpretation; if you want interpretable importances,
  pick one representation, not all three.
- **`dakota` already blends EPA and CPOE** — including raw `passing_epa` and NGS's
  `completion_percentage_above_expectation` alongside `dakota` is partially redundant; fine for prediction,
  redundant for interpretation.
- **PROE and `xpass`-derived features vs. Vegas spread**: teams that are big underdogs pass more
  (game-script-driven), and the spread already prices in expected game script — a team-week PROE feature
  and a "team is trailing/leading" score-differential feature can end up capturing much of the same
  variance as the spread itself. Not wrong to include all three, but be aware they're correlated and
  interpret importances accordingly.
- **Rest-day features and the spread** (§4 above) — Vegas already incorporates rest/travel into the line;
  including rest days as a separate feature is testing whether there's *residual* signal beyond what the
  market priced, which is a much weaker prior than treating it as an independent strong feature.
- **Age and years-of-experience and draft-class-age-at-draft**: three overlapping proxies for "how much
  wear/development has this player had." Keep all three if you're letting the GBM sort it out, but recognize
  they're highly collinear, especially age and experience.

---

## 8. Contract-year effect — honest assessment

resources.md doesn't currently mention this; flagging since it's a natural feature idea and the evidence is
mixed enough to be worth calling out explicitly rather than including on faith.

- The strongest lay treatment ([Slate, "Contract year effect: Do sports free agents try harder?"](http://www.slate.com/articles/sports/sports_nut/2012/05/contract_year_effect_do_sports_free_agents_try_harder_.html))
  cites an undergraduate thesis at Brown finding **no evidence** for a classic contract-year performance
  bump in the NFL, and offers a plausible mechanism for why: NFL contracts are largely non-guaranteed, so
  players already have a strong incentive to perform every year, diluting any extra "contract year" effect
  relative to guaranteed-contract sports like the NBA/MLB (see the
  [Wikipedia summary of the contract-year phenomenon](https://en.wikipedia.org/wiki/Contract_year_phenomenon)
  for the cross-sport comparison).
- More recent econometric work looking specifically at **contract structure** (not just "is this a walk
  year") finds real effects: performance (WPA/EPA change) is reported to be meaningfully **lower** after
  signing a contract with a higher guaranteed-money share, for QBs specifically, per work cited in
  [ScienceDaily's coverage of NFL contract-timing research](https://www.sciencedaily.com/releases/2014/07/140723123851.htm)
  and a [ScienceDirect study on NFL player pay and performance](https://www.sciencedirect.com/science/article/abs/pii/S0378426618300098).
- **Net read**: don't build a blunt "is this a contract year" binary flag expecting a performance bump —
  the better-supported effect is "guaranteed-money share of the *current* contract predicts a *decline*"
  (a discipline/incentive effect, not a hustle effect), and it's strongest for QBs specifically in the
  literature found here. This requires `load_contracts()` (OverTheCap data) joined on guaranteed-$ share.
  Mark this a **v2 stretch, low-confidence** feature — plausible mechanism, thin and somewhat conflicting
  evidence base, meaningfully harder to build correctly (needs contract terms, not just "expiring
  contract: yes/no").

---

## 9. Additional feature ideas beyond resources.md §8 — strength assessment

| Idea | Strength | Notes |
|---|---|---|
| Team-implied point total (Vegas) | **Strong** | Best free proxy for offensive context; resources.md §1 already flags it as a must-have |
| Score-differential game-script buckets | **Strong** | Explicitly named in resources.md §6 as separating top Big Data Bowl entries |
| Red-zone / inside-10 opportunity share | **Strong**, but low-n per game | TD equity concentrates here; smooth over rolling windows |
| Target competition added/lost (offseason) | **Strong**, intuitive, and already named in resources.md §8 — reinforcing it here since it's easy to under-build (a shallow "new-WR1-arrived: yes/no" flag loses most of the value vs. the full share-weighted version) | |
| PROE (team-level) | **Moderate-strong** | Real signal, but partially redundant with spread/game-script — see pitfalls §7 |
| TPRR / route participation | **Strong signal, paid-data gated** | Highest ceiling feature not free in nflverse |
| Down/distance situational splits (e.g. 3rd-down target share, 2-minute usage) | **Moderate** | Meaningful for pass-catching-RB / slot-WR roles specifically; less useful as a blanket feature |
| Weather (wind especially, outdoor games) | **Moderate** for passing volume/efficiency; **weak** for run-heavy game-plan shifts | Filter dome/indoor games out rather than imputing |
| Rest days / short week | **Weak incremental** over Vegas spread | Market already prices it; see pitfalls §7 |
| Travel distance / time-zone change | **Weak/speculative** at NFL scale | Unlike MLB's dense schedule, weekly NFL games give players a full week to recover; treat as v2 curiosity, not a v1 feature |
| Contract-year flag (simple) | **Weak as commonly framed**; contract *structure* (guaranteed $ share) has some real support | See §8 |
| Coaching/OC change | **Moderate**, already in resources.md §8 as a flag; scheme fit for specific skill sets (e.g. WR entering a route tree that suits them) is a real but hard-to-quantify sub-effect — don't over-promise a binary flag will fully capture it | |
| Strength-of-schedule-adjusted opponent defense (points/targets/yards allowed by position, trailing) | **Moderate-strong**, standard in DFS/fantasy projection systems | Build from `load_player_stats()` aggregated to opponent-week — "positional matchup" tables are a fantasy-industry staple (e.g. FantasyPros/4for4 "defense vs. position" rankings) but should be built on a trailing, opponent-adjusted basis, not raw season totals which mix in strength-of-opponent-faced noise themselves |
| Offensive line pass-block/run-block quality | **Moderate**, data-limited without PFF | See §4 |
| Combine measurables beyond draft capital | **Weak once draft capital is in the model** | See §5 |

---

## 10. Priority tiers

### v1 baseline — build first
1. **Opportunity core**: `target_share`, `air_yards_share`, `wopr`, `carries`, snap share
   (`load_player_stats()` + `load_snap_counts()`), all with proper lagging (§7) and rolling last-3/5/8-game
   + season-to-date windows.
2. **Vegas team context**: implied team point total, spread, total, derived from `load_schedules()`.
3. **Score-differential / game-script buckets** from `load_pbp()`.
4. **Age (position-specific curve)** and **draft capital** (`load_draft_picks()`), per CLAUDE.md's explicit
   call-out that boosting won't invent these on its own.
5. **Availability label + basic injury features**: games played from `load_rosters_weekly()`, current-week
   `report_status` from `load_injuries()`, depth-chart rank from `load_depth_charts()`.
6. **Target competition added/lost** (offseason roster-turnover join) — cheap to build, high signal.
7. **Red-zone/inside-10 opportunity share**, rolling-window smoothed.
8. **`load_ff_opportunity()` xFP output** pulled as a ready-made baseline/diagnostic feature and as an
   ensemble candidate against your own opportunity model.

### v2 stretch — after v1 baseline is running end-to-end
1. **TPRR / route participation** — only if paid data is acquired.
2. **Team PROE** (trailing, from `xpass`) — add once you've confirmed it isn't just re-deriving the spread.
3. **Opponent-adjusted positional matchup features** (defense vs. position, trailing/SOS-adjusted).
4. **PFR advanced stats** (pressure rate allowed, broken tackles, drop rate) from `load_pfr_advstats()`.
5. **FTN charting aggregates** (play-action rate, motion rate, contested-catch rate) from
   `load_ftn_charting()`.
6. **Weather** for outdoor games (wind primarily).
7. **Coaching/OC continuity flag** and scheme-fit features (requires manual data assembly).
8. **Contract structure (guaranteed-$ share)** — low confidence, see §8.
9. **Combine measurables** as a supplement to draft capital.
10. **Injury-history base rates by age × position**, trailing multi-season, for the availability model.
11. **Rest days / travel** — include only after confirming incremental value over the spread in an
    ablation; don't assume it adds signal.

---

## Sources referenced

- [nflreadpy docs](https://nflreadpy.nflverse.com/)
- [nflreadr reference index](https://nflreadr.nflverse.com/reference/index.html)
- [nflreadr player stats data dictionary](https://nflreadr.nflverse.com/articles/dictionary_player_stats.html)
- [nflreadr Next Gen Stats data dictionary](https://nflreadr.nflverse.com/articles/dictionary_nextgen_stats.html)
- [nflreadr FTN charting data dictionary](https://nflreadr.nflverse.com/articles/dictionary_ftn_charting.html)
- [nflreadr load_schedules reference](https://nflreadr.nflverse.com/reference/load_schedules.html)
- [nflreadr load_snap_counts reference](https://nflreadr.nflverse.com/reference/load_snap_counts.html)
- [nflreadr load_injuries reference](https://nflreadr.nflverse.com/reference/load_injuries.html)
- [nflreadr load_ff_opportunity reference](https://nflreadr.nflverse.com/reference/load_ff_opportunity.html) /
  [ffopportunity docs](https://ffopportunity.ffverse.com/) / [ffopportunity GitHub](https://github.com/ffverse/ffopportunity)
- [nflfastR PR #265 — adding RACR/WOPR](https://github.com/nflverse/nflfastR/pull/265)
- [nflfastR nfl_stats_variables reference](https://www.nflfastr.com/reference/nfl_stats_variables.html)
- [WOPR origin — RotoViz, Josh Hermsmeyer](https://www.rotoviz.com/2018/08/introducing-a-better-metric-than-targets-air-yards/)
- [NFL.com — expected rushing yards explainer](https://www.nfl.com/news/next-gen-stats-intro-to-expected-rushing-yards)
- [Establish The Run — Pass Rate Over Expectation](https://establishtherun.com/pass-rate-over-expectation/)
- [FantasyLife — What is PROE](https://www.fantasylife.com/articles/redraft/what-is-pass-rate-over-expected)
- [Open Source Football](https://opensourcefootball.com/)
- [Slate — contract year effect](http://www.slate.com/articles/sports/sports_nut/2012/05/contract_year_effect_do_sports_free_agents_try_harder_.html)
- [Wikipedia — contract year phenomenon](https://en.wikipedia.org/wiki/Contract_year_phenomenon)
- [ScienceDaily — NFL contract timing research](https://www.sciencedaily.com/releases/2014/07/140723123851.htm)
- [ScienceDirect — NFL player pay and performance](https://www.sciencedirect.com/science/article/abs/pii/S0378426618300098)
