"""Shared paths, constants, and logging setup for the data pipeline."""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

# Repo root = two levels up from this file (src/nflforecast/config.py -> repo root)
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MANUAL_DIR = DATA_DIR / "manual"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# The project is scoped to non-QB skill positions: these are the players whose
# fantasy value is driven by *opportunity share*, which is the quantity
# resources.md §4 step 2 says boosting should model. QB volume is nearly
# constant given availability and K/DST are separate problems entirely.
SKILL_POSITIONS = ("RB", "WR", "TE")

# The active historical sample and its unplayed forecast target. Raw files may
# retain older seasons as a cache, but feature construction and every model
# path operate only on this declared window.
HISTORY_START_SEASON = 2019
HISTORY_END_SEASON = 2025
TARGET_SEASON = 2026

# Trailing-window sizes used by every rolling feature block. One definition so
# opportunity/red-zone/game-script/O-line features stay directly comparable.
ROLLING_WINDOWS = (3, 5, 8)

# nflverse standardises team abbreviations in most loaders, but *not* in
# load_schedules(), which keeps the historic franchise codes. Measured on
# 2016-2024: schedules carries OAK and SD while player stats and pbp carry LV
# and LAC, so an unguarded join on `team` silently drops every Raiders
# (2016-2019) and Chargers (2016) row. Normalise everywhere before joining.
TEAM_ABBR_FIXES = {
    "OAK": "LV",
    "SD": "LAC",
    "STL": "LA",
    "LAR": "LA",
    # `load_rosters()` (seasonal) says AZ where every other loader says ARI.
    # Found via the preseason spine, where it silently nulled every team-grain
    # feature for one team's worth of players -- the exact failure mode this
    # map exists to prevent.
    "AZ": "ARI",
}


def normalize_team(col: str) -> pl.Expr:
    """Map historic/variant franchise codes onto current nflverse abbreviations."""
    return pl.col(col).replace(TEAM_ABBR_FIXES).alias(col)


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s", "%H:%M:%S")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger
