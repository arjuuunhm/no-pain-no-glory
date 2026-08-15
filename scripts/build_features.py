#!/usr/bin/env python3
"""Entrypoint: build the player-week feature and label tables from data/raw/*.parquet.

Usage:
    python scripts/build_features.py
    python scripts/build_features.py --upcoming-season 2026

Requires data/raw/*.parquet to already exist -- run build_dataset.py first.
Writes data/processed/{player_week_features,player_week_labels,player_season_labels}.parquet.

`--upcoming-season` additionally builds `preseason_features.parquet`: the same
161 columns for a season that has not been played, off the seasonal roster
file rather than the weekly one (which does not carry a season before it
starts). Those rows are written separately and never reach the label tables --
see `features/build_feature_table.py`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nflforecast.features import build_feature_table


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--upcoming-season",
        type=int,
        help="also build preseason rows for this unplayed season, e.g. 2026",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    features, weekly_labels, season_labels = build_feature_table(
        upcoming_season=args.upcoming_season
    )
    print(f"player_week_features: {features.height} rows x {features.width} cols")
    print(f"player_week_labels:   {weekly_labels.height} rows x {weekly_labels.width} cols")
    print(f"player_season_labels: {season_labels.height} rows x {season_labels.width} cols")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
