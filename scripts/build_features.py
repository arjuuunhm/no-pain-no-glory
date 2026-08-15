#!/usr/bin/env python3
"""Entrypoint: build the player-week feature and label tables from data/raw/*.parquet.

Usage:
    python scripts/build_features.py

Requires data/raw/*.parquet to already exist -- run build_dataset.py first.
Writes data/processed/{player_week_features,player_week_labels,player_season_labels}.parquet.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nflforecast.features import build_feature_table


def main() -> int:
    features, weekly_labels, season_labels = build_feature_table()
    print(f"player_week_features: {features.height} rows x {features.width} cols")
    print(f"player_week_labels:   {weekly_labels.height} rows x {weekly_labels.width} cols")
    print(f"player_season_labels: {season_labels.height} rows x {season_labels.width} cols")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
