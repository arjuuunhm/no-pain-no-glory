#!/usr/bin/env python3
"""Entrypoint: run all data pullers and land parquet files in data/raw/.

Usage:
    python scripts/build_dataset.py --seasons 2022 2023 2024
    python scripts/build_dataset.py --start 2016 --end 2024

Safe to re-run: each puller overwrites its own output file with a fresh pull.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Allow running as `python scripts/build_dataset.py` without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nflforecast.config import get_logger
from nflforecast.pullers import (
    pull_weekly_player_stats,
    pull_snap_counts,
    pull_rosters,
    pull_player_ids,
    pull_draft_picks,
    pull_nextgen_stats,
    pull_schedules,
    pull_participation,
    pull_play_by_play,
    pull_rosters_weekly,
    pull_injuries,
    pull_depth_charts,
    pull_players,
)

logger = get_logger("build_dataset")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seasons", type=int, nargs="+", help="Explicit list of seasons, e.g. --seasons 2022 2023 2024")
    p.add_argument("--start", type=int, default=2022, help="Start season (inclusive) if --seasons not given")
    p.add_argument("--end", type=int, default=2024, help="End season (inclusive) if --seasons not given")
    p.add_argument(
        "--skip",
        nargs="*",
        default=[],
        choices=[
            "weekly_stats", "snap_counts", "rosters", "player_ids", "draft_picks", "ngs",
            "schedules", "participation", "pbp", "rosters_weekly", "injuries", "depth_charts",
            "players",
        ],
        help="Puller names to skip",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    seasons = sorted(args.seasons) if args.seasons else list(range(args.start, args.end + 1))
    logger.info("Building dataset for seasons: %s", seasons)

    steps = {
        "weekly_stats": lambda: pull_weekly_player_stats(seasons),
        "snap_counts": lambda: pull_snap_counts(seasons),
        "rosters": lambda: pull_rosters(seasons),
        "player_ids": lambda: pull_player_ids(),
        "draft_picks": lambda: pull_draft_picks(seasons),
        "ngs": lambda: pull_nextgen_stats(seasons),
        "schedules": lambda: pull_schedules(seasons),
        "participation": lambda: pull_participation(seasons),
        "pbp": lambda: pull_play_by_play(seasons),
        "rosters_weekly": lambda: pull_rosters_weekly(seasons),
        "injuries": lambda: pull_injuries(seasons),
        "depth_charts": lambda: pull_depth_charts(seasons),
        "players": lambda: pull_players(),
    }

    results: dict[str, str] = {}
    for name, fn in steps.items():
        if name in args.skip:
            logger.info("Skipping %s", name)
            continue
        t0 = time.time()
        try:
            fn()
            results[name] = f"ok ({time.time() - t0:.1f}s)"
        except Exception as e:  # noqa: BLE001 - top-level pipeline, log and continue
            logger.error("Puller %s failed: %s", name, e)
            results[name] = f"FAILED: {e}"

    logger.info("=== build_dataset summary ===")
    for name, status in results.items():
        logger.info("  %-15s %s", name, status)

    return 0 if all(v.startswith("ok") for v in results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
