"""Next Gen Stats puller.

Source: nflreadpy.load_nextgen_stats(stat_type=...), one call per stat_type
("passing", "receiving", "rushing"). NGS only exists from the 2016 season
onward (nflverse raises a ValueError outside 2016-current) -- seasons are
clamped accordingly. Covers air yards / aDOT / CPOE (passing), separation /
cushion / share of team air yards (receiving), and rush yards over expected
(rushing) -- exactly the metrics resources.md flags as NGS-only.

Writes three parquet files (one per stat_type) rather than one combined file
since the column sets differ per stat_type.
"""

from __future__ import annotations

from pathlib import Path

import nflreadpy as nfl

from nflforecast.config import RAW_DIR, get_logger

logger = get_logger(__name__)

MIN_SEASON = 2016
STAT_TYPES = ("passing", "receiving", "rushing")


def _output_path(stat_type: str) -> Path:
    return RAW_DIR / f"ngs_{stat_type}.parquet"


def pull_nextgen_stats(seasons: list[int]) -> dict[str, Path]:
    usable = [s for s in seasons if s >= MIN_SEASON]
    dropped = sorted(set(seasons) - set(usable))
    if dropped:
        logger.warning("NGS unavailable before %s; dropping seasons %s", MIN_SEASON, dropped)
    if not usable:
        raise ValueError(f"No requested seasons >= {MIN_SEASON} for NGS")

    outputs: dict[str, Path] = {}
    for stat_type in STAT_TYPES:
        logger.info("Pulling NGS %s for seasons %s-%s", stat_type, min(usable), max(usable))
        df = nfl.load_nextgen_stats(seasons=usable, stat_type=stat_type)
        path = _output_path(stat_type)
        df.write_parquet(path)
        logger.info(
            "Wrote %s rows x %s cols -> %s (seasons %s-%s)",
            df.height,
            df.width,
            path,
            min(usable),
            max(usable),
        )
        outputs[stat_type] = path
    return outputs
