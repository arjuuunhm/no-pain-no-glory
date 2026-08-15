"""Weekly injury report puller.

Source: nflreadpy.load_injuries. report_status (Questionable/Doubtful/Out)
and practice participation, per player per week. Self-reported by teams and
known to be gamed for competitive reasons -- treat as a noisy pre-game risk
signal, not ground-truth severity (docs/features.md §6). Use
load_rosters_weekly()'s realized active/inactive status for the actual
availability label instead.
"""

from __future__ import annotations

from pathlib import Path

import nflreadpy as nfl

from nflforecast.config import RAW_DIR, get_logger

logger = get_logger(__name__)

OUTPUT_PATH = RAW_DIR / "injuries.parquet"


def pull_injuries(seasons: list[int], output_path: Path = OUTPUT_PATH) -> Path:
    logger.info("Pulling injury reports for seasons %s-%s", min(seasons), max(seasons))
    df = nfl.load_injuries(seasons=seasons)
    df.write_parquet(output_path)
    logger.info(
        "Wrote %s rows x %s cols -> %s (seasons %s-%s)",
        df.height,
        df.width,
        output_path,
        min(seasons),
        max(seasons),
    )
    return output_path
