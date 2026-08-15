"""Temporal walk-forward splits.

resources.md §4: random k-fold leaks catastrophically here. Same-season rows
share team context (one team's pass rate is in 5 players' features), and
same-player rows share talent. The only honest split is by time: train on
seasons <= N, predict N+1, walk forward.

There is exactly one knob -- `min_train_seasons` -- and it exists because the
benchmark ladder's rungs need different amounts of history to be defined at
all (Marcel wants 3 prior seasons per player), so a fold too early in the
panel scores a Marcel that is really a one-year average wearing a hat.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from nflforecast.config import get_logger

logger = get_logger("splits")


@dataclass(frozen=True)
class Fold:
    """One walk-forward step: fit on `train_seasons`, score on `test_season`."""

    test_season: int
    train_seasons: tuple[int, ...]

    def __str__(self) -> str:
        return f"train {self.train_seasons[0]}-{self.train_seasons[-1]} -> test {self.test_season}"


def walk_forward(seasons: Iterable[int], min_train_seasons: int = 3) -> list[Fold]:
    """All folds where at least `min_train_seasons` seasons precede the test season.

    Note this counts *panel* seasons, not raw data seasons. The panel's first
    season is one later than the data's (a row needs a prior season to have
    features at all), so a 2016-2024 pull yields panel seasons 2017-2024 and,
    at the default, test seasons 2020-2024.
    """
    ordered = sorted(set(seasons))
    folds = [
        Fold(test_season=s, train_seasons=tuple(t for t in ordered if t < s))
        for s in ordered
    ]
    usable = [f for f in folds if len(f.train_seasons) >= min_train_seasons]

    dropped = len(folds) - len(usable)
    if dropped:
        logger.info(
            "%s of %s candidate folds dropped for <%s train seasons",
            dropped,
            len(folds),
            min_train_seasons,
        )
    return usable
