"""Modelling layer: projection panel, walk-forward evaluation, benchmark ladder.

Everything here works at **player-season grain** and is trained on player-week
data only through aggregates the feature pipeline already lagged. The split
between this package and `nflforecast.features` is deliberate: features answer
"what did we know at week n", this package answers "what do we predict for
season N+1, and is it better than a monkey".

The benchmark ladder (resources.md §5) is rungs 1 and 2 -- last season
regressed to the mean, then Marcel. Rungs 3 and 4 (ADP, Vegas-informed) need
data the pipeline does not pull yet.
"""
