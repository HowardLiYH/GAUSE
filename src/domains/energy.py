"""
Energy Domain - Energy grid management with demand regimes.

Legacy compatibility wrapper.

The previous implementation imported ``synthetic_domains`` from this package,
which no longer exists under ``src/domains``. Keep this module import-safe by
providing conservative placeholders and failing only when construction is
explicitly requested.
"""

from typing import List

ENERGY_REGIMES: List[str] = ["low_demand", "normal_demand", "peak_demand", "volatile_demand"]
ENERGY_METHODS: List[str] = ["baseline", "load_shift", "peak_shave", "storage"]


class EnergyDomain:
    """Wrapper for energy domain environment."""

    def __init__(self, n_bars: int = 2000, seed: int = None):
        raise NotImplementedError(
            "EnergyDomain synthetic generator was removed in refactors. "
            "Use real-data domain modules (crypto/commodities/weather/solar/"
            "traffic/air_quality) or restore the legacy synthetic generator."
        )

    @property
    def regime_names(self):
        return ENERGY_REGIMES

    @property
    def method_names(self):
        return ENERGY_METHODS
