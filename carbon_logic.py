# =============================================================================
# carbon_logic.py
# Stochastic carbon-intensity grid simulator for three geographic zones.
#
# Each zone models real-world renewable/fossil generation mix using:
#   intensity(t) = base + amplitude × sin(2π·t / 24) + N(0, σ²)
#
# Zones (gCO2eq/kWh):
#   Zone 0 — green  : Nordic renewables-heavy grid  (base ≈ 100)
#   Zone 1 — mixed  : UK / Germany mixed grid       (base ≈ 350)
#   Zone 2 — coal   : Coal-heavy grid               (base ≈ 600)
#
# This is an upgrade on the group baseline which had no carbon awareness.
# =============================================================================

import numpy as np


class CarbonGridSimulator:
    """
    Simulates carbon intensity for three geographic energy grid zones.

    Zone intensities follow a sinusoidal daily pattern + Gaussian noise,
    modelling the real-world fluctuation of renewable energy availability.
    """

    ZONES = {
        0: {"name": "green", "base": 100,  "amplitude": 40,  "noise_std": 10},
        1: {"name": "mixed", "base": 350,  "amplitude": 80,  "noise_std": 25},
        2: {"name": "coal",  "base": 600,  "amplitude": 120, "noise_std": 40},
    }

    # Theoretical max for normalisation (coal base + amplitude + 3σ)
    MAX_INTENSITY = 600 + 120 + 3 * 40

    def __init__(self):
        # Cache per round so all queries within a round are consistent
        self._cache: dict = {}

    def reset(self):
        """Call between experiment runs to clear stale cached values."""
        self._cache.clear()

    def get_carbon_intensity(self, zone_id: int, current_round: int) -> float:
        """
        Return gCO2eq/kWh for a given zone and FL round.
        Values are cached per round for consistency within a round.
        """
        zone_id = zone_id % 3
        key = (zone_id, current_round)
        if key not in self._cache:
            z = self.ZONES[zone_id]
            sine  = z["amplitude"] * np.sin(2 * np.pi * current_round / 24.0)
            noise = np.random.normal(0, z["noise_std"])
            self._cache[key] = float(max(10.0, z["base"] + sine + noise))
        return self._cache[key]

    def get_normalised_intensity(self, zone_id: int, current_round: int) -> float:
        """Return carbon intensity normalised to [0, 1] for use in DQN state."""
        return min(self.get_carbon_intensity(zone_id, current_round) / self.MAX_INTENSITY, 1.0)

    def get_zone_name(self, zone_id: int) -> str:
        return self.ZONES[zone_id % 3]["name"]
