"""
Agents module: Method selection agents and population dynamics.

Core components:
- MethodSelector: Individual agent that selects methods using Thompson Sampling
- Population: Collection of agents with knowledge transfer
- Inventory: Shared method inventory available to all agents
- Methods: Actual trading method implementations
"""

from .method_selector import MethodSelector, SelectionResult
from .inventory import METHOD_INVENTORY, get_method_names
from .niche_population import NicheAgent, NichePopulation

# Note: ``from .population import Population, PopulationConfig`` was
# previously here but ``population.py`` no longer exists in the repo.
# The active population implementation is ``NichePopulation`` in
# ``niche_population.py``. Removing the dead import unblocks package
# loading; the broader code-level cleanup is tracked separately from V4.

__all__ = [
    "MethodSelector",
    "SelectionResult",
    "NicheAgent",
    "NichePopulation",
    "METHOD_INVENTORY",
    "get_method_names",
]
