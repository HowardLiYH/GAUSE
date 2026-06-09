"""
Compatibility shim for legacy ``inventory`` import path.

The ``inventory`` module was deleted from the repository (see git log:
``find_diff D src/agents/inventory.py``) but ``method_selector.py`` and
``__init__.py`` still reference it. This shim re-exports the v2 inventory
under the legacy names so the package imports cleanly.

This is a hygiene fix, not a V4 contribution. A proper resolution
(removing the legacy code paths and the shim) belongs in a separate
cleanup branch.
"""

from typing import List, Union

from .inventory_v2 import METHOD_INVENTORY_V2 as METHOD_INVENTORY
from .inventory_v2 import MethodCategory, TradingMethod, get_method_names_v2, get_methods_for_regime_v2


def get_method_names() -> List[str]:
    """Legacy alias for v2 method-name accessor."""
    return get_method_names_v2()


def get_methods_for_regime(regime: str) -> List[str]:
    """Legacy alias for v2 regime-method lookup."""
    return get_methods_for_regime_v2(regime)


def get_methods_by_category(category: Union[str, MethodCategory]) -> List[str]:
    """Return methods matching a category (legacy API compatibility)."""
    if isinstance(category, str):
        category = MethodCategory(category)
    return [name for name, method in METHOD_INVENTORY.items() if method.category == category]


__all__ = [
    "METHOD_INVENTORY",
    "MethodCategory",
    "TradingMethod",
    "get_method_names",
    "get_methods_for_regime",
    "get_methods_by_category",
]
