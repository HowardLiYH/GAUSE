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

from .inventory_v2 import METHOD_INVENTORY_V2 as METHOD_INVENTORY
from .inventory_v2 import get_method_names_v2 as get_method_names

__all__ = ["METHOD_INVENTORY", "get_method_names"]
