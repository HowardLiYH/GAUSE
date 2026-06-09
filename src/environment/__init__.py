"""
Environment module.

Keep imports minimal to avoid hard failures from optional/legacy modules
that may be absent in some branches.
"""

from .regime_classifier import CombinedClassifier, MAClassifier, ReturnsClassifier, VolatilityClassifier
from .synthetic_market import SyntheticMarketConfig, SyntheticMarketEnvironment

__all__ = [
    "SyntheticMarketConfig",
    "SyntheticMarketEnvironment",
    "MAClassifier",
    "VolatilityClassifier",
    "ReturnsClassifier",
    "CombinedClassifier",
]
