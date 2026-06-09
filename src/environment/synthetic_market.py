"""
Synthetic Market Environment with Controllable Regime-Switching.

This environment enables rigorous evaluation of agent specialization by:
1. Providing known ground truth regime labels
2. Allowing statistical significance through unlimited data
3. Enabling controlled parameter sweeps
4. Ensuring perfect reproducibility

Theoretical Reference:
- Hamilton, J.D. (1989). "A New Approach to the Economic Analysis of
  Nonstationary Time Series." Econometrica, 57(2), 357-384.

The regime-switching is modeled as a Hidden Markov Model where:
- Hidden states = market regimes (trend, mean_revert, volatile, sideways)
- Observations = price returns
- Transition matrix controls regime persistence
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

class BaseRegime:
    """Minimal regime interface for synthetic return generators."""

    def generate_returns(self, n_bars: int, seed: Optional[int] = None) -> np.ndarray:
        raise NotImplementedError


class TrendRegime(BaseRegime):
    def __init__(self, drift: float = 0.001, volatility: float = 0.008):
        self.drift = drift
        self.volatility = volatility

    def generate_returns(self, n_bars: int, seed: Optional[int] = None) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return rng.normal(self.drift, self.volatility, n_bars)


class MeanRevertRegime(BaseRegime):
    def __init__(self, mean: float = 0.0, speed: float = 0.12, volatility: float = 0.012):
        self.mean = mean
        self.speed = speed
        self.volatility = volatility

    def generate_returns(self, n_bars: int, seed: Optional[int] = None) -> np.ndarray:
        rng = np.random.default_rng(seed)
        x = np.zeros(n_bars)
        eps = rng.normal(0.0, self.volatility, n_bars)
        for t in range(1, n_bars):
            x[t] = x[t - 1] + self.speed * (self.mean - x[t - 1]) + eps[t]
        return x


class VolatileRegime(BaseRegime):
    def __init__(self, volatility: float = 0.03):
        self.volatility = volatility

    def generate_returns(self, n_bars: int, seed: Optional[int] = None) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return rng.normal(0.0, self.volatility, n_bars)


REGIME_OPTIMAL_METHODS = {
    "trend_up": ["BuyMomentum", "BreakoutLong", "TrendRider"],
    "trend_down": ["SellMomentum", "BreakoutShort", "TrendFader"],
    "mean_revert": ["MeanRevert", "BollingerMR", "RSI_MR"],
    "volatile": ["VolBreakout", "VolScalp", "VolFade"],
}


def create_regime(name: str) -> BaseRegime:
    """Create a regime generator from legacy regime names."""
    if name == "trend_up":
        return TrendRegime(drift=0.001, volatility=0.008)
    if name == "trend_down":
        return TrendRegime(drift=-0.001, volatility=0.008)
    if name == "mean_revert":
        return MeanRevertRegime(mean=0.0, speed=0.12, volatility=0.012)
    if name == "volatile":
        return VolatileRegime(volatility=0.03)
    # Fallback to a neutral low-drift trend regime for unknown labels
    return TrendRegime(drift=0.0, volatility=0.01)


@dataclass
class SyntheticMarketConfig:
    """Configuration for synthetic market environment."""

    # Regime configuration
    regime_names: List[str] = field(
        default_factory=lambda: ["trend_up", "trend_down", "mean_revert", "volatile"]
    )

    # Regime duration (in bars)
    regime_duration_mean: int = 50
    regime_duration_std: int = 15
    regime_duration_min: int = 10

    # Transition matrix (if None, uses uniform transitions)
    transition_matrix: Optional[np.ndarray] = None

    # Regime distinctness (0-1): How different are optimal methods?
    # 1.0 = very distinct (different methods for each regime)
    # 0.0 = all regimes have same optimal methods
    regime_distinctness: float = 1.0

    # Initial price
    initial_price: float = 100.0

    # Random seed for reproducibility
    seed: Optional[int] = None


class SyntheticMarketEnvironment:
    """
    Controllable regime-switching market for rigorous evaluation.

    Key features:
    1. Known ground truth regime labels at each timestep
    2. Controllable regime dynamics (trend, mean-revert, volatile, sideways)
    3. Configurable regime duration and transition probabilities
    4. Perfect reproducibility with seed control

    Usage:
        env = SyntheticMarketEnvironment()
        prices, regimes = env.generate(n_bars=1000, seed=42)

        # prices: DataFrame with OHLCV columns
        # regimes: Series with regime labels (ground truth)
    """

    def __init__(self, config: Optional[SyntheticMarketConfig] = None):
        self.config = config or SyntheticMarketConfig()
        self._setup_regimes()
        self._setup_transition_matrix()

    def _setup_regimes(self) -> None:
        """Initialize regime generators."""
        self.regimes: Dict[str, BaseRegime] = {}

        for name in self.config.regime_names:
            self.regimes[name] = create_regime(name)

        self.regime_names = list(self.regimes.keys())
        self.n_regimes = len(self.regime_names)

    def _setup_transition_matrix(self) -> None:
        """Setup Markov transition matrix for regime switching."""
        if self.config.transition_matrix is not None:
            self.transition_matrix = self.config.transition_matrix
        else:
            # Default: uniform transitions with self-loops
            # This gives approximately equal time in each regime
            n = self.n_regimes
            self.transition_matrix = np.ones((n, n)) / n

    def _sample_regime_duration(self, rng: np.random.Generator) -> int:
        """Sample regime duration from truncated normal."""
        duration = int(rng.normal(
            self.config.regime_duration_mean,
            self.config.regime_duration_std
        ))
        return max(self.config.regime_duration_min, duration)

    def _sample_next_regime(
        self,
        current_regime_idx: int,
        rng: np.random.Generator,
    ) -> int:
        """Sample next regime from transition matrix."""
        probs = self.transition_matrix[current_regime_idx]
        return rng.choice(self.n_regimes, p=probs)

    def generate(
        self,
        n_bars: int,
        seed: Optional[int] = None,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Generate synthetic price data with regime labels.

        Args:
            n_bars: Number of bars to generate
            seed: Random seed for reproducibility

        Returns:
            prices: DataFrame with columns [open, high, low, close, volume]
            regimes: Series with ground truth regime labels
        """
        seed = seed or self.config.seed
        rng = np.random.default_rng(seed)

        # Generate regime sequence
        regime_labels = []
        current_regime_idx = rng.choice(self.n_regimes)
        remaining_duration = self._sample_regime_duration(rng)

        for _ in range(n_bars):
            regime_labels.append(self.regime_names[current_regime_idx])
            remaining_duration -= 1

            if remaining_duration <= 0:
                current_regime_idx = self._sample_next_regime(current_regime_idx, rng)
                remaining_duration = self._sample_regime_duration(rng)

        # Generate returns for each bar based on its regime
        returns = np.zeros(n_bars)

        for i, regime_name in enumerate(regime_labels):
            regime = self.regimes[regime_name]
            # Generate single return (use unique seed for each bar)
            bar_returns = regime.generate_returns(1, seed=seed + i if seed else None)
            returns[i] = bar_returns[0]

        # Convert returns to prices
        log_prices = np.log(self.config.initial_price) + np.cumsum(returns)
        close_prices = np.exp(log_prices)

        # Generate OHLC from close prices
        # Simple model: high/low within typical range of close
        volatility = np.abs(returns) + 0.005  # Minimum spread
        high_prices = close_prices * (1 + volatility * rng.uniform(0.5, 1.0, n_bars))
        low_prices = close_prices * (1 - volatility * rng.uniform(0.5, 1.0, n_bars))
        open_prices = np.roll(close_prices, 1)
        open_prices[0] = self.config.initial_price

        # Generate volume (higher in volatile regimes)
        base_volume = 1000.0
        volume = base_volume * (1 + np.abs(returns) * 50) * rng.uniform(0.5, 1.5, n_bars)

        # Create DataFrame
        prices = pd.DataFrame({
            "open": open_prices,
            "high": high_prices,
            "low": low_prices,
            "close": close_prices,
            "volume": volume,
        })

        regimes = pd.Series(regime_labels, name="regime")

        return prices, regimes

    def generate_multiple_trials(
        self,
        n_trials: int,
        n_bars: int,
        base_seed: int = 0,
    ) -> List[Tuple[pd.DataFrame, pd.Series]]:
        """Generate multiple independent trials for statistical significance."""
        trials = []
        for i in range(n_trials):
            prices, regimes = self.generate(n_bars, seed=base_seed + i * 1000)
            trials.append((prices, regimes))
        return trials

    def get_optimal_methods(self, regime: str) -> List[str]:
        """Get optimal methods for a given regime (for Oracle baseline)."""
        return REGIME_OPTIMAL_METHODS.get(regime, [])

    def get_regime_statistics(
        self,
        regimes: pd.Series,
    ) -> Dict[str, Dict]:
        """Compute statistics about regime occurrence."""
        stats = {}

        for regime in self.regime_names:
            mask = regimes == regime
            count = mask.sum()
            proportion = count / len(regimes)

            # Compute average duration
            changes = mask.astype(int).diff().fillna(0)
            starts = (changes == 1).sum()
            avg_duration = count / max(starts, 1)

            stats[regime] = {
                "count": count,
                "proportion": proportion,
                "avg_duration": avg_duration,
            }

        return stats

    def __repr__(self) -> str:
        return (
            f"SyntheticMarketEnvironment("
            f"regimes={self.regime_names}, "
            f"duration_mean={self.config.regime_duration_mean})"
        )


def create_environment_for_experiment(
    experiment_name: str,
    **kwargs,
) -> SyntheticMarketEnvironment:
    """
    Factory function to create environment with experiment-specific settings.

    Experiments may require different regime configurations:
    - exp1_emergence: Standard 4 regimes
    - exp3_population_size: Same as exp1
    - exp5_regime_transitions: Sharp transitions (low duration_std)
    """
    if experiment_name == "exp5_regime_transitions":
        # Sharp, predictable transitions for studying handoff
        config = SyntheticMarketConfig(
            regime_duration_mean=50,
            regime_duration_std=5,  # Very consistent duration
            **kwargs,
        )
    else:
        # Default configuration
        config = SyntheticMarketConfig(**kwargs)

    return SyntheticMarketEnvironment(config)
