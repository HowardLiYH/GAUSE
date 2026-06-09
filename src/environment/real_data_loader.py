"""
Real Data Loader for Bybit Cryptocurrency Data.

Loads historical cryptocurrency data and detects regimes using HMM.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

try:
    from .hmm_regime_detector import HMMRegimeDetector
except ImportError:
    class HMMRegimeDetector:  # type: ignore[override]
        """Fallback placeholder when legacy HMM detector is unavailable."""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "src.environment.hmm_regime_detector is not present in this branch. "
                "Use domain-specific real-data regime detection paths "
                "(src/domains/* detect_regime) or restore the HMM module."
            )


# Default path to local data
DEFAULT_DATA_DIR = Path(__file__).parent.parent.parent / "data" / "bybit"


def load_bybit_data(
    symbol: str,
    data_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Load Bybit data for a symbol.

    Args:
        symbol: Coin symbol (BTC, ETH, SOL, etc.)
        data_dir: Path to bybit data directory

    Returns:
        DataFrame with OHLCV data
    """
    data_dir = data_dir or DEFAULT_DATA_DIR
    data_path = Path(data_dir) / f"Bybit_{symbol}.csv"

    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    df = pd.read_csv(data_path)

    # Parse timestamp
    if "timestamp_utc" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp_utc"])
        df = df.set_index("timestamp")

    # Ensure required columns
    required = ["open", "high", "low", "close", "volume"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    return df


def load_multi_asset_data(
    symbols: List[str] = None,
    data_dir: Optional[Path] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Load data for multiple assets.

    Args:
        symbols: List of coin symbols (default: all available)
        data_dir: Path to bybit data directory
        start_date: Optional start date filter
        end_date: Optional end date filter

    Returns:
        Dict mapping symbol to DataFrame
    """
    data_dir = data_dir or DEFAULT_DATA_DIR

    if symbols is None:
        symbols = ["BTC", "ETH", "SOL", "DOGE", "XRP"]

    data = {}

    for symbol in symbols:
        try:
            df = load_bybit_data(symbol, data_dir)

            # Apply date filters
            if start_date:
                df = df[df.index >= start_date]
            if end_date:
                df = df[df.index <= end_date]

            data[symbol] = df
        except FileNotFoundError:
            print(f"Warning: Data not found for {symbol}")

    return data


def detect_regimes_with_hmm(
    prices: pd.DataFrame,
    n_regimes: int = 4,
    random_state: int = 42,
) -> Tuple[pd.Series, HMMRegimeDetector]:
    """
    Detect regimes using HMM.

    Args:
        prices: Price DataFrame with OHLCV
        n_regimes: Number of regimes to detect
        random_state: Random seed

    Returns:
        Tuple of (regime labels, fitted detector)
    """
    detector = HMMRegimeDetector(
        n_regimes=n_regimes,
        random_state=random_state,
    )
    regimes = detector.fit_predict(prices)
    return regimes, detector


def prepare_real_data_experiment(
    symbol: str = "BTC",
    train_ratio: float = 0.7,
    n_regimes: int = 4,
    data_dir: Optional[Path] = None,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, HMMRegimeDetector]:
    """
    Prepare data for real data experiment.

    Splits data into train/test and detects regimes using HMM.

    Args:
        symbol: Coin symbol
        train_ratio: Fraction of data for training
        n_regimes: Number of regimes
        data_dir: Data directory

    Returns:
        Tuple of (train_prices, train_regimes, test_prices, test_regimes, detector)
    """
    # Load data
    prices = load_bybit_data(symbol, data_dir)

    # Split by ratio
    split_idx = int(len(prices) * train_ratio)
    train_prices = prices.iloc[:split_idx].copy()
    test_prices = prices.iloc[split_idx:].copy()

    # Fit HMM on training data
    detector = HMMRegimeDetector(n_regimes=n_regimes)
    train_regimes = detector.fit_predict(train_prices)

    # Predict on test data using trained model
    test_regimes = detector.predict(test_prices)

    return train_prices, train_regimes, test_prices, test_regimes, detector


def get_regime_statistics(
    prices: pd.DataFrame,
    regimes: pd.Series,
) -> pd.DataFrame:
    """
    Compute statistics for each regime.

    Returns DataFrame with regime characteristics.
    """
    close = prices["close"]
    returns = close.pct_change()

    stats = []
    for regime in regimes.unique():
        if regime == "unknown":
            continue

        mask = regimes == regime
        regime_returns = returns[mask]

        stats.append({
            "regime": regime,
            "count": mask.sum(),
            "proportion": mask.mean(),
            "mean_return": regime_returns.mean(),
            "std_return": regime_returns.std(),
            "sharpe": regime_returns.mean() / (regime_returns.std() + 1e-8) * np.sqrt(252 * 6),
        })

    return pd.DataFrame(stats)
