"""
Baselines module: Comparison strategies for experiments.

Baselines:
1. BuyAndHold - Naive passive investment
2. MomentumStrategy - Classic 12-1 momentum
3. MeanReversionStrategy - Bollinger bands
4. SingleAgentRL - PPO/DQN single agent
5. FinRLAgent - SOTA RL trading (optional)
6. HomogeneousPopulation - Clone best agent
7. RandomSelection - Random method selection
8. OracleSpecialist - Perfect regime knowledge (upper bound)
9. IndependentQLearning - MARL baseline (independent learners)
10. QMIX - MARL baseline (value factorization)
11. MAPPO - MARL baseline (centralized critic)
12. QualityDiversity - Explicit diversity optimization
"""

from .oracle import OracleSpecialist, EmpiricalOracle
from .random_selection import RandomSelectionPopulation
from .homogeneous import HomogeneousPopulation
from .marl_baselines import IndependentQLearning, QMIX, MAPPO, QualityDiversity, MARLConfig

__all__ = [
    "OracleSpecialist",
    "EmpiricalOracle",
    "RandomSelectionPopulation",
    "HomogeneousPopulation",
    "IndependentQLearning",
    "QMIX",
    "MAPPO",
    "QualityDiversity",
    "MARLConfig",
]
