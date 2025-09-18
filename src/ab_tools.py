from __future__ import annotations

import math

from statsmodels.stats.power import NormalIndPower, TTestIndPower
from statsmodels.stats.proportion import proportion_effectsize


def required_sample_size_proportions(
    p_baseline: float,
    mde: float,
    alpha: float = 0.05,
    power: float = 0.8,
    two_tailed: bool = True,
) -> int:
    """Per-arm sample size for a difference in proportions.

    Args:
        p_baseline: baseline conversion rate (0..1).
        mde: absolute minimum detectable effect (0..1).
        alpha: significance level.
        power: desired power (1 - beta).
        two_tailed: whether test is two-sided.
    """
    p2 = min(max(p_baseline + mde, 1e-9), 1 - 1e-9)
    es = proportion_effectsize(p_baseline, p2)
    alt = "two-sided" if two_tailed else "larger"
    n = NormalIndPower().solve_power(effect_size=es, alpha=alpha, power=power, alternative=alt)
    return int(math.ceil(n))


def required_sample_size_means(
    sd: float,
    mde: float,
    alpha: float = 0.05,
    power: float = 0.8,
    two_tailed: bool = True,
) -> int:
    """Per-arm sample size for a difference in means with known SD.

    Args:
        sd: standard deviation of the outcome.
        mde: absolute minimum detectable effect.
        alpha: significance level.
        power: desired power (1 - beta).
        two_tailed: whether test is two-sided.
    """
    d = mde / max(sd, 1e-9)
    alt = "two-sided" if two_tailed else "larger"
    n = TTestIndPower().solve_power(effect_size=d, alpha=alpha, power=power, alternative=alt)
    return int(math.ceil(n))

