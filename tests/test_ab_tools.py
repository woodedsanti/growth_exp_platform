from statsmodels.stats.power import NormalIndPower, TTestIndPower
from statsmodels.stats.proportion import proportion_effectsize

from src.ab_tools import required_sample_size_means, required_sample_size_proportions


def test_required_sample_size_proportions_matches_statsmodels():
    p0 = 0.05
    mde = 0.01
    es = proportion_effectsize(p0, p0 + mde)
    sm_n = NormalIndPower().solve_power(effect_size=es, alpha=0.05, power=0.8, alternative="two-sided")
    ours = required_sample_size_proportions(p0, mde)
    assert abs(ours - sm_n) / sm_n < 0.05


def test_required_sample_size_means_matches_statsmodels():
    sd = 1.0
    mde = 0.2
    d = mde / sd
    sm_n = TTestIndPower().solve_power(effect_size=d, alpha=0.05, power=0.8, alternative="two-sided")
    ours = required_sample_size_means(sd, mde)
    assert abs(ours - sm_n) / sm_n < 0.05

