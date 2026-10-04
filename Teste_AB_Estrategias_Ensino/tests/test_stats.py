import numpy as np
import pytest
from scipy import stats

from src import config
from src.data import generate_samples
from src.stats_tests import (describe_samples, power_right_tailed, required_n_per_group, z_test_two_sample)


@pytest.fixture(scope="module")
def samples():
    return generate_samples()


def right_tailed(a, b, **kw):
    """H1: mu_B > mu_A com Z = (media_B - media_A)/EP."""
    return z_test_two_sample(b, a, config.SIGMA_B, config.SIGMA_A, alternative="greater", **kw)


def test_result_matches_feedback_direction_and_decision(samples):
    res = right_tailed(*samples)
    assert res.z > 0                                   # B − A positivo: favorece B
    assert res.z == pytest.approx(1.5133, abs=1e-4)
    assert res.p_value == pytest.approx(0.0651, abs=1e-4)
    assert res.z_critical == pytest.approx(1.6449, abs=1e-4)
    assert res.p_value > config.ALPHA and not res.reject_h0


def test_sign_and_tail_flip_together(samples):
    """A − B com cauda esquerda dá exatamente o mesmo p-valor que B − A com cauda direita."""
    a, b = samples
    right = z_test_two_sample(b, a, config.SIGMA_B, config.SIGMA_A, alternative="greater")
    left = z_test_two_sample(a, b, config.SIGMA_A, config.SIGMA_B, alternative="less")
    assert left.z == pytest.approx(-right.z)
    assert left.p_value == pytest.approx(right.p_value)
    assert left.reject_h0 == right.reject_h0


def test_p_value_is_one_minus_cdf(samples):
    res = right_tailed(*samples)
    assert res.p_value == pytest.approx(1 - stats.norm.cdf(res.z))


def test_two_sided_is_double_one_sided(samples):
    a, b = samples
    two = z_test_two_sample(b, a, config.SIGMA_B, config.SIGMA_A, alternative="two-sided")
    assert two.p_value == pytest.approx(2 * right_tailed(a, b).p_value)


def test_invalid_alternative_raises(samples):
    with pytest.raises(ValueError):
        z_test_two_sample(*samples, 10, 12, alternative="maior")


def test_descriptive_table(samples):
    d = describe_samples(*samples, config.SIGMA_A, config.SIGMA_B, config.MEAN_A, config.MEAN_B)
    assert d.loc["Estratégia A", "variância amostral"] > d.loc["Estratégia B", "variância amostral"]   # inversão
    assert d.loc["Estratégia A", "variância populacional"] < d.loc["Estratégia B", "variância populacional"]


def test_power_and_required_n():
    assert power_right_tailed(5, 12, 10, 50) == pytest.approx(0.732, abs=0.002)
    assert required_n_per_group(5, 12, 10) == 61


def test_monte_carlo_type1_error_and_power():
    """Simulação: sob H0 rejeita ~5%; com diferença verdadeira de 5 pontos rejeita ~73%."""
    rng = np.random.RandomState(1)
    sims = 4000

    def rejection_rate(mu_b):
        rejects = 0
        for _ in range(sims):
            a = rng.normal(config.MEAN_A, config.SIGMA_A, config.N_PER_GROUP)
            b = rng.normal(mu_b, config.SIGMA_B, config.N_PER_GROUP)
            rejects += right_tailed(a, b).reject_h0
        return rejects / sims

    assert rejection_rate(config.MEAN_A) == pytest.approx(0.05, abs=0.015)
    assert rejection_rate(config.MEAN_B) == pytest.approx(0.732, abs=0.03)
