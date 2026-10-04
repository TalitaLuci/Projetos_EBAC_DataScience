"""Teste Z para duas médias (σ populacionais conhecidos), poder e tamanho de amostra.

Convenção de direção: a estatística é Z = (média_1 − média_2) / EP. Para testar H1: μ_B > μ_A,
passamos B como grupo 1 e A como grupo 2 (Z = média_B − média_A): valores positivos favorecem B e a
região crítica fica na cauda DIREITA. Se a diferença fosse definida como A − B, o mesmo teste seria
unilateral à esquerda (o sinal de Z e a cauda mudam juntos; a decisão não).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy import stats

ALTERNATIVES = ("greater", "less", "two-sided")


@dataclass
class ZTestResult:
    mean_1: float
    mean_2: float
    diff: float            # média_1 − média_2
    std_error: float
    z: float
    p_value: float
    alpha: float
    z_critical: float      # para 'two-sided' vale ±z_critical
    alternative: str
    reject_h0: bool

    def to_dict(self) -> dict:
        return {k: (bool(v) if isinstance(v, (bool, np.bool_)) else v) for k, v in asdict(self).items()}


def z_test_two_sample(sample_1, sample_2, sigma_1: float, sigma_2: float,
                      alpha: float = 0.05, alternative: str = "greater") -> ZTestResult:
    if alternative not in ALTERNATIVES:
        raise ValueError(f"alternative deve ser um de {ALTERNATIVES}")
    s1, s2 = np.asarray(sample_1, float), np.asarray(sample_2, float)
    mean_1, mean_2 = s1.mean(), s2.mean()
    se = np.sqrt(sigma_1**2 / len(s1) + sigma_2**2 / len(s2))
    z = (mean_1 - mean_2) / se
    if alternative == "greater":
        p, z_crit = stats.norm.sf(z), stats.norm.ppf(1 - alpha)
    elif alternative == "less":
        p, z_crit = stats.norm.cdf(z), stats.norm.ppf(alpha)
    else:
        p, z_crit = 2 * stats.norm.sf(abs(z)), stats.norm.ppf(1 - alpha / 2)
    return ZTestResult(mean_1, mean_2, mean_1 - mean_2, se, z, p, alpha, z_crit, alternative, bool(p < alpha))


def describe_samples(sample_a, sample_b, sigma_a: float, sigma_b: float, mu_a: float, mu_b: float) -> pd.DataFrame:
    """Média, variância/desvio amostrais (ddof=1) e comparação com os parâmetros populacionais."""
    rows = {}
    for name, s, sigma, mu in (("A", sample_a, sigma_a, mu_a), ("B", sample_b, sigma_b, mu_b)):
        s = np.asarray(s, float)
        rows[f"Estratégia {name}"] = {
            "n": len(s),
            "média amostral": s.mean(), "média populacional": mu,
            "erro padrão da média": sigma / np.sqrt(len(s)),
            "variância amostral": s.var(ddof=1), "variância populacional": sigma**2,
            "desvio padrão amostral": s.std(ddof=1), "desvio padrão populacional": sigma,
        }
    return pd.DataFrame(rows).T


def confidence_interval_diff(result: ZTestResult, level: float = 0.95) -> tuple[float, float]:
    """IC bilateral da diferença de médias (σ conhecidos)."""
    zc = stats.norm.ppf(1 - (1 - level) / 2)
    return result.diff - zc * result.std_error, result.diff + zc * result.std_error


def cohens_d(sample_1, sample_2) -> float:
    """Tamanho do efeito (d de Cohen, desvio padrão combinado das amostras)."""
    a, b = np.asarray(sample_1, float), np.asarray(sample_2, float)
    pooled = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return (a.mean() - b.mean()) / pooled


def power_right_tailed(true_diff: float, sigma_1: float, sigma_2: float, n_per_group, alpha: float = 0.05):
    """Poder do teste unilateral à direita quando a diferença verdadeira é `true_diff` (μ_1 − μ_2)."""
    se = np.sqrt((sigma_1**2 + sigma_2**2) / np.asarray(n_per_group, float))
    return stats.norm.sf(stats.norm.ppf(1 - alpha) - true_diff / se)


def required_n_per_group(true_diff: float, sigma_1: float, sigma_2: float, alpha: float = 0.05,
                         power: float = 0.80) -> int:
    z_a, z_b = stats.norm.ppf(1 - alpha), stats.norm.ppf(power)
    return int(np.ceil((z_a + z_b) ** 2 * (sigma_1**2 + sigma_2**2) / true_diff**2))
