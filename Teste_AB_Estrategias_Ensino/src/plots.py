"""Gráficos do teste Z."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from .stats_tests import ZTestResult, power_right_tailed


def plot_z_distribution(result: ZTestResult, ax=None, path: Path | None = None):
    """Normal padrão com região crítica à DIREITA (teste unilateral, Z = média_B − média_A)."""
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(9, 5))
    x = np.linspace(-4, 4, 800)
    pdf = stats.norm.pdf(x)
    ax.plot(x, pdf, color="black", lw=1.8, label="Distribuição de Z sob H0 — N(0, 1)")
    ax.fill_between(x, pdf, where=x >= result.z_critical, color="#d62728", alpha=0.35,
                    label=f"Região crítica (α = {result.alpha:.2f}): Z > {result.z_critical:.3f}")
    ax.fill_between(x, pdf, where=x >= result.z, facecolor="none", edgecolor="#1f77b4", hatch="///",
                    linewidth=0, label=f"p-valor = {result.p_value:.4f} (área hachurada, à direita de Z obs.)")
    ax.axvline(result.z_critical, color="#d62728", ls="--", lw=1.5)
    ax.axvline(result.z, color="#1f77b4", lw=2.2, label=f"Z observado = {result.z:.3f}")
    decision = "rejeita H0" if result.reject_h0 else "não rejeita H0"
    ax.set(title=f"Teste Z unilateral à direita (H1: μB > μA) — {decision}",
           xlabel="Z = (média B − média A) / erro padrão", ylabel="Densidade")
    ax.legend(loc="upper right", fontsize=9)
    if own_fig:
        fig.tight_layout()
        if path is not None:
            fig.savefig(path, dpi=130)
            plt.close(fig)
    return ax


def plot_power_curve(sigma_1: float, sigma_2: float, true_diff: float, n_observed: int,
                     alpha: float = 0.05, path: Path | None = None):
    n = np.arange(10, 201)
    power = power_right_tailed(true_diff, sigma_1, sigma_2, n, alpha)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(n, power, lw=2)
    ax.axhline(0.8, color="gray", ls="--", label="Poder de 80%")
    ax.axvline(n_observed, color="#d62728", ls=":", label=f"n usado = {n_observed} por grupo")
    ax.scatter([n_observed], [power_right_tailed(true_diff, sigma_1, sigma_2, n_observed, alpha)], color="#d62728", zorder=3)
    ax.set(title=f"Poder do teste (diferença verdadeira = {true_diff:g} pontos, α = {alpha})",
           xlabel="Alunos por estratégia", ylabel="Poder", ylim=(0, 1.02))
    ax.legend(loc="lower right")
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=130)
        plt.close(fig)
    return fig
