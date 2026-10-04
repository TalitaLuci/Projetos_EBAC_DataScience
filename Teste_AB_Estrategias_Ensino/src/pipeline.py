"""Pipeline: dados -> estatística descritiva -> teste Z unilateral à direita -> relatórios.

Uso (a partir da raiz do projeto):
    python -m src.pipeline
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import config
from .data import load_samples, save_samples
from .plots import plot_power_curve, plot_z_distribution
from .stats_tests import (cohens_d, confidence_interval_diff, describe_samples, power_right_tailed,
                          required_n_per_group, z_test_two_sample)


def run_pipeline(reports_dir: Path = config.REPORTS_DIR, data_path: Path = config.DATA_FILE,
                 make_plots: bool = True, verbose: bool = True) -> dict:
    log = print if verbose else (lambda *a, **k: None)
    reports_dir = Path(reports_dir)
    figures_dir = reports_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    if not Path(data_path).exists():
        save_samples(data_path)

    a, b = load_samples(data_path)
    desc = describe_samples(a, b, config.SIGMA_A, config.SIGMA_B, config.MEAN_A, config.MEAN_B)
    log("Estatística descritiva:\n" + desc.round(3).T.to_string())

    # H1: mu_B > mu_A  ->  Z = (media_B - media_A)/EP, cauda direita
    res = z_test_two_sample(b, a, config.SIGMA_B, config.SIGMA_A, config.ALPHA, config.ALTERNATIVE)
    log(f"\nZ = {res.z:.4f} | p-valor = {res.p_value:.4f} | Z crítico = {res.z_critical:.4f} | "
        f"{'rejeita' if res.reject_h0 else 'NÃO rejeita'} H0 (α = {res.alpha})")

    ci = confidence_interval_diff(res)
    true_diff = config.MEAN_B - config.MEAN_A
    extras = {
        "ic95_diferenca": list(map(float, ci)),
        "cohens_d": float(cohens_d(b, a)),
        "poder_diferenca_verdadeira": float(power_right_tailed(true_diff, config.SIGMA_B, config.SIGMA_A,
                                                                 config.N_PER_GROUP, config.ALPHA)),
        "n_por_grupo_para_80pct_poder": required_n_per_group(true_diff, config.SIGMA_B, config.SIGMA_A, config.ALPHA),
    }
    log(f"IC 95% da diferença: [{ci[0]:.2f}, {ci[1]:.2f}] | d de Cohen = {extras['cohens_d']:.2f} | "
        f"poder = {extras['poder_diferenca_verdadeira']:.1%} | n p/ 80% de poder = {extras['n_por_grupo_para_80pct_poder']} por grupo")

    if make_plots:
        plot_z_distribution(res, path=figures_dir / "distribuicao_z.png")
        plot_power_curve(config.SIGMA_B, config.SIGMA_A, true_diff, config.N_PER_GROUP, config.ALPHA,
                         path=figures_dir / "curva_poder.png")

    results = {"teste_z": res.to_dict(), "descritiva": desc.to_dict("index"), **extras}
    (reports_dir / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Teste A/B (teste Z) — estratégias de ensino")
    parser.add_argument("--no-plots", action="store_true")
    run_pipeline(make_plots=not parser.parse_args().no_plots)


if __name__ == "__main__":
    main()
