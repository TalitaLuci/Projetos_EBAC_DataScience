"""Pipeline completo: dados -> treino/teste -> avaliação -> validação cruzada -> relatórios.

Uso (a partir da raiz do projeto):
    python -m src.pipeline              # modelos com padronização (recomendado)
    python -m src.pipeline --unscaled   # inclui também os SVMs sem padronização (lento: ~30 s)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from . import config
from .data import load_prepared, split
from .evaluate import compute_metrics, plot_confusion_matrices, plot_decision_regions
from .models import build_models, build_unscaled_svms, cross_validate_models


def run_pipeline(include_unscaled: bool = False, make_plots: bool = True,
                 reports_dir: Path = config.REPORTS_DIR, data_path: Path = config.DATA_FILE,
                 verbose: bool = True) -> dict:
    log = print if verbose else (lambda *a, **k: None)
    reports_dir = Path(reports_dir)
    figures_dir = reports_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    df = load_prepared(data_path)
    X_train, X_test, y_train, y_test = split(df)
    log(f"Base: {len(df)} clientes | compradores = {df[config.TARGET].mean():.1%} | treino {len(X_train)} / teste {len(X_test)}")

    models = build_models()
    fitted, test_metrics = {}, {}
    for name, model in models.items():
        fitted[name] = model.fit(X_train, y_train)
        test_metrics[name] = compute_metrics(y_test, fitted[name].predict(X_test))
    test_table = pd.DataFrame(test_metrics).T
    log("\nMétricas no teste:\n" + test_table.round(3).to_string())

    cv_table = cross_validate_models(models, df.drop(columns=[config.TARGET]), df[config.TARGET])
    log("\nValidação cruzada (5 folds, base completa):\n" + cv_table.round(3).to_string())

    unscaled_table = None
    if include_unscaled:
        u = {n: m.fit(X_train, y_train) for n, m in build_unscaled_svms().items()}
        unscaled_table = pd.DataFrame({n: compute_metrics(y_test, m.predict(X_test)) for n, m in u.items()}).T
        log("\nSVMs sem padronização (referência do exercício original):\n" + unscaled_table.round(3).to_string())

    if make_plots:
        plot_confusion_matrices(fitted, X_test, y_test, figures_dir / "matrizes_confusao.png")
        svms = {k: v for k, v in fitted.items() if k.startswith("SVM")}
        plot_decision_regions(svms, X_train, y_train, figures_dir / "regioes_decisao_svm.png")

    to_dict = lambda t: {k: {m: float(v) for m, v in row.items()} for k, row in t.to_dict("index").items()}
    results = {
        "n_clientes": int(len(df)),
        "taxa_compra": float(df[config.TARGET].mean()),
        "test_metrics": to_dict(test_table),
        "cv_metrics": to_dict(cv_table),
        "svm_sem_escala": to_dict(unscaled_table) if unscaled_table is not None else None,
    }
    (reports_dir / "metrics.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline SVM — propensão de compra de carros")
    parser.add_argument("--unscaled", action="store_true", help="incluir SVMs sem padronização (lento)")
    parser.add_argument("--no-plots", action="store_true", help="não gerar figuras")
    args = parser.parse_args()
    run_pipeline(include_unscaled=args.unscaled, make_plots=not args.no_plots)


if __name__ == "__main__":
    main()
