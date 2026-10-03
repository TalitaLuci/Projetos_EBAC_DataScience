"""Pipeline completo: dados brutos -> limpeza -> treino/teste -> CV -> avaliação -> relatórios.

Uso (a partir da raiz do projeto):
    python -m src.pipeline
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from . import config
from .data import load_clean, split_xy
from .evaluate import (compute_metrics, feature_importances, plot_confusion_matrices, plot_importances,
                       plot_roc)
from .models import build_pipelines, cross_validate_auc


def run_pipeline(make_plots: bool = True, reports_dir: Path = config.REPORTS_DIR,
                 data_path: Path = config.DATA_FILE, verbose: bool = True) -> dict:
    log = print if verbose else (lambda *a, **k: None)
    reports_dir = Path(reports_dir)
    figures_dir = reports_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    df = load_clean(data_path)
    X, y = split_xy(df)
    log(f"Base limpa: {len(df)} clientes | churn = {y.mean():.1%} | {X.shape[1]} atributos")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, stratify=y, random_state=config.RANDOM_STATE)

    pipelines = build_pipelines(X)
    cv = cross_validate_auc(pipelines, X_train, y_train)
    log("\nROC-AUC (5-fold CV, treino):\n" + cv.round(4).to_string())

    metrics, fitted = {}, {}
    for name, pipe in pipelines.items():
        pipe.fit(X_train, y_train)
        fitted[name] = pipe
        metrics[name] = compute_metrics(y_test, pipe.predict(X_test), pipe.predict_proba(X_test)[:, 1])
    table = pd.DataFrame(metrics).T
    log("\nMétricas no teste:\n" + table.round(4).to_string())

    importances = feature_importances(fitted["Random Forest"])
    log("\nTop variáveis (Random Forest):\n" + importances.round(4).to_string())

    if make_plots:
        plot_confusion_matrices(fitted, X_test, y_test, figures_dir / "matriz_confusao.png")
        plot_roc(fitted, X_test, y_test, figures_dir / "curva_roc.png")
        plot_importances(importances, figures_dir / "importancia_variaveis.png")

    results = {
        "n_clientes": int(len(df)),
        "taxa_churn": float(y.mean()),
        "cv_roc_auc": {k: {m: float(v) for m, v in row.items()} for k, row in cv.to_dict("index").items()},
        "test_metrics": {k: {m: float(v) for m, v in row.items()} for k, row in table.to_dict("index").items()},
        "top_features": {k: float(v) for k, v in importances.items()},
    }
    (reports_dir / "metrics.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline de churn de telecom")
    parser.add_argument("--no-plots", action="store_true", help="não gerar figuras")
    args = parser.parse_args()
    run_pipeline(make_plots=not args.no_plots)


if __name__ == "__main__":
    main()
