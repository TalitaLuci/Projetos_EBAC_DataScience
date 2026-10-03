"""Pipeline completo: duelo de modelos -> tuning -> ensemble -> holdout -> submissão.

Uso (a partir da raiz do projeto):
    python -m src.pipeline            # execução completa
    python -m src.pipeline --quick    # grades mínimas, ~1 min (teste de fumaça)

Diferença metodológica em relação ao notebook v2: aqui o holdout é *realmente* tratado como o
test.csv do Kaggle — os rótulos dos passageiros do holdout NÃO alimentam o ``Family_Survival`` de
ninguém durante a avaliação. A submissão final é refeita com todos os rótulos do train.csv.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import train_test_split

from . import config
from .data import load_gender_submission, load_raw
from .evaluate import (compute_metrics, plot_confusion_roc, plot_duel, plot_permutation_importance)
from .features import build_features
from .models import build_models, duel, make_ensemble, param_grids, scale_pos_weight, tune


def run_pipeline(quick: bool = False, make_plots: bool = True, n_jobs: int = -1,
                 submissions_dir: Path = config.SUBMISSIONS_DIR,
                 reports_dir: Path = config.REPORTS_DIR,
                 verbose: bool = True) -> dict:
    log = print if verbose else (lambda *a, **k: None)
    submissions_dir, reports_dir = Path(submissions_dir), Path(reports_dir)
    figures_dir = reports_dir / "figures"
    for d in (submissions_dir, figures_dir):
        d.mkdir(parents=True, exist_ok=True)

    train, test = load_raw()

    # ---------------------------------------------------------------- 1) avaliação com holdout
    idx_train, idx_hold = train_test_split(
        np.arange(len(train)), test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE, stratify=train["Survived"])
    X_all, y_all, _ = build_features(train, test, known_labels_idx=idx_train)  # holdout = "sem rótulo"
    X_tr, y_tr = X_all.iloc[idx_train], y_all.iloc[idx_train]
    X_ho, y_ho = X_all.iloc[idx_hold], y_all.iloc[idx_hold]
    log(f"Treino: {len(X_tr)} | Holdout: {len(X_ho)} | Atributos: {X_tr.shape[1]}")

    models = build_models(scale_pos_weight(y_tr))
    cv_df = duel(models, X_tr, y_tr, n_jobs=n_jobs)
    ranking = cv_df.mean().sort_values(ascending=False)
    log("\nDuelo (CV, sem tuning):\n" + ranking.round(4).to_string())

    finalists = ranking.index[:2].tolist()
    grids = param_grids(quick)
    tuned, tuned_scores, tuned_params = {}, {}, {}
    for name in finalists:
        search = tune(models[name], grids[name], X_tr, y_tr, n_jobs=n_jobs)
        tuned[name], tuned_scores[name], tuned_params[name] = search.best_estimator_, search.best_score_, search.best_params_
        log(f"Tuning {name}: CV={search.best_score_:.4f} | {search.best_params_}")

    ensemble = make_ensemble(tuned, finalists)
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    cv5 = StratifiedKFold(n_splits=config.CV_TUNING_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    ens_score = cross_val_score(ensemble, X_tr, y_tr, cv=cv5, scoring="accuracy", n_jobs=n_jobs).mean()

    candidates = {**tuned, "Ensemble (Voting Soft)": ensemble}
    cand_scores = {**tuned_scores, "Ensemble (Voting Soft)": ens_score}
    champion_name = max(cand_scores, key=cand_scores.get)
    champion = candidates[champion_name].fit(X_tr, y_tr)
    log(f"\nCampeão (CV): {champion_name} = {cand_scores[champion_name]:.4f}")

    y_pred = champion.predict(X_ho)
    y_proba = champion.predict_proba(X_ho)[:, 1]
    holdout_metrics = compute_metrics(y_ho, y_pred, y_proba)
    log("Holdout: " + " | ".join(f"{k}={v:.4f}" for k, v in holdout_metrics.items()))

    if make_plots:
        plot_duel(cv_df, figures_dir / "duelo_cv.png")
        plot_confusion_roc(y_ho, y_pred, y_proba, champion_name, figures_dir / "holdout_confusao_roc.png")
        plot_permutation_importance(champion, X_ho, y_ho, champion_name, figures_dir / "permutation_importance.png")

    # ---------------------------------------------------------------- 2) submissão final
    X_full, y_full, X_test = build_features(train, test)  # todos os rótulos do train.csv conhecidos
    final_model = clone(champion).fit(X_full, y_full)
    submission = pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": final_model.predict(X_test).astype(int)})
    sub_path = submissions_dir / "submission_pipeline.csv"
    submission.to_csv(sub_path, index=False)

    baseline = load_gender_submission().merge(submission, on="PassengerId", suffixes=("_sexo", "_modelo"))
    agreement_gender = float((baseline["Survived_sexo"] == baseline["Survived_modelo"]).mean())
    agreement_v2 = None
    v2_path = submissions_dir / "submission_v2.csv"
    if v2_path.exists():
        m = pd.read_csv(v2_path).merge(submission, on="PassengerId", suffixes=("_v2", "_pipe"))
        agreement_v2 = float((m["Survived_v2"] == m["Survived_pipe"]).mean())
    log(f"\nSubmissão salva em {sub_path} | taxa prevista de sobreviventes: {submission['Survived'].mean():.1%}"
        f" | concordância c/ baseline-sexo: {agreement_gender:.1%}"
        + (f" | c/ submission_v2: {agreement_v2:.1%}" if agreement_v2 is not None else ""))

    results = {
        "quick_mode": quick,
        "n_features": int(X_tr.shape[1]),
        "duel_cv_accuracy": {k: float(v) for k, v in ranking.items()},
        "finalists": finalists,
        "tuned_cv_accuracy": {k: float(v) for k, v in cand_scores.items()},
        "best_params": {k: {p: (v.item() if hasattr(v, "item") else v) for p, v in d.items()} for k, d in tuned_params.items()},
        "champion": champion_name,
        "holdout_metrics": {k: float(v) for k, v in holdout_metrics.items()},
        "submission_survival_rate": float(submission["Survived"].mean()),
        "agreement_with_gender_baseline": agreement_gender,
        "agreement_with_submission_v2": agreement_v2,
    }
    (reports_dir / "metrics.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline Titanic (duelo de modelos + submissão Kaggle)")
    parser.add_argument("--quick", action="store_true", help="grades mínimas de hiperparâmetros (teste rápido)")
    parser.add_argument("--no-plots", action="store_true", help="não gerar figuras")
    parser.add_argument("--n-jobs", type=int, default=-1, help="paralelismo do scikit-learn (padrão: todos os núcleos)")
    args = parser.parse_args()
    run_pipeline(quick=args.quick, make_plots=not args.no_plots, n_jobs=args.n_jobs)


if __name__ == "__main__":
    main()
