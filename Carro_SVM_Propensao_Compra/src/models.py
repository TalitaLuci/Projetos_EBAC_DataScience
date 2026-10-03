"""Modelos do exercício: SVM (linear, poly, rbf) e XGBoost de referência.

Os SVMs são ``Pipeline(StandardScaler -> SVC)``: SVM depende de distâncias/produtos internos, e
``AnnualSalary`` (dezenas de milhares) domina ``Age`` (dezenas) quando não há padronização.
O scaler é reajustado dentro de cada fold da validação cruzada.
"""
from __future__ import annotations

import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from .config import CV_FOLDS, RANDOM_STATE


def scaled_svm(kernel: str) -> Pipeline:
    return Pipeline([("scaler", StandardScaler()), ("svm", SVC(kernel=kernel, random_state=RANDOM_STATE))])


def build_models() -> dict:
    return {
        "SVM linear": scaled_svm("linear"),
        "SVM poly": scaled_svm("poly"),
        "SVM rbf": scaled_svm("rbf"),
        "XGBoost": XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss"),
    }


def build_unscaled_svms() -> dict:
    """Versão do exercício original (sem padronização), mantida só para comparação."""
    return {
        "SVM linear (sem escala)": SVC(kernel="linear", random_state=RANDOM_STATE),
        "SVM poly (sem escala)": SVC(kernel="poly", random_state=RANDOM_STATE),
    }


def cross_validate_models(models: dict, X, y, n_jobs: int = 1) -> pd.DataFrame:
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    rows = {}
    for name, model in models.items():
        r = cross_validate(model, X, y, cv=cv, scoring=["accuracy", "recall", "f1"], n_jobs=n_jobs)
        rows[name] = {
            "cv_accuracy_mean": r["test_accuracy"].mean(), "cv_accuracy_std": r["test_accuracy"].std(),
            "cv_recall_mean": r["test_recall"].mean(), "cv_f1_mean": r["test_f1"].mean(),
        }
    return pd.DataFrame(rows).T
