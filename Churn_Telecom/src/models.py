"""Definição dos modelos e utilitários de treino/validação cruzada."""
from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from .config import CV_FOLDS, RANDOM_STATE
from .features import build_preprocessor


def build_models() -> dict:
    """Ambos com class_weight='balanced' para compensar ~26% de churn."""
    return {
        "Regressão Logística": LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=RANDOM_STATE),
    }


def build_pipelines(X: pd.DataFrame) -> dict[str, Pipeline]:
    return {name: Pipeline([("preprocessor", build_preprocessor(X)), ("model", model)])
            for name, model in build_models().items()}


def cross_validate_auc(pipelines: dict, X, y, n_jobs: int = 1) -> pd.DataFrame:
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    rows = {}
    for name, pipe in pipelines.items():
        scores = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc", n_jobs=n_jobs)
        rows[name] = {"roc_auc_mean": scores.mean(), "roc_auc_std": scores.std()}
    return pd.DataFrame(rows).T
