"""Definição dos modelos, grades de hiperparâmetros, duelo (CV), tuning e ensemble.

Todos os modelos são ``Pipeline(StandardScaler -> classificador)``: o scaler é reajustado dentro de
cada fold da validação cruzada, evitando vazar estatísticas do fold de validação para o treino.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from .config import CV_DUEL_FOLDS, CV_TUNING_FOLDS, RANDOM_STATE


def build_models(scale_pos_weight: float) -> dict[str, Pipeline]:
    """Os quatro competidores do duelo, balanceados por peso de classe (sem SMOTE)."""
    def pipe(clf):
        return Pipeline([("scaler", StandardScaler()), ("clf", clf)])

    return {
        "Regressão Logística": pipe(LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)),
        "Random Forest": pipe(RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE)),
        "SVM (RBF)": pipe(SVC(probability=True, class_weight="balanced", random_state=RANDOM_STATE)),
        "XGBoost": pipe(XGBClassifier(eval_metric="logloss", scale_pos_weight=scale_pos_weight, random_state=RANDOM_STATE)),
    }


def param_grids(quick: bool = False) -> dict[str, dict]:
    """Grades do GridSearchCV. ``quick=True`` usa grades mínimas (teste de fumaça / CI)."""
    if quick:
        return {
            "Regressão Logística": {"clf__C": [0.1, 1]},
            "Random Forest": {"clf__n_estimators": [100], "clf__max_depth": [4, None]},
            "SVM (RBF)": {"clf__C": [1, 10], "clf__gamma": ["scale"]},
            "XGBoost": {"clf__n_estimators": [100], "clf__max_depth": [2, 3], "clf__learning_rate": [0.1]},
        }
    return {
        "Regressão Logística": {"clf__C": [0.01, 0.1, 1, 10]},
        "Random Forest": {
            "clf__n_estimators": [200, 400],
            "clf__max_depth": [4, 6, None],
            "clf__min_samples_split": [2, 5],
            "clf__min_samples_leaf": [1, 2, 4],
            "clf__max_features": ["sqrt", "log2"],
        },
        "SVM (RBF)": {"clf__C": [0.1, 1, 10, 50], "clf__gamma": ["scale", 0.01, 0.1]},
        "XGBoost": {
            "clf__n_estimators": [200, 400],
            "clf__max_depth": [2, 3, 4],
            "clf__learning_rate": [0.03, 0.1],
            "clf__subsample": [0.8, 1.0],
            "clf__colsample_bytree": [0.8, 1.0],
            "clf__reg_lambda": [1, 5],
        },
    }


def duel(models: dict, X: pd.DataFrame, y: pd.Series, n_jobs: int = -1) -> pd.DataFrame:
    """Validação cruzada estratificada de todos os modelos -> DataFrame (folds x modelos)."""
    cv = StratifiedKFold(n_splits=CV_DUEL_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    return pd.DataFrame({
        name: cross_val_score(model, X, y, cv=cv, scoring="accuracy", n_jobs=n_jobs)
        for name, model in models.items()
    })


def tune(model, grid: dict, X: pd.DataFrame, y: pd.Series, n_jobs: int = -1) -> GridSearchCV:
    cv = StratifiedKFold(n_splits=CV_TUNING_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    return GridSearchCV(model, grid, cv=cv, scoring="accuracy", n_jobs=n_jobs).fit(X, y)


def make_ensemble(tuned: dict, finalists: list[str]) -> VotingClassifier:
    """Votação *soft* entre os finalistas já otimizados."""
    return VotingClassifier(estimators=[(name, tuned[name]) for name in finalists], voting="soft")


def scale_pos_weight(y: pd.Series) -> float:
    neg, pos = np.bincount(y)
    return neg / pos
