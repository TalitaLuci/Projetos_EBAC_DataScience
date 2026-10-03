"""Pré-processamento (imputação + padronização + one-hot) como ColumnTransformer."""
from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import NUMERIC_FEATURES


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Numéricas: mediana -> StandardScaler. Categóricas: moda -> one-hot (binárias viram 1 coluna)."""
    categorical = [c for c in X.columns if c not in NUMERIC_FEATURES]
    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", drop="if_binary")),
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, NUMERIC_FEATURES),
        ("cat", categorical_pipe, categorical),
    ])
