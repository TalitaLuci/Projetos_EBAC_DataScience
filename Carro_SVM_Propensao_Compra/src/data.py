"""Leitura, limpeza e separação X/y da base de clientes."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from .config import DATA_FILE, ID_COLUMN, RANDOM_STATE, TARGET, TEST_SIZE


def load_raw(path: Path = DATA_FILE) -> pd.DataFrame:
    """Lê o CSV bruto (UTF-8 com BOM e terminadores CRLF)."""
    return pd.read_csv(path, encoding="utf-8-sig")


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Remove o ID (sem valor preditivo) e codifica Gender com LabelEncoder (Female=0, Male=1).

    O mapeamento do LabelEncoder é determinístico (ordem alfabética), por isso aplicá-lo antes da
    separação treino/teste não vaza informação.
    """
    out = df.drop(columns=[ID_COLUMN]).copy()
    out["Gender_encoded"] = LabelEncoder().fit_transform(out["Gender"])
    return out.drop(columns=["Gender"])


def load_prepared(path: Path = DATA_FILE) -> pd.DataFrame:
    return prepare(load_raw(path))


def split(df: pd.DataFrame):
    """X/y e divisão estratificada treino/teste (75/25)."""
    X, y = df.drop(columns=[TARGET]), df[TARGET]
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
