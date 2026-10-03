"""Carregamento e limpeza da base bruta.

A limpeza aqui NÃO imputa valores ausentes de features: isso é feito dentro da Pipeline do
scikit-learn (ajustado só no treino), evitando vazamento de informação para o teste.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import (COLUMN_RENAME, DATA_FILE, DROP_RAW_COLUMNS, GENERO_FIX, INTERNET_FIX,
                     NON_FEATURES, TARGET)


def load_raw(path: Path = DATA_FILE) -> pd.DataFrame:
    """Lê o CSV bruto (separador ';', com BOM UTF-8 e terminadores CRLF)."""
    return pd.read_csv(path, delimiter=";", encoding="utf-8-sig")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Padroniza categorias e nomes, remove linhas sem alvo e colunas inviáveis."""
    out = df.copy()
    out = out.dropna(subset=["Churn"])                       # sem rótulo -> inutilizável
    out = out.drop(columns=DROP_RAW_COLUMNS)
    out["Genero"] = out["Genero"].replace(GENERO_FIX)
    out["Servico_Internet"] = out["Servico_Internet"].replace(INTERNET_FIX)
    out = out.rename(columns=COLUMN_RENAME).reset_index(drop=True)
    return out


def split_xy(clean_df: pd.DataFrame):
    """Retorna (X, y) com y binário (1 = cancelou) e sem colunas que não são features."""
    y = (clean_df[TARGET] == "Yes").astype(int)
    X = clean_df.drop(columns=[TARGET] + NON_FEATURES)
    return X, y


def load_clean(path: Path = DATA_FILE) -> pd.DataFrame:
    return clean(load_raw(path))
