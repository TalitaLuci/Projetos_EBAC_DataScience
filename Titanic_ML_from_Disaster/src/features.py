"""Engenharia de atributos (versão v2 do projeto), sem vazamento de dados.

Princípios:
* As estatísticas de imputação (medianas de idade/tarifa, moda do porto) vêm SÓ do train.csv.
* ``Family_Survival`` usa apenas rótulos de *outros* passageiros do grupo e apenas rótulos
  declarados como conhecidos (``known_labels_mask``). Assim é possível simular fielmente o
  cenário de produção: passageiros do holdout/teste entram como "sem rótulo".
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import (CATEGORICAL_COLS, DECK_GROUP_MAP, FEATURES, KNOWN_TITLES, TITLE_MAP)


# ------------------------------------------------------------------ passos individuais
def extract_title(names: pd.Series) -> pd.Series:
    title = names.str.extract(r",\s*([^\.]*)\.", expand=False).replace(TITLE_MAP)
    return title.where(title.isin(KNOWN_TITLES), "Outro")


def impute_missing(combined: pd.DataFrame, train_mask: pd.Series) -> pd.DataFrame:
    """Imputa Age (mediana por Title+Pclass), Fare (mediana por Pclass) e Embarked (moda)."""
    out = combined.copy()
    train = out.loc[train_mask]

    age_medians = train.groupby(["Title", "Pclass"])["Age"].median()
    global_age = train["Age"].median()
    age_lookup = pd.Series(
        list(zip(out["Title"], out["Pclass"])), index=out.index
    ).map(age_medians.to_dict()).fillna(global_age)
    out["Age"] = out["Age"].fillna(age_lookup)

    fare_medians = train.groupby("Pclass")["Fare"].median()
    out["Fare"] = out["Fare"].fillna(out["Pclass"].map(fare_medians)).fillna(train["Fare"].median())

    out["Embarked"] = out["Embarked"].fillna(train["Embarked"].mode()[0])
    return out


def add_cabin_features(df: pd.DataFrame) -> pd.DataFrame:
    df["HasCabin"] = df["Cabin"].notnull().astype(int)
    df["Deck"] = df["Cabin"].str[0].map(DECK_GROUP_MAP).fillna("Desconhecido")
    return df


def add_family_features(df: pd.DataFrame) -> pd.DataFrame:
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    df["FarePerPerson"] = df["Fare"] / df["FamilySize"]
    return df


def add_family_survival(df: pd.DataFrame, known_labels_mask: pd.Series) -> pd.DataFrame:
    """Taxa de sobrevivência conhecida dos OUTROS integrantes do grupo (sobrenome+tarifa, depois bilhete).

    1.0 -> algum outro integrante conhecido sobreviveu
    0.0 -> nenhum sobreviveu e ao menos um morreu
    0.5 -> sem grupo identificável ou sem rótulos conhecidos (neutro)
    """
    labels = df["Survived"].where(known_labels_mask)  # rótulo só onde é "conhecido"
    df["Last_Name"] = df["Name"].str.split(",").str[0]
    result = pd.Series(0.5, index=df.index)

    def _apply(group_keys, only_neutral: bool):
        for _, idx in df.groupby(group_keys).groups.items():
            idx = list(idx)
            if len(idx) < 2:
                continue
            for i in idx:
                if only_neutral and result[i] != 0.5:
                    continue
                others = labels.loc[[j for j in idx if j != i]]
                if others.max() == 1.0:
                    result[i] = 1.0
                elif others.min() == 0.0:
                    result[i] = 0.0

    _apply(["Last_Name", "Fare"], only_neutral=False)   # passo 1: famílias que compraram juntas
    _apply("Ticket", only_neutral=True)                  # passo 2: mesmo bilhete, para quem sobrou neutro
    df["Family_Survival"] = result
    return df


# ------------------------------------------------------------------ função principal
def build_features(train: pd.DataFrame, test: pd.DataFrame, known_labels_idx=None):
    """Constrói (X_train, y_train, X_test) com colunas idênticas e sem valores ausentes.

    Parameters
    ----------
    known_labels_idx : índices (posições 0..len(train)-1) de ``train`` cujos rótulos podem alimentar
        ``Family_Survival``. ``None`` -> todos (uso na submissão final). Para avaliar num holdout
        sem vazamento, passe apenas os índices do conjunto de treino.
    """
    n_train = len(train)
    combined = pd.concat(
        [train.assign(_source="train"), test.assign(_source="test", Survived=np.nan)],
        sort=False, ignore_index=True,
    )
    train_mask = combined["_source"] == "train"

    known = pd.Series(False, index=combined.index)
    if known_labels_idx is None:
        known.iloc[:n_train] = True
    else:
        known.iloc[np.asarray(list(known_labels_idx), dtype=int)] = True

    combined["Title"] = extract_title(combined["Name"])
    combined = impute_missing(combined, train_mask)
    combined = add_cabin_features(combined)
    combined = add_family_features(combined)
    combined = add_family_survival(combined, known)

    encoded = pd.get_dummies(
        combined[FEATURES + ["Survived", "_source"]], columns=CATEGORICAL_COLS, drop_first=True
    )
    X_train = encoded[encoded["_source"] == "train"].drop(columns=["_source", "Survived"]).reset_index(drop=True)
    X_test = encoded[encoded["_source"] == "test"].drop(columns=["_source", "Survived"]).reset_index(drop=True)
    y_train = train["Survived"].astype(int).reset_index(drop=True)

    # dummies como float: evita problemas de dtype bool no scikit-learn/XGBoost
    X_train = X_train.astype(float)
    X_test = X_test.astype(float)
    assert list(X_train.columns) == list(X_test.columns)
    assert not X_train.isnull().any().any() and not X_test.isnull().any().any()
    return X_train, y_train, X_test
