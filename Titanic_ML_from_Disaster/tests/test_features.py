import numpy as np
import pandas as pd
import pytest

from src.data import load_raw
from src.features import build_features, extract_title


@pytest.fixture(scope="module")
def raw():
    return load_raw()


def test_shapes_and_columns(raw):
    train, test = raw
    X, y, Xt = build_features(train, test)
    assert X.shape == (891, 21) and Xt.shape == (418, 21)
    assert list(X.columns) == list(Xt.columns)
    assert set(y.unique()) == {0, 1}


def test_no_missing_values(raw):
    X, _, Xt = build_features(*raw)
    assert not X.isnull().any().any()
    assert not Xt.isnull().any().any()


def test_title_extraction():
    names = pd.Series(["Braund, Mr. Owen Harris", "Heikkinen, Miss. Laina", "Rothes, the Countess. of (Lucy)",
                       "Palsson, Master. Gosta Leonard", "Weird, Xyz. Someone"])
    assert extract_title(names).tolist() == ["Mr", "Miss", "Nobre", "Master", "Outro"]


def test_family_survival_ignores_unknown_labels(raw):
    """Passageiros fora de known_labels_idx não podem influenciar o Family_Survival dos demais."""
    train, test = raw
    idx_train = np.arange(0, 700)
    X_a, _, _ = build_features(train, test, known_labels_idx=idx_train)

    altered = train.copy()
    altered.loc[700:, "Survived"] = 1 - altered.loc[700:, "Survived"]   # inverte rótulos "desconhecidos"
    X_b, _, _ = build_features(altered, test, known_labels_idx=idx_train)
    pd.testing.assert_frame_equal(X_a, X_b)


def test_family_survival_never_uses_own_label(raw):
    """Inverter o rótulo de um passageiro não pode alterar o PRÓPRIO Family_Survival dele."""
    train, test = raw
    base, _, _ = build_features(train, test)
    in_group = base.index[base["Family_Survival"] != 0.5][:25]   # passageiros com grupo identificado
    for i in in_group:
        flipped = train.copy()
        flipped.loc[i, "Survived"] = 1 - flipped.loc[i, "Survived"]
        X_b, _, _ = build_features(flipped, test)
        assert X_b.loc[i, "Family_Survival"] == base.loc[i, "Family_Survival"]
