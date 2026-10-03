import pandas as pd
import pytest

from src.data import clean, load_raw, split_xy


@pytest.fixture(scope="module")
def raw():
    return load_raw()


@pytest.fixture(scope="module")
def cleaned(raw):
    return clean(raw)


def test_raw_shape_and_columns(raw):
    assert raw.shape == (2500, 16)
    assert "customerID" in raw.columns          # BOM do arquivo não pode contaminar o 1º nome de coluna


def test_clean_removes_target_nulls_and_phone(raw, cleaned):
    assert len(cleaned) == len(raw) - raw["Churn"].isna().sum() == 2495
    assert cleaned["Churn"].notna().all()
    assert "PhoneService" not in cleaned.columns


def test_clean_standardizes_categories(cleaned):
    assert set(cleaned["Genero"].dropna().unique()) == {"Female", "Male"}
    assert set(cleaned["Servico_Internet"].unique()) == {"DSL", "Fiber optic", "No"}


def test_clean_renames_columns(cleaned):
    for col in ("Cliente_ID", "Dependentes", "Tempo_Cliente", "Streaming_TV", "Forma_Pagamento"):
        assert col in cleaned.columns


def test_clean_keeps_feature_nulls_for_pipeline(cleaned):
    """Imputação de features é responsabilidade da Pipeline (evita vazamento) — a limpeza não imputa."""
    assert cleaned["Pagamento_Mensal"].isna().sum() > 0


def test_split_xy(cleaned):
    X, y = split_xy(cleaned)
    assert set(y.unique()) == {0, 1} and abs(y.mean() - 0.26) < 0.01
    assert not {"Churn", "Cliente_ID", "Total_Pago"} & set(X.columns)
    assert X.shape == (2495, 12)
