import pytest

from src.data import load_prepared, load_raw, prepare, split


@pytest.fixture(scope="module")
def raw():
    return load_raw()


def test_raw_shape_and_columns(raw):
    assert raw.shape == (1000, 5)
    assert list(raw.columns) == ["User ID", "Gender", "Age", "AnnualSalary", "Purchased"]   # BOM não contamina


def test_no_missing_values(raw):
    assert raw.isnull().sum().sum() == 0


def test_prepare_drops_id_and_encodes_gender(raw):
    df = prepare(raw)
    assert "User ID" not in df.columns and "Gender" not in df.columns
    assert set(df["Gender_encoded"].unique()) == {0, 1}
    female_rows = raw["Gender"] == "Female"
    assert (df.loc[female_rows, "Gender_encoded"] == 0).all()      # ordem alfabética: Female=0, Male=1


def test_split_is_stratified_and_sized():
    X_train, X_test, y_train, y_test = split(load_prepared())
    assert len(X_train) == 750 and len(X_test) == 250
    assert abs(y_train.mean() - y_test.mean()) < 0.01
    assert list(X_train.columns) == ["Age", "AnnualSalary", "Gender_encoded"]
