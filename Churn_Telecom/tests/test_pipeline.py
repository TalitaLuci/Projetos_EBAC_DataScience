import json

import numpy as np
import pandas as pd

from src.data import load_clean, split_xy
from src.features import build_preprocessor
from src.pipeline import run_pipeline


def test_preprocessor_imputes_from_train_only():
    """A mediana usada na imputação deve vir só do treino, nunca do teste."""
    X, _ = split_xy(load_clean())
    X_train, X_test = X.iloc[:2000].copy(), X.iloc[2000:].copy()
    X_test["Pagamento_Mensal"] = 9999.0                      # valores extremos só no "teste"
    pre = build_preprocessor(X).fit(X_train)
    median_train = X_train["Pagamento_Mensal"].median()
    X_probe = X_train.iloc[:1].copy()
    X_probe["Pagamento_Mensal"] = np.nan
    out = pre.transform(X_probe)
    idx = list(pre.get_feature_names_out()).index("num__Pagamento_Mensal")
    scaler = pre.named_transformers_["num"].named_steps["scaler"]
    expected = (median_train - scaler.mean_[2]) / scaler.scale_[2]
    assert np.isclose(out[0, idx], expected)


def test_preprocessor_handles_unseen_category():
    X, _ = split_xy(load_clean())
    pre = build_preprocessor(X).fit(X)
    probe = X.iloc[:1].copy()
    probe["Forma_Pagamento"] = "Pix"                          # categoria nunca vista
    assert pre.transform(probe).shape[0] == 1


def test_pipeline_end_to_end(tmp_path):
    results = run_pipeline(make_plots=True, reports_dir=tmp_path, verbose=False)
    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "figures" / "curva_roc.png").exists()
    lr, rf = results["test_metrics"]["Regressão Logística"], results["test_metrics"]["Random Forest"]
    assert lr["roc_auc"] > 0.80 and rf["roc_auc"] > 0.80
    assert lr["recall"] > rf["recall"]                        # trade-off documentado no README/notebook
    assert rf["accuracy"] > lr["accuracy"]
    assert json.loads((tmp_path / "metrics.json").read_text())["n_clientes"] == 2495
