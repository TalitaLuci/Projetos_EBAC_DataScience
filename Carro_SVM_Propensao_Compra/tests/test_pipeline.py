import json

import numpy as np

from src.data import load_prepared, split
from src.models import build_models, scaled_svm
from src.pipeline import run_pipeline


def test_scaler_is_inside_pipeline_and_fit_on_train_only():
    X_train, X_test, y_train, _ = split(load_prepared())
    model = scaled_svm("linear").fit(X_train, y_train)
    scaler = model.named_steps["scaler"]
    assert np.allclose(scaler.mean_, X_train.mean().values)         # estatísticas só do treino


def test_scaling_fixes_poly_kernel():
    """Regressão do problema original: sem padronização o kernel poly tinha recall ~0,43."""
    from sklearn.metrics import recall_score
    from sklearn.svm import SVC

    X_train, X_test, y_train, y_test = split(load_prepared())
    scaled = scaled_svm("poly").fit(X_train, y_train)
    unscaled = SVC(kernel="poly", random_state=42).fit(X_train, y_train)
    assert recall_score(y_test, scaled.predict(X_test)) > recall_score(y_test, unscaled.predict(X_test)) + 0.15


def test_pipeline_end_to_end(tmp_path):
    results = run_pipeline(make_plots=True, reports_dir=tmp_path, verbose=False)
    assert (tmp_path / "figures" / "regioes_decisao_svm.png").exists()
    test, cv = results["test_metrics"], results["cv_metrics"]
    assert set(test) == set(build_models())
    assert test["SVM rbf"]["accuracy"] > test["SVM linear"]["accuracy"]
    assert cv["SVM rbf"]["cv_accuracy_mean"] > cv["SVM linear"]["cv_accuracy_mean"]
    assert cv["SVM rbf"]["cv_accuracy_mean"] > cv["XGBoost"]["cv_accuracy_mean"]
    assert json.loads((tmp_path / "metrics.json").read_text())["n_clientes"] == 1000
