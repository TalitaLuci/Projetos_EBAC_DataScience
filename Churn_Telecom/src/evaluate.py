"""Métricas e gráficos de avaliação."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score, roc_curve)


def compute_metrics(y_true, y_pred, y_proba) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }


def plot_confusion_matrices(fitted: dict, X_test, y_test, path: Path) -> None:
    fig, axes = plt.subplots(1, len(fitted), figsize=(6 * len(fitted), 5))
    for ax, (name, pipe) in zip(axes, fitted.items()):
        cm = confusion_matrix(y_test, pipe.predict(X_test))
        ConfusionMatrixDisplay(cm, display_labels=["Não Churn", "Churn"]).plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title(f"Matriz de Confusão — {name}")
        ax.set_xlabel("Previsto")
        ax.set_ylabel("Real")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_roc(fitted: dict, X_test, y_test, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for name, pipe in fitted.items():
        proba = pipe.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, proba)
        ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_test, proba):.3f})")
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Modelo aleatório")
    ax.set(xlabel="Taxa de Falsos Positivos", ylabel="Recall (Churn)", title="Curva ROC — Comparação de Modelos")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def feature_importances(rf_pipe, top: int = 12) -> pd.Series:
    names = rf_pipe.named_steps["preprocessor"].get_feature_names_out()
    imp = pd.Series(rf_pipe.named_steps["model"].feature_importances_, index=names)
    return imp.sort_values(ascending=False).head(top)


def plot_importances(imp: pd.Series, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    imp.sort_values().plot(kind="barh", color="steelblue", ax=ax)
    ax.set(title=f"Top {len(imp)} variáveis mais importantes — Random Forest", xlabel="Importância")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
