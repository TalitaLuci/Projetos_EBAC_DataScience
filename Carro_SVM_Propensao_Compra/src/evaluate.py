"""Métricas e gráficos."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)


def compute_metrics(y_true, y_pred) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
    }


def plot_confusion_matrices(fitted: dict, X_test, y_test, path: Path) -> None:
    n = len(fitted)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4))
    for ax, (name, model) in zip(np.atleast_1d(axes), fitted.items()):
        cm = confusion_matrix(y_test, model.predict(X_test))
        ConfusionMatrixDisplay(cm, display_labels=["Não compra", "Compra"]).plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set(title=name, xlabel="Previsto", ylabel="Real")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_decision_regions(fitted: dict, X_train, y_train, path: Path, gender: int = 0) -> None:
    """Regiões de decisão no plano Age x AnnualSalary (Gender fixo; ele quase não influencia)."""
    age = np.linspace(X_train["Age"].min() - 2, X_train["Age"].max() + 2, 250)
    sal = np.linspace(X_train["AnnualSalary"].min() - 5000, X_train["AnnualSalary"].max() + 5000, 250)
    A, S = np.meshgrid(age, sal)
    grid = pd.DataFrame({"Age": A.ravel(), "AnnualSalary": S.ravel(), "Gender_encoded": gender})[X_train.columns]
    n = len(fitted)
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 4.2), sharey=True)
    for ax, (name, model) in zip(np.atleast_1d(axes), fitted.items()):
        Z = model.predict(grid).reshape(A.shape)
        ax.contourf(A, S, Z, levels=[-0.5, 0.5, 1.5], colors=["#cfe3f5", "#f8d3c5"], alpha=0.9)
        ax.scatter(X_train["Age"], X_train["AnnualSalary"], c=y_train, cmap="coolwarm", s=8, edgecolor="none")
        ax.set(title=name, xlabel="Idade")
    np.atleast_1d(axes)[0].set_ylabel("Salário anual")
    fig.suptitle("Regiões de decisão (azul = não compra, laranja = compra)", y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
