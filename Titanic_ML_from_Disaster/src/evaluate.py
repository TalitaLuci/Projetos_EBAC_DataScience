"""Métricas e gráficos de avaliação."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # backend sem janela: funciona em servidor/CI
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)

from .config import RANDOM_STATE

LABELS = ["Não sobreviveu", "Sobreviveu"]


def compute_metrics(y_true, y_pred, y_proba) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }


def plot_duel(cv_df: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.boxplot(data=cv_df, ax=ax)
    sns.stripplot(data=cv_df, ax=ax, color="black", alpha=0.4, size=4)
    ax.set_title("Duelo de Modelos — Acurácia em Validação Cruzada")
    ax.set_ylabel("Acurácia")
    plt.xticks(rotation=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_confusion_roc(y_true, y_pred, y_proba, title: str, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt="d", cmap="Blues", ax=axes[0],
                xticklabels=LABELS, yticklabels=LABELS)
    axes[0].set(title=f"Matriz de Confusão — {title}", xlabel="Previsto", ylabel="Real")
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    axes[1].plot(fpr, tpr, linewidth=2, label=f"{title} (AUC = {roc_auc_score(y_true, y_proba):.3f})")
    axes[1].plot([0, 1], [0, 1], "--", color="gray", label="Classificador aleatório")
    axes[1].set(title="Curva ROC", xlabel="Taxa de Falsos Positivos", ylabel="Taxa de Verdadeiros Positivos")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_permutation_importance(model, X, y, title: str, path: Path, top: int = 15) -> None:
    result = permutation_importance(model, X, y, n_repeats=30, random_state=RANDOM_STATE,
                                    scoring="accuracy", n_jobs=1)
    imp = pd.Series(result.importances_mean, index=X.columns).sort_values(ascending=False).head(top)
    fig, ax = plt.subplots(figsize=(8, 6))
    imp.sort_values().plot(kind="barh", ax=ax, color="#2c7fb8")
    ax.set_title(f"Top {top} atributos (Permutation Importance) — {title}")
    ax.set_xlabel("Queda média de acurácia ao embaralhar o atributo")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
