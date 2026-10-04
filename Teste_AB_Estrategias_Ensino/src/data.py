"""Geração (simulada, reprodutível) e leitura das notas das duas estratégias."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import (DATA_FILE, MEAN_A, MEAN_B, N_PER_GROUP, SEED, SIGMA_A, SIGMA_B)


def generate_samples(seed: int = SEED, n: int = N_PER_GROUP):
    """Reproduz exatamente os dados do exercício (np.random.seed(0), A e depois B)."""
    rng = np.random.RandomState(seed)             # mesmo gerador do np.random.seed(seed)
    sample_a = rng.normal(loc=MEAN_A, scale=SIGMA_A, size=n)
    sample_b = rng.normal(loc=MEAN_B, scale=SIGMA_B, size=n)
    return sample_a, sample_b


def to_frame(sample_a, sample_b) -> pd.DataFrame:
    """Formato longo: uma linha por aluno."""
    return pd.DataFrame({
        "aluno": np.arange(1, len(sample_a) + len(sample_b) + 1),
        "estrategia": ["A"] * len(sample_a) + ["B"] * len(sample_b),
        "nota": np.concatenate([sample_a, sample_b]),
    })


def save_samples(path: Path = DATA_FILE) -> pd.DataFrame:
    df = to_frame(*generate_samples())
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def load_samples(path: Path = DATA_FILE):
    """Lê o CSV e devolve (nota_A, nota_B) como arrays."""
    df = pd.read_csv(path)
    return (df.loc[df["estrategia"] == "A", "nota"].to_numpy(),
            df.loc[df["estrategia"] == "B", "nota"].to_numpy())
