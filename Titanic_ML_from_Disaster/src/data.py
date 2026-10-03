"""Leitura dos arquivos brutos do Kaggle (data/raw/)."""
from pathlib import Path

import pandas as pd

from .config import DATA_RAW


def load_train(data_dir: Path = DATA_RAW) -> pd.DataFrame:
    return pd.read_csv(Path(data_dir) / "train.csv")


def load_test(data_dir: Path = DATA_RAW) -> pd.DataFrame:
    return pd.read_csv(Path(data_dir) / "test.csv")


def load_gender_submission(data_dir: Path = DATA_RAW) -> pd.DataFrame:
    """Baseline oficial do Kaggle ("apenas sexo"), usado como checagem de sanidade."""
    return pd.read_csv(Path(data_dir) / "gender_submission.csv")


def load_raw(data_dir: Path = DATA_RAW):
    """Retorna (train, test)."""
    return load_train(data_dir), load_test(data_dir)
