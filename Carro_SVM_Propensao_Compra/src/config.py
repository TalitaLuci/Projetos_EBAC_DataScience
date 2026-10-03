"""Configurações centrais: caminhos, semente e parâmetros de validação."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_ROOT / "data" / "raw" / "CARRO_CLIENTES.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

RANDOM_STATE = 42
TEST_SIZE = 0.25
CV_FOLDS = 5

TARGET = "Purchased"
ID_COLUMN = "User ID"
