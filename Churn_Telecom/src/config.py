"""Configurações centrais: caminhos, semente aleatória e dicionários de padronização."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_FILE = DATA_RAW / "CHURN_TELECON.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

RANDOM_STATE = 0
TEST_SIZE = 0.2
CV_FOLDS = 5

# --- Limpeza ------------------------------------------------------------------
TARGET = "Churn"
# PhoneService tem ~59% de valores ausentes: removida em vez de imputada
DROP_RAW_COLUMNS = ["PhoneService"]
GENERO_FIX = {"F": "Female", "f": "Female", "M": "Male"}
INTERNET_FIX = {"dsl": "DSL"}

COLUMN_RENAME = {
    "customerID": "Cliente_ID", "Dependents": "Dependentes", "Tempo_como_Cliente": "Tempo_Cliente",
    "StreamingTV": "Streaming_TV", "PaymentMethod": "Forma_Pagamento",
}

# --- Modelagem ----------------------------------------------------------------
# Total_Pago e Cliente_ID ficam de fora das features (ver README, seção de decisões)
NON_FEATURES = ["Cliente_ID", "Total_Pago"]
NUMERIC_FEATURES = ["Idoso", "Tempo_Cliente", "Pagamento_Mensal"]
