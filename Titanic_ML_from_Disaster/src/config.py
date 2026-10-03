"""Configurações centrais do projeto: caminhos, semente aleatória e listas de atributos."""
from pathlib import Path

# --- Caminhos -----------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# --- Reprodutibilidade --------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.2          # fração do train.csv reservada como holdout interno
CV_DUEL_FOLDS = 10       # validação cruzada do "duelo" de modelos
CV_TUNING_FOLDS = 5      # validação cruzada do GridSearchCV

# --- Engenharia de atributos --------------------------------------------------
TITLE_MAP = {
    "Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs",
    "Lady": "Nobre", "the Countess": "Nobre", "Countess": "Nobre", "Sir": "Nobre",
    "Jonkheer": "Nobre", "Don": "Nobre", "Dona": "Nobre",
    "Capt": "Oficial", "Col": "Oficial", "Major": "Oficial", "Dr": "Oficial", "Rev": "Oficial",
}
KNOWN_TITLES = ["Mr", "Mrs", "Miss", "Master", "Nobre", "Oficial"]

DECK_GROUP_MAP = {
    "A": "ABC", "B": "ABC", "C": "ABC", "T": "ABC",
    "D": "DE", "E": "DE",
    "F": "FG", "G": "FG",
}

FEATURES = [
    "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "FarePerPerson", "Embarked",
    "Title", "HasCabin", "Deck", "FamilySize", "IsAlone", "Family_Survival",
]
CATEGORICAL_COLS = ["Sex", "Embarked", "Title", "Deck"]
