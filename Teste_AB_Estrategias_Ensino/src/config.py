"""Parâmetros do enunciado e caminhos do projeto."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_ROOT / "data" / "raw" / "notas_estrategias.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# --- Parâmetros populacionais (dados pelo enunciado, usados no teste Z) -----------
MEAN_A, SIGMA_A = 70, 10
MEAN_B, SIGMA_B = 75, 12
N_PER_GROUP = 50
SEED = 0

ALPHA = 0.05
# H0: mu_B = mu_A   vs   H1: mu_B > mu_A  -> teste unilateral à direita com Z = (media_B - media_A) / EP
ALTERNATIVE = "greater"
