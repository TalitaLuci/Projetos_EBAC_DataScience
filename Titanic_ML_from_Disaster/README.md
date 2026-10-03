# 🚢 Duelo de Modelos — Previsão de Sobrevivência no Titanic

Projeto de Machine Learning ponta a ponta para a competição [Titanic — Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic) (Kaggle), desenvolvido no curso **EBAC — Profissão: Cientista de Dados**.

O repositório documenta um **ciclo de iteração guiado por feedback real do Kaggle**: a v1 pontuou abaixo do esperado, o gap entre validação interna e leaderboard foi diagnosticado, e a v2 corrigiu o problema — com ganho comprovado por uma nova submissão.

## 📊 Resultados

| Versão | CV (tunada) | Holdout interno | **Kaggle** |
|---|---|---|---|
| v1 — XGBoost + SMOTE | 85,3% | 79,9% | **0,755** |
| v2 — Ensemble (SVM + LogReg), sem SMOTE | 86,7% | 79,3%* | **0,780** |

\* O holdout da v2 no notebook é levemente otimista (ver [nota metodológica](#-nota-metodológica-holdout-sem-vazamento)). O pipeline em `src/` reavalia sem esse viés: **≈77%**, bem mais próximo do 0,780 real do Kaggle.

### 🔍 Por que a v2 foi melhor
- **SMOTE → pesos de classe.** SMOTE depois do one-hot gerava valores fracionários sem sentido em colunas binárias (ex.: `Title_Mr = 0.4`).
- **Menos atributos redundantes** (sem `AgeBin`/`FareBin`; `Deck` de 8 → 4 categorias): 31 → 21 atributos.
- **`Family_Survival`**: sobrevivência conhecida dos *outros* integrantes do mesmo grupo (sobrenome+tarifa, ou bilhete).
- **Ensemble soft** (SVM + Regressão Logística) em vez de depender de um único "vencedor".

## 📁 Estrutura

```
Titanic_ML_from_Disaster/
├── data/raw/                 # train.csv, test.csv, gender_submission.csv (Kaggle)
├── notebooks/
│   ├── 01_titanic_v1_xgboost_smote.ipynb
│   └── 02_titanic_v2_ensemble_family_survival.ipynb
├── src/
│   ├── config.py             # caminhos, sementes, listas de atributos
│   ├── data.py               # leitura dos dados
│   ├── features.py           # engenharia de atributos (sem vazamento)
│   ├── models.py             # modelos, grades, duelo, tuning, ensemble
│   ├── evaluate.py           # métricas e gráficos
│   └── pipeline.py           # pipeline completo (CLI)
├── tests/                    # pytest
├── submissions/submission_v2.csv   # submissão enviada ao Kaggle (0,780)
├── reports/figures/          # gráficos gerados pelo pipeline
├── requirements.txt
└── README.md
```

## ▶️ Como executar

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 1) Notebooks (abra a partir da raiz ou de notebooks/ — os caminhos se ajustam sozinhos)
jupyter notebook notebooks/

# 2) Pipeline em linha de comando
python -m src.pipeline --quick     # ~15 s, grades mínimas (teste rápido)
python -m src.pipeline             # completo (~20 s com 1 núcleo)

# 3) Testes
pytest
```

O pipeline gera `submissions/submission_pipeline.csv`, `reports/metrics.json` e os gráficos em `reports/figures/`.

## 🧪 Nota metodológica: holdout sem vazamento

No notebook v2, `Family_Survival` é calculado com **todos** os rótulos do `train.csv`, inclusive os do holdout; assim, os rótulos do holdout alimentam os atributos de seus próprios familiares e o holdout fica otimista (79,3%). No `src/pipeline.py`, passageiros do holdout entram como **sem rótulo** durante a avaliação (exatamente como o `test.csv` no Kaggle), e só a submissão final usa todos os rótulos. O resultado (≈77%) é uma estimativa mais honesta. Os notebooks foram mantidos como estavam para preservar o histórico da v1 → v2.

## 🛠️ Stack
`Python` · `pandas` · `scikit-learn` · `XGBoost` · `imbalanced-learn` (v1) · `seaborn`/`matplotlib` · `pytest`

---
**Autor:** Talita Luci · Mestrando em Engenharia Mecânica (UDESC) · [github.com/TalitaLuci](https://github.com/TalitaLuci)
