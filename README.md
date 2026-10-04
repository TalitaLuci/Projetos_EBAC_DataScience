# Projetos EBAC — Profissão: Cientista de Dados

Repositório com os projetos desenvolvidos durante o curso **EBAC — Profissão: Cientista de Dados**.
Cada projeto vive na sua própria pasta, com README, dependências, dados, notebooks e código reutilizável.

| # | Projeto | Tema | Destaque |
|---|---------|------|----------|
| 1 | [Titanic_ML_from_Disaster](Titanic_ML_from_Disaster/) | Classificação (Kaggle) | Duelo de modelos, diagnóstico de overfitting v1 → v2, Kaggle 0,755 → **0,780** |
| 2 | [Churn_Telecom](Churn_Telecom/) | Classificação (churn) | Pipeline sem vazamento, Reg. Logística vs Random Forest, recall de **88%** na classe Churn |
| 3 | [Carro_SVM_Propensao_Compra](Carro_SVM_Propensao_Compra/) | Classificação (SVM) | Padronização corrige o kernel `poly` (74,8% → 83,2%); SVM `rbf` supera o XGBoost (90,3% vs 88,1% em CV) |
| 4 | [Teste_AB_Estrategias_Ensino](Teste_AB_Estrategias_Ensino/) | Estatística (teste Z) | Teste unilateral à direita (B − A), Z = 1,51, p = 0,065; análise de poder |

## Estrutura padrão de cada projeto

```
Nome_do_Projeto/
├── data/raw/          # dados originais (imutáveis)
├── notebooks/         # análises numeradas (01_..., 02_...)
├── src/               # código reutilizável (dados, features, modelos, pipeline)
├── tests/             # testes automatizados (pytest)
├── reports/figures/   # gráficos e métricas geradas
├── submissions/       # arquivos de submissão
├── requirements.txt
└── README.md
```

**Autor:** Talita Luci — [github.com/TalitaLuci](https://github.com/TalitaLuci)
