# 📉 Modelagem de Churn — Telecomunicações

Projeto de Machine Learning (classificação binária) que prevê o **cancelamento de clientes (churn)** de uma operadora de telecom a partir do perfil, dos serviços contratados e das condições comerciais, comparando **Regressão Logística** e **Random Forest**. Desenvolvido no curso **EBAC — Profissão: Cientista de Dados**.

## 📊 Resultados (conjunto de teste, 499 clientes)

| Modelo | Acurácia | Precisão (Churn) | Recall (Churn) | F1 (Churn) | ROC-AUC |
|---|---|---|---|---|---|
| **Regressão Logística** | 76,4% | 52,8% | **87,7%** | **65,9%** | **0,874** |
| Random Forest | **78,8%** | 59,5% | 57,7% | 58,6% | 0,846 |

- Validação cruzada (5 folds, treino): ROC-AUC de **0,844 ± 0,015** (Reg. Logística) e **0,816 ± 0,022** (Random Forest).
- **Trade-off:** a Regressão Logística captura ~88% dos cancelamentos reais (ao custo de mais alarmes falsos); o Random Forest acerta mais no geral, mas deixa escapar ~42% dos clientes que cancelam. Como a ação de retenção costuma custar bem menos que perder o cliente, o recall pesa mais aqui, e a acurácia sozinha engana (classe majoritária = 74%).

### Principais achados da análise exploratória
- Churn da base: **26,0%** dos clientes.
- **Contrato mensal:** 42% de churn, contra 11% (1 ano) e 1,5% (2 anos).
- **Tempo de casa:** mediana de 10 meses entre quem cancela, contra 37 meses entre quem fica.
- **Mensalidade:** mediana de ≈ R$ 81 entre quem cancela, contra ≈ R$ 65 entre quem fica.
- **Internet fibra ótica:** 42% de churn, contra 17% (DSL) e 6% (sem internet).
- **Gênero:** diferença pequena (27,2% vs 24,8%) e **não significativa** (qui-quadrado, p = 0,19).

## 📁 Estrutura

```
Churn_Telecom/
├── data/raw/CHURN_TELECON.csv          # base bruta (2.500 clientes, separador ';')
├── notebooks/
│   └── 01_modelagem_churn_telecom.ipynb   # análise completa, com saídas
├── src/
│   ├── config.py      # caminhos, semente, mapas de padronização
│   ├── data.py        # leitura e limpeza (sem imputar features)
│   ├── features.py    # ColumnTransformer: imputação + escala + one-hot
│   ├── models.py      # modelos e validação cruzada
│   ├── evaluate.py    # métricas e gráficos
│   └── pipeline.py    # pipeline completo (CLI)
├── tests/             # pytest (9 testes)
├── reports/           # metrics.json + figures/ gerados pelo pipeline
├── requirements.txt
└── README.md
```

## ▶️ Como executar

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

jupyter notebook notebooks/        # análise completa (abre de notebooks/ ou da raiz)
python -m src.pipeline             # pipeline em linha de comando (~5 s)
pytest                             # testes
```

O pipeline gera `reports/metrics.json` e os gráficos em `reports/figures/`.

## 🧠 Decisões e correções em relação à versão original

| Tema | Antes | Agora |
|---|---|---|
| Caminho dos dados | o notebook lia `CHURN_TELECON_MOD08_TAREFA.csv`, que não existe no repositório | lê `data/raw/CHURN_TELECON.csv` por caminho relativo à raiz do projeto |
| Imputação | `Genero` (moda) e `Pagamento_Mensal` (média) imputados na base inteira, **antes** da separação treino/teste | imputação dentro da `Pipeline` (moda/mediana), calculada só no treino e refeita em cada fold |
| `Pagamento_Mensal` | média, com o argumento de que a distribuição era simétrica | **mediana**: a distribuição é assimétrica (média 65,6 vs mediana 71,5, com um grupo de mensalidades baixas) |
| `Total_Pago` | removido "por vazamento" | removido por **redundância** (correlação de 0,998 com `Pagamento_Mensal × Tempo_Cliente`); incluí-lo muda o ROC-AUC em apenas +0,002 a +0,006 |
| Gênero | "praticamente igual" | teste qui-quadrado explícito no notebook (p = 0,19) |
| Importância de variáveis | lida como confirmação | com ressalva: a importância por impureza favorece variáveis contínuas e dá algum peso a `Genero`, sem efeito real |
| Outros | referência de seção errada (Seção 6 → 7); sem testes; dependências sem `scipy` | corrigidos; 9 testes; `requirements.txt` completo |

As métricas de teste mudaram muito pouco com a imputação refeita (ROC-AUC de 0,8739 → 0,8737 na Reg. Logística e de 0,8472 → 0,8456 no Random Forest).

## 🔭 Próximos passos
- Gradient Boosting/XGBoost e ajuste de hiperparâmetros (`GridSearchCV`).
- Escolher o ponto de corte (threshold) pelo custo real de retenção vs. perda do cliente.
- Testar `PhoneService` como categoria "Desconhecido" (em validação cruzada, o ROC-AUC do Random Forest sobe ~0,009; o da Reg. Logística não muda).
- Validar com dados mais recentes antes de qualquer uso em produção.

## 🛠️ Stack
`Python` · `pandas` · `NumPy` · `scikit-learn` · `SciPy` · `matplotlib`/`seaborn` · `pytest`

---
**Autor:** Talita Luci · [github.com/TalitaLuci](https://github.com/TalitaLuci)
