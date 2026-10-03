# 🚗 SVM — Propensão de Compra de Carros

Exercício do **Módulo 40** do curso **EBAC — Profissão: Cientista de Dados**: prever se um cliente vai comprar um carro (idade, salário anual e gênero) com **SVM** (kernels `linear`, `poly` e, como extra, `rbf`), comparando com o **XGBoost** da atividade anterior.

## 📊 Resultados

Base: 1.000 clientes, 40,2% compradores. Treino/teste estratificado 75/25 e validação cruzada de 5 folds na base completa. Todos os SVMs usam `StandardScaler` dentro de uma `Pipeline`.

| Modelo | Acurácia (teste) | Recall compra (teste) | Acurácia (CV 5 folds) | F1 compra (CV) |
|---|---|---|---|---|
| SVM linear | 82,8% | 0,68 | 82,6% ± 2,7 | 0,764 |
| SVM poly | 83,2% | 0,67 | 84,2% ± 2,2 | 0,786 |
| **SVM rbf** *(extra)* | **91,2%** | **0,91** | **90,3% ± 1,9** | **0,882** |
| XGBoost | 89,6% | 0,88 | 88,1% ± 2,3 | 0,853 |

**Conclusões**
- Entre os kernels pedidos, **linear e poly empatam**; ambos deixam de identificar ~30% dos compradores.
- O XGBoost supera os dois SVMs pedidos, como no resultado original.
- A fronteira de decisão **não é linear** (a taxa de compra salta de 29% para 77% entre 41–45 e 46–48 anos, e entre os menores de 40 a compra depende do salário). O **SVM `rbf`** acompanha essa curva e supera o XGBoost padrão. A diferença na validação cruzada é de ~2 p.p., com desvio de ~2 p.p. em cada modelo, então o ganho é consistente, mas pequeno.

## 🔧 Correção principal em relação à versão original

O notebook original treinava os SVMs **sem padronizar** as variáveis: `AnnualSalary` (dezenas de milhares) dominava `Age` (dezenas). Efeitos medidos:

| | Sem padronização | Com padronização |
|---|---|---|
| SVM poly — acurácia / recall | 74,8% / 0,43 | **83,2% / 0,67** |
| SVM linear — acurácia / recall | 82,4% / 0,72 | 82,8% / 0,68 |
| SVM linear — tempo de treino | ~34 s | < 1 s |

O texto original atribuía o mau resultado do `poly` à relação ser "quase linear". Isso era, em boa parte, efeito da escala. O ranking final também mudou: de **XGBoost > linear > poly** para **rbf > XGBoost > poly ≈ linear**. Outras mudanças: comparação por validação cruzada (e não só um único teste de 250 linhas), caminho relativo para os dados e título do notebook corrigido (dizia "Módulo 39").

## 📁 Estrutura

```
Carro_SVM_Propensao_Compra/
├── data/raw/CARRO_CLIENTES.csv
├── notebooks/01_svm_propensao_compra_carros.ipynb   # exercício completo, com saídas
├── src/
│   ├── config.py     # caminhos, semente, parâmetros
│   ├── data.py       # leitura, LabelEncoder, split
│   ├── models.py     # SVMs padronizados, XGBoost, validação cruzada
│   ├── evaluate.py   # métricas e gráficos (matrizes, regiões de decisão)
│   └── pipeline.py   # pipeline completo (CLI)
├── tests/            # pytest (7 testes)
├── reports/          # metrics.json + figures/
├── requirements.txt
└── README.md
```

## ▶️ Como executar

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

jupyter notebook notebooks/       # abre de notebooks/ ou da raiz (a seção 7.1 leva ~35 s)
python -m src.pipeline            # pipeline em linha de comando (~5 s)
python -m src.pipeline --unscaled # inclui os SVMs sem padronização (~35 s)
pytest                            # testes
```

## 🔭 Próximos passos
- Ajustar `C` e `gamma` do `rbf` e os hiperparâmetros do XGBoost com `GridSearchCV` antes de declarar um vencedor definitivo.
- Escolher o ponto de corte pelo custo de perder um comprador vs. o de abordar um não comprador.

## 🛠️ Stack
`Python` · `pandas` · `scikit-learn` · `XGBoost` · `matplotlib`/`seaborn` · `pytest`

---
**Autor:** Talita Luci · [github.com/TalitaLuci](https://github.com/TalitaLuci)
