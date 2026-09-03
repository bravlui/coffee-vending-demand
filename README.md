# coffee-vending-demand

Previsão de demanda e segmentação de clientes para operações de máquinas de café.

Projeto de ciência de dados ponta a ponta e reprodutível sobre o dataset público
**[Coffee Sales](https://www.kaggle.com/datasets/ihelon/coffee-sales)** — o log de
transações de uma única máquina de café (mar/2024 – mar/2025).

---

## 1. Problema

Um operador de máquinas de café decide **quanto de cada produto abastecer e
quando**. Hoje a decisão é manual e reativa:

- **Ruptura de estoque** → venda perdida e um cliente que vai embora (78% da
  receita desta máquina vem de clientes *recorrentes*, então uma experiência
  ruim é cara).
- **Excesso de estoque** → capital parado e desperdício de um insumo perecível.

O pipeline produz, a cada ciclo de reposição:

1. uma **previsão de demanda por produto** para os próximos 7 dias, e
2. uma **recomendação de "abastecer até" (order-up-to)** que atinge um nível de
   serviço-alvo, calibrada pelo erro real observado nos backtests.

Uma segunda saída, complementar, segmenta a **base de clientes** (RFM + K-means)
para que a área comercial veja quem gera receita e quem está se afastando.

Os dados cobrem uma máquina; o desenho generaliza para uma frota (ver
[docs/architecture.md](docs/architecture.md)).

---

## 2. Abordagem em um parágrafo

As transações são agregadas em um **painel diário por produto** (com zeros
preenchidos — um dia sem venda é demanda zero real). As features são termos de
calendário, feriados da Ucrânia, demanda defasada (7/14/28 dias), médias/desvios
móveis e a razão preço-vs-tendência. Todas as features são defasadas em pelo
menos o horizonte de previsão, então um único modelo **direto** atende todo o
próximo ciclo sem acúmulo recursivo de erro. Um **LightGBM** agrupado (pooled) é
ajustado nas janelas de origem móvel mais antigas e comparado com os baselines
seasonal-naive, média móvel, **Croston** e **TSB**. As duas janelas mais recentes
ficam como holdout final intocado. A promoção do modelo é decidida apenas nas
janelas de desenvolvimento; uma **política de estoque-base (base-stock)**
transparente então converte a previsão escolhida em quantidades de reposição.

As decisões de projeto e os trade-offs estão registrados como ADRs em
[docs/decisions/](docs/decisions/).

---

## 3. Estrutura do repositório

```
config/config.yaml            Todos os parâmetros do pipeline (fonte única de verdade)
src/coffee_intel/
  config.py                   Carregador de config tipado (pydantic)
  data/                       ingestão · validação (data quality) · limpeza
  features/                   painel e features de previsão · features RFM de cliente
  models/                     baselines · forecaster LightGBM · backtest · métricas · segmentação
  policy/replenishment.py     previsão -> recomendação de "abastecer até"
  pipelines/                  prepare · forecasting · segmentation (orquestração)
  reporting/                  figuras matplotlib · snapshot de evidências
  cli.py                      comando `coffee-intel`
tests/                        testes unitários pytest (dados sintéticos, dispensam o arquivo bruto)
docs/                         arquitetura + Architecture Decision Records + resultados
notebooks/01_eda.py           análise exploratória (script no formato `# %%`)
reports/                      figures/ e metrics/ (gerados) · evidence/ (snapshot versionado)
presentation/coffee-vending-demand.pptx  apresentação executiva do case
```

### Documentação

- [presentation/coffee-vending-demand.pptx](presentation/coffee-vending-demand.pptx) — apresentação executiva revisada
- [docs/results.md](docs/results.md) — resultados e números consolidados
- [docs/architecture.md](docs/architecture.md) — arquitetura de produção + diagrama
- [docs/decisions/](docs/decisions/) — Architecture Decision Records (o "porquê")

Gerar PDF de qualquer documento (requer `pip install -e ".[docs]"` e Chrome/Edge):

```bash
python scripts/build_report_pdf.py docs/results.md
```

---

## 4. Instalação

Requer Python 3.11+.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -e ".[dev]"
```

As versões das dependências vêm do `pyproject.toml` (Python 3.11+). O CI valida
3.11 e 3.12 de forma independente.

### Dados

O arquivo bruto **não** é versionado. Baixe do Kaggle e coloque aqui:

```
data/raw/index_1.csv        # obrigatório (tem o id anonimizado do cartão)
data/raw/index_2.csv        # opcional (segunda máquina, sem id de cartão)
```

```bash
# com o Kaggle CLI configurado (~/.kaggle/kaggle.json)
kaggle datasets download -d ihelon/coffee-sales -p data/raw --unzip
```

---

## 5. Execução

```bash
coffee-intel prepare      # bruto -> data/processed/transactions.parquet + relatório de data quality
coffee-intel forecast     # backtest, treino, previsão, plano de reposição
coffee-intel segment      # segmentação de clientes
coffee-intel run-all      # tudo acima

coffee-intel forecast --no-plots           # pular as figuras
coffee-intel forecast -c config/config.yaml
```

Saídas:

| Arquivo | O quê |
|---|---|
| `data/processed/forecast_next_cycle.csv` | previsão por produto e por dia para os próximos 7 dias |
| `data/processed/replenishment_recommendation.csv` | nível "abastecer até" e quantidade sugerida de pedido |
| `data/processed/customer_segments.csv` | uma linha por cliente com seu segmento |
| `reports/metrics/backtest_summary_cycle.csv` | WAPE / MAE / RMSE / viés por modelo (nível ciclo) |
| `reports/metrics/*.json` | resumos das execuções em formato legível por máquina |
| `reports/figures/*.png` | EDA e diagnósticos de modelo |

---

## 6. Desenvolvimento

```bash
make lint          # ruff + black --check
make format        # ruff --fix + black
make test          # pytest com cobertura
make all           # lint + test
pre-commit install # roda os hooks a cada commit
```

O CI (GitHub Actions) roda lint + testes em Python 3.11 e 3.12 a cada push e PR —
ver [.github/workflows/ci.yml](.github/workflows/ci.yml).

---

## 7. Resultados (esta máquina)

Relatório completo: [docs/results.md](docs/results.md). Artefatos brutos em
[reports/evidence/](reports/evidence/) e em `reports/metrics/` após uma execução.

Resumo após o ajuste temporal de hiperparâmetros: o LightGBM ajustado vence o
holdout intocado de duas janelas (WAPE de ciclo 0,168 vs. 0,202), mas nas seis
janelas de desenvolvimento seu WAPE é 0,348 contra 0,324 do seasonal-naive. O
pipeline, portanto, ainda entrega o modelo simples, pela regra explícita de
parcimônia. A visão de cliente mostra que **6% dos clientes identificados geram
41% da receita no cartão**.

## 8. Limitações

- Uma máquina, ~13 meses — pouco para ML; os resultados são direcionais.
- Reajustes de preço são sistêmicos e confundidos com o tempo, então efeitos de
  preço não são estimativas causais.
- Vendas em dinheiro (~2,5%) não têm id de cliente e ficam fora da segmentação.
- Sem ground truth real de estoque/ruptura — o ganho de nível de serviço é
  modelado, não medido.

Ver [docs/architecture.md](docs/architecture.md) §"Evolutions" para os próximos
passos.
