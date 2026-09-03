# coffee-vending-demand

Previsão de demanda e política de reposição para operações de máquinas de café.

Projeto de ciência de dados ponta a ponta e reprodutível sobre o dataset público
**[Coffee Sales](https://www.kaggle.com/datasets/ihelon/coffee-sales)** — o log de
transações de uma única máquina de café (mar/2024 – mar/2025).

---

## 1. Problema

Um operador de máquinas de café decide **quanto de cada produto abastecer e
quando**. Hoje a decisão é manual e reativa:

- **Ruptura de estoque** → item indisponível e possível venda perdida.
- **Excesso de estoque** → capital parado e desperdício de um insumo perecível.

O pipeline produz, a cada ciclo de reposição:

1. uma **previsão de demanda por produto** para os próximos 7 dias, e
2. uma **recomendação de "abastecer até" (order-up-to)** que atinge um nível de
   serviço-alvo, calibrada pelo erro real observado nos backtests.

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

| Caminho | Onde procurar / responsabilidade |
|---|---|
| [`config/config.yaml`](config/config.yaml) | horizonte, janelas, features, LightGBM, regra de seleção e nível de serviço |
| [`src/coffee_intel/cli.py`](src/coffee_intel/cli.py) | comandos `prepare`, `forecast` e `run-all` |
| [`src/coffee_intel/data/ingest.py`](src/coffee_intel/data/ingest.py) | leitura do CSV e padronização dos nomes das colunas |
| [`src/coffee_intel/data/validate.py`](src/coffee_intel/data/validate.py) | nove verificações de qualidade |
| [`src/coffee_intel/data/clean.py`](src/coffee_intel/data/clean.py) | filtros, normalizações e remoção de duplicatas |
| [`src/coffee_intel/features/forecasting.py`](src/coffee_intel/features/forecasting.py) | painel diário, variável resposta `units` e 19 variáveis explicativas |
| [`src/coffee_intel/models/baselines.py`](src/coffee_intel/models/baselines.py) | seasonal-naive, média móvel, Croston e TSB |
| [`src/coffee_intel/models/forecaster.py`](src/coffee_intel/models/forecaster.py) | treinamento e predição do LightGBM |
| [`src/coffee_intel/models/tuning.py`](src/coffee_intel/models/tuning.py) | ajuste de hiperparâmetros apenas nas janelas de desenvolvimento |
| [`src/coffee_intel/models/backtest.py`](src/coffee_intel/models/backtest.py) | origem móvel, cálculo por janela e regra de promoção |
| [`src/coffee_intel/models/metrics.py`](src/coffee_intel/models/metrics.py) | WAPE, MAE, RMSE e viés |
| [`src/coffee_intel/policy/replenishment.py`](src/coffee_intel/policy/replenishment.py) | previsão → estoque de segurança → nível “abastecer até” |
| [`src/coffee_intel/pipelines/prepare.py`](src/coffee_intel/pipelines/prepare.py) | orquestra ingestão, validação e limpeza |
| [`src/coffee_intel/pipelines/forecasting.py`](src/coffee_intel/pipelines/forecasting.py) | orquestra o fluxo completo de previsão e reposição |
| [`src/coffee_intel/reporting/`](src/coffee_intel/reporting/) | gráficos e snapshot versionável das evidências |
| [`tests/`](tests/) | testes organizados pelas mesmas responsabilidades do código |
| [`docs/code-guide.md`](docs/code-guide.md) | leitura guiada do código, etapa por etapa |

### Ordem recomendada para ler o código

`cli.py` → `pipelines/prepare.py` → `features/forecasting.py` →
`models/backtest.py` → `models/forecaster.py` →
`policy/replenishment.py` → `pipelines/forecasting.py`.

Os comentários explicam decisões que não são óbvias — especialmente ordem temporal,
prevenção de vazamento e regra de seleção. O guia de código explica cada etapa sem
repetir em comentários aquilo que a própria instrução Python já expressa.

### Documentação

- [docs/results.md](docs/results.md) — resultados e números consolidados
- [docs/architecture.md](docs/architecture.md) — arquitetura de produção + diagrama
- [docs/code-guide.md](docs/code-guide.md) — mapa do fluxo e explicação dos módulos
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
coffee-intel run-all      # prepare + forecast + snapshot de evidências

coffee-intel forecast --no-plots           # pular as figuras
coffee-intel forecast -c config/config.yaml
```

Saídas:

| Arquivo | O quê |
|---|---|
| `data/processed/forecast_next_cycle.csv` | previsão por produto e por dia para os próximos 7 dias |
| `data/processed/replenishment_recommendation.csv` | nível "abastecer até" e quantidade sugerida de pedido |
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

Relatório completo: [docs/results.md](docs/results.md). Os artefatos brutos
(métricas e snapshot de evidências) são gerados em `reports/` após uma execução.

Resumo após o ajuste temporal de hiperparâmetros: o LightGBM ajustado vence o
holdout intocado de duas janelas (WAPE de ciclo 0,168 vs. 0,202), mas nas seis
janelas de desenvolvimento seu WAPE é 0,348 contra 0,324 do seasonal-naive. O
pipeline, portanto, ainda entrega o modelo simples, pela regra explícita de
parcimônia.

## 8. Limitações

- Uma máquina, ~13 meses — pouco para ML; os resultados são direcionais.
- Reajustes de preço são sistêmicos e confundidos com o tempo, então efeitos de
  preço não são estimativas causais.
- Sem ground truth real de estoque/ruptura — o ganho de nível de serviço é
  modelado, não medido.

Ver [docs/architecture.md](docs/architecture.md) §"Evolutions" para os próximos
passos.
