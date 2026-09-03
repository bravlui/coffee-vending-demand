# Guia de leitura do código

Este documento liga cada decisão de projeto ao arquivo que a implementa.

## Fluxo completo

```text
CSV bruto
  -> ingestão e nomes canônicos
  -> validação e limpeza
  -> painel diário por produto
  -> 19 variáveis explicativas sem informação futura
  -> backtest de origem móvel
  -> seleção entre baselines, LightGBM e ensemble
  -> previsão dos próximos 7 dias
  -> estoque de segurança e nível de reposição
```

## 1. Entrada e qualidade

- `data/ingest.py`: lê o CSV, converte data e preço e padroniza os nomes
  `datetime`, `money`, `coffee_name`, `cash_type` e `card`.
- `data/validate.py`: executa as nove regras de qualidade e registra por que cada
  uma passou ou falhou.
- `data/clean.py`: remove preços fora da faixa configurada, normaliza categorias
  e elimina duplicatas exatas.
- `pipelines/prepare.py`: chama essas três etapas na ordem correta e grava a
  tabela limpa e o relatório de qualidade.

## 2. Variável resposta e variáveis explicativas

Em `features/forecasting.py`, cada transação representa uma unidade vendida.
O agrupamento por `date` e `product` cria `units`, a variável resposta diária.
Dias sem transação são incluídos com `units = 0` para formar um painel contínuo.

As 19 variáveis explicativas são:

- 7 de calendário: dia da semana, fim de semana, dia do mês, semana do ano,
  mês, tempo transcorrido e feriado;
- 9 do histórico de vendas: defasagens de 7/14/28 dias, mesmo dia da semana há
  uma e duas semanas, médias e desvios móveis de 7/28 dias;
- 2 de preço: preço médio e preço relativo à média móvel;
- 1 de produto: código categórico.

Todas as variáveis históricas são deslocadas por pelo menos sete dias. Assim,
uma linha usada para prever o próximo ciclo contém apenas informação que já
existia na data da decisão.

## 3. Modelos e métrica

- `models/baselines.py`: implementa referências simples e interpretáveis.
- `models/forecaster.py`: treina um LightGBM único para todos os produtos.
- `models/metrics.py`: define WAPE, MAE, RMSE e viés. O WAPE decide a promoção
  porque mede o erro absoluto em relação ao volume total do ciclo.
- `models/tuning.py`: testa hiperparâmetros somente nas janelas de
  desenvolvimento.

O modelo não é escolhido por complexidade. Ele precisa reduzir o WAPE além do
limite configurado, vencer em quantidade suficiente de janelas e não piorar o
viés além da tolerância.

## 4. Backtest temporal

`models/backtest.py` cria oito origens móveis. Em cada origem, o treino contém
somente datas anteriores e o teste contém os sete dias seguintes. As seis
primeiras janelas servem para ajuste e seleção; as duas últimas são holdout e
servem apenas para reporte.

Esse desenho simula como a previsão seria usada semanalmente e evita a mistura
aleatória de passado e futuro.

## 5. Reposição

`policy/replenishment.py` transforma a previsão em decisão operacional:

```text
nível-alvo = demanda prevista no período de proteção + estoque de segurança
pedido     = máximo(0, nível-alvo - estoque disponível)
```

O estoque de segurança é calibrado com erros anteriores do backtest. Sem
telemetria de estoque, o projeto usa `on_hand = 0` como hipótese explícita; por
isso o resultado atual é um nível recomendado, não impacto operacional medido.

## 6. Orquestração e saídas

`pipelines/forecasting.py` conecta features, tuning, backtest, seleção, treino
final, previsão e política de reposição. `cli.py` expõe esse fluxo pelos comandos
`prepare`, `forecast` e `run-all`. `reporting/` grava gráficos e evidências que
permitem reproduzir os números dos relatórios em `docs/`.

## Lendo uma função

Para cada função, quatro perguntas orientam a leitura:

1. Qual dado ela recebe?
2. Qual transformação executa?
3. Qual hipótese ou proteção de qualidade aplica?
4. Qual objeto devolve para a próxima etapa?

Esse padrão cobre o código sem adicionar comentários que apenas repetem a
sintaxe do Python.
