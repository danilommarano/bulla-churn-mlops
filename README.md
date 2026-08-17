# Churn MLOps — produtização do modelo de turnover (teste técnico Bulla)

Produtização de um modelo de **previsão de churn/turnover de clientes** (regressão logística)
como uma **pipeline de MLOps robusta que roda inteira no seu notebook**. O ponto de partida
eram dois scripts de ciência de dados (`legacy/train_model_churn.py` e
`legacy/infer_model_churn.py`) que funcionavam, mas eram frágeis: davam boas respostas por
motivos errados (vazamento de dados, pré-processo re-ajustado na inferência, métrica enganosa).
Eles foram **auditados** e transformados numa esteira de produção **versionada, testada,
orquestrada e monitorada**.

A vaga é toda em **Google Vertex AI** e o teste proíbe usar cloud. A estratégia: **para cada peça
do Vertex AI, um equivalente open-source rodando localmente** — provando domínio da plataforma
sem depender dela. O mapeamento está na seção [Arquitetura](#arquitetura).

> O score de retenção vai de **0 a 10**: quanto mais alto, menor a chance de turnover e melhor o
> cliente.

## O desafio

O enunciado pede para produtizar o modelo com boas práticas de Engenharia de Software e MLOps —
com liberdade para melhorar também o treino, auditando o trabalho existente. Duas restrições
importantes: **não** trocar a regressão por um algoritmo mais complexo (o foco é a pipeline, não
caçar acurácia), e a solução **roda 100% local, sem cloud**. Os entregáveis pedidos são um repo
Git e este README cobrindo: problemas conceituais do script original, desenho da arquitetura,
justificativa das etapas/melhorias e como rodar. O enunciado original está preservado em
[`legacy/README.md`](legacy/README.md).

## Arquitetura

O ciclo de vida completo, do CSV à predição e de volta ao re-treino:

```
                          ┌───────────────────────────────────────────┐
                          │        PIPELINE DE TREINO (KFP local)       │
   Customer-Churn.csv ──► │  prepare → split → train → evaluate →       │
        │                 │                            register (gate)  │
        │                 └───────┬──────────────┬───────────┬─────────┘
        │                         │              │           │
        │                         ▼              ▼           ▼
        │                    Feature Store    MLflow      MLflow
        │                      (Feast)        Tracking    Registry
        │                    offline/online   (métricas)  (@production)
        │                         │                          │
        ▼                         ▼                          ▼
   (referência p/          ┌──────────────────────────────────────┐
    monitoramento)         │   SERVING — API FastAPI (/predict)    │
        │                  │  carrega @production + busca features │
        │                  └───────────────┬──────────────────────┘
        │                                  ▼
        │                            predições + score 0–10
        ▼                                  ▼
   ┌─────────────────────────────────────────────────┐
   │  MONITORAMENTO — Evidently (drift & performance)  │
   │  referência (treino) vs. produção → quality gate  │
   └───────────────────────┬───────────────────────────┘
                           │ detectou drift?
                           └────────► dispara RE-TREINO (volta ao topo)
```

Tudo empacotável em **Docker**, operável por **Makefile**, testado por **pytest** (78 testes) e
automatizado por **CI/CD (GitHub Actions + Jenkinsfile documental)**.

### Mapeamento Vertex AI ↔ open-source

| Componente Vertex AI (vaga)   | Equivalente local (entrega)      | Papel                                   |
| ----------------------------- | -------------------------------- | --------------------------------------- |
| Vertex Pipelines              | **KFP SDK** (mesma engine)       | Orquestra as etapas do treino           |
| Vertex AI Experiments         | **MLflow Tracking**              | Rastreia params/métricas/artefatos      |
| Vertex Model Registry         | **MLflow Model Registry**        | Versiona e promove modelos (`@production`) |
| Vertex AI Feature Store       | **Feast**                        | Serve features offline/online sem skew  |
| Vertex Model Monitoring       | **Evidently AI**                 | Detecta data/target/score drift         |
| Vertex Endpoint               | **FastAPI + Docker**             | Serve predições REST                    |
| Cloud Build / Vertex CI       | **GitHub Actions + Jenkinsfile** | CI/CD                                    |

Cada marco documenta o mapeamento em detalhe (KFP↔Pipelines, Feast↔Feature Store, Evidently↔Model
Monitoring, GitHub Actions↔Cloud Build) nas specs em [`docs/superpowers/specs/`](docs/superpowers/specs/).

## Problemas conceituais no script original

A auditoria dos dois scripts legados encontrou 10 problemas. Cada um foi corrigido por uma escolha
de arquitetura concreta:

| #  | Problema no script original                                                                         | Correção na entrega                                            |
| -- | --------------------------------------------------------------------------------------------------- | ------------------------------------------------------------- |
| 1  | **Target leakage**: `geography_churn_rate` calculado com o `turnover` do dataset **inteiro** antes do split | Feature aprendida **só no treino**, materializada e servida via **Feast** |
| 2  | **Train/serve skew**: `StandardScaler`/`LabelEncoder`/`qcut` re-ajustados na inferência; só `model.pkl` salvo | Tudo dentro de um **`sklearn.Pipeline`** persistido; nada re-ajustado |
| 3  | **`CustomerId` como feature** — identificador sem poder preditivo                                    | Removido das features (nunca selecionado)                     |
| 4  | **`surname_encoded`**: `LabelEncoder` no sobrenome (altíssima cardinalidade), re-ajustado com ordem diferente | Removido das features                                         |
| 5  | **Métrica enganosa**: `accuracy` numa base ~20% de churn, sem AUC/precision/recall                  | **AUC + precision + recall + F1 + matriz de confusão**        |
| 6  | **`train_test_split` sem `random_state` nem `stratify`** — não reproduzível, não preserva as classes | `random_state=42`, `stratify=y`                               |
| 7  | **`churn_rate_por_uf` hardcoded** na inferência, com códigos que não batem com o encoder re-ajustado | Substituído pela feature servida do **Feast**                 |
| 8  | **`model.pkl` via pickle solto** — sem versão, metadados, assinatura ou estágio                     | **MLflow Model Registry** com assinatura e alias `@production` |
| 9  | **Score de retenção** calculado no treino e descartado; regra duplicada entre treino e inferência   | Função única e testada em `scoring.py`                        |
| 10 | **Sem `Pipeline`** encapsulando pré-processo + modelo; sem validação de schema; sem testes          | KFP + Pandera/Pydantic + pytest                               |

## Etapas e melhorias

**Fluxo 1 — Treino (DAG do KFP: `prepare_data_op → split_data_op → train_model_op → evaluate_model_op → register_model_op`).**
O CSV é validado (Pandera), **dividido antes de qualquer agregação** (matando o leakage do Bug #1),
e só então a taxa de churn por geografia é calculada usando **apenas o treino** e gravada no Feast.
O restante do pré-processo (one-hot, scaler) vive dentro de um `sklearn.Pipeline` que faz `fit` só
no treino, virando um **objeto único e persistível** (Bug #2). A `LogisticRegression`
(`class_weight="balanced"`) é avaliada com **métricas honestas** (Bug #5) logadas no MLflow, e só é
**promovida a `@production`** se passar no *quality gate* (`roc_auc >= min_roc_auc`) — um deploy
condicional como o do Vertex.

**Fluxo 2 — Serving (API FastAPI).** `POST /predict` recebe uma lista de clientes (validada por
Pydantic), busca no Feast **online** as features de agregação (o **mesmo** valor do treino, sem
recalcular — mata skew e leakage de vez), aplica o Pipeline `@production` carregado do MLflow, e
converte a probabilidade no **score 0–10** pela função única `scoring.py`. Retorna `turnover_pred`,
`prob_churn` e `score_retencao`.

**Fluxo 3 — Monitoramento (Evidently).** Os dados de treino viram **referência**; o Evidently
compara a produção contra ela, mede **data/score drift** e qualidade, e um *quality gate* em Python
**falha (exit code ≠ 0)** acima do limiar — o gatilho de re-treino que fecha o ciclo.

## Como rodar

Pré-requisito: [`uv`](https://docs.astral.sh/uv/) (gerencia Python e dependências).

```bash
make setup      # uv sync + cria .env a partir do .env.example
make pipeline   # roda o DAG de treino KFP ponta a ponta e registra o modelo @production
make serve      # sobe a API em http://localhost:8000 (docs em /docs)
```

Com a API no ar, um exemplo de predição:

```bash
curl -s http://localhost:8000/predict -H 'Content-Type: application/json' -d '[{
  "CreditScore": 619, "Age": 42, "Balance": 0.0, "EstimatedSalary": 101348.88,
  "Tenure": 2, "NumOfProducts": 1, "HasCrCard": 1,
  "Satisfaction Score": 2, "Point Earned": 464,
  "Geography": "France", "Gender": "Female", "Card Type": "DIAMOND"
}]'
# -> [{"turnover_pred":1,"prob_churn":0.71,"score_retencao":3}]
```

Demais alvos (`make help` lista todos):

| Target              | O que faz                                                                       |
| ------------------- | ------------------------------------------------------------------------------- |
| `make train`        | Treina e registra no MLflow direto (sem o DAG), backend SQLite local            |
| `make feast-materialize` | Popula e materializa a feature store (offline → online)                    |
| `make monitor`      | Relatório de drift/qualidade num holdout saudável (o gate passa)                |
| `make monitor-drift`| Idem com drift simulado — demonstra a detecção (o gate falha de propósito)      |
| `make test`         | Roda a suíte (78 testes)                                                         |
| `make ci`           | Espelha o GitHub Actions localmente: lint + format-check + test + build + pipeline |
| `make docker-build` / `make docker-run` | Constrói e roda a API em container (monta o registry MLflow local) |

## Limitações honestas

Este é um espelho **local** do Vertex AI, então alguns pontos ficam documentados em vez de
implementados:

- **Monitoramento**: a API não captura predições automaticamente (o holdout `X_test` serve de proxy
  de "produção"); não há scheduler gerenciado (`make monitor` roda sob demanda / cron manual); o
  Evidently usa Jensen-Shannon também para drift categórico (o Vertex usa L-infinity, que o Evidently
  não expõe nativamente).
- **CI/CD**: o `publish` empurra a imagem para o **GHCR**, mas o deploy a um endpoint rodando é
  manual (`docker run`); o **Jenkinsfile** é declarativo e documental (não há servidor Jenkins ativo).
- **Kubernetes**: a vaga cita K8s; aqui a API roda como container único (Docker) — o passo para um
  cluster fica descrito como próximo passo, não construído (o teste é local).

## TensorFlow vs scikit-learn

O modelo é uma regressão logística do scikit-learn, uma escolha deliberada para este churn tabular.
O documento [`docs/tensorflow_variant.md`](docs/tensorflow_variant.md) discute honestamente o que o
TensorFlow traria, por que a regressão logística **é** uma rede de uma camada densa sigmoide, e
quando a balança viraria para o TF.

## Estrutura do repositório

```
src/churn/
  data.py schema.py         # carga e validação (Pandera)
  features/                 # engenharia de features sem leakage (transformer sklearn)
  feature_store/            # Feast: definições, materialização e serving
  training/                 # pipeline, treino, avaliação, registry (MLflow)
  orchestration/            # DAG do KFP + steps
  serving/                  # API FastAPI + contratos Pydantic
  monitoring/               # Evidently: datasets, relatório, quality gate
  scoring.py                # probabilidade -> score 0–10 (fonte única)
tests/                      # 78 testes (pytest)
docs/                       # specs de design de cada marco + doc do TensorFlow
legacy/                     # scripts originais + enunciado do teste
```
