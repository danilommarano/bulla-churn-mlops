# Design — Orquestração da pipeline com KFP (Marco 5)

- **Data:** 2026-08-17
- **Autor:** Danilo Marano (com Claude Code)
- **Status:** Aprovado — pronto para plano de implementação
- **Contexto:** Teste técnico para vaga de Machine Learning Engineer (MLOps) na Bulla.
- **Issue:** #11 · **Branch:** `feat/pipeline-orchestration`
- **Design geral:** [`2026-08-15-bulla-churn-mlops-design.md`](./2026-08-15-bulla-churn-mlops-design.md)

---

## 1. Objetivo

Transformar o treino monolítico (`churn.training.train`) numa **pipeline orquestrada como DAG
reproduzível** com KFP (Kubeflow Pipelines), espelhando localmente o **Vertex AI Pipelines** da vaga.
A entrega prova o argumento central de orquestração de MLOps: o ciclo de treino vira um grafo de
**componentes isolados** que passam **artefatos tipados** (Dataset, Model, Metrics) entre si, com
**linhagem** rastreável, executável ponta a ponta **sem cluster** via `kfp local`.

O escopo é **aditivo e de baixo risco**: a lógica de ML (feature engineering, `build_pipeline`,
`evaluate`, tracking/registry MLflow) **não muda de comportamento** — é reusada e apenas reorganizada
atrás dos componentes. O `train.py` existente continua funcionando (os 30 testes do Marco 2 seguem
verdes após um refactor DRY comportamentalmente idêntico).

### 1.1 Abordagem escolhida

Das opções levantadas no brainstorming:

- **Runner:** **`SubprocessRunner`** (não `DockerRunner`) + **documentar** o mapeamento Docker/Vertex.
  `local.init(runner=local.SubprocessRunner(use_venv=False))` roda cada componente como um subprocesso
  no venv atual, então os componentes conseguem `import churn` sem empacotar imagem. É o runner mais
  simples que ainda exercita a semântica de componentes/artefatos do KFP; a produção (Vertex) usaria
  containers, o que é documentado na tabela de mapeamento (§2) e nos riscos (§8).
- **Granularidade:** **5 estágios com artifact lineage** —
  `prepare_data → split_data → train_model → evaluate_model → register_model`, passando artefatos
  Dataset/Model/Metrics entre eles. É a granularidade que melhor demonstra um DAG de MLOps (cada passo
  do ciclo é um nó com entradas/saídas versionadas), sem cair no exagero de um nó por transformação.

### 1.2 Fora de escopo (YAGNI)

- Não trocar o algoritmo nem tunar hiperparâmetro (mantém `LogisticRegression` do Marco 2).
- Sem cluster Kubernetes, sem Vertex real, sem `DockerRunner`, sem submeter a pipeline compilada a um
  backend KFP. Tudo local via `SubprocessRunner`.
- Sem materialização Feast como estágio do DAG — o Feast (Marco 4) continua no `make feast-materialize`
  à parte. Acoplá-lo ao DAG de treino não agrega à narrativa de orquestração e aumenta o risco.
- Sem `dsl.If`/`dsl.Condition` para o gate de qualidade (pode não rodar de forma confiável no
  `kfp local`); o gate é feito em código Python dentro do componente `register_model` (§3.4).
- Sem reescrever `/predict` nem o serving (Marco 3) — a pipeline produz o mesmo modelo registrado que
  a API já consome via `models:/churn-model@production`.

---

## 2. Mapeamento Vertex AI ↔ KFP local

| Conceito Vertex AI Pipelines | Equivalente KFP local (este projeto) |
|---|---|
| Pipeline (grafo de componentes) | `@dsl.pipeline def churn_training_pipeline(...)` em `orchestration/dag.py` |
| Componente containerizado | `@dsl.component` (função Python) executado pelo `SubprocessRunner` |
| Executor de componente (container na cloud) | Subprocesso no venv atual (`SubprocessRunner(use_venv=False)`) |
| Parâmetros de pipeline | Parâmetros primitivos do `@dsl.pipeline` (data_path, test_size, seed, thresholds...) |
| Artefatos (ML Metadata / GCS) | `dsl.Dataset` / `dsl.Model` / `dsl.Metrics` passados por `Input[...]`/`Output[...]` |
| Linhagem (ML Metadata lineage) | Grafo de `.outputs[...]` ligando os componentes + `dsl.Metrics` logadas |
| Conditional deploy (gate de promoção) | Checagem `roc_auc >= min_roc_auc` em código no `register_model` |
| Vertex Pipelines run | `local.init(...)` + chamada da pipeline (roda o DAG em processo) |
| Model Registry (Vertex) | MLflow Model Registry (reusa Marco 2): `set_registered_model_alias` |

Esse mapeamento vai também para o README final (Marco 9).

---

## 3. Arquitetura

**Princípio-chave — separar lógica de orquestração da lógica de ML.** Cada estágio tem duas camadas:

1. **Função pura** em `orchestration/steps/<stage>.py` — opera sobre **caminhos de arquivo** e valores
   primitivos (sem tipos KFP), reusa o código de ML existente, é rápida e testável em isolamento.
2. **Wrapper `@dsl.component`** em `orchestration/dag.py` — fino; só traduz artefatos KFP
   (`Input/Output[...]`) em caminhos e delega para a função pura.

Isso dá o **fan-out/fan-in** da Fase D: 5 pares (função pura, teste unitário) independentes + 1 fan-in
(o `dag.py` + o teste e2e). Também mantém o KFP fora dos testes unitários dos passos (só o teste e2e
exercita o runtime do KFP).

### 3.1 Novo pacote `src/churn/orchestration/`

```
src/churn/orchestration/
  __init__.py
  steps/
    __init__.py
    prepare_data.py     # prepare_data(out_path, data_path) -> escreve parquet validado
    split_data.py       # split_data(in_path, train_out, test_out, test_size, random_state)
    train_model.py      # train_model(train_path, model_out, random_state, n_age_bins)
    evaluate_model.py   # evaluate_model(model_path, test_path, metrics_out) -> dict
    register_model.py   # register_model(model_path, metrics_path, train_path, cfg) -> dict (gate interno)
  dag.py                # @dsl.component wrappers + @dsl.pipeline + run_local()/main()
```

O nome `orchestration` evita colisão com `churn.training.pipeline` (que é o estimador sklearn).

### 3.2 Funções puras (`steps/`) e artefatos

Cada função pura lê/escreve arquivos nos caminhos que o KFP fornece para os artefatos.

- **`prepare_data(out_path: str, data_path: str) -> None`**
  Reusa `load_raw(data_path)` (que já aplica o contrato Pandera). Seleciona
  `INPUT_COLUMNS + ["turnover"]` e escreve um parquet em `out_path`.
  → Produz o artefato **`Dataset`** "prepared".

- **`split_data(in_path, train_out, test_out, test_size, random_state) -> None`**
  Lê o parquet preparado, refaz **exatamente** o split do `train.py`
  (`train_test_split(..., test_size, random_state, stratify=y)` sobre `y = df["turnover"]`) e escreve
  dois parquets (`train_out`, `test_out`), cada um com `INPUT_COLUMNS + ["turnover"]`.
  → Produz dois artefatos **`Dataset`** ("train", "test").

- **`train_model(train_path, model_out, random_state, n_age_bins) -> None`**
  Lê o parquet de train, separa `X/y`, chama `build_pipeline(random_state, n_age_bins).fit(X, y)` e
  serializa o `Pipeline` sklearn fitado com `joblib.dump` em `model_out`. O `ChurnFeatureBuilder`
  custom é importável (pickla sem problema, como já acontece no MLflow do Marco 2).
  → Produz o artefato **`Model`**.

- **`evaluate_model(model_path, test_path, metrics_out) -> dict`**
  Carrega o modelo (`joblib.load`), lê o parquet de test, chama `evaluate(model, X, y)` (reuso direto
  do Marco 2) e grava o dict como JSON em `metrics_out`. Retorna o dict (para o wrapper logar
  `dsl.Metrics`).
  → Produz o artefato **`Metrics`** (JSON) + métricas escalares logadas como `dsl.Metrics`.

- **`register_model(model_path, metrics_path, train_path, cfg) -> dict`**
  Carrega o modelo, as métricas e um pequeno sample do parquet de train (para a signature/
  input_example), computa o gate internamente
  (`promote = metrics["roc_auc"] >= cfg.min_roc_auc`) e chama o helper DRY
  `log_and_register(..., promote=promote)` (§3.3) para logar no MLflow e registrar a versão.
  **Gate de qualidade:** promove ao alias `@production` **apenas se** o gate passar; abaixo do limiar
  registra a versão mas **não** promove. Retorna `{run_id, version, promoted, roc_auc}`.

### 3.3 Refactor DRY — `src/churn/training/registry.py`

Para não duplicar o bloco MLflow entre `train.py` e o `register_model`, extrair um helper:

```python
def log_and_register(
    pipeline, X_sample, metrics: dict, cfg: Settings, promote: bool = True
) -> dict:
    """Log the fitted pipeline + metrics/params to MLflow, register a version,
    and (optionally) move the production alias. Returns run_id, version, promoted."""
```

Responsabilidades (extraídas verbatim do atual `train.py`, comportamento idêntico):
`set_tracking_uri`/`set_experiment` → `start_run` → `log_params` → `log_metrics` (exceto
`confusion_matrix`) → `infer_signature(X_sample, pipeline.predict(X_sample))` →
`log_model(name="model", signature, input_example=X_sample.head(3), registered_model_name,
serialization_format=CLOUDPICKLE)` → se `promote`: `set_registered_model_alias(...)`.

- **`train.py`** passa a chamar `log_and_register(pipeline, X_train, metrics, cfg, promote=True)`,
  mantendo o retorno `{run_id, version, metrics}` (comportamento inalterado; 30 testes do Marco 2 verdes).
- **`register_model`** chama `log_and_register(pipeline, X_train_sample, metrics, cfg,
  promote=(roc_auc >= cfg.min_roc_auc))`.

Nota sobre `X_sample`: `log_and_register` precisa de um pequeno DataFrame de exemplo para a signature
e o `input_example`. O `register_model` deriva-o do próprio parquet de train (algumas linhas),
mantendo a signature idêntica à do `train.py`.

### 3.4 Config (`src/churn/config.py`)

Adicionar ao `Settings` (prefixo `CHURN_`):

- `min_roc_auc: float = 0.70` — limiar do gate de promoção. O modelo atual (roc_auc ≈ 0.764) passa;
  o valor é conservador o suficiente para não ser um gate decorativo.

Os componentes leem os demais parâmetros já existentes (`data_path`, `test_size`, `random_state`,
`n_age_bins`, `mlflow_*`, `model_*`). A pipeline os expõe como parâmetros (§3.5).

### 3.5 `dag.py` — componentes, pipeline e entrypoint

- **Wrappers `@dsl.component`** (um por estágio): assinaturas com `Input/Output[Dataset|Model|Metrics]`
  e parâmetros primitivos; cada um só resolve `artifact.path` e chama a função pura correspondente. O
  `evaluate_model` também loga `metrics_out.log_metric(k, v)` para os escalares (linhagem KFP).
- **`@dsl.pipeline def churn_training_pipeline(...)`** — recebe os parâmetros como primitivos
  (`data_path`, `test_size`, `random_state`, `n_age_bins`, `mlflow_tracking_uri`, `mlflow_experiment`,
  `model_name`, `model_alias`, `min_roc_auc`) e conecta os componentes via `.outputs[...]`, formando
  a linhagem:
  ```
  prep = prepare_data(data_path=data_path)
  split = split_data(dataset=prep.outputs["output"], test_size=..., random_state=...)
  trained = train_model(train_set=split.outputs["train"], random_state=..., n_age_bins=...)
  ev = evaluate_model(model=trained.outputs["model"], test_set=split.outputs["test"])
  register_model(model=trained.outputs["model"], metrics=ev.outputs["metrics"],
                 train_set=split.outputs["train"],  # sample para signature/input_example
                 mlflow_tracking_uri=..., ..., min_roc_auc=min_roc_auc)
  ```
- **`run_local(cfg=settings)`** — chama `local.init(runner=local.SubprocessRunner(use_venv=False))` e
  executa `churn_training_pipeline(**params_de(cfg))`. Passar os parâmetros a partir do `settings`
  torna trivial o override no teste e2e (sem depender de env nos subprocessos).
- **`main()`** — entrypoint de `python -m churn.orchestration.dag` (chamado pelo `make pipeline`).

### 3.6 Makefile

Novo target:

```makefile
pipeline: ## Roda o DAG de treino KFP local ponta a ponta (prepare -> ... -> register)
	uv run python -m churn.orchestration.dag
```

### 3.7 Artefatos gerados e `.gitignore`

Já feito (commit `593b4f7`): `local_outputs/` (diretório de runs do `kfp local`) está no `.gitignore`.
Os artefatos MLflow (`mlflow.db`, `mlruns/`) já eram ignorados. Nenhuma ação nova.

---

## 4. Fluxo de dados

```
Customer-Churn-Records.csv
        │  prepare_data (load_raw + contrato Pandera; seleciona INPUT_COLUMNS + turnover)
        ▼
   Dataset "prepared" (parquet)
        │  split_data (mesmo seed/test_size/stratify do train.py)
        ├──────────────► Dataset "train" (parquet)
        └──────────────► Dataset "test"  (parquet)
                              │
   Dataset "train" ──► train_model (build_pipeline().fit) ──► Model (joblib)
                                                                 │
   Model + Dataset "test" ──► evaluate_model (evaluate) ──► Metrics (JSON + dsl.Metrics)
                                                                 │
   Model + Metrics ──► register_model (log_and_register; gate roc_auc >= min_roc_auc)
                              │
                              ▼
        MLflow: run logado + versão registrada; alias @production movido SÓ se passar o gate
```

**Linhagem (o argumento do Marco):** cada seta é um artefato KFP tipado ligando `.outputs[...]` de um
componente à entrada do próximo — exatamente o grafo de ML Metadata que o Vertex Pipelines materializa.

**Reprodutibilidade / consistência com o treino atual:** `prepare_data`+`split_data` reproduzem o
mesmo `load_raw` + `train_test_split(stratify, seed)` do `train.py`; `train_model` usa o mesmo
`build_pipeline`; `register_model` usa o mesmo helper de logging. Logo, o modelo produzido pelo DAG é
equivalente ao do `make train`.

---

## 5. Tratamento de erros

- **`pip` ausente no venv:** o `SubprocessRunner` faz shell-out para `pip`; o `uv venv` não traz `pip`.
  Já mitigado no spike (Fase A): `pip` está nas dependências do `pyproject.toml`, então `uv sync` +
  `make pipeline` funciona do zero. Sem ação nova.
- **Caminhos relativos e CWD:** `make pipeline` roda da raiz do repo; `data_path` e
  `mlflow_tracking_uri` relativos resolvem para a raiz (o spike confirmou que o `SubprocessRunner`
  preserva o CWD do processo pai). Mesma classe do gotcha Docker+MLflow do Marco 3.
- **Gate de qualidade reprovado** (`roc_auc < min_roc_auc`): **não é erro** — a versão é registrada,
  mas o alias `@production` **não** é movido (`promoted=False`). Comportamento explícito e testado.
- **Pickle do modelo:** o `Pipeline` contém `ChurnFeatureBuilder` custom; como no Marco 2, ele é
  importável a partir de `churn.features.builder`, então `joblib.dump/load` funciona nos subprocessos.

---

## 6. Estratégia de testes (`tests/test_orchestration.py`)

Duas camadas, espelhando o fan-out/fan-in:

**Unitários por passo (rápidos, sem KFP)** — usam `tmp_path` e o CSV real; um por função pura:

1. **`test_prepare_data`** — escreve o parquet; ao reler, tem as colunas `INPUT_COLUMNS + ["turnover"]`
   e ~10000 linhas.
2. **`test_split_data`** — a partir do parquet preparado, gera train/test com proporção ≈ `test_size`,
   sem sobreposição de índices, e ambos com as colunas esperadas.
3. **`test_train_model`** — produz um arquivo de modelo que, ao `joblib.load`, é um `Pipeline` sklearn
   com `predict`/`predict_proba` sobre `INPUT_COLUMNS`.
4. **`test_evaluate_model`** — dado um modelo treinado e um parquet de test, escreve um JSON de métricas
   com as chaves esperadas e `roc_auc` em `(0, 1)`.
5. **`test_register_model_gate`** — dois casos: (a) `min_roc_auc` baixo → `promoted=True` e o alias
   `@production` aponta para a versão; (b) `min_roc_auc` alto (ex.: 0.99) → `promoted=False` e o alias
   não é movido. Roda contra um `mlflow_tracking_uri` temporário via `Settings`.

**E2E do DAG (fan-in, exercita o `kfp local`)** — um teste:

6. **`test_pipeline_end_to_end`** — roda `run_local(cfg)` com um `Settings` de teste apontando
   `mlflow_tracking_uri` para um SQLite em `tmp_path` e `data_path` para o CSV real; afirma que uma
   versão foi registrada e promovida a `@production` com `roc_auc` plausível (≳ 0.7). Marcado como
   lento (pode usar `@pytest.mark.slow` se útil), mas roda no `make test`.

**Regressão:** os 30 testes do treino/registro (Marco 2) continuam verdes após o refactor DRY —
`log_and_register` é comportamentalmente idêntico ao bloco extraído.

Critério de pronto: os 53 testes existentes continuam verdes + os novos passam (`make test`) e
`make lint` limpo.

---

## 7. Estrutura de arquivos

| Arquivo | Ação | Responsabilidade |
|---|---|---|
| `src/churn/orchestration/__init__.py` | criar | Marca o pacote |
| `src/churn/orchestration/steps/__init__.py` | criar | Marca o subpacote |
| `src/churn/orchestration/steps/prepare_data.py` | criar | `prepare_data` (load_raw → parquet) |
| `src/churn/orchestration/steps/split_data.py` | criar | `split_data` (mesmo split do train.py) |
| `src/churn/orchestration/steps/train_model.py` | criar | `train_model` (build_pipeline().fit → joblib) |
| `src/churn/orchestration/steps/evaluate_model.py` | criar | `evaluate_model` (evaluate → JSON) |
| `src/churn/orchestration/steps/register_model.py` | criar | `register_model` (log_and_register + gate) |
| `src/churn/orchestration/dag.py` | criar | `@dsl.component` wrappers + `@dsl.pipeline` + `run_local`/`main` |
| `src/churn/training/registry.py` | criar | Helper DRY `log_and_register` |
| `src/churn/training/train.py` | modificar | Passa a chamar `log_and_register` (comportamento idêntico) |
| `src/churn/config.py` | modificar | `min_roc_auc: float = 0.70` |
| `tests/test_orchestration.py` | criar | Testes 1–6 acima |
| `Makefile` | modificar | Target `pipeline` |

Reuso (sem alteração de comportamento): `churn.data.load_raw`, `churn.features.builder`
(`INPUT_COLUMNS`, `ChurnFeatureBuilder`), `churn.training.pipeline.build_pipeline`,
`churn.training.evaluate.evaluate`.

---

## 8. Riscos e mitigações

- **Compat de dependências KFP** — já resolvido no spike (Fase A): `kfp>=2.17.0` convive com
  mlflow/feast/pandas 2.x/numpy 2.x **sem** rebaixar protobuf. `pip` adicionado às deps. Sem ação nova.
- **`SubprocessRunner` ≠ produção (Docker/Vertex)** — o runner local roda no venv atual, não em
  container. Trade-off consciente (simplicidade > fidelidade total): a semântica de
  componentes/artefatos/linhagem é a mesma; a diferença (imagem por componente) é **documentada** na
  tabela de mapeamento (§2) e destacada como o passo que a produção acrescentaria. Evita empacotar/
  publicar imagens só para um teste técnico local.
- **CWD dos subprocessos** — se um subprocesso do `SubprocessRunner` não herdasse o CWD, caminhos
  relativos (CSV, `sqlite:///mlflow.db`) quebrariam. O spike confirmou herança do CWD; ainda assim, o
  teste e2e usa caminhos sob `tmp_path` para não depender disso e para não sujar o `mlflow.db` real.
- **Divergência com o `train.py`** — se o split/pipeline do DAG divergir do `train.py`, o modelo do
  DAG deixa de ser equivalente ao do `make train`. Mitigação: `split_data`/`train_model` reusam
  `train_test_split`(mesmos args) e `build_pipeline`; `register_model` reusa `log_and_register`.
- **Regressão no refactor DRY** — extrair o bloco MLflow do `train.py` pode alterar comportamento.
  Mitigação: extração verbatim + os 30 testes do Marco 2 como rede de segurança (rodar antes/depois).
