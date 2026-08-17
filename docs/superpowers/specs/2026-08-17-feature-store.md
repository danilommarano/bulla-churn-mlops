# Design — Feature Store local com Feast (Marco 4)

- **Data:** 2026-08-17
- **Autor:** Danilo Marano (com Claude Code)
- **Status:** Aprovado — pronto para plano de implementação
- **Contexto:** Teste técnico para vaga de Machine Learning Engineer (MLOps) na Bulla.
- **Issue:** #9 · **Branch:** `feat/feature-store`
- **Design geral:** [`2026-08-15-bulla-churn-mlops-design.md`](./2026-08-15-bulla-churn-mlops-design.md)

---

## 1. Objetivo

Adicionar uma **feature store local** (Feast) que serve a feature agregada `geography_churn_rate`
por uma camada offline/online, espelhando o **Vertex AI Feature Store** da vaga. A entrega prova o
argumento central de uma feature store: **consistência offline/online** — o valor servido em produção
(online) é idêntico ao valor que o modelo aprendeu no treino (offline), eliminando conceitualmente o
*training-serving skew* para essa feature.

O escopo é **aditivo e de baixo risco**: o treino e o modelo persistido (Marcos 2 e 3) **não mudam**.
O `GeographyChurnRateEncoder` (leakage-safe, fitado só no split de train) continua sendo a fonte da
verdade; o Feast registra, materializa e serve o mesmo mapa aprendido por ele.

### 1.1 Abordagem escolhida

Das três opções levantadas no brainstorming (A: reescrever treino+serving numa única fonte Feast;
B: Feast como fonte de serving + prova de consistência; C: Feast decorativo/desacoplado), foi escolhida
a **B**. Ela entrega a narrativa anti-skew completa com risco de regressão mínimo: reusa o encoder já
existente e testado, sem reescrever a pipeline verde de 42 testes.

### 1.2 Fora de escopo (YAGNI)

- Não trocar o algoritmo nem tunar hiperparâmetro.
- Não mover a derivação de `geography_churn_rate` do `ChurnFeatureBuilder` para o Feast em tempo de
  treino (isso seria a abordagem A). O builder segue derivando a feature no treino; o Feast serve a
  mesma feature na camada online.
- Sem Feast Registry remoto, sem provider cloud, sem streaming/push source. Tudo local (SQLite + parquet).
- Sem reescrever `/predict` para consumir o Feast (o `/predict` continua derivando features do payload
  via a pipeline persistida). O consumo online é demonstrado por um endpoint dedicado e pelos testes.

---

## 2. Mapeamento Vertex AI ↔ Feast

| Conceito Vertex AI Feature Store | Equivalente Feast (local) |
|---|---|
| Featurestore / Entity Type | `Entity(name="geography", join_keys=["Geography"])` |
| Feature | `Field(name="geography_churn_rate", dtype=Float32)` numa `FeatureView` |
| Offline store (BigQuery) | `FileSource` apontando para um parquet local |
| Online store (Bigtable) | `SqliteOnlineStoreConfig` (arquivo `.db` local) |
| Ingestion / materialization job | `store.materialize_incremental(end_date=...)` |
| Online Serving API | `store.get_online_features(...)` |
| Feature Registry | `registry.db` (SQLite) gerado pelo `store.apply(...)` |

Esse mapeamento vai também para o README final (Marco de documentação).

---

## 3. Arquitetura

### 3.1 Novo pacote `src/churn/feature_store/`

Três módulos com responsabilidades isoladas:

- **`definitions.py`** — objetos declarativos do Feast e a factory do store.
  - `Entity` `geography` (join key `Geography`, `ValueType.STRING`).
  - `FileSource` para o parquet offline (`timestamp_field="event_timestamp"`).
  - `FeatureView` `geo_churn_stats` com `schema=[Field("geography_churn_rate", Float32)]`,
    `ttl` amplo, `online=True`.
  - `build_store(cfg) -> FeatureStore` — monta `FeatureStore(config=RepoConfig(...))`
    programaticamente (registry SQLite, `SqliteOnlineStoreConfig`,
    `entity_key_serialization_version=3`), dispensando `feature_store.yaml`.
  - Sentinela de fallback: constante `GLOBAL_KEY = "__global__"`.

- **`materialize.py`** — popula o offline store e materializa para o online.
  - `build_offline_frame(cfg) -> pd.DataFrame`: carrega o CSV, refaz o **mesmo split de train**
    (seed, `test_size`, `stratify` idênticos ao `train.py`), fita `GeographyChurnRateEncoder` no
    train e monta um DataFrame `{Geography, geography_churn_rate, event_timestamp}`. Inclui uma linha
    extra com `Geography=GLOBAL_KEY` e o `global_rate_` do encoder (o fallback para geografia não vista).
  - `materialize(cfg)`: escreve o parquet, roda `store.apply([...])` e
    `store.materialize_incremental(end_date=<timestamp fixo>)`.
  - `main()`: entrypoint chamado por `python -m churn.feature_store.materialize`.

- **`store.py`** — leitura online (a "Online Serving API" local).
  - `get_geography_churn_rate(geographies: list[str], cfg=settings) -> dict[str, float]`:
    chama `store.get_online_features(features=["geo_churn_stats:geography_churn_rate"],
    entity_rows=[{"Geography": g} for g in geographies])`, converte via `.to_dict()`. Para cada
    geografia cujo valor online vier nulo (não materializada), usa o valor da linha `GLOBAL_KEY`.

### 3.2 Config (`src/churn/config.py`)

Adicionar ao `Settings` (prefixo `CHURN_`):

- `feast_repo_path: str = "feature_repo"` — diretório raiz dos artefatos Feast.

Derivar (propriedades ou campos computados) os caminhos:
- registry: `<feast_repo_path>/registry.db`
- online store: `<feast_repo_path>/online_store.db`
- offline parquet: `<feast_repo_path>/data/geo_churn_stats.parquet`

Manter os campos existentes (`data_path`, `random_state`, `test_size`, etc.) — o `materialize.py`
reusa `random_state`, `test_size` e `data_path` para garantir o mesmo split do treino.

### 3.3 Serving (`src/churn/serving/api.py`)

Endpoint **aditivo e defensivo**:

- `GET /features/geography/{geography}` → `{"geography": "...", "geography_churn_rate": 0.16...}`.
- Carrega o store de forma **lazy** (na primeira chamada, não no `lifespan`) para não acoplar a
  subida da API à existência do `feature_repo/`.
- Se o store/online não estiver materializado, responde **503** (mesmo padrão do modelo não carregado).
  Não toca `/health` nem `/predict`.

### 3.4 Artefatos gerados e `.gitignore`

Os artefatos Feast são **gerados** (como `mlflow.db`/`mlruns/`) e **nunca** entram no git. Adicionar ao
`.gitignore`:

```gitignore
# Feast local artifacts (generated)
feature_repo/
```

### 3.5 Makefile

- Novo target `feast-materialize`:

  ```makefile
  feast-materialize: ## Popula e materializa a feature store Feast (offline -> online)
  	uv run python -m churn.feature_store.materialize
  ```

- `docker-run`: montar o `feature_repo` para o endpoint online funcionar no container (uma linha,
  consistente com o mount do `mlruns`):

  ```makefile
  -v "$(CURDIR)/feature_repo:$(CURDIR)/feature_repo" \
  ```

---

## 4. Fluxo de dados

```
Customer-Churn-Records.csv
        │  (mesmo split de train: seed, test_size, stratify)
        ▼
GeographyChurnRateEncoder.fit(train)         ← fonte da verdade, leakage-safe
        │  mapping_ + global_rate_
        ▼
build_offline_frame → parquet (FileSource)   ← OFFLINE store
        │  store.apply(...) + materialize_incremental(...)
        ▼
SQLite online_store.db                        ← ONLINE store
        │  get_online_features(...)
        ▼
get_geography_churn_rate(["France", ...])     ← serving online por entity key
        │
        ▼
GET /features/geography/France  →  {"geography_churn_rate": 0.16...}
```

**Garantia anti-skew (por construção):** o parquet offline é derivado do **mesmo**
`GeographyChurnRateEncoder.fit(train)` que o `ChurnFeatureBuilder` usa no treino. Logo, offline ==
valor de treino por definição; o teste de consistência prova que online == offline após materialização.

---

## 5. Tratamento de erros

- **Store não materializado** (`feature_repo/` ausente ou vazio): `get_geography_churn_rate` levanta
  erro claro; o endpoint converte em **503**. Documentado que `make feast-materialize` é pré-requisito.
- **Geografia não vista** (não está no online store → valor nulo): **não é erro** — cai no valor
  `GLOBAL_KEY` (`global_rate_`), espelhando a semântica de fallback do encoder. Consistência de
  comportamento entre encoder e feature store.
- **Materialização idempotente:** rodar `make feast-materialize` de novo re-aplica as definitions e
  re-materializa sem quebrar (sobrescreve o parquet e atualiza o online store).

---

## 6. Estratégia de testes (`tests/test_feature_store.py`)

Fixture module-scoped que materializa numa `tmp_path` via `Settings` parametrizado (mesmo padrão de
`test_load_production_model_round_trips` em `tests/test_api.py`, que sobrescreve caminhos por `Settings`).

1. **`test_definitions_build`** — a entity, o source e a feature view são construídos com nomes, join
   keys e schema esperados.
2. **`test_materialize_end_to_end`** — após materializar contra o CSV real num repo temporário,
   `get_geography_churn_rate(["France", "Germany", "Spain"])` retorna 3 floats em `(0, 1)`.
3. **`test_online_matches_training_encoder`** *(a prova anti-skew)* — fita
   `GeographyChurnRateEncoder` no mesmo split de train; afirma que, para cada geografia, o valor
   servido **online pelo Feast** == `encoder.mapping_[geografia]` (tolerância `1e-6`).
4. **`test_unseen_geography_falls_back_to_global`** — consulta uma geografia fora do train
   (ex.: `"Sao Paulo"`) → retorna o `global_rate_`.
5. **`test_api_feature_endpoint`** — com store materializado (via `dependency_overrides`/`Settings`
   de teste), `GET /features/geography/France` responde 200 com o rate; sem store → 503.

Critério de pronto: os 42 testes existentes continuam verdes + os novos passam (`make test`), e
`make lint` limpo.

---

## 7. Estrutura de arquivos

| Arquivo | Ação | Responsabilidade |
|---|---|---|
| `src/churn/feature_store/__init__.py` | criar | Marca o pacote |
| `src/churn/feature_store/definitions.py` | criar | Entity, FileSource, FeatureView, `build_store` |
| `src/churn/feature_store/materialize.py` | criar | Offline frame + apply + materialize + `main()` |
| `src/churn/feature_store/store.py` | criar | `get_geography_churn_rate` (leitura online + fallback) |
| `src/churn/config.py` | modificar | `feast_repo_path` + caminhos derivados |
| `src/churn/serving/api.py` | modificar | Endpoint `GET /features/geography/{geography}` |
| `tests/test_feature_store.py` | criar | Testes 1–5 acima |
| `.gitignore` | modificar | Ignorar `feature_repo/` |
| `Makefile` | modificar | Target `feast-materialize` + mount no `docker-run` |

Reuso (sem alteração): `src/churn/features/aggregations.py` (`GeographyChurnRateEncoder`),
`src/churn/data.py` (`load_raw`), `src/churn/features/builder.py` (`INPUT_COLUMNS`).

---

## 8. Riscos e mitigações

- **Compat de dependências Feast** — já resolvido no spike (Fase A): `feast>=0.54` +
  `uvicorn[standard]>=0.30.6,<=0.34.0` no `pyproject.toml`, com pandas/numpy/protobuf modernos. Sem
  ação nova.
- **Docker + caminho do online store** — mesmo padrão do gotcha MLflow: o `feature_repo` precisa ser
  montado no mesmo caminho absoluto (mount adicionado no `docker-run`). O endpoint é defensivo (503) se
  ausente, então a ausência não derruba a API.
- **Divergência de split** — se o split do `materialize.py` divergir do `train.py`, o teste anti-skew
  (teste 3) falha. Mitigação: reusar `random_state`/`test_size`/`stratify` a partir do mesmo `Settings`.
