# Model Monitoring com Evidently — Design (Marco 6)

**Data:** 2026-08-17
**Issue:** #13
**Branch:** `feat/monitoring-evidently`
**Equivalente Vertex AI:** Model Monitoring (data/prediction drift + model quality)

## 1. Objetivo

Produzir o equivalente local do **Vertex AI Model Monitoring**: um relatório de
*drift* (dados de entrada + score) e de *qualidade do modelo* comparando um
conjunto de **referência** (baseline de treino) contra um conjunto **corrente**,
com um *quality gate* em Python que falha (exit code ≠ 0) quando o drift ultrapassa
o limiar — o papel do "alerta" gerenciado do Vertex.

Escopo do teste técnico: **pipeline de MLOps**, não trocar o modelo. O
monitoramento é *on-demand* (batch, via CLI), não um serviço agendado.

## 2. Mapeamento Vertex AI ↔ Evidently (com gaps honestos)

| Vertex AI Model Monitoring | Equivalente local (este marco) | Gap honesto |
|---|---|---|
| Feature/input drift (skew & drift) | `DataDriftPreset` sobre as 12 colunas raw | — |
| Prediction (score) drift | `ValueDrift('prob_churn')` | — |
| Model quality (labels chegando) | `ClassificationQuality()` com holdout rotulado | Em produção o label chega atrasado; aqui uso o holdout como proxy |
| Métrica de drift numérica | Jensen-Shannon distance | Vertex usa **Jensen-Shannon** — paridade OK |
| Métrica de drift categórica | Jensen-Shannon (Evidently) | Vertex usa **L-infinity**; Evidently **não tem L-infinity nativo** — documentado como limitação |
| Threshold por feature (default Vertex 0.3) | `drift_threshold=0.3` explícito no config | Evidently default é 0.1 — é preciso setar 0.3 |
| Captura automática no endpoint | Holdout `X_test` como proxy de "produção" | A API não loga predições; sem captura real |
| Agendamento gerenciado | `make monitor` (CLI / cron manual) | Sem scheduler gerenciado |
| Alerta (Cloud Monitoring / e-mail) | *Quality gate* Python → exit code ≠ 0 | Sem canal de alerta gerenciado |

Estes gaps alimentam a tabela de mapeamento do README final (Marco 9).

## 3. Arquitetura — pacote `src/churn/monitoring/`

Espelha `feature_store/` e `orchestration/`: funções puras + entrypoint fino.
Cinco módulos, cada um com uma responsabilidade única:

| Arquivo | Responsabilidade | Depende de |
|---|---|---|
| `datasets.py` | Monta os DataFrames **reference** e **current**, cada um já com as 12 colunas raw + `prob_churn` (score do modelo) + `turnover` (label). | `data.load_raw`, `train_test_split`, `load_production_model`, `INPUT_COLUMNS` |
| `perturb.py` | `simulate_drift(df)` — perturbação determinística de features numéricas para demonstrar detecção (`--simulate-drift`). Função pura. | — |
| `report.py` | `build_report(reference_df, current_df, cfg)` — monta `DataDefinition` + `Report`, roda e devolve o snapshot Evidently. | evidently 0.7, `INPUT_COLUMNS` |
| `gate.py` | `summarize(snapshot)` → dict de métricas-chave; `evaluate_gate(summary, threshold)` → bool (aprovado?). | — |
| `__main__.py` | CLI: parseia `--simulate-drift`; orquestra datasets → report → escreve HTML+JSON → aplica o gate → exit code. | os módulos acima, `cfg` |

### 3.1 Fluxo de dados

```
load_raw(cfg.data_path)
  → train_test_split(random_state=42, test_size=0.2, stratify=y)   # MESMO split do treino
  → reference = X_train + turnover ;  current = X_test + turnover
  → model = load_production_model(cfg)                             # models:/churn-model@production
  → prob_churn = model.predict_proba(frame[INPUT_COLUMNS])[:, 1]   # em ref e current
  → [--simulate-drift]  current = simulate_drift(current)          # perturba ANTES do predict? ver §3.4
  → build_report(reference, current, cfg) → snapshot
  → snapshot.save_html(cfg.monitoring_report_path)                 # reports/drift.html
  → write metrics.json (summarize(snapshot))                       # reports/metrics.json
  → gate = evaluate_gate(summary, cfg.drift_threshold)
  → exit(0 if gate else 1)
```

### 3.2 Por que holdout como "produção"

A API (`serving/api.py`) **não loga predições**, logo não há dados de produção
reais. A escolha honesta: usar o **holdout `X_test`** (que o modelo nunca viu no
fit) como um conjunto "corrente saudável". Sem perturbação, o drift é ~0 e o gate
passa (`make monitor`). Com `--simulate-drift`, perturbo o holdout para demonstrar
detecção e o gate falhar (`make monitor-drift`). Isto é declarado como proxy, não
como captura real de produção.

### 3.3 DataDefinition (Evidently 0.7)

- `numerical_columns` = `RAW_NUMERIC` (9) + `"prob_churn"` (o score entra como coluna numérica para o `ValueDrift`)
- `categorical_columns` = `RAW_CATEGORICAL` (3)
- classificação: `BinaryClassification(target="turnover", prediction_probas="prob_churn", pos_label=1)`

`Report([DataDriftPreset(), ValueDrift(column="prob_churn"), ClassificationQuality()], include_tests=True)`.
A chamada de execução usa **argumentos nomeados** — `report.run(current_data=cur_ds, reference_data=ref_ds)` — para não depender da ordem posicional.

> Nota de implementação: a API exata de configuração do método de drift
> (Jensen-Shannon) e do threshold 0.3 por coluna no `DataDriftPreset` do Evidently
> 0.7 será fixada no plano/execução via TDD com chamadas reais (a superfície de API
> muda entre minors). O design fixa a **intenção** (JS + threshold 0.3 + gate por
> drift share); o plano fixa a **assinatura**.

### 3.4 Perturbação (`simulate_drift`)

Determinística (sem `random` global — usa deslocamentos fixos ou `np.random.default_rng(seed)`),
preserva o schema (mesmas colunas, mesmos dtypes). Desloca a distribuição de
algumas numéricas de forma visível (ex.: `Age += 15`, `Balance *= 1.5`,
`CreditScore -= 100`). Aplicada ao `current` **antes** do `predict_proba`, para que
tanto o input drift quanto o score drift apareçam (o modelo reage às features
perturbadas). O label `turnover` do holdout é preservado.

### 3.5 Quality gate

- `summarize(snapshot)` extrai de `snapshot.dict()`: nº de colunas com drift, drift
  share, e as métricas de qualidade (roc_auc/accuracy/precision/recall quando
  disponíveis). Retorna um dict simples e serializável (vai pro `metrics.json`).
- `evaluate_gate(summary, threshold)` → `False` (falha) se `drift_share > threshold`,
  `True` caso contrário. É o "alerta": o `__main__` traduz `False` em exit code 1.

## 4. Config (`config.py`) — aditivo

```python
# Model monitoring (Vertex AI Model Monitoring equivalent)
reports_dir: str = "reports"
drift_threshold: float = 0.3  # Jensen-Shannon per-feature, Vertex default parity

@property
def monitoring_report_path(self) -> str:
    return str(Path(self.reports_dir) / "drift.html")

@property
def monitoring_metrics_path(self) -> str:
    return str(Path(self.reports_dir) / "metrics.json")
```

`reports/` já está no `.gitignore` (junto com `*.html`) — nada versionado.

## 5. Makefile — aditivo

```makefile
monitor: ## Gera o relatório de drift/qualidade (holdout saudável; gate deve passar)
	uv run python -m churn.monitoring

monitor-drift: ## Idem com drift simulado (demonstra detecção; gate falha de propósito)
	uv run python -m churn.monitoring --simulate-drift
```

Adicionar `monitor monitor-drift` ao `.PHONY`.

## 6. Testes (TDD)

Funções puras testadas sem tocar Evidently pesado; um smoke end-to-end marcado.

- **`perturb.py`**: `simulate_drift` é determinístico (duas chamadas → igual),
  muda a média de ≥1 coluna numérica, preserva colunas e dtypes, não altera
  `turnover`.
- **`datasets.py`**: `build_reference_current(cfg)` devolve dois frames com as
  colunas esperadas (`prob_churn`, `turnover` presentes; 12 raw presentes);
  `prob_churn ∈ [0,1]`; tamanho do split reproduz `test_size`. (Usa um CSV/fixture
  pequeno ou o dataset real via `cfg` de teste; modelo pode ser mockado para não
  depender de `@production` — decidir no plano.)
- **`gate.py`**: `evaluate_gate` retorna `False` quando `drift_share > threshold` e
  `True` quando `≤`; `summarize` extrai as chaves esperadas de um snapshot real
  pequeno.
- **`report.py`** (smoke, pode ser lento): `build_report` sobre um mini-dataset
  gera um snapshot cujo `save_html` produz arquivo não-vazio.
- **integração do gate**: holdout puro → gate passa; holdout perturbado → drift
  detectado e gate falha.

`make test` continua verde (os 62 atuais + novos). `ruff check` limpo.

## 7. Fora de escopo (YAGNI)

Sem logging no MLflow, sem step no KFP, sem *feature attribution drift*, sem
streaming, sem *target drift* isolado, sem agendamento. Monitoramento é standalone
via CLI.
