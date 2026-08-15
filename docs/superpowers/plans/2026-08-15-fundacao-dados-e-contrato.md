# Fundação — Contrato de Dados, Scoring e Agregação sem Leakage — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Estabelecer a fundação do projeto MLOps de churn — projeto `uv`, configuração tipada, contrato de dados (Pandera), a regra de scoring única e a agregação `geography_churn_rate` comprovadamente livre de target leakage — tudo coberto por testes.

**Architecture:** Pacote Python `src/churn/` instalável via `uv`. Camada de dados isola o CSV do resto: `schema.py` (Pandera) valida a entrada, `data.py` carrega+valida, `scoring.py` centraliza a regra prob→score (fecha bug #9), e `features/aggregations.py` implementa o encoder de taxa de churn por geografia com `fit` **apenas no treino** (fecha bug #1). Sem Feast/MLflow/API ainda — esses entram em marcos seguintes que constroem sobre esta base.

**Tech Stack:** Python 3.12+, uv, pandas, numpy, scikit-learn, pandera, pydantic-settings, pytest, ruff.

**Escopo deste plano:** Fases 1 (parcial) e o núcleo da Fase 2 da spec (`docs/superpowers/specs/2026-08-15-bulla-churn-mlops-design.md`). Fora deste plano (marcos futuros, cada um com seu próprio plano/branch/PR): Feast, treino sklearn Pipeline + MLflow, KFP, FastAPI, Docker, Evidently, CI/CD, `tensorflow_variant.md`, README final completo.

---

### Task 1: Scaffold do projeto (uv + estrutura + pyproject)

**Files:**
- Create: `pyproject.toml` (via `uv init`, depois editado)
- Create: `src/churn/__init__.py`
- Create: `src/churn/features/__init__.py`
- Create: `tests/__init__.py`
- Create: `.env.example`

- [ ] **Step 1: Inicializar o projeto uv**

Run:
```bash
uv init --package --name churn --python 3.12 .
```
Isso cria `pyproject.toml` e `src/churn/`. Se `uv init` reclamar de diretório não vazio, prossiga (ele cria os arquivos que faltam) — os arquivos existentes (`README`, `.gitignore`, CSV, `legacy` futuro) são preservados.

- [ ] **Step 2: Adicionar dependências de runtime e dev**

Run:
```bash
uv add pandas numpy scikit-learn "pandera[pandas]>=0.22" pydantic-settings
uv add --dev pytest ruff
```

- [ ] **Step 3: Criar a estrutura de pacotes**

Run:
```bash
mkdir -p src/churn/features tests
touch src/churn/features/__init__.py tests/__init__.py
```
Garanta que `src/churn/__init__.py` existe (o `uv init --package` já o cria). Se não existir, crie vazio.

- [ ] **Step 4: Criar `.env.example`**

Create `.env.example`:
```dotenv
# Configuração da pipeline de churn (lida por src/churn/config.py via pydantic-settings)
CHURN_DATA_PATH=Customer-Churn-Records.csv
CHURN_RANDOM_STATE=42
CHURN_TEST_SIZE=0.2
```

- [ ] **Step 5: Verificar que o ambiente resolve**

Run: `uv run python -c "import pandas, numpy, sklearn, pandera, pydantic_settings; print('ok')"`
Expected: imprime `ok` sem erro.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock src/churn/__init__.py src/churn/features/__init__.py tests/__init__.py .env.example
git commit -m "chore: scaffold uv project structure and core deps"
```

---

### Task 2: Configuração tipada (`config.py`)

**Files:**
- Create: `src/churn/config.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Escrever o teste que falha**

Create `tests/test_config.py`:
```python
from churn.config import Settings


def test_defaults():
    s = Settings()
    assert s.data_path == "Customer-Churn-Records.csv"
    assert s.random_state == 42
    assert s.test_size == 0.2


def test_env_override(monkeypatch):
    monkeypatch.setenv("CHURN_RANDOM_STATE", "7")
    monkeypatch.setenv("CHURN_TEST_SIZE", "0.3")
    s = Settings()
    assert s.random_state == 7
    assert s.test_size == 0.3
```

- [ ] **Step 2: Rodar o teste e ver falhar**

Run: `uv run pytest tests/test_config.py -v`
Expected: FAIL com `ModuleNotFoundError: No module named 'churn.config'`.

- [ ] **Step 3: Implementar `config.py`**

Create `src/churn/config.py`:
```python
"""Configuração tipada da pipeline, lida de variáveis de ambiente / .env."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Parâmetros da pipeline. Prefixo CHURN_ nas variáveis de ambiente."""

    model_config = SettingsConfigDict(env_prefix="CHURN_", env_file=".env", extra="ignore")

    data_path: str = "Customer-Churn-Records.csv"
    random_state: int = 42
    test_size: float = 0.2


settings = Settings()
```

- [ ] **Step 4: Rodar o teste e ver passar**

Run: `uv run pytest tests/test_config.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/churn/config.py tests/test_config.py
git commit -m "feat: add typed settings via pydantic-settings"
```

---

### Task 3: Preservar scripts legados + Makefile

**Files:**
- Move: `train_model_churn.py` → `legacy/train_model_churn.py`
- Move: `infer_model_churn.py` → `legacy/infer_model_churn.py`
- Create: `legacy/README.md`
- Create: `Makefile`

- [ ] **Step 1: Mover os scripts originais para `legacy/`**

Run:
```bash
mkdir -p legacy
git mv train_model_churn.py legacy/train_model_churn.py
git mv infer_model_churn.py legacy/infer_model_churn.py
```
(Se `git mv` falhar porque os arquivos não estão rastreados, use `mv` simples — eles já estão no repo, então `git mv` deve funcionar.)

- [ ] **Step 2: Documentar o porquê da pasta legacy**

Create `legacy/README.md`:
```markdown
# Código original (preservado para referência)

Estes são os dois scripts originais recebidos no teste, **intocados**. Servem de baseline
para a auditoria dos problemas conceituais (ver o design em
`docs/superpowers/specs/2026-08-15-bulla-churn-mlops-design.md`, seção 2) e de rastreabilidade
"antes → depois". Não são executados pela pipeline nova.
```

- [ ] **Step 3: Criar o Makefile**

Create `Makefile`:
```makefile
.DEFAULT_GOAL := help

.PHONY: help setup test lint format

help: ## Lista os targets disponíveis
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

setup: ## Prepara o ambiente do zero (deps + .env)
	uv sync
	@test -f .env || cp .env.example .env
	@echo "Ambiente pronto. Edite .env se precisar."

test: ## Roda a suíte de testes
	uv run pytest -v

lint: ## Checa estilo com ruff
	uv run ruff check src tests

format: ## Formata com ruff
	uv run ruff format src tests
```

- [ ] **Step 4: Verificar o Makefile**

Run: `make help`
Expected: lista os targets `setup`, `test`, `lint`, `format` com suas descrições.

- [ ] **Step 5: Commit**

```bash
git add legacy Makefile
git commit -m "chore: preserve original scripts in legacy/ and add Makefile"
```

---

### Task 4: Contrato de dados do CSV (`schema.py`)

**Files:**
- Create: `src/churn/schema.py`
- Test: `tests/test_schema.py`

- [ ] **Step 1: Escrever o teste que falha**

Create `tests/test_schema.py`:
```python
import pandas as pd
import pytest
from pandera.errors import SchemaError

from churn.schema import validate_raw


def _valid_row() -> dict:
    return {
        "RowNumber": 1,
        "CustomerId": 15634602,
        "Surname": "Hargrave",
        "CreditScore": 619,
        "Geography": "Sao Paulo",
        "Gender": "Female",
        "Age": 42,
        "Tenure": 2,
        "Balance": 0.0,
        "NumOfProducts": 1,
        "HasCrCard": 1,
        "IsActiveMember": 1,
        "EstimatedSalary": 101348.88,
        "turnover": 1,
        "Complain": 1,
        "Satisfaction Score": 2,
        "Card Type": "DIAMOND",
        "Point Earned": 464,
    }


def test_valid_frame_passes():
    df = pd.DataFrame([_valid_row()])
    out = validate_raw(df)
    assert len(out) == 1


def test_unknown_geography_rejected():
    row = _valid_row()
    row["Geography"] = "Lisboa"
    with pytest.raises(SchemaError):
        validate_raw(pd.DataFrame([row]))


def test_target_out_of_domain_rejected():
    row = _valid_row()
    row["turnover"] = 2
    with pytest.raises(SchemaError):
        validate_raw(pd.DataFrame([row]))


def test_impossible_age_rejected():
    row = _valid_row()
    row["Age"] = -5
    with pytest.raises(SchemaError):
        validate_raw(pd.DataFrame([row]))
```

- [ ] **Step 2: Rodar o teste e ver falhar**

Run: `uv run pytest tests/test_schema.py -v`
Expected: FAIL com `ModuleNotFoundError: No module named 'churn.schema'`.

- [ ] **Step 3: Implementar `schema.py`**

Create `src/churn/schema.py`:
```python
"""Contrato de dados do CSV bruto (Pandera). Barra dados fora do padrão antes do treino."""

import pandas as pd
from pandera.pandas import Check, Column, DataFrameSchema

GEOGRAPHIES = ["Minas Gerais", "Rio de Janeiro", "Sao Paulo"]
GENDERS = ["Male", "Female"]
CARD_TYPES = ["DIAMOND", "GOLD", "PLATINUM", "SILVER"]

RAW_SCHEMA = DataFrameSchema(
    {
        "RowNumber": Column(int, Check.ge(1)),
        "CustomerId": Column(int),
        "Surname": Column(str),
        "CreditScore": Column(int, Check.in_range(300, 900)),
        "Geography": Column(str, Check.isin(GEOGRAPHIES)),
        "Gender": Column(str, Check.isin(GENDERS)),
        "Age": Column(int, Check.in_range(18, 120)),
        "Tenure": Column(int, Check.in_range(0, 10)),
        "Balance": Column(float, Check.ge(0), nullable=True),
        "NumOfProducts": Column(int, Check.in_range(1, 4)),
        "HasCrCard": Column(int, Check.isin([0, 1])),
        "IsActiveMember": Column(int, Check.isin([0, 1])),
        "EstimatedSalary": Column(float, Check.ge(0), nullable=True),
        "turnover": Column(int, Check.isin([0, 1])),
        "Complain": Column(int, Check.isin([0, 1])),
        "Satisfaction Score": Column(int, Check.in_range(1, 5)),
        "Card Type": Column(str, Check.isin(CARD_TYPES)),
        "Point Earned": Column(int, Check.ge(0)),
    },
    strict=False,  # tolera colunas extras (ex.: índice)
    coerce=True,
)


def validate_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Valida o DataFrame bruto contra RAW_SCHEMA. Lança SchemaError se inválido."""
    return RAW_SCHEMA.validate(df, lazy=False)
```

- [ ] **Step 4: Rodar o teste e ver passar**

Run: `uv run pytest tests/test_schema.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add src/churn/schema.py tests/test_schema.py
git commit -m "feat: add Pandera raw-data contract for the churn CSV"
```

---

### Task 5: Carregamento validado do CSV (`data.py`)

**Files:**
- Create: `src/churn/data.py`
- Test: `tests/test_data.py`

- [ ] **Step 1: Escrever o teste que falha**

Create `tests/test_data.py`:
```python
from churn.data import load_raw


def test_load_raw_validates_and_loads():
    df = load_raw("Customer-Churn-Records.csv")
    assert len(df) == 10000
    assert "turnover" in df.columns
    # o schema garante domínio: nenhuma geografia fora das 3 conhecidas
    assert set(df["Geography"].unique()) <= {"Minas Gerais", "Rio de Janeiro", "Sao Paulo"}
```

- [ ] **Step 2: Rodar o teste e ver falhar**

Run: `uv run pytest tests/test_data.py -v`
Expected: FAIL com `ModuleNotFoundError: No module named 'churn.data'`.

- [ ] **Step 3: Implementar `data.py`**

Create `src/churn/data.py`:
```python
"""Carregamento do CSV com validação de schema na porta de entrada."""

import pandas as pd

from churn.schema import validate_raw


def load_raw(path: str) -> pd.DataFrame:
    """Lê o CSV e valida contra o contrato. Lança SchemaError se algo estiver fora do padrão."""
    df = pd.read_csv(path)
    return validate_raw(df)
```

- [ ] **Step 4: Rodar o teste e ver passar**

Run: `uv run pytest tests/test_data.py -v`
Expected: PASS (1 passed). Se falhar por coerção de tipo (ex.: `Balance` como int), ajuste apenas se necessário — o schema usa `coerce=True`, então deve passar.

- [ ] **Step 5: Commit**

```bash
git add src/churn/data.py tests/test_data.py
git commit -m "feat: add validated CSV loader"
```

---

### Task 6: Regra de scoring única (`scoring.py`)

**Files:**
- Create: `src/churn/scoring.py`
- Test: `tests/test_scoring.py`

- [ ] **Step 1: Escrever o teste que falha**

Create `tests/test_scoring.py`:
```python
import numpy as np

from churn.scoring import retention_score


def test_scalar_bounds():
    assert retention_score(0.0) == 0   # nenhuma chance de ficar → score 0
    assert retention_score(1.0) == 10  # certeza de ficar → score 10


def test_scalar_rounding():
    assert retention_score(0.42) == 4  # 4.2 → 4
    assert retention_score(0.78) == 8  # 7.8 → 8


def test_array_input():
    out = retention_score(np.array([0.0, 0.42, 1.0]))
    assert out.tolist() == [0, 4, 10]
    assert out.dtype == np.int64 or out.dtype == int
```

- [ ] **Step 2: Rodar o teste e ver falhar**

Run: `uv run pytest tests/test_scoring.py -v`
Expected: FAIL com `ModuleNotFoundError: No module named 'churn.scoring'`.

- [ ] **Step 3: Implementar `scoring.py`**

Create `src/churn/scoring.py`:
```python
"""Regra única de score de retenção: probabilidade de PERMANECER → inteiro 0–10.

Importada por treino, inferência batch e API, para não duplicar a regra (fecha bug #9).
"""

import numpy as np


def retention_score(prob_stay):
    """Converte a probabilidade de o cliente permanecer (0..1) em um score inteiro 0–10.

    Aceita escalar ou array. Retorna int (escalar) ou ndarray de int (array).
    """
    scores = np.round(np.asarray(prob_stay, dtype=float) * 10).astype(int)
    if scores.ndim == 0:
        return int(scores)
    return scores
```

- [ ] **Step 4: Rodar o teste e ver passar**

Run: `uv run pytest tests/test_scoring.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/churn/scoring.py tests/test_scoring.py
git commit -m "feat: add single retention-scoring rule (prob->0-10)"
```

---

### Task 7: Agregação `geography_churn_rate` sem leakage (`features/aggregations.py`)

**Files:**
- Create: `src/churn/features/aggregations.py`
- Test: `tests/test_aggregations.py`

Este é o núcleo conceitual: o encoder aprende a taxa de churn por geografia **só no treino**
(`fit`) e apenas **aplica** no teste/produção (`transform`). Isso fecha o bug #1 (target leakage):
o valor visto pelo teste vem do treino, nunca recalculado com o rótulo do próprio teste.

- [ ] **Step 1: Escrever o teste que falha**

Create `tests/test_aggregations.py`:
```python
import pandas as pd

from churn.features.aggregations import GeographyChurnRateEncoder


def _frame(geos, targets) -> pd.DataFrame:
    return pd.DataFrame({"Geography": geos, "turnover": targets})


def test_fit_learns_train_group_means():
    train = _frame(
        ["Sao Paulo", "Sao Paulo", "Minas Gerais", "Minas Gerais"],
        [1, 0, 0, 0],  # SP: 0.5, MG: 0.0
    )
    enc = GeographyChurnRateEncoder().fit(train)
    assert enc.mapping_["Sao Paulo"] == 0.5
    assert enc.mapping_["Minas Gerais"] == 0.0


def test_transform_uses_train_mapping_not_test_labels():
    # No treino, SP tem churn 0.5. No teste, SP tem churn 1.0.
    # O transform DEVE devolver 0.5 (aprendido no treino), nunca 1.0 (rótulo do teste).
    train = _frame(["Sao Paulo", "Sao Paulo"], [1, 0])  # SP: 0.5
    test = _frame(["Sao Paulo", "Sao Paulo"], [1, 1])   # SP: 1.0 (não pode vazar)
    enc = GeographyChurnRateEncoder().fit(train)
    out = enc.transform(test)
    assert list(out) == [0.5, 0.5]


def test_unseen_geography_falls_back_to_global_rate():
    train = _frame(["Sao Paulo", "Sao Paulo", "Minas Gerais"], [1, 0, 0])  # global = 1/3
    enc = GeographyChurnRateEncoder().fit(train)
    out = enc.transform(_frame(["Rio de Janeiro"], [0]))
    assert abs(out.iloc[0] - (1 / 3)) < 1e-9
```

- [ ] **Step 2: Rodar o teste e ver falhar**

Run: `uv run pytest tests/test_aggregations.py -v`
Expected: FAIL com `ModuleNotFoundError: No module named 'churn.features.aggregations'`.

- [ ] **Step 3: Implementar `aggregations.py`**

Create `src/churn/features/aggregations.py`:
```python
"""Encoder da taxa de churn por geografia — aprendido SÓ no treino (anti-leakage, bug #1)."""

import pandas as pd


class GeographyChurnRateEncoder:
    """Aprende a taxa média de `turnover` por `Geography` no conjunto de treino e a aplica.

    - `fit(df_train)`: memoriza a média do alvo por grupo e a taxa global (fallback).
    - `transform(df)`: mapeia cada linha para a taxa aprendida; geografia não vista → taxa global.

    Como o mapeamento vem exclusivamente do treino, o conjunto de teste/produção nunca
    influencia o valor da feature — não há vazamento do rótulo.
    """

    def __init__(self, geography_col: str = "Geography", target_col: str = "turnover"):
        self.geography_col = geography_col
        self.target_col = target_col

    def fit(self, df: pd.DataFrame) -> "GeographyChurnRateEncoder":
        self.mapping_ = df.groupby(self.geography_col)[self.target_col].mean().to_dict()
        self.global_rate_ = float(df[self.target_col].mean())
        return self

    def transform(self, df: pd.DataFrame) -> pd.Series:
        return df[self.geography_col].map(self.mapping_).fillna(self.global_rate_)
```

- [ ] **Step 4: Rodar o teste e ver passar**

Run: `uv run pytest tests/test_aggregations.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Rodar a suíte completa e o lint**

Run: `uv run pytest -v && uv run ruff check src tests`
Expected: todos os testes passam (config, schema, data, scoring, aggregations); ruff sem erros.

- [ ] **Step 6: Commit**

```bash
git add src/churn/features/aggregations.py tests/test_aggregations.py
git commit -m "feat: add leakage-free geography churn-rate encoder"
```

---

## Notas de execução

- **Branch:** todo este marco vive numa feature branch (`feat/fundacao-dados`), integrada por PR no fim — nunca commitar direto no `main`. Sem crédito ao Claude nos commits/PR (regra deste repo).
- **Ordem:** as tasks são sequenciais; cada uma deixa a suíte verde antes da próxima.
- **Definition of done do marco:** `make test` verde, `make lint` limpo, scripts originais preservados em `legacy/`, e o contrato de dados + scoring + encoder anti-leakage prontos para o marco de treino consumir.
