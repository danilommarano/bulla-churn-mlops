# Site de apresentação (deck + dashboard) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Construir, em `presentation/`, um site Astro + Tailwind que É a apresentação do projeto Churn MLOps — um deck de 12 slides keynote-minimalista + um dashboard de métricas — com interação ao vivo no `/predict` e fallback offline, mais a integração mínima no repo Python (CORS + alvo de Makefile de métricas).

**Architecture:** Duas faces sobre a mesma base visual: `/` (deck, 12 seções full-screen governadas por `deck.ts`) e `/dashboard` (cards + matriz + gauge + timeline). Componentes Astro pequenos e isolados; dados reais (`metrics.json`, `presets_fallback.json`, `slides.ts`, `bugs.ts`, `vertex-map.ts`) embutidos no build. A única lógica de runtime é client-side: navegação por teclado e o `ModelPlayground` (fetch ao `/predict` com fallback pré-gravado). No lado Python, uma adição de CORS coberta por teste e um alvo `make presentation-metrics` que reusa `churn.training.evaluate.evaluate`.

**Tech Stack:** Astro 5, Tailwind CSS v4 (`@tailwindcss/vite`), `astro-expressive-code` (highlight Dracula + marcação de linha), Mermaid (client-side), FastAPI/pytest (lado repo). Sem framework de slides de terceiros.

---

## Convenção de verificação (leia antes de começar)

Este plano tem **duas naturezas de tarefa**, cada uma com sua forma honesta de verificar:

- **Tarefas Python (1 e 2):** TDD clássico — teste falhando primeiro, depois implementação, `uv run pytest` verde. Rode da raiz do repo.
- **Tarefas Astro (3–14):** não force pytest em `.astro`. A verificação real é **`npm run build` sem erro** dentro de `presentation/`, mais uma checagem manual pontual descrita na tarefa (ex.: "abra e veja o gauge animar"). Onde houver lógica JS pura testável de forma barata (o `deck.ts` e o resolvedor de score do playground), a tarefa inclui um teste com `node --test`.

Todos os comandos Astro rodam **dentro de `presentation/`**. Todos os comandos Python rodam **na raiz do repo**. Cada tarefa diz onde.

**Branch:** já estamos em `feat/presentation-site` (issue #21). Commite pequeno a cada tarefa. Sem trailer `Co-Authored-By: Claude` neste repo.

---

## Estrutura de arquivos (o que cada arquivo faz)

**Lado repo Python (raiz):**
- `src/churn/serving/api.py` — MODIFICAR: adicionar `CORSMiddleware` liberando as origens locais do Astro.
- `src/churn/reporting.py` — CRIAR: `export_production_metrics(cfg, out_path) -> dict` (reusa `evaluate`) + CLI `python -m churn.reporting`.
- `tests/test_api.py` — MODIFICAR: teste do header CORS.
- `tests/test_reporting.py` — CRIAR: teste do export de métricas.
- `Makefile` — MODIFICAR: alvo `presentation-metrics`.
- `.gitignore` — MODIFICAR: ignorar artefatos de build do site.

**Lado site (`presentation/`):**
- `package.json`, `astro.config.mjs`, `tsconfig.json` — scaffold; `base` configurável.
- `src/styles/global.css` — `@import "tailwindcss"` + tokens do tema claro.
- `src/layouts/Base.astro` — casca comum (fontes, reset, css global).
- `src/pages/index.astro` — deck (12 seções full-screen).
- `src/pages/dashboard.astro` — dashboard.
- `src/components/Slide.astro` — casca de 1 slide.
- `src/components/Deck.astro` + `src/scripts/deck.ts` — navegação/teclas/progresso.
- `src/components/CodeDiff.astro` — antes→depois (expressive-code Dracula).
- `src/components/Mermaid.astro` — 1 diagrama client-side.
- `src/components/VertexOssMap.astro` — mapa Vertex↔OSS interativo.
- `src/components/ModelPlayground.astro` + `src/scripts/playground.ts` — ilha interativa.
- `src/components/RetentionGauge.astro` — gauge 0–10 animado.
- `src/components/MetricCard.astro`, `src/components/ConfusionMatrix.astro`, `src/components/Timeline.astro` — peças do dashboard/métricas.
- `src/data/metrics.json`, `src/data/presets_fallback.json` — dados reais (movidos do staging).
- `src/data/slides.ts`, `src/data/bugs.ts`, `src/data/vertex-map.ts`, `src/data/timeline.ts` — conteúdo.
- `src/scripts/deck.test.mjs`, `src/scripts/playground.test.mjs` — testes de lógica pura.
- `src/assets/mlflow-runs.png`, `src/assets/evidently-drift.png` — prints reais (Task 14).
- `README.md` — nota curta de como rodar o site.

---

## Task 1: CORS no serving/api.py

**Files:**
- Modify: `src/churn/serving/api.py`
- Test: `tests/test_api.py`

- [ ] **Step 1: Escrever o teste falhando**

Adicione ao final de `tests/test_api.py`:

```python
def test_predict_sends_cors_header_for_local_astro_origin(client):
    r = client.post(
        "/predict",
        json=[_VALID],
        headers={"Origin": "http://localhost:4321"},
    )
    assert r.status_code == 200
    assert r.headers["access-control-allow-origin"] == "http://localhost:4321"
```

- [ ] **Step 2: Rodar o teste e ver falhar**

Run (na raiz): `uv run pytest tests/test_api.py::test_predict_sends_cors_header_for_local_astro_origin -v`
Expected: FAIL — `KeyError: 'access-control-allow-origin'` (o middleware ainda não existe).

- [ ] **Step 3: Implementar o middleware**

Em `src/churn/serving/api.py`, adicione o import junto aos outros de fastapi:

```python
from fastapi.middleware.cors import CORSMiddleware
```

E logo após a linha `app = FastAPI(title="Churn scoring API", lifespan=lifespan)`:

```python
# Libera o site de apresentação (Astro em dev) a consumir /predict ao vivo.
# Origens locais fixas — não é uma API pública; o site apresenta-se da própria máquina.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4321", "http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)
```

- [ ] **Step 4: Rodar o teste e ver passar**

Run (na raiz): `uv run pytest tests/test_api.py -v`
Expected: PASS — o novo teste e todos os antigos de `test_api.py` verdes.

- [ ] **Step 5: Commit**

```bash
git add src/churn/serving/api.py tests/test_api.py
git commit -m "feat(serving): allow local Astro origins via CORS for live /predict"
```

---

## Task 2: Export de métricas versionado + alvo de Makefile

**Files:**
- Create: `src/churn/reporting.py`
- Create: `tests/test_reporting.py`
- Modify: `Makefile`

Contexto: hoje `estudo-local/presentation-data/capture_metrics.py` faz isso de forma ad-hoc. Esta tarefa versiona a lógica reusando `evaluate`, exposta como função testável + CLI, e liga ao Makefile. A função reproduz o mesmo split do resto do projeto (`train_test_split(..., test_size=cfg.test_size, random_state=cfg.random_state, stratify=y)`) e avalia o modelo `@production`.

- [ ] **Step 1: Escrever o teste falhando**

Crie `tests/test_reporting.py`:

```python
import json
from pathlib import Path

from churn.config import Settings
from churn.reporting import export_production_metrics
from churn.training.train import train

CSV_PATH = str(Path(__file__).resolve().parents[1] / "Customer-Churn-Records.csv")


def test_export_production_metrics_writes_expected_keys(tmp_path):
    cfg = Settings(
        data_path=CSV_PATH,
        mlflow_tracking_uri=f"sqlite:///{tmp_path}/mlflow.db",
        mlflow_experiment="report-test",
        model_name="churn-model-report-test",
        model_alias="production",
    )
    train(cfg)
    out = tmp_path / "metrics.json"
    metrics = export_production_metrics(cfg, out)

    assert out.exists()
    on_disk = json.loads(out.read_text())
    assert on_disk == metrics
    for key in ("roc_auc", "precision", "recall", "f1", "accuracy", "confusion_matrix"):
        assert key in metrics
    assert len(metrics["confusion_matrix"]) == 2
    assert 0.0 <= metrics["roc_auc"] <= 1.0
```

- [ ] **Step 2: Rodar o teste e ver falhar**

Run (na raiz): `uv run pytest tests/test_reporting.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'churn.reporting'`.

- [ ] **Step 3: Implementar o módulo**

Crie `src/churn/reporting.py`:

```python
"""Exporta as métricas do modelo @production para o site de apresentação.

Versiona o que o helper ad-hoc de staging fazia: reproduz o mesmo split do projeto,
avalia o @production com o `evaluate` canônico (sem duplicar lógica de métrica) e grava
o JSON consumido pelo build do site (`presentation/src/data/metrics.json`).
"""

import json
from pathlib import Path

from sklearn.model_selection import train_test_split

from churn.config import Settings, settings
from churn.data import load_raw
from churn.features.builder import INPUT_COLUMNS
from churn.serving.api import load_production_model
from churn.training.evaluate import evaluate

DEFAULT_OUT = Path("presentation/src/data/metrics.json")


def export_production_metrics(cfg: Settings = settings, out_path: Path = DEFAULT_OUT) -> dict:
    """Avalia o @production no holdout de teste e grava as métricas em `out_path`."""
    model = load_production_model(cfg)
    df = load_raw(cfg.data_path)
    X, y = df[INPUT_COLUMNS], df["turnover"]
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.random_state, stratify=y
    )
    metrics = evaluate(model, X_test, y_test)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    metrics = export_production_metrics()
    print(json.dumps(metrics, indent=2))
    print(f"-> {DEFAULT_OUT}")
```

- [ ] **Step 4: Rodar o teste e ver passar**

Run (na raiz): `uv run pytest tests/test_reporting.py -v`
Expected: PASS.

- [ ] **Step 5: Adicionar o alvo de Makefile**

No `Makefile`, adicione `presentation-metrics` à lista `.PHONY` (edite a linha que começa com `.PHONY:` incluindo o novo alvo) e acrescente o alvo logo após `monitor-drift`:

```makefile
presentation-metrics: ## Exporta métricas do @production para o site (presentation/src/data/metrics.json)
	uv run python -m churn.reporting
```

- [ ] **Step 6: Commit**

```bash
git add src/churn/reporting.py tests/test_reporting.py Makefile
git commit -m "feat(reporting): versioned @production metrics export + make presentation-metrics"
```

---

## Task 3: Scaffold do Astro + Tailwind + Base layout

**Files:**
- Create: `presentation/package.json`, `presentation/astro.config.mjs`, `presentation/tsconfig.json`
- Create: `presentation/src/styles/global.css`
- Create: `presentation/src/layouts/Base.astro`
- Create: `presentation/src/pages/index.astro` (placeholder mínimo desta tarefa)
- Modify: `.gitignore` (raiz)

- [ ] **Step 1: Ignorar artefatos de build do site**

No `.gitignore` da raiz, adicione ao final:

```gitignore
# Site de apresentação (Astro) — artefatos gerados
presentation/node_modules/
presentation/dist/
presentation/.astro/
```

- [ ] **Step 2: Criar `presentation/package.json`**

```json
{
  "name": "churn-presentation",
  "type": "module",
  "version": "0.1.0",
  "private": true,
  "scripts": {
    "dev": "astro dev",
    "build": "astro build",
    "preview": "astro preview",
    "test": "node --test src/scripts"
  },
  "dependencies": {
    "astro": "^5.0.0",
    "astro-expressive-code": "^0.38.0",
    "mermaid": "^11.0.0"
  },
  "devDependencies": {
    "@tailwindcss/vite": "^4.0.0",
    "tailwindcss": "^4.0.0"
  }
}
```

- [ ] **Step 3: Criar `presentation/astro.config.mjs`**

`base` sai de `SITE_BASE` (deploy-ready; default `/` para rodar local sem prefixo). Expressive-code configurado com tema Dracula e marcação de linha.

```js
import { defineConfig } from "astro/config";
import tailwindcss from "@tailwindcss/vite";
import expressiveCode from "astro-expressive-code";

// Deploy-ready: em GitHub Pages/Vercel defina SITE_BASE (ex.: "/churn/"); local fica "/".
const base = process.env.SITE_BASE ?? "/";

export default defineConfig({
  base,
  integrations: [
    expressiveCode({
      themes: ["dracula"],
      styleOverrides: { borderRadius: "0.5rem" },
    }),
  ],
  vite: { plugins: [tailwindcss()] },
});
```

- [ ] **Step 4: Criar `presentation/tsconfig.json`**

```json
{
  "extends": "astro/tsconfigs/strict",
  "include": [".astro/types.d.ts", "**/*"],
  "exclude": ["dist"]
}
```

- [ ] **Step 5: Criar `presentation/src/styles/global.css`**

```css
@import "tailwindcss";

:root {
  --ink: #1a1a1a;
  --muted: #6b7280;
  --accent: #2563eb;
  --bg: #ffffff;
}

html {
  scroll-behavior: smooth;
}

body {
  color: var(--ink);
  background: var(--bg);
  font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
  -webkit-font-smoothing: antialiased;
}

@media (prefers-reduced-motion: reduce) {
  html {
    scroll-behavior: auto;
  }
}
```

- [ ] **Step 6: Criar `presentation/src/layouts/Base.astro`**

```astro
---
import "../styles/global.css";
interface Props {
  title?: string;
}
const { title = "Churn MLOps — apresentação" } = Astro.props;
---

<!doctype html>
<html lang="pt-BR">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>{title}</title>
  </head>
  <body>
    <slot />
  </body>
</html>
```

- [ ] **Step 7: Criar `presentation/src/pages/index.astro` (placeholder)**

```astro
---
import Base from "../layouts/Base.astro";
---

<Base>
  <main class="grid min-h-screen place-items-center">
    <h1 class="text-4xl font-semibold tracking-tight">Churn MLOps</h1>
  </main>
</Base>
```

- [ ] **Step 8: Instalar e verificar o build**

Run (dentro de `presentation/`): `npm install && npm run build`
Expected: instala as deps e gera `dist/` sem erro (linha final típica: "Complete!").

- [ ] **Step 9: Commit**

```bash
git add .gitignore presentation/package.json presentation/package-lock.json presentation/astro.config.mjs presentation/tsconfig.json presentation/src/styles/global.css presentation/src/layouts/Base.astro presentation/src/pages/index.astro
git commit -m "feat(presentation): scaffold Astro + Tailwind v4 + expressive-code"
```

---

## Task 4: Dados reais + conteúdo dos slides

**Files:**
- Create: `presentation/src/data/metrics.json` (mover do staging)
- Create: `presentation/src/data/presets_fallback.json` (mover do staging)
- Create: `presentation/src/data/slides.ts`
- Create: `presentation/src/data/bugs.ts`
- Create: `presentation/src/data/vertex-map.ts`
- Create: `presentation/src/data/timeline.ts`

- [ ] **Step 1: Mover os dados capturados do staging**

Run (na raiz):

```bash
mkdir -p presentation/src/data
cp estudo-local/presentation-data/metrics.json presentation/src/data/metrics.json
cp estudo-local/presentation-data/presets_fallback.json presentation/src/data/presets_fallback.json
```

Confira que `presentation/src/data/metrics.json` contém `"roc_auc": 0.7643640999113213` e `"confusion_matrix": [[1043, 549], [106, 302]]`.

- [ ] **Step 2: Criar `presentation/src/data/slides.ts`**

Tipos + conteúdo textual e speaker notes dos 12 slides. `track5` marca a trilha de 5 min.

```ts
export interface SlideMeta {
  id: number;
  kicker?: string;
  title: string;
  subtitle?: string;
  track5: boolean;
  notes: string;
}

export const SLIDES: SlideMeta[] = [
  { id: 1, kicker: "Churn MLOps", title: "De 2 scripts frágeis a uma esteira de produção", subtitle: "Produtização do modelo de turnover — teste técnico Bulla", track5: true, notes: "Abertura. Uma frase: peguei dois scripts legados e transformei numa esteira MLOps local que espelha o Vertex AI." },
  { id: 2, kicker: "O problema", title: "O que a Bulla pediu", subtitle: "Duas restrições: não trocar o modelo, rodar 100% local", track5: true, notes: "Produtizar o modelo existente. Restrição 1: manter a regressão logística. Restrição 2: espelho local do Vertex, sem nuvem paga." },
  { id: 3, kicker: "A grande sacada", title: "Cada peça do Vertex tem um par open-source", subtitle: "Passe o mouse para acender o par; clique para ir ao slide", track5: true, notes: "O mapa é o coração da estratégia: KFP=Pipelines, MLflow=Experiments+Registry, Feast=Feature Store, Evidently=Monitoring, FastAPI+Docker=Endpoint, Actions/Jenkins=CI/CD." },
  { id: 4, kicker: "A auditoria", title: "Achamos 10 bugs nos scripts originais", subtitle: "Cada um virou uma decisão de arquitetura", track5: false, notes: "Transição. Os 10 problemas conceituais não são teóricos — cada correção é uma peça concreta da esteira." },
  { id: 5, kicker: "Bugs antes→depois", title: "Do bug à correção, em código", subtitle: "Use ←/→ para navegar pelos 10", track5: true, notes: "Coração técnico. Foque no bug #1 (leakage) e #5 (métrica enganosa) se o tempo apertar. O contador mostra N/10." },
  { id: 6, kicker: "A arquitetura", title: "O ciclo de vida completo", subtitle: "Treino → registro → serving → monitoramento → re-treino", track5: false, notes: "Diagrama Mermaid. Mostra o loop fechado: o gate de drift dispara o re-treino." },
  { id: 7, kicker: "Modelo ao vivo", title: "Fale com o modelo real", subtitle: "3 perfis, 4 ajustes, score 0–10 na hora", track5: true, notes: "Demo. Clique nos presets. Se a API estiver no ar, os sliders recalculam ao vivo; senão, cai no fallback pré-gravado real. Nunca quebra." },
  { id: 8, kicker: "Métricas honestas", title: "Accuracy engana; recall é o que importa", subtitle: "Base ~20% churn — 0.67 de accuracy esconde o jogo", track5: false, notes: "Resolve o bug #5. AUC 0.76, recall 0.74. A matriz mostra: pegamos 302 de 408 que saem (recall), ao custo de falsos positivos — trade-off consciente com class_weight=balanced." },
  { id: 9, kicker: "O momento do drift", title: "O gate que falha de propósito", subtitle: "make monitor (PASS) vs make monitor-drift (ALERT, exit 2)", track5: true, notes: "Clímax. Saudável: drift_share 0.000, exit 0. Drift simulado: drift_share 0.462, 6 colunas, exit 2 — e a qualidade despenca (accuracy 0.23, recall 0.99 = prevê que todos saem). É o gatilho de re-treino." },
  { id: 10, kicker: "O que foi construído", title: "9 marcos, 78 testes, CI verde", subtitle: "Cada marco: issue → branch → PR → merge", track5: false, notes: "Timeline dos 9 marcos entregues. Enfatize disciplina: tudo por PR, testado, CI espelhando o pipeline." },
  { id: 11, kicker: "Limitações honestas", title: "O que ficou documentado, não construído", subtitle: "K8s, captura de predições, scheduler gerenciado", track5: false, notes: "Honestidade: espelho local. A API não captura predições (holdout como proxy); sem scheduler gerenciado; K8s descrito como próximo passo, não construído." },
  { id: 12, kicker: "Fecho", title: "De frágil a confiável", subtitle: "Esteira reprodutível, testada e observável — pronta para a nuvem", track5: true, notes: "Recap: mapa Vertex↔OSS, 10 bugs corrigidos, loop de monitoramento fechado. Próximo passo: subir os equivalentes gerenciados no Vertex de verdade." },
];
```

- [ ] **Step 3: Criar `presentation/src/data/bugs.ts`**

Os 10 bugs com trechos antes→depois (do README §"Problemas conceituais"). `badLine`/`goodLine` marcam a linha-chave para o CodeDiff destacar.

```ts
export interface Bug {
  n: number;
  title: string;
  problem: string;
  fix: string;
  lang: string;
  before: string;
  after: string;
  badLine: number;
  goodLine: number;
}

export const BUGS: Bug[] = [
  {
    n: 1,
    title: "Target leakage na taxa por geografia",
    problem: "geography_churn_rate calculado com o turnover do dataset inteiro, antes do split.",
    fix: "Feature aprendida só no treino, materializada e servida via Feast.",
    lang: "python",
    before: "rate = df.groupby('Geography')['turnover'].mean()\ndf['geo_churn'] = df['Geography'].map(rate)\nX_train, X_test = train_test_split(df)",
    after: "X_train, X_test = train_test_split(df, stratify=y)\n# rate aprendida SÓ no treino, dentro do Pipeline\nbuilder.fit(X_train, y_train)  # Feast serve o mesmo valor",
    badLine: 1,
    goodLine: 3,
  },
  {
    n: 2,
    title: "Train/serve skew",
    problem: "StandardScaler/LabelEncoder/qcut re-ajustados na inferência; só model.pkl salvo.",
    fix: "Tudo dentro de um sklearn.Pipeline persistido; nada re-ajustado.",
    lang: "python",
    before: "scaler = StandardScaler().fit(X_infer)  # re-fit no serving!\nX = scaler.transform(X_infer)\npred = model.predict(X)",
    after: "pipe = mlflow.sklearn.load_model('models:/churn@production')\npred = pipe.predict(X_infer)  # scaler já fitado no treino",
    badLine: 1,
    goodLine: 2,
  },
  {
    n: 3,
    title: "CustomerId como feature",
    problem: "Identificador sem poder preditivo entrando no modelo.",
    fix: "Removido das features (nunca selecionado).",
    lang: "python",
    before: "FEATURES = ['CustomerId', 'CreditScore', 'Age', ...]\nX = df[FEATURES]",
    after: "INPUT_COLUMNS = RAW_NUMERIC + RAW_CATEGORICAL  # sem CustomerId\nX = df[INPUT_COLUMNS]",
    badLine: 1,
    goodLine: 1,
  },
  {
    n: 4,
    title: "surname_encoded",
    problem: "LabelEncoder no sobrenome (altíssima cardinalidade), re-ajustado com ordem diferente.",
    fix: "Removido das features.",
    lang: "python",
    before: "df['surname_encoded'] = LabelEncoder().fit_transform(df['Surname'])",
    after: "# Surname nunca é selecionado — ruído de alta cardinalidade fora",
    badLine: 1,
    goodLine: 1,
  },
  {
    n: 5,
    title: "Métrica enganosa",
    problem: "accuracy numa base ~20% de churn, sem AUC/precision/recall.",
    fix: "AUC + precision + recall + F1 + matriz de confusão.",
    lang: "python",
    before: "print('accuracy:', accuracy_score(y_test, pred))  # 0.67, engana",
    after: "metrics = evaluate(model, X_test, y_test)\n# roc_auc, precision, recall, f1, confusion_matrix",
    badLine: 1,
    goodLine: 1,
  },
  {
    n: 6,
    title: "Split não reprodutível",
    problem: "train_test_split sem random_state nem stratify — não reproduz, não preserva classes.",
    fix: "random_state=42, stratify=y.",
    lang: "python",
    before: "X_train, X_test, y_train, y_test = train_test_split(X, y)",
    after: "X_train, X_test, y_train, y_test = train_test_split(\n    X, y, test_size=0.2, random_state=42, stratify=y)",
    badLine: 1,
    goodLine: 2,
  },
  {
    n: 7,
    title: "churn_rate_por_uf hardcoded",
    problem: "Dicionário fixo na inferência, com códigos que não batem com o encoder re-ajustado.",
    fix: "Substituído pela feature servida do Feast.",
    lang: "python",
    before: "CHURN_UF = {'SP': 0.18, 'RJ': 0.22, ...}  # fixo, desatualiza\nrate = CHURN_UF[uf]",
    after: "rate = get_geography_churn_rate([geo])[geo]  # Feast online",
    badLine: 1,
    goodLine: 1,
  },
  {
    n: 8,
    title: "model.pkl via pickle solto",
    problem: "Sem versão, metadados, assinatura ou estágio.",
    fix: "MLflow Model Registry com assinatura e alias @production.",
    lang: "python",
    before: "pickle.dump(model, open('model.pkl', 'wb'))",
    after: "mlflow.sklearn.log_model(pipe, name='model', signature=sig,\n    registered_model_name='churn-model')  # -> @production",
    badLine: 1,
    goodLine: 1,
  },
  {
    n: 9,
    title: "Score de retenção duplicado",
    problem: "Calculado no treino e descartado; regra duplicada entre treino e inferência.",
    fix: "Função única e testada em scoring.py.",
    lang: "python",
    before: "# treino: score = round(p_stay*10)\n# serving: score = int(p*10)  # regra diferente!",
    after: "from churn.scoring import retention_score\nscore = retention_score(proba[:, 0])  # fonte única, testada",
    badLine: 2,
    goodLine: 2,
  },
  {
    n: 10,
    title: "Sem Pipeline, sem schema, sem testes",
    problem: "Pré-processo e modelo soltos; sem validação de schema; sem testes.",
    fix: "KFP + Pandera/Pydantic + pytest.",
    lang: "python",
    before: "X = preprocess(df)      # função solta\nmodel.fit(X, y)         # sem validação, sem teste",
    after: "pipe = build_pipeline()  # ChurnFeatureBuilder + scaler + LR\n# Pandera valida o schema; 78 testes cobrem o fluxo",
    badLine: 1,
    goodLine: 1,
  },
];
```

- [ ] **Step 4: Criar `presentation/src/data/vertex-map.ts`**

```ts
export interface VertexPair {
  vertex: string;
  oss: string;
  role: string;
  slide?: number;
}

export const VERTEX_MAP: VertexPair[] = [
  { vertex: "Vertex Pipelines", oss: "KFP SDK", role: "Orquestra as etapas do treino", slide: 6 },
  { vertex: "Vertex AI Experiments", oss: "MLflow Tracking", role: "Rastreia params/métricas/artefatos", slide: 10 },
  { vertex: "Vertex Model Registry", oss: "MLflow Model Registry", role: "Versiona e promove modelos (@production)", slide: 8 },
  { vertex: "Vertex AI Feature Store", oss: "Feast", role: "Serve features offline/online sem skew", slide: 5 },
  { vertex: "Vertex Model Monitoring", oss: "Evidently AI", role: "Detecta data/target/score drift", slide: 9 },
  { vertex: "Vertex Endpoint", oss: "FastAPI + Docker", role: "Serve predições REST", slide: 7 },
  { vertex: "Cloud Build / Vertex CI", oss: "GitHub Actions + Jenkinsfile", role: "CI/CD", slide: 10 },
];
```

- [ ] **Step 5: Criar `presentation/src/data/timeline.ts`**

```ts
export interface Milestone {
  marco: number;
  title: string;
  detail: string;
}

export const TIMELINE: Milestone[] = [
  { marco: 1, title: "Fundação + auditoria", detail: "Estrutura do repo, os 10 bugs mapeados" },
  { marco: 2, title: "Features sem leakage", detail: "ChurnFeatureBuilder (transformer sklearn)" },
  { marco: 3, title: "Treino + Pipeline", detail: "sklearn.Pipeline persistível, métricas honestas" },
  { marco: 4, title: "MLflow Tracking + Registry", detail: "Params/métricas + alias @production com gate" },
  { marco: 5, title: "Feast Feature Store", detail: "Feature de geografia offline→online sem skew" },
  { marco: 6, title: "Orquestração KFP", detail: "DAG prepare→split→train→evaluate→register" },
  { marco: 7, title: "Serving FastAPI + Docker", detail: "/predict, /health, score 0–10; imagem Docker" },
  { marco: 8, title: "Monitoramento Evidently", detail: "Drift + quality gate (exit≠0 dispara re-treino)" },
  { marco: 9, title: "CI/CD + docs", detail: "GitHub Actions + Jenkinsfile, README final, 78 testes" },
];
```

- [ ] **Step 6: Verificar o build**

Run (dentro de `presentation/`): `npm run build`
Expected: build sem erro (os `.ts`/`.json` compilam; nada os consome ainda, mas o TS estrito valida a sintaxe quando importados nas próximas tarefas — aqui só garantimos que o projeto ainda builda).

- [ ] **Step 7: Commit**

```bash
git add presentation/src/data/
git commit -m "feat(presentation): real captured data + slide/bug/map/timeline content"
```

---

## Task 5: Deck — Slide, Deck, deck.ts e as 12 seções

**Files:**
- Create: `presentation/src/components/Slide.astro`
- Create: `presentation/src/scripts/deck.ts`
- Create: `presentation/src/scripts/deck.test.mjs`
- Create: `presentation/src/components/Deck.astro`
- Modify: `presentation/src/pages/index.astro` (montar as 12 seções)

- [ ] **Step 1: Criar `presentation/src/components/Slide.astro`**

Casca de 1 slide: branco, full-screen, centralizado, com kicker/título/subtítulo e um `<slot />` para o conteúdo rico. Notas ficam num `<template data-notes>` (não renderizado; o deck.ts as revela sob tecla `S`).

```astro
---
import type { SlideMeta } from "../data/slides";
interface Props {
  meta: SlideMeta;
}
const { meta } = Astro.props;
---

<section
  id={`slide-${meta.id}`}
  class="slide relative flex min-h-screen w-full snap-start flex-col items-center justify-center px-8 py-16 text-center"
  data-track5={meta.track5 ? "1" : "0"}
>
  {meta.kicker && (
    <p class="mb-4 text-sm font-medium uppercase tracking-widest text-[var(--muted)]">
      {meta.kicker}
    </p>
  )}
  <h2 class="max-w-4xl text-4xl font-semibold tracking-tight sm:text-5xl">
    {meta.title}
  </h2>
  {meta.subtitle && (
    <p class="mt-5 max-w-2xl text-lg text-[var(--muted)]">{meta.subtitle}</p>
  )}
  <div class="mt-10 w-full max-w-5xl">
    <slot />
  </div>
  <template data-notes set:html={meta.notes} />
</section>
```

- [ ] **Step 2: Escrever o teste da lógica pura do deck**

O `deck.ts` exporta helpers puros (cálculo de índice) testáveis sem DOM. Crie `presentation/src/scripts/deck.test.mjs`:

```js
import { test } from "node:test";
import assert from "node:assert/strict";
import { clampIndex, nextIndex } from "./deck.mjs";

test("clampIndex mantém dentro dos limites", () => {
  assert.equal(clampIndex(-1, 12), 0);
  assert.equal(clampIndex(99, 12), 11);
  assert.equal(clampIndex(5, 12), 5);
});

test("nextIndex anda e satura nas pontas", () => {
  assert.equal(nextIndex(0, 1, 12), 1);
  assert.equal(nextIndex(11, 1, 12), 11);
  assert.equal(nextIndex(0, -1, 12), 0);
});
```

Nota: o teste importa `./deck.mjs`. Para manter uma fonte única de lógica testável sem toolchain de bundling, extraia os helpers puros num arquivo `.mjs` separado e o `deck.ts` os reexporta/usa.

- [ ] **Step 3: Rodar o teste e ver falhar**

Run (dentro de `presentation/`): `node --test src/scripts/deck.test.mjs`
Expected: FAIL — `Cannot find module './deck.mjs'`.

- [ ] **Step 4: Criar `presentation/src/scripts/deck.mjs` (helpers puros)**

```js
export function clampIndex(i, total) {
  if (i < 0) return 0;
  if (i > total - 1) return total - 1;
  return i;
}

export function nextIndex(current, delta, total) {
  return clampIndex(current + delta, total);
}
```

- [ ] **Step 5: Rodar o teste e ver passar**

Run (dentro de `presentation/`): `node --test src/scripts/deck.test.mjs`
Expected: PASS (2 testes).

- [ ] **Step 6: Criar `presentation/src/scripts/deck.ts` (comportamento no cliente)**

Usa os helpers puros e liga teclado/progresso/hash/teclas S/B/D. Respeita `prefers-reduced-motion` deixando o scroll instantâneo.

```ts
import { clampIndex, nextIndex } from "./deck.mjs";

function initDeck() {
  const slides = Array.from(document.querySelectorAll<HTMLElement>(".slide"));
  const total = slides.length;
  if (total === 0) return;

  const dotsWrap = document.getElementById("deck-dots");
  const counter = document.getElementById("deck-counter");
  const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  let current = 0;
  let notesOn = false;

  // bolinhas de progresso
  if (dotsWrap) {
    slides.forEach((_, i) => {
      const dot = document.createElement("button");
      dot.className = "deck-dot h-2 w-2 rounded-full bg-black/20 transition";
      dot.setAttribute("aria-label", `Ir ao slide ${i + 1}`);
      dot.addEventListener("click", () => go(i));
      dotsWrap.appendChild(dot);
    });
  }

  function paint() {
    if (counter) counter.textContent = `${current + 1} / ${total}`;
    dotsWrap?.querySelectorAll(".deck-dot").forEach((d, i) => {
      d.classList.toggle("bg-black/20", i !== current);
      d.classList.toggle("bg-[var(--accent)]", i === current);
    });
    location.hash = `slide-${current + 1}`;
  }

  function go(i: number) {
    current = clampIndex(i, total);
    slides[current].scrollIntoView({ behavior: reduce ? "auto" : "smooth" });
    paint();
  }

  function toggleNotes() {
    notesOn = !notesOn;
    slides.forEach((s) => {
      const tpl = s.querySelector<HTMLTemplateElement>("template[data-notes]");
      let box = s.querySelector<HTMLElement>(".deck-notes");
      if (notesOn && tpl && !box) {
        box = document.createElement("div");
        box.className =
          "deck-notes absolute bottom-24 left-1/2 -translate-x-1/2 max-w-2xl rounded-lg bg-black/85 px-5 py-3 text-sm text-white";
        box.innerHTML = tpl.innerHTML;
        s.appendChild(box);
      } else if (!notesOn && box) {
        box.remove();
      }
    });
  }

  function toggleBlackout() {
    document.body.classList.toggle("deck-blackout");
  }

  document.addEventListener("keydown", (e) => {
    if (e.key === "ArrowRight" || e.key === " ") {
      e.preventDefault();
      go(nextIndex(current, 1, total));
    } else if (e.key === "ArrowLeft") {
      e.preventDefault();
      go(nextIndex(current, -1, total));
    } else if (e.key.toLowerCase() === "s") {
      toggleNotes();
    } else if (e.key.toLowerCase() === "b") {
      toggleBlackout();
    } else if (e.key.toLowerCase() === "d") {
      window.location.href = import.meta.env.BASE_URL + "dashboard";
    }
  });

  // deep-link inicial (#slide-7)
  const m = location.hash.match(/slide-(\d+)/);
  if (m) current = clampIndex(Number(m[1]) - 1, total);
  go(current);
}

if (document.readyState !== "loading") initDeck();
else document.addEventListener("DOMContentLoaded", initDeck);
```

- [ ] **Step 7: Criar `presentation/src/components/Deck.astro`**

Casca do deck: container com scroll-snap, rodapé fixo (contador + bolinhas + dica de atalhos), estilo de blackout, e o script.

```astro
---
---

<div class="deck snap-y snap-mandatory">
  <slot />
</div>

<footer
  class="fixed inset-x-0 bottom-0 z-10 flex items-center justify-between px-6 py-4 text-xs text-[var(--muted)]"
>
  <span id="deck-counter">1 / 12</span>
  <div id="deck-dots" class="flex items-center gap-2"></div>
  <span class="hidden sm:inline">←/→ navegar · S notas · B blackout · D dashboard</span>
</footer>

<style is:global>
  body.deck-blackout > *:not(script) {
    visibility: hidden;
  }
  body.deck-blackout {
    background: #000;
  }
</style>

<script>
  import "../scripts/deck.ts";
</script>
```

- [ ] **Step 8: Montar as 12 seções em `presentation/src/pages/index.astro`**

Substitua o placeholder. Cada slide usa `Slide` com seu `meta`; o conteúdo rico entra nas próximas tarefas (por ora, placeholders textuais mínimos para os slides que terão componentes).

```astro
---
import Base from "../layouts/Base.astro";
import Deck from "../components/Deck.astro";
import Slide from "../components/Slide.astro";
import { SLIDES } from "../data/slides";
---

<Base>
  <Deck>
    {SLIDES.map((meta) => <Slide meta={meta} />)}
  </Deck>
</Base>
```

- [ ] **Step 9: Verificar build + navegação**

Run (dentro de `presentation/`): `npm run build && npm run test`
Expected: build sem erro; `node --test` verde.
Manual (opcional agora): `npm run dev`, abrir `http://localhost:4321`, apertar `→` várias vezes (avança), `S` (mostra nota), `B` (blackout), `D` (tenta ir ao dashboard — 404 até a Task 13, ok).

- [ ] **Step 10: Commit**

```bash
git add presentation/src/components/Slide.astro presentation/src/components/Deck.astro presentation/src/scripts/ presentation/src/pages/index.astro
git commit -m "feat(presentation): keyboard-driven deck with 12 slide shells"
```

---

## Task 6: CodeDiff + slide 5 (os 10 bugs)

**Files:**
- Create: `presentation/src/components/CodeDiff.astro`
- Modify: `presentation/src/pages/index.astro` (injetar no slide 5)

- [ ] **Step 1: Criar `presentation/src/components/CodeDiff.astro`**

Navega pelos 10 bugs (client-side); cada bug mostra "Antes" e "Depois" com expressive-code (tema Dracula) e a linha-chave destacada via meta `{n}`. O componente renderiza todos os bugs e alterna a visibilidade.

```astro
---
import { Code } from "astro-expressive-code/components";
import { BUGS } from "../data/bugs";
---

<div class="codediff text-left">
  <div class="mb-4 flex items-center justify-between">
    <span id="cd-counter" class="text-sm font-medium text-[var(--muted)]">1 / 10</span>
    <div class="flex gap-2">
      <button id="cd-prev" class="rounded-md border px-3 py-1 text-sm">← Anterior</button>
      <button id="cd-next" class="rounded-md border px-3 py-1 text-sm">Próximo →</button>
    </div>
  </div>

  {BUGS.map((bug, i) => (
    <div class="cd-item" data-index={i} hidden={i !== 0}>
      <h3 class="mb-1 text-xl font-semibold">
        Bug #{bug.n} — {bug.title}
      </h3>
      <p class="mb-1 text-sm text-red-600">Problema: {bug.problem}</p>
      <p class="mb-4 text-sm text-emerald-600">Correção: {bug.fix}</p>
      <div class="grid gap-4 md:grid-cols-2">
        <div>
          <p class="mb-1 text-xs uppercase tracking-widest text-[var(--muted)]">Antes</p>
          <Code code={bug.before} lang={bug.lang} meta={`{${bug.badLine}}`} />
        </div>
        <div>
          <p class="mb-1 text-xs uppercase tracking-widest text-[var(--muted)]">Depois</p>
          <Code code={bug.after} lang={bug.lang} meta={`{${bug.goodLine}}`} />
        </div>
      </div>
    </div>
  ))}
</div>

<script>
  function initCodeDiff() {
    const items = Array.from(document.querySelectorAll<HTMLElement>(".cd-item"));
    if (items.length === 0) return;
    const counter = document.getElementById("cd-counter");
    let i = 0;
    function show(n: number) {
      i = (n + items.length) % items.length;
      items.forEach((el, k) => (el.hidden = k !== i));
      if (counter) counter.textContent = `${i + 1} / ${items.length}`;
    }
    document.getElementById("cd-prev")?.addEventListener("click", () => show(i - 1));
    document.getElementById("cd-next")?.addEventListener("click", () => show(i + 1));
    show(0);
  }
  if (document.readyState !== "loading") initCodeDiff();
  else document.addEventListener("DOMContentLoaded", initCodeDiff);
</script>
```

- [ ] **Step 2: Injetar no slide 5**

Em `index.astro`, troque o `.map` simples por um render que insere o `CodeDiff` como filho do Slide 5. Substitua o bloco `<Deck>...</Deck>` por:

```astro
---
import Base from "../layouts/Base.astro";
import Deck from "../components/Deck.astro";
import Slide from "../components/Slide.astro";
import CodeDiff from "../components/CodeDiff.astro";
import { SLIDES } from "../data/slides";

const meta = (id: number) => SLIDES.find((s) => s.id === id)!;
---

<Base>
  <Deck>
    <Slide meta={meta(1)} />
    <Slide meta={meta(2)} />
    <Slide meta={meta(3)} />
    <Slide meta={meta(4)} />
    <Slide meta={meta(5)}><CodeDiff /></Slide>
    <Slide meta={meta(6)} />
    <Slide meta={meta(7)} />
    <Slide meta={meta(8)} />
    <Slide meta={meta(9)} />
    <Slide meta={meta(10)} />
    <Slide meta={meta(11)} />
    <Slide meta={meta(12)} />
  </Deck>
</Base>
```

Nota: as próximas tarefas preenchem os demais `<Slide meta={meta(N)}>` com seus componentes, seguindo este mesmo padrão.

- [ ] **Step 3: Verificar build**

Run (dentro de `presentation/`): `npm run build`
Expected: build sem erro; expressive-code gera os blocos Dracula.
Manual: `npm run dev` → slide 5 → botões "Próximo/Anterior" trocam os 10 bugs; a linha destacada aparece.

- [ ] **Step 4: Commit**

```bash
git add presentation/src/components/CodeDiff.astro presentation/src/pages/index.astro
git commit -m "feat(presentation): CodeDiff before/after Dracula for the 10 bugs (slide 5)"
```

---

## Task 7: VertexOssMap + slide 3

**Files:**
- Create: `presentation/src/components/VertexOssMap.astro`
- Modify: `presentation/src/pages/index.astro` (slide 3)

- [ ] **Step 1: Criar `presentation/src/components/VertexOssMap.astro`**

Tabela de pares; hover acende o par (linha destacada) e clique navega pro slide da peça via hash.

```astro
---
import { VERTEX_MAP } from "../data/vertex-map";
---

<div class="vmap mx-auto max-w-3xl text-left">
  {VERTEX_MAP.map((p) => (
    <a
      href={p.slide ? `#slide-${p.slide}` : "#"}
      class="vmap-row grid grid-cols-[1fr_auto_1fr] items-center gap-3 rounded-lg px-4 py-3 transition hover:bg-black/5"
    >
      <span class="font-medium">{p.vertex}</span>
      <span class="text-[var(--muted)]">→</span>
      <span class="font-semibold text-[var(--accent)]">{p.oss}</span>
      <span class="col-span-3 text-sm text-[var(--muted)]">{p.role}</span>
    </a>
  ))}
</div>
```

- [ ] **Step 2: Injetar no slide 3**

Em `index.astro`, importe `VertexOssMap` e troque `<Slide meta={meta(3)} />` por `<Slide meta={meta(3)}><VertexOssMap /></Slide>`.

- [ ] **Step 3: Verificar build**

Run (dentro de `presentation/`): `npm run build`
Expected: sem erro.
Manual: slide 3 → hover acende a linha; clicar num par pula pro slide correspondente (`#slide-7` etc.).

- [ ] **Step 4: Commit**

```bash
git add presentation/src/components/VertexOssMap.astro presentation/src/pages/index.astro
git commit -m "feat(presentation): interactive Vertex↔OSS map (slide 3)"
```

---

## Task 8: Mermaid + slide 6

**Files:**
- Create: `presentation/src/components/Mermaid.astro`
- Modify: `presentation/src/pages/index.astro` (slide 6)

- [ ] **Step 1: Criar `presentation/src/components/Mermaid.astro`**

Renderiza 1 diagrama no cliente (evita dor de build). O grafo do ciclo de vida vem como prop `chart` com default embutido.

```astro
---
interface Props {
  chart?: string;
}
const defaultChart = `flowchart LR
  A[CSV validado<br/>Pandera] --> B[Split<br/>stratify+seed]
  B --> C[Feast<br/>feature sem leakage]
  C --> D[Pipeline sklearn<br/>fit no treino]
  D --> E[MLflow<br/>tracking + registry]
  E -->|gate roc_auc>=0.70| F[@production]
  F --> G[FastAPI /predict<br/>score 0-10]
  G --> H[Evidently<br/>drift + quality gate]
  H -->|exit != 0| B`;
const { chart = defaultChart } = Astro.props;
---

<div class="mermaid-wrap flex justify-center">
  <pre class="mermaid" set:html={chart} />
</div>

<script>
  import mermaid from "mermaid";
  const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  mermaid.initialize({
    startOnLoad: true,
    theme: "neutral",
    flowchart: { curve: reduce ? "linear" : "basis" },
  });
</script>
```

- [ ] **Step 2: Injetar no slide 6**

Em `index.astro`, importe `Mermaid` e troque `<Slide meta={meta(6)} />` por `<Slide meta={meta(6)}><Mermaid /></Slide>`.

- [ ] **Step 3: Verificar build**

Run (dentro de `presentation/`): `npm run build`
Expected: sem erro (o mermaid roda no cliente, não no build).
Manual: slide 6 → o fluxograma do ciclo de vida renderiza, com o loop `Evidently → Split`.

- [ ] **Step 4: Commit**

```bash
git add presentation/src/components/Mermaid.astro presentation/src/pages/index.astro
git commit -m "feat(presentation): client-side Mermaid lifecycle diagram (slide 6)"
```

---

## Task 9: RetentionGauge (gauge 0–10 animado)

**Files:**
- Create: `presentation/src/components/RetentionGauge.astro`

- [ ] **Step 1: Criar `presentation/src/components/RetentionGauge.astro`**

Gauge SVG semicircular 0–10. Recebe `score` inicial e expõe um id para o playground atualizar. Anima o arco via CSS transition; sob `prefers-reduced-motion` a transition é removida.

```astro
---
interface Props {
  score?: number;
  id?: string;
}
const { score = 0, id = "gauge" } = Astro.props;
// semicírculo: raio 90, comprimento do arco = π*r ≈ 282.7
const ARC = 282.74;
const initialOffset = ARC * (1 - score / 10);
---

<div class="gauge inline-flex flex-col items-center" data-arc={ARC}>
  <svg width="220" height="130" viewBox="0 0 220 130">
    <path
      d="M 20 120 A 90 90 0 0 1 200 120"
      fill="none"
      stroke="#e5e7eb"
      stroke-width="16"
      stroke-linecap="round"
    />
    <path
      id={`${id}-arc`}
      d="M 20 120 A 90 90 0 0 1 200 120"
      fill="none"
      stroke="var(--accent)"
      stroke-width="16"
      stroke-linecap="round"
      stroke-dasharray={ARC}
      stroke-dashoffset={initialOffset}
      class="gauge-arc"
    />
  </svg>
  <div class="-mt-8 text-center">
    <span id={`${id}-value`} class="text-4xl font-semibold">{score}</span>
    <span class="text-[var(--muted)]">/10</span>
  </div>
</div>

<style>
  .gauge-arc {
    transition: stroke-dashoffset 700ms ease;
  }
  @media (prefers-reduced-motion: reduce) {
    .gauge-arc {
      transition: none;
    }
  }
</style>
```

- [ ] **Step 2: Verificar build**

Run (dentro de `presentation/`): `npm run build`
Expected: sem erro (o componente ainda não é consumido; a Task 10 e a 13 o usam).

- [ ] **Step 3: Commit**

```bash
git add presentation/src/components/RetentionGauge.astro
git commit -m "feat(presentation): animated 0-10 retention gauge (reduced-motion aware)"
```

---

## Task 10: ModelPlayground + slide 7

**Files:**
- Create: `presentation/src/scripts/playground.ts`
- Create: `presentation/src/scripts/playground.mjs` (resolvedor puro)
- Create: `presentation/src/scripts/playground.test.mjs`
- Create: `presentation/src/components/ModelPlayground.astro`
- Modify: `presentation/src/pages/index.astro` (slide 7)

- [ ] **Step 1: Escrever o teste do resolvedor puro**

O playground tem uma decisão pura: dado o preset seleccionado e se a API respondeu, qual resultado exibir (resposta ao vivo vs fallback pré-gravado). Crie `presentation/src/scripts/playground.test.mjs`:

```js
import { test } from "node:test";
import assert from "node:assert/strict";
import { pickResult } from "./playground.mjs";

const preset = { response: { turnover_pred: 1, prob_churn: 0.55, score_retencao: 5 } };

test("usa a resposta ao vivo quando a API responde", () => {
  const live = { turnover_pred: 0, prob_churn: 0.2, score_retencao: 8 };
  const r = pickResult(preset, live);
  assert.deepEqual(r, { data: live, live: true });
});

test("cai no fallback pré-gravado quando não há resposta ao vivo", () => {
  const r = pickResult(preset, null);
  assert.deepEqual(r, { data: preset.response, live: false });
});
```

- [ ] **Step 2: Rodar o teste e ver falhar**

Run (dentro de `presentation/`): `node --test src/scripts/playground.test.mjs`
Expected: FAIL — `Cannot find module './playground.mjs'`.

- [ ] **Step 3: Criar `presentation/src/scripts/playground.mjs`**

```js
// Decisão pura: resposta ao vivo tem prioridade; senão, fallback pré-gravado do preset.
export function pickResult(preset, liveResponse) {
  if (liveResponse) return { data: liveResponse, live: true };
  return { data: preset.response, live: false };
}
```

- [ ] **Step 4: Rodar o teste e ver passar**

Run (dentro de `presentation/`): `node --test src/scripts/playground.test.mjs`
Expected: PASS (2 testes).

- [ ] **Step 5: Criar `presentation/src/scripts/playground.ts`**

Liga presets, sliders, fetch ao `/predict` e o gauge. Ao selecionar preset: preenche sliders e tenta `fetch`; se falhar, usa fallback e mostra o selo. Mexer nos sliders com API no ar re-faz o fetch (debounced); sem API, mantém o resultado do preset e exibe "resposta pré-gravada".

```ts
import { pickResult } from "./playground.mjs";

interface PresetResponse {
  turnover_pred: number;
  prob_churn: number;
  score_retencao: number;
}
interface Preset {
  id: string;
  label: string;
  request: Record<string, unknown>;
  response: PresetResponse;
}

const API = "http://localhost:8000/predict";
const ARC = 282.74;
const SLIDERS = ["Age", "Balance", "NumOfProducts", "Satisfaction Score"];

function initPlayground() {
  const root = document.getElementById("playground");
  if (!root) return;
  const presets: Preset[] = JSON.parse(root.dataset.presets ?? "[]");
  let current: Preset = presets[0];

  const seal = document.getElementById("pg-seal")!;
  const predEl = document.getElementById("pg-pred")!;
  const probEl = document.getElementById("pg-prob")!;
  const arc = document.getElementById("pg-gauge-arc") as unknown as SVGPathElement | null;
  const valueEl = document.getElementById("pg-gauge-value")!;

  function paint(data: PresetResponse, live: boolean) {
    predEl.textContent = data.turnover_pred === 1 ? "Sai (churn)" : "Fica";
    probEl.textContent = `${(data.prob_churn * 100).toFixed(1)}%`;
    valueEl.textContent = String(data.score_retencao);
    if (arc) arc.style.strokeDashoffset = String(ARC * (1 - data.score_retencao / 10));
    seal.hidden = live;
  }

  function currentRequest(): Record<string, unknown> {
    const req = { ...current.request };
    for (const key of SLIDERS) {
      const el = document.querySelector<HTMLInputElement>(`[data-slider="${key}"]`);
      if (el) req[key] = key === "Balance" ? Number(el.value) : Number(el.value);
    }
    return req;
  }

  async function run() {
    let live: PresetResponse | null = null;
    try {
      const res = await fetch(API, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify([currentRequest()]),
      });
      if (res.ok) live = (await res.json())[0];
    } catch {
      live = null; // API fora / sem CORS / offline -> fallback
    }
    const { data, live: isLive } = pickResult(current, live);
    paint(data, isLive);
  }

  function selectPreset(p: Preset) {
    current = p;
    for (const key of SLIDERS) {
      const el = document.querySelector<HTMLInputElement>(`[data-slider="${key}"]`);
      if (el && key in p.request) el.value = String(p.request[key]);
    }
    run();
  }

  root.querySelectorAll<HTMLButtonElement>("[data-preset]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const p = presets.find((x) => x.id === btn.dataset.preset);
      if (p) selectPreset(p);
    });
  });

  let t: number | undefined;
  root.querySelectorAll<HTMLInputElement>("[data-slider]").forEach((el) => {
    el.addEventListener("input", () => {
      window.clearTimeout(t);
      t = window.setTimeout(run, 250);
    });
  });

  selectPreset(current);
}

if (document.readyState !== "loading") initPlayground();
else document.addEventListener("DOMContentLoaded", initPlayground);
```

- [ ] **Step 6: Criar `presentation/src/components/ModelPlayground.astro`**

Reusa o `RetentionGauge` (Task 9) com `id="pg-gauge"` — ele gera exatamente os ids `pg-gauge-arc` e `pg-gauge-value` que o `playground.ts` atualiza. Sem SVG duplicado.

```astro
---
import RetentionGauge from "./RetentionGauge.astro";
import fallback from "../data/presets_fallback.json";
const presets = fallback.presets;
const first = presets[0];
const SLIDERS: { key: string; min: number; max: number; step: number }[] = [
  { key: "Age", min: 18, max: 75, step: 1 },
  { key: "Balance", min: 0, max: 250000, step: 1000 },
  { key: "NumOfProducts", min: 1, max: 4, step: 1 },
  { key: "Satisfaction Score", min: 1, max: 5, step: 1 },
];
---

<div id="playground" data-presets={JSON.stringify(presets)} class="text-left">
  <div class="mb-6 flex flex-wrap gap-2">
    {presets.map((p) => (
      <button
        data-preset={p.id}
        class="rounded-full border px-4 py-2 text-sm font-medium hover:bg-black/5"
      >
        {p.label}
      </button>
    ))}
  </div>

  <div class="grid gap-8 md:grid-cols-2">
    <div class="space-y-4">
      {SLIDERS.map((s) => (
        <label class="block">
          <span class="text-sm text-[var(--muted)]">{s.key}</span>
          <input
            type="range"
            data-slider={s.key}
            min={s.min}
            max={s.max}
            step={s.step}
            value={String(first.request[s.key as keyof typeof first.request])}
            class="w-full"
          />
        </label>
      ))}
    </div>

    <div class="flex flex-col items-center justify-center">
      <RetentionGauge id="pg-gauge" score={0} />
      <p class="mt-4 text-sm">
        Previsão: <strong id="pg-pred">—</strong> · churn <strong id="pg-prob">—</strong>
      </p>
      <p id="pg-seal" hidden class="mt-2 rounded bg-amber-100 px-3 py-1 text-xs text-amber-800">
        resposta pré-gravada (API offline)
      </p>
    </div>
  </div>
</div>

<script>
  import "../scripts/playground.ts";
</script>
```

Nota: o `RetentionGauge` renderiza `<span id="pg-gauge-value">` e "/10" separados; o texto "/10 retenção" fica a cargo do próprio componente ("/10"). Não há duplicação de `<style>` — a transição do arco vive no `RetentionGauge`.

- [ ] **Step 7: Injetar no slide 7**

Em `index.astro`, importe `ModelPlayground` e troque `<Slide meta={meta(7)} />` por `<Slide meta={meta(7)}><ModelPlayground /></Slide>`.

- [ ] **Step 8: Verificar build + testes**

Run (dentro de `presentation/`): `npm run build && npm run test`
Expected: build sem erro; `node --test src/scripts` roda deck + playground verdes.
Manual (com fallback): `npm run dev` com a API **desligada** → slide 7 → clicar nos 3 presets muda gauge/prob/pred e mostra o selo "resposta pré-gravada"; valores batem com `presets_fallback.json` (fiel score 8, borderline 5, risco 3).
Manual (ao vivo, opcional): na raiz `make serve`; recarregar → mexer nos sliders recalcula e o selo some.

- [ ] **Step 9: Commit**

```bash
git add presentation/src/components/ModelPlayground.astro presentation/src/scripts/playground.ts presentation/src/scripts/playground.mjs presentation/src/scripts/playground.test.mjs presentation/src/pages/index.astro
git commit -m "feat(presentation): live ModelPlayground with offline fallback (slide 7)"
```

---

## Task 11: MetricCard + ConfusionMatrix + slide 8

**Files:**
- Create: `presentation/src/components/MetricCard.astro`
- Create: `presentation/src/components/ConfusionMatrix.astro`
- Modify: `presentation/src/pages/index.astro` (slide 8)

- [ ] **Step 1: Criar `presentation/src/components/MetricCard.astro`**

```astro
---
interface Props {
  label: string;
  value: string;
  note?: string;
}
const { label, value, note } = Astro.props;
---

<div class="rounded-xl border border-black/10 px-6 py-5 text-left">
  <p class="text-sm uppercase tracking-widest text-[var(--muted)]">{label}</p>
  <p class="mt-1 text-3xl font-semibold">{value}</p>
  {note && <p class="mt-1 text-sm text-[var(--muted)]">{note}</p>}
</div>
```

- [ ] **Step 2: Criar `presentation/src/components/ConfusionMatrix.astro`**

Lê `metrics.json` (fonte única; sem números hard-coded). Matriz `[[TN, FP], [FN, TP]]`.

```astro
---
import metrics from "../data/metrics.json";
const [[tn, fp], [fn, tp]] = metrics.confusion_matrix as [
  [number, number],
  [number, number],
];
---

<div class="mx-auto max-w-md text-center">
  <div class="grid grid-cols-[auto_1fr_1fr] gap-1 text-sm">
    <div></div>
    <div class="font-medium text-[var(--muted)]">Prev. fica</div>
    <div class="font-medium text-[var(--muted)]">Prev. sai</div>
    <div class="flex items-center font-medium text-[var(--muted)]">Real fica</div>
    <div class="rounded bg-emerald-50 py-4 text-lg font-semibold">{tn}<br /><span class="text-xs font-normal text-[var(--muted)]">TN</span></div>
    <div class="rounded bg-red-50 py-4 text-lg font-semibold">{fp}<br /><span class="text-xs font-normal text-[var(--muted)]">FP</span></div>
    <div class="flex items-center font-medium text-[var(--muted)]">Real sai</div>
    <div class="rounded bg-red-50 py-4 text-lg font-semibold">{fn}<br /><span class="text-xs font-normal text-[var(--muted)]">FN</span></div>
    <div class="rounded bg-emerald-50 py-4 text-lg font-semibold">{tp}<br /><span class="text-xs font-normal text-[var(--muted)]">TP</span></div>
  </div>
  <p class="mt-3 text-sm text-[var(--muted)]">
    Pegamos {tp} dos {tp + fn} que saem (recall {(tp / (tp + fn) * 100).toFixed(0)}%), ao custo de {fp} falsos positivos.
  </p>
</div>
```

- [ ] **Step 3: Injetar no slide 8**

Em `index.astro`, importe `MetricCard`, `ConfusionMatrix` e `metrics` (`import metrics from "../data/metrics.json"`). Troque `<Slide meta={meta(8)} />` por:

```astro
<Slide meta={meta(8)}>
  <div class="grid grid-cols-2 gap-4 sm:grid-cols-4">
    <MetricCard label="ROC AUC" value={metrics.roc_auc.toFixed(3)} note="separa bem" />
    <MetricCard label="Recall" value={metrics.recall.toFixed(3)} note="o que importa" />
    <MetricCard label="Precision" value={metrics.precision.toFixed(3)} />
    <MetricCard label="Accuracy" value={metrics.accuracy.toFixed(3)} note="engana" />
  </div>
  <div class="mt-8">
    <ConfusionMatrix />
  </div>
</Slide>
```

- [ ] **Step 4: Verificar build**

Run (dentro de `presentation/`): `npm run build`
Expected: sem erro; os valores na tela batem com `metrics.json` (AUC 0.764, recall 0.740, matriz 1043/549/106/302).

- [ ] **Step 5: Commit**

```bash
git add presentation/src/components/MetricCard.astro presentation/src/components/ConfusionMatrix.astro presentation/src/pages/index.astro
git commit -m "feat(presentation): honest metrics + confusion matrix from metrics.json (slide 8)"
```

---

## Task 12: Timeline + slide 9 (drift) + slides de texto

**Files:**
- Create: `presentation/src/components/Timeline.astro`
- Modify: `presentation/src/pages/index.astro` (slides 9 e 10)

- [ ] **Step 1: Criar `presentation/src/components/Timeline.astro`**

```astro
---
import { TIMELINE } from "../data/timeline";
---

<ol class="mx-auto max-w-2xl space-y-3 text-left">
  {TIMELINE.map((m) => (
    <li class="flex gap-4 rounded-lg px-4 py-2 hover:bg-black/5">
      <span class="grid h-8 w-8 shrink-0 place-items-center rounded-full bg-[var(--accent)] text-sm font-semibold text-white">
        {m.marco}
      </span>
      <span>
        <strong>{m.title}</strong>
        <span class="block text-sm text-[var(--muted)]">{m.detail}</span>
      </span>
    </li>
  ))}
</ol>
```

- [ ] **Step 2: Injetar no slide 9 (drift) e slide 10 (timeline)**

Em `index.astro`, importe `Timeline`. Troque `<Slide meta={meta(9)} />` por um comparativo textual PASS vs ALERT (números reais capturados), e `<Slide meta={meta(10)} />` pela timeline:

```astro
<Slide meta={meta(9)}>
  <div class="grid gap-6 text-left md:grid-cols-2">
    <div class="rounded-xl border border-emerald-200 bg-emerald-50 p-6">
      <p class="text-sm font-semibold text-emerald-700">make monitor — holdout saudável</p>
      <p class="mt-2 font-mono text-sm">[PASS] drift_share=0.000 · drifted_columns=0 · exit 0</p>
      <p class="mt-2 text-sm text-[var(--muted)]">accuracy 0.67 · roc_auc 0.76 · o gate deixa passar.</p>
    </div>
    <div class="rounded-xl border border-red-200 bg-red-50 p-6">
      <p class="text-sm font-semibold text-red-700">make monitor-drift — drift simulado</p>
      <p class="mt-2 font-mono text-sm">[ALERT] drift_share=0.462 · drifted_columns=6 · exit 2</p>
      <p class="mt-2 text-sm text-[var(--muted)]">accuracy 0.23 · recall 0.99 = "prevê que todos saem". Gatilho de re-treino.</p>
    </div>
  </div>
</Slide>
```

E:

```astro
<Slide meta={meta(10)}><Timeline /></Slide>
```

- [ ] **Step 3: Verificar build**

Run (dentro de `presentation/`): `npm run build`
Expected: sem erro.
Manual: slide 9 mostra o PASS verde vs ALERT vermelho; slide 10 lista os 9 marcos.

- [ ] **Step 4: Commit**

```bash
git add presentation/src/components/Timeline.astro presentation/src/pages/index.astro
git commit -m "feat(presentation): drift moment (slide 9) + milestone timeline (slide 10)"
```

---

## Task 13: Dashboard

**Files:**
- Create: `presentation/src/pages/dashboard.astro`

- [ ] **Step 1: Criar `presentation/src/pages/dashboard.astro`**

Reusa os componentes de métrica/matriz/timeline num layout de dashboard (não full-screen). Tecla `D` volta ao deck.

```astro
---
import Base from "../layouts/Base.astro";
import MetricCard from "../components/MetricCard.astro";
import ConfusionMatrix from "../components/ConfusionMatrix.astro";
import Timeline from "../components/Timeline.astro";
import metrics from "../data/metrics.json";
---

<Base title="Churn MLOps — dashboard">
  <main class="mx-auto max-w-5xl px-6 py-12">
    <header class="mb-10 flex items-baseline justify-between">
      <h1 class="text-3xl font-semibold tracking-tight">Churn MLOps — dashboard</h1>
      <a href={import.meta.env.BASE_URL} class="text-sm text-[var(--accent)]">← deck (tecla D)</a>
    </header>

    <section class="mb-10 grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-5">
      <MetricCard label="ROC AUC" value={metrics.roc_auc.toFixed(3)} note="separa bem" />
      <MetricCard label="Recall" value={metrics.recall.toFixed(3)} note="o que importa" />
      <MetricCard label="Precision" value={metrics.precision.toFixed(3)} />
      <MetricCard label="F1" value={metrics.f1.toFixed(3)} />
      <MetricCard label="Accuracy" value={metrics.accuracy.toFixed(3)} note="engana" />
    </section>

    <section class="mb-10 grid gap-8 md:grid-cols-2">
      <div>
        <h2 class="mb-4 text-lg font-semibold">Matriz de confusão</h2>
        <ConfusionMatrix />
      </div>
      <div>
        <h2 class="mb-4 text-lg font-semibold">Status</h2>
        <ul class="space-y-2 text-sm">
          <li>✅ 9 marcos entregues (issue → PR → merge)</li>
          <li>✅ 78 testes · CI verde (GitHub Actions + Jenkinsfile)</li>
          <li>✅ Modelo @production no MLflow Registry (gate roc_auc ≥ 0.70)</li>
          <li>✅ Monitoramento Evidently com quality gate (exit ≠ 0 dispara re-treino)</li>
        </ul>
      </div>
    </section>

    <section>
      <h2 class="mb-4 text-lg font-semibold">Marcos</h2>
      <Timeline />
    </section>
  </main>

  <script>
    document.addEventListener("keydown", (e) => {
      if (e.key.toLowerCase() === "d") window.location.href = import.meta.env.BASE_URL;
    });
  </script>
</Base>
```

- [ ] **Step 2: Verificar build**

Run (dentro de `presentation/`): `npm run build`
Expected: sem erro; `dist/dashboard/index.html` gerado.
Manual: `npm run dev` → `http://localhost:4321/dashboard` mostra cards + matriz + status + timeline; tecla `D` volta ao deck; no deck, `D` vai pro dashboard.

- [ ] **Step 3: Commit**

```bash
git add presentation/src/pages/dashboard.astro
git commit -m "feat(presentation): metrics dashboard face reusing metric components"
```

---

## Task 14: Prints reais + verificação final + README do site

**Files:**
- Create: `presentation/src/assets/mlflow-runs.png`
- Create: `presentation/src/assets/evidently-drift.png`
- Modify: `presentation/src/pages/index.astro` (embutir prints nos slides 10 e 9)
- Create: `presentation/README.md`

- [ ] **Step 1: Capturar o print das runs do MLflow**

Run (na raiz), garantindo que há runs registradas (`make pipeline` já rodou no staging):

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Abra `http://localhost:5000`, navegue até o experimento com as runs, e salve um screenshot em `presentation/src/assets/mlflow-runs.png`. Encerre o servidor (Ctrl-C).

- [ ] **Step 2: Capturar o print do relatório de drift do Evidently**

O HTML já existe no staging (`estudo-local/presentation-data/evidently_drift.html`). Abra-o no navegador e salve um screenshot da seção de drift em `presentation/src/assets/evidently-drift.png`.

- [ ] **Step 3: Embutir os prints nos slides**

Em `index.astro`, adicione os imports de imagem e o componente `Image` do Astro no topo:

```astro
import { Image } from "astro:assets";
import mlflowRuns from "../assets/mlflow-runs.png";
import evidentlyDrift from "../assets/evidently-drift.png";
```

No slide 9, adicione abaixo do grid PASS/ALERT (dentro do mesmo `<Slide meta={meta(9)}>`):

```astro
<Image src={evidentlyDrift} alt="Relatório de drift do Evidently" class="mx-auto mt-6 max-h-64 w-auto rounded-lg border" />
```

No slide 10, adicione abaixo do `<Timeline />` (dentro do mesmo `<Slide meta={meta(10)}>`):

```astro
<Image src={mlflowRuns} alt="Runs registradas no MLflow" class="mx-auto mt-6 max-h-64 w-auto rounded-lg border" />
```

- [ ] **Step 4: Criar `presentation/README.md`**

```markdown
# Site de apresentação — Churn MLOps

Deck (12 slides) + dashboard que apresentam o projeto. Astro + Tailwind, roda local.

## Rodar

```bash
cd presentation
npm install
npm run dev      # http://localhost:4321
```

Atalhos no deck: `←/→/espaço` navegam · `S` speaker notes · `B` blackout · `D` alterna deck↔dashboard.

## Interação ao vivo (opcional)

O slide 7 (ModelPlayground) chama `POST http://localhost:8000/predict`. Para a demo ao vivo,
suba a API na raiz do repo (`make serve`). Sem a API, o playground usa respostas reais
pré-gravadas (`src/data/presets_fallback.json`) — a demo nunca quebra.

## Dados

- `src/data/metrics.json` — métricas reais do `@production` (regenere com `make presentation-metrics` na raiz).
- `src/data/presets_fallback.json` — respostas reais do `/predict` (fallback offline).

## Build / deploy

`npm run build` gera `dist/`. Deploy-ready: defina `SITE_BASE` (ex.: `/churn/`) para publicar sob subpath.
```

- [ ] **Step 5: Verificação final**

Run (dentro de `presentation/`): `npm run build && npm run test`
Expected: build gera `dist/` sem erro; `node --test src/scripts` verde (deck + playground).
Run (na raiz): `uv run pytest tests/test_api.py tests/test_reporting.py -v`
Expected: CORS e export de métricas verdes.

- [ ] **Step 6: Commit**

```bash
git add presentation/src/assets/ presentation/src/pages/index.astro presentation/README.md
git commit -m "feat(presentation): real MLflow/Evidently screenshots + site README"
```

---

## Encerramento

Após a Task 14, use **superpowers:finishing-a-development-branch** para abrir o PR fechando a issue #21 (`Closes #21`), com resumo do que foi construído e como foi verificado (build do site, testes Python, checagem manual do fallback).
