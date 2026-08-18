# Site de apresentação (deck + dashboard) — design

> Fecha a issue #21. Entregável versionado em `presentation/`. Números embutidos aqui são reais,
> capturados do modelo `@production` em 2026-08-18 (ver §7).

## 1. Objetivo

Um site que roda no navegador e **é a apresentação** do projeto Churn MLOps. Substitui slides de
PowerPoint por uma página única, apresentável da própria máquina, com duas faces:

1. **Deck** — 12 slides no estilo keynote minimalista (OpenAI/Claude): telas brancas, muito espaço
   em branco, uma ideia por tela, tipografia elegante, navegação por teclado.
2. **Dashboard** — métricas honestas, timeline dos marcos e status do projeto num relance.

O diferencial: durante a fala dá pra **interagir com o modelo real** (`/predict`), mostrar os **bugs
como código** (antes→depois, tema Dracula) e ver o **momento do drift** (o gate que falha de
propósito). Tudo funciona **offline** — a interação ao vivo é bônus, nunca dependência.

## 2. Requisitos

**Funcionais**
- Deck de 12 slides navegável por teclado (←/→/espaço), com trilha de 5 min demarcada.
- Dashboard com cards de métrica, matriz de confusão, gauge 0–10 e timeline dos 9 marcos.
- Playground do modelo: 3 presets narrativos + 4 sliders-chave + gauge animado; chama `/predict`
  ao vivo com fallback offline pré-gravado.
- Bugs como código: diff antes→depois (Shiki + Dracula), linha do bug destacada, navegável pelos 10.
- Diagrama Mermaid do ciclo de vida; 2–3 prints reais (MLflow runs, Evidently drift).
- Mapa Vertex↔OSS interativo (hover acende o par; clique navega pro slide da peça).

**Não-funcionais**
- **Resiliência offline total:** métricas, fallback de inferência e imagens embutidos no build; o
  site abre e apresenta sem internet e sem a API.
- **Acessibilidade:** respeita `prefers-reduced-motion` (gauge e transições); navegação por teclado.
- **Simplicidade:** componentes pequenos e isolados; fácil de apresentar sob pressão.
- **Isolamento do repo:** vive em `presentation/` com `package.json` próprio; o CI Python não encosta.

## 3. Decisões travadas (do brainstorming)

| # | Decisão | Resolução |
|---|---------|-----------|
| 1 | Onde mora | Versionado em `presentation/` (branch+PR). "Colas" ficam em speaker notes ocultas (tecla `S`), fora do modo plateia. Build padrão = modo plateia, sem notas. |
| 2 | Deploy | Deploy-ready (base path configurável), mas **não publicar agora** — repo privado, inferência ao vivo só local, apresenta-se da própria máquina. |
| 3 | Duração | Sem requisito da Bulla. 12 slides; os marcados `[5min]` formam a trilha de 5 min, o deck inteiro é a de 10 min. |
| 4 | Imagens | **Híbrido:** render nativo (dashboard/matriz/gauge/timeline) + só 2–3 prints reais (MLflow runs, Evidently drift). |

## 4. Arquitetura

**Localização e stack.** App **Astro + Tailwind** em `presentation/`, `package.json` próprio. Shiki
(nativo do Astro) faz o highlight Dracula com marcação de linha em build time; Mermaid renderiza no
cliente (evita dor de integração no build).

```
presentation/
  package.json                # deps próprias (astro, tailwind); isolado do projeto Python
  astro.config.mjs            # base path configurável (deploy-ready)
  tailwind.config.mjs
  src/
    layouts/Base.astro        # casca comum (fontes, reset, tema claro)
    pages/
      index.astro             # deck (12 slides como seções full-screen)
      dashboard.astro         # dashboard de métricas + status
    components/
      Slide.astro             # casca de 1 slide (branco, centralizado, título + conteúdo)
      Deck.astro + deck.ts    # navegação por teclado, progresso, hash, teclas S/B/D
      CodeDiff.astro          # antes→depois lado a lado (Shiki+Dracula, linha destacada)
      Mermaid.astro           # renderiza 1 diagrama no cliente
      VertexOssMap.astro      # mapa interativo (hover acende par; clique navega)
      ModelPlayground.astro   # ilha interativa (ver §5)
      MetricCard.astro        # 1 métrica (label + valor + nota)
      ConfusionMatrix.astro   # matriz 2×2 a partir de metrics.json
      RetentionGauge.astro    # gauge 0–10 animado (respeita reduced-motion)
      Timeline.astro          # 9 marcos entregues
    data/
      metrics.json            # AUC/precision/recall/f1/accuracy + matriz (§7)
      presets_fallback.json   # 3 presets com respostas reais do /predict (§7)
      slides.ts               # conteúdo textual + speaker notes dos 12 slides
    assets/
      mlflow-runs.png         # print real (captura manual — §7)
      evidently-drift.png     # print real do relatório de drift
```

**Rotas (2 faces, mesma base visual).**
- `/` → deck (slides como seções full-screen; navegação por seta/scroll-snap).
- `/dashboard` → métricas e status do projeto.
- Tecla `D` alterna deck↔dashboard.

**Mecânica do deck (`deck.ts`, client-side).**
- `←`/`→`/`espaço` navegam; bolinhas de progresso no rodapé; número do slide.
- Deep-link por hash (`#slide-7`) pra pular direto num ensaio.
- Tecla `S` = speaker notes ocultas (só modo pessoal; ausentes no build de plateia).
- Tecla `B` = blackout (tela branca/preta pra pausa dramática ou Q&A).
- Tecla `D` = toggle deck↔dashboard.
- Transições sóbrias (fade/slide curtos); desligadas sob `prefers-reduced-motion`.

## 5. ModelPlayground (ilha interativa)

- **3 presets narrativos** (botões) que preenchem o form. Valores e respostas reais em §7:
  - **Cliente fiel** → score 8, `prob_churn` 0.23, fica.
  - **Cliente borderline** → score 5, `prob_churn` 0.55, no muro da decisão.
  - **Cliente em risco** → score 3, `prob_churn` 0.73, sai.
- **4 sliders-chave** editáveis: `Age`, `Balance`, `NumOfProducts`, `Satisfaction Score`. Os demais
  campos ficam fixos em defaults sensatos (do preset selecionado).
- Ao acionar: `fetch('http://localhost:8000/predict')` → mostra `turnover_pred`, `prob_churn` e o
  **score 0–10** num **gauge animado** (`RetentionGauge`).
- **Fallback offline:** se o `fetch` falhar (API fora, sem CORS, sem rede), usa a resposta real
  pré-gravada do preset (`presets_fallback.json`). A demo **nunca quebra ao vivo**. No modo fallback,
  mexer nos sliders exibe o selo "resposta pré-gravada" e mantém o resultado do preset selecionado
  (não há re-fetch nem interpolação) — a interação ao vivo com sliders exige a API no ar.

## 6. Espinha do deck (12 slides)

`[5min]` = trilha curta. Uma ideia por tela.

1. `[5min]` **Título** — "De 2 scripts frágeis a uma esteira de produção".
2. `[5min]` **O problema** — o que a Bulla pediu + 2 restrições (não trocar o modelo, 100% local).
3. `[5min]` **A grande sacada** — mapa Vertex↔OSS interativo (`VertexOssMap`).
4. **A auditoria** — "achamos 10 bugs" (abertura da seção).
5. `[5min]` **Bugs antes→depois** — `CodeDiff` Dracula, navega pelos 10, contador "N/10" + link pro
   commit/PR. ⭐ coração.
6. **A arquitetura** — diagrama `Mermaid` do ciclo de vida.
7. `[5min]` **Modelo ao vivo** — `ModelPlayground` (presets + gauge 0–10). ⭐ demo.
8. **Métricas honestas** — cards AUC/precision/recall/f1 + `ConfusionMatrix` (resolve o bug #5:
   accuracy 0.67 engana; recall 0.74 é o que importa numa base ~20% churn).
9. `[5min]` **O momento do drift** — `make monitor` (gate PASS, exit 0) vs `make monitor-drift`
   (gate ALERT, exit 2, qualidade despenca) + print do Evidently. ⭐ clímax.
10. **O que foi construído** — `Timeline` dos 9 marcos + 78 testes + status do CI.
11. **Limitações honestas** — K8s, captura de predições, scheduler (do README).
12. `[5min]` **Fecho** — recap + próximos passos.

## 7. Dados capturados (reais, 2026-08-18)

Rodados uma vez contra o modelo `@production` e versionados como JSON no build (staging atual em
`estudo-local/presentation-data/`, movido pra `presentation/src/data/` na implementação):

- **`metrics.json` (produção, holdout de teste):**
  `roc_auc` 0.7644 · `precision` 0.3549 · `recall` 0.7402 · `f1` 0.4797 · `accuracy` 0.6725 ·
  matriz de confusão `[[1043, 549], [106, 302]]` (TN, FP / FN, TP).
- **`presets_fallback.json` (3 presets, respostas reais do `/predict`):**
  fiel `{turnover_pred:0, prob_churn:0.233, score:8}`;
  borderline `{turnover_pred:1, prob_churn:0.549, score:5}`;
  risco `{turnover_pred:1, prob_churn:0.730, score:3}`.
- **Monitoramento (slide 9):** saudável → `[PASS] drift_share=0.000 drifted_columns=0`, exit 0;
  drift simulado → `[ALERT] drift_share=0.462 drifted_columns=6`, exit 2, com colapso de qualidade
  (accuracy 0.23, roc_auc 0.61, recall 0.99 = "prevê que todos saem"). 2 relatórios Evidently HTML
  capturados (saudável + drift).

**Captura visual pendente (manual, durante o build):**
- Print da tela de **runs do MLflow** (`mlflow ui` → screenshot) → `assets/mlflow-runs.png`.
- Print do **relatório de drift do Evidently** (abrir `evidently_drift.html` → screenshot) →
  `assets/evidently-drift.png`.

## 8. Integração com o repo

- **CORS (adição versionada em `serving/api.py`):** `CORSMiddleware` liberando as origens locais do
  Astro (`http://localhost:4321`, `http://localhost:3000`). Pequena, legítima, coberta por teste.
- **Alvo de Makefile `presentation-metrics`:** exporta as métricas do `@production`
  (o que `capture_metrics.py` faz hoje) pra `presentation/src/data/metrics.json`. Reusa
  `churn.training.evaluate.evaluate` — não duplica lógica de métrica.
- **`.gitignore`:** ignorar `presentation/node_modules/`, `presentation/dist/`, `presentation/.astro/`.
  Os artefatos gerados do projeto Python (mlruns/, mlflow.db, reports/, *.html) continuam ignorados.
- O CI Python (GitHub Actions) **não** roda nada de `presentation/` (paths distintos). Um lint/build
  do site é opcional e fica fora de escopo neste marco (ver §11).

## 9. Deploy

Deploy-ready, sem publicar agora. `astro.config.mjs` com `base` configurável por variável de
ambiente (um comando pra GitHub Pages/Vercel depois). Justificativa: repo privado (processo
seletivo), inferência ao vivo só roda local, apresenta-se da própria máquina.

## 10. Verificação

- **Build:** `npm run build` em `presentation/` gera `dist/` sem erro.
- **CORS:** teste em `tests/` confirma que a resposta de `/predict` traz os headers
  `access-control-allow-origin` pra origem local.
- **Fallback offline:** com a API fora, o `ModelPlayground` renderiza os 3 presets a partir do JSON
  (verificação manual + os valores do JSON são reais, batem com o `/predict`).
- **Métricas na tela** conferem com `metrics.json` (fonte única; sem números hard-coded no HTML).

## 11. Fora de escopo (YAGNI)

- Publicar o site (deploy real) — só deploy-ready.
- CI/lint/build do site no GitHub Actions — o foco é o entregável apresentável.
- Editor livre de todos os 12 campos do modelo — só 4 sliders + presets.
- Framework de slides de terceiros (reveal.js etc.) — deck próprio, mínimo, em Astro.
- Autenticação, backend próprio do site, persistência — o site é estático + fetch ao `/predict`.

## 12. Riscos e mitigação

| Risco | Mitigação |
|-------|-----------|
| API fora durante a demo | Fallback pré-gravado real (`presets_fallback.json`) — demo nunca quebra. |
| CORS bloqueia o fetch | `CORSMiddleware` versionado + o fallback cobre mesmo se esquecerem de subir a API. |
| Mermaid quebra no build | Renderizar no cliente, não no build. |
| Animação nauseante no projetor | `prefers-reduced-motion` desliga gauge/transições. |
| Números divergentes entre slides | Fonte única (`metrics.json`); nada hard-coded. |
