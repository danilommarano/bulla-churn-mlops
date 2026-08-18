# Site de apresentação — Churn MLOps

Deck (12 slides) + dashboard que apresentam o projeto. Astro + Tailwind, roda local.

## Rodar

    cd presentation
    npm install
    npm run dev      # http://localhost:4321

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
