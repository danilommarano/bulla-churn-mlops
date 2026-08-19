# Site de apresentação — Churn MLOps

Deck (12 slides) + dashboard que apresentam o projeto. Astro + Tailwind, roda local.

## Rodar

Tudo de uma vez, da raiz do repo (sobe API + site juntos; Ctrl-C encerra os dois):

    make present

Ou só o site:

    cd presentation
    npm install
    npm run dev      # http://localhost:4321

Atalhos no deck: `←/→/espaço` navegam · `S` speaker notes · `B` blackout · `D` alterna deck↔dashboard · `E` modo edição.

## Editar a apresentação (modo edição — tecla `E`)

Aperte `E` no deck para abrir o modo edição (painel no canto):

- **Texto** — clique em qualquer título/subtítulo/kicker (contorno tracejado) e digite.
- **Cor e fonte** — o painel muda acento, fundo, cor do texto e fonte (corpo/títulos) ao vivo.
- **Selecionar** — clique num texto/bloco para selecioná-lo (contorno sólido); os controles abaixo agem sobre a seleção:
  - **Alinhar texto** — ⬅ / ⬛ / ➡ (esquerda / centro / direita).
  - **Tamanho** — A− / A+ por elemento.
  - **Bloco** — ↑ / ↓ move o componente selecionado dentro do slide.
- **Slide** — Topo / Centro (alinhamento vertical do conteúdo do slide atual).
- **Ordem** — ◀ / ▶ move o slide atual na sequência.
- **Persistência** — as edições ficam salvas no navegador (localStorage) na hora.

Para tornar as edições **permanentes e versionadas**, clique em **Exportar `overrides.json`**
e substitua `src/data/overrides.json` pelo arquivo baixado. O deck carrega esse arquivo
(por baixo do localStorage) em toda visita — então vira a fonte da verdade no repo.
**Resetar edições** limpa o localStorage e volta ao `overrides.json` versionado.

> Limite: o modo edição é para uso local. Sem exportar, as edições vivem só naquele
> navegador. É de propósito enxuto — não é um editor visual "arrasta qualquer coisa".

## Interação ao vivo (opcional)

O slide 7 (ModelPlayground) chama `POST http://localhost:8000/predict`. O `make present` já
sobe a API junto; se quiser só a API, use `make serve` na raiz. Sem a API, o playground usa
respostas reais pré-gravadas (`src/data/presets_fallback.json`) — a demo nunca quebra.

## Dados

- `src/data/metrics.json` — métricas reais do `@production` (regenere com `make presentation-metrics` na raiz).
- `src/data/presets_fallback.json` — respostas reais do `/predict` (fallback offline).

## Build / deploy

`npm run build` gera `dist/`. Deploy-ready: defina `SITE_BASE` (ex.: `/churn/`) para publicar sob subpath.
