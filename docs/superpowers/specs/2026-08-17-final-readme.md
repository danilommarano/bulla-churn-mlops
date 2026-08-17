# README-solução final + docs de apresentação — Design (Marco 9)

> Entregável do teste técnico Bulla (ML Engineer / MLOps). Fecha o projeto com o
> **README-solução** (o entregável central pedido pelo teste), deixa o `estudo-local/`
> fiel ao que foi construído para a apresentação, e faz uma higiene real de warnings.
> Closes #19.

## 1. Objetivo

O `README.md` atual ainda é o **brief do teste** (o enunciado), não a solução. O teste pede
explicitamente um README com: (a) problemas conceituais do script original, (b) desenho da
arquitetura, (c) justificativa das etapas/melhorias e (d) como rodar. Este marco entrega esse
README, atualiza a documentação de estudo local para a apresentação, e limpa os
`FutureWarning` do KFP na suíte.

## 2. Escopo

- **Versionado (vai no PR):**
  - `README.md` — reescrito como README-solução.
  - `src/churn/orchestration/dag.py` — `base_image` explícito nos 5 `@dsl.component`.
  - `docs/superpowers/specs/2026-08-17-final-readme.md` — esta spec.
- **Local, não versionado (regra do projeto — `estudo-local/` está em `.git/info/exclude`):**
  - `estudo-local/04-as-tecnologias.md`, `05-arquitetura-e-fluxo.md` e um novo módulo de
    roteiro de apresentação.

Não se toca em código de produção (só o `base_image` do `dag.py`, que é cosmético — o runner
local usa `SubprocessRunner(use_venv=False)`, então `base_image` nunca é usado em execução;
serve só para silenciar o warning e fixar o comportamento futuro do KFP).

## 3. Conteúdo do README

Prosa direta, em português; identificadores e comandos em inglês. Seções:

1. **Título + pitch** — 1 parágrafo: o que é (produtização de um modelo de churn como pipeline
   MLOps local) e a ideia-chave (espelhar o Vertex AI com equivalentes open-source rodando no
   notebook).
2. **O desafio** — resumo curto do que o teste pede (preserva o contexto do brief original;
   restrição de não trocar o algoritmo, foco em MLOps, tudo local).
3. **Arquitetura** — o diagrama de fluxo ponta a ponta (treino → serving → monitoramento →
   re-treino) + a **tabela consolidada Vertex AI ↔ OSS** (7 linhas do mapa geral).
4. **Problemas conceituais no script original** — tabela dos **10 bugs → correção**
   (leakage, train/serve skew, CustomerId, surname_encoded, accuracy enganosa, split sem
   seed/stratify, churn_rate hardcoded, pickle solto, score duplicado, sem Pipeline/validação/teste).
5. **Etapas e melhorias** — narrativa curta dos 3 fluxos justificando cada escolha
   (Feast contra skew/leakage; Pipeline persistível; MLflow tracking+registry; quality gate;
   Evidently; FastAPI+Docker; CI/CD).
6. **Como rodar** — pré-requisitos (uv), depois `make setup` → `make pipeline` → `make serve`,
   com `make monitor`/`monitor-drift` e `docker-build`/`docker-run`; o que esperar de cada um.
   Tabela curta dos targets do Makefile.
7. **Limitações honestas** — os gaps documentados nos marcos (monitoring: sem captura real de
   predições, sem scheduler gerenciado, L-infinity categórico ausente no Evidently; CI/CD: sem
   deploy automático a endpoint, Jenkinsfile documental) + ponteiro para
   `docs/tensorflow_variant.md`.

## 4. Conteúdo das docs de estudo (local)

- **Módulo 4 (`04-as-tecnologias.md`)** — correções de acurácia: remover a afirmação de
  `docker-compose` (existe só um `docker/Dockerfile` + targets `docker-build`/`docker-run`) e a
  de "manifests K8s documentados" (não existem); descrever K8s honestamente como próximo passo
  não implementado. Garantir que toda dependência relevante do `pyproject.toml` aparece.
- **Módulo 5 (`05-arquitetura-e-fluxo.md`)** — atualizar nomes reais dos steps (`prepare_data_op`
  … `register_model_op`) e alinhar o texto ao que foi construído.
- **Novo módulo 07 — roteiro de apresentação/demo** — ordem do que rodar ao vivo, o que dizer em
  cada tela (MLflow UI, pipeline, API `/predict`, monitor com e sem drift), a contagem real de
  testes (78) e respostas de 1 linha para perguntas-armadilha, incluindo os gaps honestos.

## 5. Higiene

`base_image="python:3.12"` explícito em `prepare_data_op`, `split_data_op`, `train_model_op`,
`evaluate_model_op`, `register_model_op`. Objetivo: `make test`/`make ci` sem os 30 `FutureWarning`
do KFP (troca de base_image default em out/2027). O item `.fillna` do `builder.py` foi
descartado: verificado empiricamente que não emite warning (colunas já são `float64`).

## 6. Verificação

- `make ci` verde; `uv run pytest` sem `FutureWarning` do KFP.
- README cobre os 4 itens de entregável do teste (checagem editorial).
- `estudo-local/` sem afirmações falsas (docker-compose / K8s manifests).
- Referências (targets, steps, tabelas) batem com o código real.

## 7. Fora de escopo (YAGNI)

- Nenhum `docker-compose.yml` ou manifest K8s novo (o teste é local; documentar honestamente > construir).
- Nenhuma mudança em lógica de produção além do `base_image` cosmético.
- Nenhum diagrama gerado por ferramenta (o ASCII do Módulo 5 basta).
