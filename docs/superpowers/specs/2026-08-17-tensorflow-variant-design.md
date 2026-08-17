# TensorFlow vs scikit-learn — Design do documento (Marco 8)

> Entregável do teste técnico Bulla (ML Engineer / MLOps). Documenta, de forma honesta,
> **quando e por que** TensorFlow valeria mais que scikit-learn — e por que, para este
> problema de churn, o sklearn é a escolha certa.

## 1. Objetivo

O teste menciona TensorFlow. O modelo entregue é uma regressão logística do scikit-learn
(decisão da spec principal §7). Este marco entrega **um único documento**
(`docs/tensorflow_variant.md`) que:

1. reconhece a equivalência (uma regressão logística é uma rede de uma camada densa
   sigmoide — dá pra escrever em Keras trivialmente);
2. lista, de forma objetiva, os **pontos fortes do TensorFlow frente ao sklearn**;
3. é honesto sobre o **trade-off neste caso**: churn tabular (~10k linhas, sinal
   majoritariamente linear) é território onde o sklearn ganha — o TF só passa a valer
   quando o problema cresce em escala, modalidade ou exigência de produção.

**Sem código executável.** Nada de exemplo Keras, teste de paridade, dependência
opcional ou make target. É documentação — o objetivo é demonstrar critério de
engenharia (saber escolher a ferramenta), não construir a variante.

## 2. Escopo

Um arquivo: `docs/tensorflow_variant.md`. Nenhum outro artefato. `pyproject.toml`,
`Makefile`, `tests/` e `examples/` **não** são tocados. `make ci`/`make test` seguem
idênticos.

## 3. Conteúdo do documento

Seções propostas (prosa curta e direta, não academic):

1. **Contexto e decisão.** O modelo entregue é uma LogReg sklearn. Por quê: dataset
   tabular pequeno, sinal linear, interpretabilidade e simplicidade > capacidade. Este
   doc explica quando a balança viraria pro TensorFlow.

2. **A equivalência.** Uma regressão logística é literalmente uma rede neural de uma
   camada: `Dense(1, activation="sigmoid")` treinada com `binary_crossentropy`. Ou
   seja, migrar o *modelo atual* pra TF é trivial e não traria ganho — o ganho do TF
   aparece nos modelos que o sklearn **não** faz.

3. **Pontos fortes do TensorFlow frente ao sklearn.** O núcleo do documento. Cada item
   com uma frase de "quando isso importa":
   - **Deep learning de verdade** — redes profundas, embeddings, camadas convolucionais
     e recorrentes, atenção. O sklearn não faz aprendizado profundo; para texto, imagem,
     sequências ou features de altíssima cardinalidade, é TF (ou PyTorch).
   - **Escala além da memória** — `tf.data` faz streaming de datasets que não cabem em
     RAM; treino distribuído (multi-GPU/TPU, multi-nó) via `tf.distribute`. O sklearn é
     single-node, in-memory.
   - **Aceleração por hardware** — GPU/TPU nativo. Para modelos grandes, ordens de
     magnitude mais rápido.
   - **Diferenciação automática e customização** — losses, camadas e loops de treino
     customizados com autodiff. O sklearn expõe estimadores fechados; o TF deixa você
     definir o objetivo.
   - **Aprendizado incremental / online** — treino por mini-batches permite atualização
     contínua com dados novos (retraining incremental), útil quando o churn deriva no
     tempo. A maioria dos estimadores sklearn re-treina do zero.
   - **Ecossistema de produção** — SavedModel + TensorFlow Serving (serving de alta
     performance), TFLite (mobile/edge), TF.js (browser), TFX (pipelines de produção com
     validação de dados e análise de modelo). O caminho do sklearn pra produção é mais
     manual.
   - **Integração com Vertex AI** — o Vertex treina containers TF nativamente, tem
     AutoML e serviços gerenciados afinados no ecossistema TF. Amarra ao tema do teste.

4. **O trade-off neste caso (honestidade).** Para *este* problema — churn tabular,
   volume pequeno, forte baseline linear — o TF seria overkill: mais código, mais
   dependências (CUDA, versões), treino mais lento sem ganho de acurácia, e perda de
   interpretabilidade. A regra prática: **comece simples; migre pro TF quando** (a) o
   sinal for claramente não-linear e complexo, (b) os dados não couberem em memória, (c)
   a modalidade for texto/imagem/sequência, ou (d) o serving exigir edge/mobile/altíssima
   escala.

5. **Como a migração se encaixaria no MLOps existente.** Nota curta: o resto do pipeline
   é agnóstico de framework — o MLflow registra modelos TF (`mlflow.tensorflow`) como
   registra sklearn, a mesma API FastAPI serviria, e no KFP só o step de treino trocaria.
   Trocar o modelo não obriga a reescrever o MLOps. (Explicado, não executado.)

## 4. Verificação

Documento não tem teste automatizado. A verificação é editorial:

1. **Precisão técnica** — cada ponto forte do TF é factualmente correto e não
   caricato; cada afirmação sobre o sklearn é justa (não um espantalho).
2. **Honestidade** — o doc conclui que o sklearn é a escolha certa *aqui*, não vende o
   TF gratuitamente. Um avaliador deve ver critério, não hype.
3. **Consistência com o repo** — as referências (LogReg entregue, MLflow, FastAPI, KFP,
   Vertex) batem com o que está implementado nos marcos anteriores.
4. **Idioma** — o doc em si segue a convenção do projeto (README/docs em português;
   identificadores e trechos de código em inglês).

## 5. Fora de escopo (YAGNI)

- Exemplo Keras executável (`examples/tensorflow_variant.py`).
- Teste de paridade (`tests/test_tensorflow_variant.py`).
- Dependência opcional `tf` no `pyproject.toml`.
- Alvo `make tf-variant`.
- Qualquer prova empírica de equivalência (AUC/correlação). O documento **afirma** a
  equivalência teórica; não a mede.
- Treinar, registrar ou servir um modelo TF de fato.
