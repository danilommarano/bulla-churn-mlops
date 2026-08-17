# TensorFlow vs scikit-learn neste projeto

> Por que o modelo de churn entregue usa **scikit-learn**, o que o **TensorFlow**
> ofereceria no lugar, e quando a balança viraria. Documento de decisão — não há
> variante TF construída aqui (ver [Escopo](#escopo)).

## Contexto e decisão

O modelo em produção neste repositório é uma **regressão logística** do scikit-learn
(`churn.training.pipeline.build_pipeline`). A escolha é deliberada: o dataset de churn é
**tabular e pequeno** (~10 mil clientes), o sinal é majoritariamente **linear**, e o
valor de negócio depende tanto de acertar quanto de **explicar** por que um cliente foi
sinalizado. Nesse cenário, um modelo linear regularizado entrega quase toda a acurácia
alcançável com uma fração da complexidade — e coeficientes que o time de negócio
consegue ler.

O teste técnico cita TensorFlow, então vale registrar honestamente **o que o TF traria**
e **quando** ele passaria a valer a pena. É uma decisão de engenharia — escolher a
ferramenta certa para o problema —, não uma limitação.

## A equivalência

Antes dos contrastes, um ponto que desarma a falsa dicotomia: uma regressão logística
**é** uma rede neural de uma única camada. Em Keras, o modelo entregue seria:

```python
model = keras.Sequential([
    keras.layers.Input(shape=(n_features,)),
    keras.layers.Dense(1, activation="sigmoid"),
])
model.compile(optimizer="adam", loss="binary_crossentropy")
```

Uma camada `Dense(1, activation="sigmoid")` com perda `binary_crossentropy` otimiza
exatamente o mesmo objetivo que a `LogisticRegression`. Ou seja: **portar o modelo atual
pra TensorFlow é trivial e não traz ganho nenhum** — treinaria mais devagar, dependeria
de mais bibliotecas e chegaria na mesma fronteira de decisão. O ganho do TensorFlow não
está em reescrever este modelo; está nos modelos que o scikit-learn **não** consegue
construir. É disso que a próxima seção trata.

## Pontos fortes do TensorFlow frente ao scikit-learn

### Deep learning de verdade
O scikit-learn cobre modelos lineares, árvores, ensembles e SVMs — não faz aprendizado
profundo. O TensorFlow foi desenhado para isso: redes profundas, *embeddings* de
variáveis categóricas de alta cardinalidade, camadas convolucionais (imagem), recorrentes
e de atenção (sequência, texto). **Quando importa:** se as features fossem texto de
tickets de suporte, cliques em log, ou categóricas com milhares de níveis, uma rede
capturaria interações que um modelo linear não alcança.

### Escala além da memória
O scikit-learn é *single-node* e *in-memory*: o `X` inteiro precisa caber na RAM. O
`tf.data` faz *streaming* de datasets que não cabem em memória, e o `tf.distribute`
distribui o treino por múltiplas GPUs, TPUs ou nós. **Quando importa:** com dezenas de
milhões de clientes e centenas de features, o treino sklearn começa a esbarrar em memória
muito antes do TF.

### Aceleração por hardware
Treino nativo em **GPU/TPU**. Para modelos grandes, é a diferença entre minutos e horas —
ou entre viável e inviável. O scikit-learn roda em CPU. **Quando importa:** modelos
profundos com muitos parâmetros, onde cada época é cara.

### Diferenciação automática e customização
O TensorFlow expõe *autodiff*: você define uma *loss* customizada, uma camada nova ou um
laço de treino sob medida, e o gradiente é calculado automaticamente. Os estimadores do
sklearn são "caixas" com objetivos fixos. **Quando importa:** quando o problema pede uma
função de custo específica de negócio (por exemplo, ponderar erros de churn por *lifetime
value* do cliente de um jeito que a `class_weight` padrão não expressa).

### Aprendizado incremental / online
O treino por *mini-batches* do TF permite atualizar o modelo continuamente conforme novos
dados chegam, sem re-treinar do zero. Churn é um alvo que **deriva no tempo** (o
comportamento do cliente muda), então retraining incremental é um recurso natural. A
maioria dos estimadores do sklearn re-ajusta o modelo inteiro a cada atualização.

### Ecossistema de produção
O TensorFlow carrega um ecossistema de *deployment* maduro: **SavedModel** + **TensorFlow
Serving** (serving de alta performance, com *batching* e versionamento), **TFLite**
(mobile/edge), **TF.js** (browser) e **TFX** (pipelines de produção com validação de
dados e análise de modelo). O caminho do sklearn para produção — como o desta entrega,
via MLflow + FastAPI — é perfeitamente funcional, mas mais montado à mão. **Quando
importa:** serving em altíssima escala, inferência no dispositivo, ou pipelines de ML
padronizados de ponta a ponta.

### Integração com Vertex AI
Como este projeto espelha o Vertex AI, vale notar: o Vertex treina containers TensorFlow
**nativamente**, oferece AutoML e serviços gerenciados afinados no ecossistema TF/Keras.
Num ambiente GCP de verdade, um modelo TF encaixa nos trilhos gerenciados do Vertex com
menos fricção.

## O trade-off neste caso

Para **este** problema, adotar TensorFlow seria *overkill*, e a honestidade pede dizer
por quê:

- **Sem ganho de acurácia.** Com sinal linear e ~10k linhas, uma rede profunda tende a
  empatar (ou pior, *overfittar*) frente à regressão logística regularizada.
- **Mais complexidade operacional.** Dependências pesadas (TensorFlow, e CUDA para GPU),
  mais versões para gerenciar, treino mais lento.
- **Menos interpretabilidade.** Perde-se a leitura direta dos coeficientes, que aqui tem
  valor de negócio.

A regra prática que este projeto adota: **comece simples; migre para o TensorFlow
quando** —

1. o sinal for claramente **não-linear e complexo**, e árvores/*boosting* já não bastarem;
2. os dados **não couberem em memória** ou o treino precisar de escala distribuída;
3. a modalidade for **texto, imagem ou sequência**;
4. o *serving* exigir **edge/mobile** ou escala/latência que o TF Serving atende melhor.

Nenhuma dessas condições vale para o churn tabular de hoje — por isso, scikit-learn.

## Como a migração se encaixaria no MLOps existente

Um ponto tranquilizador: **trocar o modelo não obriga a reescrever o MLOps**. A
arquitetura desta entrega é agnóstica de framework:

- **Registro:** o MLflow versiona modelos TensorFlow (`mlflow.tensorflow`) do mesmo jeito
  que versiona sklearn (`mlflow.sklearn`) — mesmo *registry*, mesmos *aliases*
  (`@production`).
- **Serving:** a mesma API FastAPI carregaria o modelo pelo *alias* do MLflow; o contrato
  de entrada/saída não muda.
- **Orquestração:** no DAG do KFP, apenas o *step* de treino (`train_model_op`) trocaria; os
  passos de preparação, split, avaliação e registro permanecem.

Ou seja, a decisão "sklearn vs TF" é local ao passo de treino. O restante do ciclo de
vida — *feature store*, *pipeline*, *registry*, *serving*, *monitoring* — foi construído
para não depender dela. (Esta seção descreve o encaixe; a migração não é executada neste
repositório.)

## Escopo

Este documento é uma **análise de decisão**, não uma implementação. Não há exemplo Keras
executável, teste de paridade, dependência opcional de TensorFlow nem *target* de
`make` associado. O modelo entregue e testado continua sendo a regressão logística do
scikit-learn; `make ci` e `make test` seguem sem qualquer dependência de TensorFlow.
