# Variante TensorFlow do modelo de churn — Design (Marco 8)

> Entregável do teste técnico Bulla (ML Engineer / MLOps). Documenta a variante
> **TensorFlow/Keras** do modelo entregue (regressão logística sklearn) com **prova
> empírica de equivalência**, sem trocar o modelo de produção.

## 1. Objetivo

O teste técnico menciona TensorFlow. O modelo entregue é uma regressão logística do
scikit-learn (decisão da spec principal §7). Este marco fecha a lacuna **documentando**
como esse mesmo modelo seria construído em TF/Keras e **provando** que são o mesmo
modelo — sem promover o TF a dependência de primeira classe nem tocar no que roda em
produção.

A equivalência é exata em teoria: uma regressão logística **é** uma única camada densa
com ativação sigmoide treinada com perda de entropia cruzada binária. A prova é
empírica: treinar as duas versões sobre o mesmo `X` e mostrar que concordam.

**Restrição central:** `make ci` e `make test` seguem **TF-free e verdes**. O
TensorFlow é um extra opcional; o teste de paridade dá skip quando ele está ausente.
A prova roda sob demanda em um comando (`make tf-variant`).

## 2. Mapeamento Vertex AI ↔ este repo

| Vertex AI / GCP | Equivalente neste repo | Gap honesto |
|---|---|---|
| Vertex Training com container TF (treina Keras nativamente) | Variante Keras **documentada + verificável** (`examples/tensorflow_variant.py`) | Roda localmente sob demanda, não como job de treino gerenciado |
| Model Registry (framework-agnóstico: sklearn, TF, XGBoost) | MLflow já registra o modelo sklearn; o doc mostra que trocar pra `mlflow.tensorflow` seria simétrico | O flavor TF é **explicado, não executado** |
| Prediction container (serving agnóstico de framework) | A mesma API FastAPI serviria o modelo TF sem mudança de contrato | Não empacotamos um container servindo TF |

Mensagem: no Vertex, trocar sklearn por TF é trocar o container de treino e o flavor do
registry — o resto do MLOps (pipeline, registry, serving, monitoring) não muda. Este
marco demonstra isso localmente com a peça central (o modelo) de fato treinada e
verificada.

## 3. A equivalência (o que muda e o que é reusado)

O pipeline de treino atual (`churn.training.pipeline.build_pipeline`) é um `Pipeline`
sklearn de três passos:

```
features (ChurnFeatureBuilder) -> preprocess (ColumnTransformer) -> model (LogisticRegression)
```

**Só o último passo muda.** A variante TF reusa `features` + `preprocess` sem alteração
e troca `LogisticRegression` por uma camada `Dense(1, activation="sigmoid")` treinada
com `binary_crossentropy`. Isso garante que a comparação isola o classificador: mesmas
features, mesmo scaling, mesmo one-hot, mesmo split.

Como obter o `X` transformado reusando o código existente: cortar a cabeça do pipeline
sklearn já ajustado.

```python
sk_pipeline = build_pipeline(random_state=cfg.random_state, n_age_bins=cfg.n_age_bins)
sk_pipeline.fit(X_train, y_train)          # ajusta features + preprocess + LogReg
preproc = sk_pipeline[:-1]                  # sub-pipeline já ajustado: features + preprocess
X_train_t = preproc.transform(X_train)
X_test_t = preproc.transform(X_test)
```

**Gotcha (densificação):** o `OneHotEncoder` do `ColumnTransformer` produz saída
**esparsa** por padrão; Keras precisa de denso. Densificar após o transform:

```python
import scipy.sparse as sp
if sp.issparse(X_train_t):
    X_train_t = X_train_t.toarray()
    X_test_t = X_test_t.toarray()
```

**Espelhar `class_weight="balanced"`:** o LogReg usa pesos balanceados; para paridade
justa, o Keras recebe os mesmos pesos via `class_weight` no `.fit`, computados como o
sklearn faz (`n_samples / (n_classes * bincount(y))`):

```python
from sklearn.utils.class_weight import compute_class_weight
classes = np.array([0, 1])
weights = compute_class_weight("balanced", classes=classes, y=y_train)
class_weight = {0: weights[0], 1: weights[1]}
```

**Determinismo:** `keras.utils.set_random_seed(cfg.random_state)` antes de construir o
modelo, para a prova ser reproduzível.

### 3.1 O modelo Keras

Uma única camada densa, sem camadas ocultas — é literalmente a regressão logística:

```python
model = keras.Sequential([
    keras.layers.Input(shape=(X_train_t.shape[1],)),
    keras.layers.Dense(1, activation="sigmoid"),
])
model.compile(optimizer="adam", loss="binary_crossentropy")
model.fit(
    X_train_t, y_train,
    epochs=100, batch_size=64,
    class_weight=class_weight, verbose=0,
)
proba_keras = model.predict(X_test_t, verbose=0).ravel()
```

Os hiperparâmetros de otimização (optimizer, epochs, batch_size) são livres: o objetivo
é convergir a fronteira de decisão perto da do LBFGS, não replicar o otimizador. A
prova mede concordância nas **predições**, não nos pesos (LBFGS vs Adam nunca chegam
aos mesmos pesos).

## 4. Artefatos

Quatro artefatos + um aditivo no Makefile. Princípio: **reusar** o preprocessing
existente; **isolar** o TF atrás de um extra opcional.

### 4.1 `pyproject.toml` (aditivo)

```toml
[project.optional-dependencies]
tf = ["tensorflow-cpu>=2.16"]
```

`tensorflow-cpu` (não `tensorflow` cheio): a prova roda em CPU, sem CUDA, e o pacote é
menor. Keras 3 já vem embutido no TF ≥ 2.16 (`from tensorflow import keras`). O install
default (`uv sync`) e o `make ci` **não** instalam isso — só `uv sync --extra tf`.

### 4.2 `examples/tensorflow_variant.py`

Script rodável (não módulo do pacote — é um exemplo). Estrutura:

1. Carrega os dados e faz o **mesmo split** do `train.py` (`load_raw`, `INPUT_COLUMNS`,
   `df["turnover"]`, `train_test_split` com `test_size`/`random_state`/`stratify=y` de
   `Settings`).
2. Ajusta `build_pipeline(...)` completo → obtém `proba_sklearn` e `auc_sklearn` no teste.
3. Corta a cabeça (`sk_pipeline[:-1]`), transforma e densifica → `X_train_t`, `X_test_t`.
4. Treina o Keras `Dense(1, sigmoid)` com `class_weight` balanceado → `proba_keras`.
5. Calcula `auc_keras` (via `roc_auc_score`) e `pearson(proba_sklearn, proba_keras)`.
6. Imprime os três números e um veredito de paridade legível.

Expõe uma função reutilizável (ex.: `run_parity()`) que retorna um dict com
`auc_sklearn`, `auc_keras`, `auc_gap`, `pearson` — para o teste importar sem duplicar
lógica. O `if __name__ == "__main__"` chama e imprime.

### 4.3 `tests/test_tensorflow_variant.py`

```python
import pytest

tf = pytest.importorskip("tensorflow")  # skipa o módulo inteiro se o TF não estiver instalado
```

Um teste que chama `run_parity()` e afirma:

- `abs(result["auc_keras"] - result["auc_sklearn"]) < 0.02`
- `result["pearson"] > 0.98`

Como o `importorskip` está no topo, `make test`/`make ci` sem o extra `tf` **coletam 0
testes deste arquivo** (skip limpo, verde). Com o extra instalado, o teste roda e prova
a paridade de verdade.

### 4.4 `docs/tensorflow_variant.md`

O documento voltado ao avaliador. Seções:

1. **Decisão e restrição** — por que sklearn é o modelo entregue e por que o TF é
   documentado + verificável, não promovido a produção.
2. **Equivalência matemática** — LogReg = camada densa sigmoide + BCE; a fórmula e a
   intuição.
3. **O que muda** — só a cabeça classificadora; features/split/scaling reusados
   (referência ao `build_pipeline()[:-1]`).
4. **A prova** — como rodar (`make tf-variant`), o que ela afirma (gap de AUC < 0.02 e
   correlação de Pearson > 0.98) e os números observados na execução real (preenchidos
   com a saída de verdade, não inventados).
5. **Como plugaria no resto do MLOps** — `mlflow.tensorflow` no lugar de
   `mlflow.sklearn`; a mesma API FastAPI serve o modelo; no KFP só o step `train_model`
   troca. Explicado, não executado.
6. **Mapeamento Vertex** — a tabela da §2 deste design.

### 4.5 `Makefile` (aditivo)

```make
tf-variant: ## Instala o extra TF e roda a prova de paridade sklearn vs Keras (sob demanda)
	uv sync --extra tf
	uv run python examples/tensorflow_variant.py
	uv run pytest tests/test_tensorflow_variant.py -v
```

`.PHONY` ganha `tf-variant`. Reproduz a prova em um comando.

## 5. Verificação (honesta)

A prova é operacional, em camadas:

1. **`make test` verde sem o extra TF** — confirma que o teste de paridade dá skip
   limpo e não quebra a suíte TF-free (o CI segue idêntico).
2. **`make tf-variant` verde com o extra TF** — instala `tensorflow-cpu`, roda o
   exemplo (imprime os números reais) e o teste (afirma as tolerâncias). Esta é a prova
   de que a equivalência se sustenta empiricamente.
3. **Números reais no doc** — os valores de AUC e Pearson em `tensorflow_variant.md` são
   os observados na execução de verdade, colados da saída — não estimativas.

Ponto a confirmar na implementação: que o `tensorflow-cpu` instala de fato no ambiente
(Python 3.12, Linux) e que os números batem as tolerâncias. Se a correlação ficar
abaixo de 0.98 ou o gap de AUC acima de 0.02, ajustar epochs/batch_size (não afrouxar a
tolerância sem justificativa) até convergir — a fronteira de decisão de uma LogReg bem
condicionada é reproduzível por SGD/Adam com folga.

## 6. Fora de escopo (YAGNI)

- Integrar o TF ao pipeline de treino, ao DAG KFP ou ao serving de produção (continuam
  sklearn).
- Log real no MLflow via `mlflow.tensorflow` (explicado no doc, não executado).
- GPU / CUDA (a prova roda em CPU).
- Tentar bater os pesos entre LBFGS e Adam (a prova é sobre predições, não parâmetros).
- Adicionar o TF ao `make ci` ou ao workflow do GitHub Actions (a suíte segue TF-free).
- Tuning de arquitetura (camadas ocultas, regularização) — descaracterizaria a
  equivalência com a regressão logística.
