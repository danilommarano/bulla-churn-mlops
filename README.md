# Teste Técnico Bullla: Machine Learning Engineer (MLOps)

O Bullla está desenvolvendo um modelo de **previsão de churn/turnover de clientes** utilizando regressão logística.

O modelo gera a probabilidade de o cliente encerrar o relacionamento baseado em um **score que varia de 0 a 10**, em que quanto mais alto o score, menor a chance de turnover e melhor o cliente.

Um dos cientistas de dados do seu time elaborou os seguintes scripts:

- `train_model_churn.py`: lê `Customer-Churn-Records.csv`, faz feature engineering, treina o modelo, reporta a acurácia e salva `model.pkl`
- `infer_model_churn.py`: carrega o pickle, prepara uma base, gera o score e grava `predictions.csv`

Como Engenheiro de Machine Learning do time, você é responsável por trabalhar na otimização e produtização do modelo, contemplando as boas práticas de desenvolvimento de Software e MLOps. 

Você tem a liberdade criativa de propor melhorias não só para a inferência e disponibilização do modelo, como também no próprio script de treinamento, auditando todo o trabalho que foi desenvolvido.

Ademais, você também pode atuar em possíveis automações na configuração e setup do ambiente, bem como na gerência do ciclo de vida de modelos.

Porém, sua missão **não** é trocar a regressão por um algoritmo mais complexo como XGBoost e nem caçar o melhor hiperparâmetro. O foco é criar uma pipeline robusta de MLOps, garantindo o ciclo de vida do modelo.

## Entregáveis

- Repositório Git com a solução proposta.
- README com:
  - Problemas conceituais encontrados no script original
  - Desenho da arquitetura
  - Explicação e justificativa das etapas e das melhorias que foram propostas
  - Como rodar a solução

A solução deve poder ser executada localmente e não é necessário nenhuma integração com provedores Cloud.

## Arquivos do teste

| Arquivo | Descrição |
|---------|-----------|
| `train_model_churn.py` | Script legado de treino |
| `infer_model_churn.py` | Script legado de inferência batch |
| `Customer-Churn-Records.csv` | Base de dados de clientes |

Boa sorte!
