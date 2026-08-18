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
