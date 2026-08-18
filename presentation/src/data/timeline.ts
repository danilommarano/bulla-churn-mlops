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
