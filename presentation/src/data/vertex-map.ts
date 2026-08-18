export interface VertexPair {
  vertex: string;
  oss: string;
  role: string;
  slide?: number;
}

export const VERTEX_MAP: VertexPair[] = [
  { vertex: "Vertex Pipelines", oss: "KFP SDK", role: "Orquestra as etapas do treino", slide: 6 },
  { vertex: "Vertex AI Experiments", oss: "MLflow Tracking", role: "Rastreia params/métricas/artefatos", slide: 10 },
  { vertex: "Vertex Model Registry", oss: "MLflow Model Registry", role: "Versiona e promove modelos (@production)", slide: 8 },
  { vertex: "Vertex AI Feature Store", oss: "Feast", role: "Serve features offline/online sem skew", slide: 5 },
  { vertex: "Vertex Model Monitoring", oss: "Evidently AI", role: "Detecta data/target/score drift", slide: 9 },
  { vertex: "Vertex Endpoint", oss: "FastAPI + Docker", role: "Serve predições REST", slide: 7 },
  { vertex: "Cloud Build / Vertex CI", oss: "GitHub Actions + Jenkinsfile", role: "CI/CD", slide: 10 },
];
