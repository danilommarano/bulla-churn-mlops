export interface SlideMeta {
  id: number;
  kicker?: string;
  title: string;
  subtitle?: string;
  track5: boolean;
  notes: string;
}

export const SLIDES: SlideMeta[] = [
  { id: 1, kicker: "Churn MLOps", title: "Produtização do modelo de turnover", subtitle: "Teste técnico Bulla · pipeline de MLOps 100% local", track5: true, notes: "Abertura. Uma frase: peguei dois scripts legados e transformei numa esteira MLOps local que espelha o Vertex AI." },
  { id: 2, kicker: "O problema", title: "O que a Bulla pediu", subtitle: "Duas restrições: não trocar o modelo, rodar 100% local", track5: true, notes: "Produtizar o modelo existente. Restrição 1: manter a regressão logística. Restrição 2: espelho local do Vertex, sem nuvem paga." },
  { id: 3, kicker: "O ponto de partida", title: "O modelo que recebi", subtitle: "Regressão logística de churn, servida como um único Pipeline sklearn", track5: true, notes: "Detalhe do modelo recebido antes da estratégia. Regressão logística com class_weight=balanced; alvo turnover (0/1); 12 colunas de entrada + 3 features derivadas. A restrição do teste é não trocar o modelo — então produtizo ESTE, sem re-fit na inferência." },
  { id: 4, kicker: "A grande sacada", title: "Cada peça do Vertex tem um par open-source", subtitle: "Passe o mouse para acender o par; clique para ir ao slide", track5: true, notes: "O mapa é o coração da estratégia: KFP=Pipelines, MLflow=Experiments+Registry, Feast=Feature Store, Evidently=Monitoring, FastAPI+Docker=Endpoint, Actions/Jenkins=CI/CD." },
  { id: 5, kicker: "A auditoria", title: "Achamos 10 bugs nos scripts originais", subtitle: "Cada um virou uma decisão de arquitetura", track5: false, notes: "Transição. Os 10 problemas conceituais não são teóricos — cada correção é uma peça concreta da esteira." },
  { id: 6, kicker: "Bugs antes→depois", title: "Do bug à correção, em código", subtitle: "Use ←/→ para navegar pelos 10", track5: true, notes: "Coração técnico. Foque no bug #1 (leakage) e #5 (métrica enganosa) se o tempo apertar. O contador mostra N/10." },
  { id: 7, kicker: "A arquitetura", title: "O ciclo de vida completo", subtitle: "Treino → registro → serving → monitoramento → re-treino", track5: false, notes: "Diagrama Mermaid. Mostra o loop fechado: o gate de drift dispara o re-treino." },
  { id: 8, kicker: "Modelo ao vivo", title: "Fale com o modelo real", subtitle: "3 perfis, 4 ajustes, score 0–10 na hora", track5: true, notes: "Demo. Clique nos presets. Se a API estiver no ar, os sliders recalculam ao vivo; senão, cai no fallback pré-gravado real. Nunca quebra." },
  { id: 9, kicker: "Métricas honestas", title: "Accuracy engana; recall é o que importa", subtitle: "Base ~20% churn — 0.67 de accuracy esconde o jogo", track5: false, notes: "Resolve o bug #5. AUC 0.76, recall 0.74. A matriz mostra: pegamos 302 de 408 que saem (recall), ao custo de falsos positivos — trade-off consciente com class_weight=balanced." },
  { id: 10, kicker: "O momento do drift", title: "O gate que falha de propósito", subtitle: "make monitor (PASS) vs make monitor-drift (ALERT, exit 2)", track5: true, notes: "Clímax. Saudável: drift_share 0.000, exit 0. Drift simulado: drift_share 0.462, 6 colunas, exit 2 — e a qualidade despenca (accuracy 0.23, recall 0.99 = prevê que todos saem). É o gatilho de re-treino." },
  { id: 11, kicker: "O que foi construído", title: "9 marcos, 78 testes, CI verde", subtitle: "Cada marco: issue → branch → PR → merge", track5: false, notes: "Timeline dos 9 marcos entregues. Enfatize disciplina: tudo por PR, testado, CI espelhando o pipeline." },
  { id: 12, kicker: "Limitações honestas", title: "O que ficou documentado, não construído", subtitle: "K8s, captura de predições, scheduler gerenciado", track5: false, notes: "Honestidade: espelho local. A API não captura predições (holdout como proxy); sem scheduler gerenciado; K8s descrito como próximo passo, não construído." },
  { id: 13, kicker: "Fecho", title: "De frágil a confiável", subtitle: "Esteira reprodutível, testada e observável — pronta para a nuvem", track5: true, notes: "Recap: mapa Vertex↔OSS, 10 bugs corrigidos, loop de monitoramento fechado. Próximo passo: subir os equivalentes gerenciados no Vertex de verdade." },
];
