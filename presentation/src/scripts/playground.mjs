// Decisão pura: resposta ao vivo tem prioridade; senão, fallback pré-gravado do preset.
export function pickResult(preset, liveResponse) {
  if (liveResponse) return { data: liveResponse, live: true };
  return { data: preset.response, live: false };
}
