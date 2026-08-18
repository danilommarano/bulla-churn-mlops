import { test } from "node:test";
import assert from "node:assert/strict";
import { pickResult } from "./playground.mjs";

const preset = { response: { turnover_pred: 1, prob_churn: 0.55, score_retencao: 5 } };

test("usa a resposta ao vivo quando a API responde", () => {
  const live = { turnover_pred: 0, prob_churn: 0.2, score_retencao: 8 };
  const r = pickResult(preset, live);
  assert.deepEqual(r, { data: live, live: true });
});

test("cai no fallback pré-gravado quando não há resposta ao vivo", () => {
  const r = pickResult(preset, null);
  assert.deepEqual(r, { data: preset.response, live: false });
});
