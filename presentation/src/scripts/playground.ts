import { pickResult } from "./playground.mjs";

interface PresetResponse {
  turnover_pred: number;
  prob_churn: number;
  score_retencao: number;
}
interface Preset {
  id: string;
  label: string;
  request: Record<string, unknown>;
  response: PresetResponse;
}

const API = "http://localhost:8000/predict";
const ARC = 282.74;
const SLIDERS = ["Age", "Balance", "NumOfProducts", "Satisfaction Score"];

function initPlayground() {
  const root = document.getElementById("playground");
  if (!root) return;
  const presets: Preset[] = JSON.parse(root.dataset.presets ?? "[]");
  let current: Preset = presets[0];

  const seal = document.getElementById("pg-seal")!;
  const predEl = document.getElementById("pg-pred")!;
  const probEl = document.getElementById("pg-prob")!;
  const arc = document.getElementById("pg-gauge-arc") as unknown as SVGPathElement | null;
  const valueEl = document.getElementById("pg-gauge-value")!;

  function paint(data: PresetResponse, live: boolean) {
    predEl.textContent = data.turnover_pred === 1 ? "Sai (churn)" : "Fica";
    probEl.textContent = `${(data.prob_churn * 100).toFixed(1)}%`;
    valueEl.textContent = String(data.score_retencao);
    if (arc) arc.style.strokeDashoffset = String(ARC * (1 - data.score_retencao / 10));
    seal.hidden = live;
  }

  function currentRequest(): Record<string, unknown> {
    const req = { ...current.request };
    for (const key of SLIDERS) {
      const el = document.querySelector<HTMLInputElement>(`[data-slider="${key}"]`);
      if (el) req[key] = key === "Balance" ? Number(el.value) : Number(el.value);
    }
    return req;
  }

  async function run() {
    let live: PresetResponse | null = null;
    try {
      const res = await fetch(API, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify([currentRequest()]),
      });
      if (res.ok) live = (await res.json())[0];
    } catch {
      live = null; // API fora / sem CORS / offline -> fallback
    }
    const { data, live: isLive } = pickResult(current, live);
    paint(data, isLive);
  }

  function selectPreset(p: Preset) {
    current = p;
    for (const key of SLIDERS) {
      const el = document.querySelector<HTMLInputElement>(`[data-slider="${key}"]`);
      if (el && key in p.request) el.value = String(p.request[key]);
    }
    run();
  }

  root.querySelectorAll<HTMLButtonElement>("[data-preset]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const p = presets.find((x) => x.id === btn.dataset.preset);
      if (p) selectPreset(p);
    });
  });

  let t: number | undefined;
  root.querySelectorAll<HTMLInputElement>("[data-slider]").forEach((el) => {
    el.addEventListener("input", () => {
      window.clearTimeout(t);
      t = window.setTimeout(run, 250);
    });
  });

  selectPreset(current);
}

if (document.readyState !== "loading") initPlayground();
else document.addEventListener("DOMContentLoaded", initPlayground);
