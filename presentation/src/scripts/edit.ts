// Modo edição do deck (tecla E): edita texto inline, cor/fonte global e ordem
// dos slides, com persistência em localStorage e exportação para overrides.json.
// Aplicação em runtime: overrides.json (versionado) + localStorage (sessão) por cima.
import committed from "../data/overrides.json";

export interface Theme {
  accent?: string;
  bg?: string;
  ink?: string;
  muted?: string;
  font?: string; // chave de FONTS ou stack CSS
  fontDisplay?: string;
}
export interface Overrides {
  theme?: Theme;
  text?: Record<string, string>;
  order?: number[];
}

const LS_KEY = "deck-overrides-v1";

const FONTS: Record<string, string> = {
  sans: 'ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif',
  serif: 'ui-serif, Georgia, Cambria, "Times New Roman", serif',
  mono: 'ui-monospace, "SF Mono", Menlo, Consolas, monospace',
};

const DEFAULTS = {
  accent: "#c15f3c",
  bg: "#f0eee6",
  ink: "#1f1e1c",
  font: "sans",
  fontDisplay: "serif",
};

function readLocal(): Overrides {
  try {
    return JSON.parse(localStorage.getItem(LS_KEY) ?? "{}") ?? {};
  } catch {
    return {};
  }
}
function writeLocal(o: Overrides) {
  localStorage.setItem(LS_KEY, JSON.stringify(o));
}

/** overrides.json (versionado) com localStorage (sessão) sobrescrevendo. */
export function getMerged(): Overrides {
  const c = committed as Overrides;
  const l = readLocal();
  return {
    theme: { ...(c.theme ?? {}), ...(l.theme ?? {}) },
    text: { ...(c.text ?? {}), ...(l.text ?? {}) },
    order: l.order && l.order.length ? l.order : (c.order ?? []),
  };
}

function applyTheme(t: Theme) {
  const root = document.documentElement.style;
  if (t.accent) root.setProperty("--accent", t.accent);
  if (t.bg) root.setProperty("--bg", t.bg);
  if (t.ink) root.setProperty("--ink", t.ink);
  if (t.muted) root.setProperty("--muted", t.muted);
  if (t.font) root.setProperty("--font", FONTS[t.font] ?? t.font);
  if (t.fontDisplay) root.setProperty("--font-display", FONTS[t.fontDisplay] ?? t.fontDisplay);
}

function applyText(text: Record<string, string>) {
  for (const [key, val] of Object.entries(text)) {
    const el = document.querySelector<HTMLElement>(`[data-edit="${CSS.escape(key)}"]`);
    if (el) el.textContent = val;
  }
}

function applyOrder(order: number[]) {
  const deck = document.querySelector<HTMLElement>(".deck");
  if (!deck) return;
  const byId = new Map<number, HTMLElement>();
  deck.querySelectorAll<HTMLElement>(".slide").forEach((s) => byId.set(Number(s.dataset.slideId), s));
  for (const id of order) {
    const el = byId.get(id);
    if (el) deck.appendChild(el); // reanexa na ordem desejada
  }
}

/** Aplica os overrides efetivos ao DOM. Chamado no início da init do deck. */
export function applyStoredOverrides() {
  const o = getMerged();
  if (o.order && o.order.length) applyOrder(o.order);
  applyTheme(o.theme ?? {});
  applyText(o.text ?? {});
}

function patchTheme(part: Theme) {
  const l = readLocal();
  l.theme = { ...(l.theme ?? {}), ...part };
  writeLocal(l);
  applyTheme(part);
}
function patchText(key: string, val: string) {
  const l = readLocal();
  l.text = { ...(l.text ?? {}), [key]: val };
  writeLocal(l);
}
/** Persiste a nova ordem dos slides (chamado pelo deck ao mover um slide). */
export function saveOrder(order: number[]) {
  const l = readLocal();
  l.order = order;
  writeLocal(l);
}

function exportOverrides() {
  const data = JSON.stringify(getMerged(), null, 2);
  const blob = new Blob([data], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "overrides.json";
  a.click();
  URL.revokeObjectURL(a.href);
}

export interface EditCtx {
  moveCurrent: (delta: number) => void;
}

const PANEL_HTML = `
<div style="font-weight:600;margin-bottom:10px;font-size:13px">Modo edição</div>
<label style="display:flex;align-items:center;justify-content:space-between;margin:6px 0">Acento
  <input type="color" id="edit-accent" style="width:34px;height:24px;border:none;background:none;padding:0"></label>
<label style="display:flex;align-items:center;justify-content:space-between;margin:6px 0">Fundo
  <input type="color" id="edit-bg" style="width:34px;height:24px;border:none;background:none;padding:0"></label>
<label style="display:flex;align-items:center;justify-content:space-between;margin:6px 0">Texto
  <input type="color" id="edit-ink" style="width:34px;height:24px;border:none;background:none;padding:0"></label>
<label style="display:flex;align-items:center;justify-content:space-between;margin:6px 0">Fonte corpo
  <select id="edit-font" style="font:inherit"><option value="sans">Sans</option><option value="serif">Serif</option><option value="mono">Mono</option></select></label>
<label style="display:flex;align-items:center;justify-content:space-between;margin:6px 0">Fonte títulos
  <select id="edit-font-display" style="font:inherit"><option value="sans">Sans</option><option value="serif">Serif</option><option value="mono">Mono</option></select></label>
<div style="display:flex;align-items:center;justify-content:space-between;margin:10px 0 6px">Slide atual
  <span><button id="edit-move-left" style="font:inherit;cursor:pointer;border:1px solid rgba(0,0,0,.2);border-radius:6px;padding:2px 8px;background:#fff">◀</button>
  <button id="edit-move-right" style="font:inherit;cursor:pointer;border:1px solid rgba(0,0,0,.2);border-radius:6px;padding:2px 8px;background:#fff">▶</button></span></div>
<button id="edit-export" style="width:100%;margin-top:8px;font:inherit;cursor:pointer;border:none;border-radius:8px;padding:8px;background:#1f1e1c;color:#fff">Exportar overrides.json</button>
<button id="edit-reset" style="width:100%;margin-top:6px;font:inherit;cursor:pointer;border:1px solid rgba(0,0,0,.2);border-radius:8px;padding:6px;background:#fff">Resetar edições</button>
<p style="margin:10px 0 0;font-size:11px;color:#6f6d64">Clique nos textos para editar · <b>E</b> sai</p>
`;

function buildPanel(): HTMLElement {
  const panel = document.createElement("div");
  panel.id = "edit-panel";
  panel.style.cssText =
    "position:fixed;top:16px;right:16px;z-index:60;width:240px;background:#fff;color:#1f1e1c;" +
    "border:1px solid rgba(0,0,0,.15);border-radius:12px;box-shadow:0 10px 40px rgba(0,0,0,.18);" +
    'padding:14px 16px;font:13px/1.4 ui-sans-serif,system-ui,sans-serif;display:none';
  panel.innerHTML = PANEL_HTML;
  return panel;
}

export function initEditMode(ctx: EditCtx) {
  const panel = buildPanel();
  document.body.appendChild(panel);
  let on = false;

  const merged = getMerged();
  const theme = merged.theme ?? {};
  const $ = <T extends HTMLElement>(id: string) => panel.querySelector<T>("#" + id)!;

  const accent = $<HTMLInputElement>("edit-accent");
  const bg = $<HTMLInputElement>("edit-bg");
  const ink = $<HTMLInputElement>("edit-ink");
  const font = $<HTMLSelectElement>("edit-font");
  const fontDisplay = $<HTMLSelectElement>("edit-font-display");

  accent.value = theme.accent ?? DEFAULTS.accent;
  bg.value = theme.bg ?? DEFAULTS.bg;
  ink.value = theme.ink ?? DEFAULTS.ink;
  font.value = theme.font ?? DEFAULTS.font;
  fontDisplay.value = theme.fontDisplay ?? DEFAULTS.fontDisplay;

  accent.addEventListener("input", () => patchTheme({ accent: accent.value }));
  bg.addEventListener("input", () => patchTheme({ bg: bg.value }));
  ink.addEventListener("input", () => patchTheme({ ink: ink.value }));
  font.addEventListener("change", () => patchTheme({ font: font.value }));
  fontDisplay.addEventListener("change", () => patchTheme({ fontDisplay: fontDisplay.value }));

  $<HTMLButtonElement>("edit-move-left").addEventListener("click", () => ctx.moveCurrent(-1));
  $<HTMLButtonElement>("edit-move-right").addEventListener("click", () => ctx.moveCurrent(1));
  $<HTMLButtonElement>("edit-export").addEventListener("click", exportOverrides);
  $<HTMLButtonElement>("edit-reset").addEventListener("click", () => {
    localStorage.removeItem(LS_KEY);
    location.reload();
  });

  const editables = () => Array.from(document.querySelectorAll<HTMLElement>("[data-edit]"));

  function setEditing(state: boolean) {
    on = state;
    panel.style.display = state ? "block" : "none";
    document.body.classList.toggle("deck-editing", state);
    editables().forEach((el) => {
      el.contentEditable = state ? "true" : "false";
      if (state && !el.dataset.editBound) {
        el.dataset.editBound = "1";
        el.addEventListener("input", () => patchText(el.dataset.edit!, el.textContent ?? ""));
      }
    });
  }

  document.addEventListener("keydown", (e) => {
    if (e.key.toLowerCase() !== "e") return;
    const t = e.target as HTMLElement | null;
    if (t && (t.isContentEditable || /^(INPUT|SELECT|TEXTAREA)$/.test(t.tagName))) return;
    e.preventDefault();
    setEditing(!on);
  });

  setEditing(false);
}
