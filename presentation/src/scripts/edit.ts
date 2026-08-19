// Modo edição do deck (tecla E): edita texto inline, cor/fonte global, alinhamento
// e tamanho por elemento, alinhamento vertical do slide e ordem de slides/blocos.
// Persistência: localStorage (sessão) por cima de overrides.json (versionado).
// Aplicação em runtime, sem build-time merge.
import committed from "../data/overrides.json";

export type Align = "left" | "center" | "right";
export type SlideAlign = "top" | "center";

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
  align?: Record<string, Align>;
  size?: Record<string, number>; // passo; mult = 1.1^passo
  slideAlign?: Record<string, SlideAlign>;
  order?: number[]; // ordem dos slides (por id)
  blocks?: Record<string, string[]>; // ordem dos blocos por slide (por block-id)
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

const ALIGN_SELF: Record<Align, string> = {
  left: "flex-start",
  center: "center",
  right: "flex-end",
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
    align: { ...(c.align ?? {}), ...(l.align ?? {}) },
    size: { ...(c.size ?? {}), ...(l.size ?? {}) },
    slideAlign: { ...(c.slideAlign ?? {}), ...(l.slideAlign ?? {}) },
    order: l.order && l.order.length ? l.order : (c.order ?? []),
    blocks: { ...(c.blocks ?? {}), ...(l.blocks ?? {}) },
  };
}

// ---- aplicação ao DOM ----

function elByKey(key: string) {
  return document.querySelector<HTMLElement>(`[data-edit="${CSS.escape(key)}"]`);
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
    const el = elByKey(key);
    if (el) el.textContent = val;
  }
}

function alignEl(el: HTMLElement, val: Align) {
  el.style.alignSelf = ALIGN_SELF[val];
  el.style.textAlign = val;
}
function applyAlign(map: Record<string, Align>) {
  for (const [key, val] of Object.entries(map)) {
    const el = elByKey(key);
    if (el) alignEl(el, val);
  }
}

function sizeEl(el: HTMLElement, step: number) {
  if (!el.dataset.baseFs) {
    el.dataset.baseFs = String(parseFloat(getComputedStyle(el).fontSize));
  }
  const base = parseFloat(el.dataset.baseFs);
  el.style.fontSize = step ? `${(base * Math.pow(1.1, step)).toFixed(2)}px` : "";
}
function applySize(map: Record<string, number>) {
  for (const [key, step] of Object.entries(map)) {
    const el = elByKey(key);
    if (el) sizeEl(el, step);
  }
}

function slideAlignEl(section: HTMLElement, val: SlideAlign) {
  section.style.justifyContent = val === "top" ? "flex-start" : "center";
}
function applySlideAlign(map: Record<string, SlideAlign>) {
  for (const [id, val] of Object.entries(map)) {
    const s = document.querySelector<HTMLElement>(`.slide[data-slide-id="${id}"]`);
    if (s) slideAlignEl(s, val);
  }
}

function tagBlocks() {
  document.querySelectorAll<HTMLElement>("[data-blocks]").forEach((wrap) => {
    const slideId = wrap.closest<HTMLElement>(".slide")?.dataset.slideId;
    Array.from(wrap.children).forEach((c, i) => {
      const el = c as HTMLElement;
      if (!el.dataset.blockId) el.dataset.blockId = `${slideId}:${i}`;
    });
  });
}
function applyBlocks(map: Record<string, string[]>) {
  for (const [slideId, order] of Object.entries(map)) {
    const wrap = document.querySelector<HTMLElement>(`.slide[data-slide-id="${slideId}"] [data-blocks]`);
    if (!wrap) continue;
    const byId = new Map<string, HTMLElement>();
    Array.from(wrap.children).forEach((c) => byId.set((c as HTMLElement).dataset.blockId!, c as HTMLElement));
    for (const id of order) {
      const el = byId.get(id);
      if (el) wrap.appendChild(el);
    }
  }
}

function applyOrder(order: number[]) {
  const deck = document.querySelector<HTMLElement>(".deck");
  if (!deck) return;
  const byId = new Map<number, HTMLElement>();
  deck.querySelectorAll<HTMLElement>(".slide").forEach((s) => byId.set(Number(s.dataset.slideId), s));
  for (const id of order) {
    const el = byId.get(id);
    if (el) deck.appendChild(el);
  }
}

/** Aplica todos os overrides efetivos ao DOM. Chamado no início da init do deck. */
export function applyStoredOverrides() {
  const o = getMerged();
  tagBlocks();
  if (o.order && o.order.length) applyOrder(o.order);
  if (o.blocks) applyBlocks(o.blocks);
  applyTheme(o.theme ?? {});
  applyText(o.text ?? {});
  applyAlign(o.align ?? {});
  applySize(o.size ?? {});
  applySlideAlign(o.slideAlign ?? {});
}

// ---- persistência ----

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
function patchAlign(key: string, val: Align) {
  const l = readLocal();
  l.align = { ...(l.align ?? {}), [key]: val };
  writeLocal(l);
}
function patchSize(key: string, step: number) {
  const l = readLocal();
  l.size = { ...(l.size ?? {}) };
  if (step) l.size[key] = step;
  else delete l.size[key];
  writeLocal(l);
}
function patchSlideAlign(id: string, val: SlideAlign) {
  const l = readLocal();
  l.slideAlign = { ...(l.slideAlign ?? {}), [id]: val };
  writeLocal(l);
}
/** Persiste a nova ordem dos slides (chamado pelo deck ao mover um slide). */
export function saveOrder(order: number[]) {
  const l = readLocal();
  l.order = order;
  writeLocal(l);
}
function saveBlocks(slideId: string, order: string[]) {
  const l = readLocal();
  l.blocks = { ...(l.blocks ?? {}), [slideId]: order };
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

const BTN =
  "font:inherit;cursor:pointer;border:1px solid rgba(0,0,0,.2);border-radius:6px;padding:3px 8px;background:#fff;min-width:30px";

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
<hr style="border:none;border-top:1px solid rgba(0,0,0,.1);margin:10px 0">
<div style="font-size:12px;color:#6f6d64;margin-bottom:6px">Texto selecionado</div>
<div style="display:flex;gap:5px;margin-bottom:6px">
  <button id="edit-align-left" title="Alinhar à esquerda" style="${BTN}">⬅</button>
  <button id="edit-align-center" title="Centralizar" style="${BTN}">⬛</button>
  <button id="edit-align-right" title="Alinhar à direita" style="${BTN}">➡</button>
  <button id="edit-size-down" title="Diminuir" style="${BTN}">A−</button>
  <button id="edit-size-up" title="Aumentar" style="${BTN}">A+</button>
</div>
<div style="display:flex;align-items:center;justify-content:space-between;margin:6px 0">Bloco
  <span><button id="edit-block-up" title="Subir bloco" style="${BTN}">↑</button>
  <button id="edit-block-down" title="Descer bloco" style="${BTN}">↓</button></span></div>
<div style="display:flex;align-items:center;justify-content:space-between;margin:6px 0">Slide
  <span><button id="edit-slide-top" style="${BTN}">Topo</button>
  <button id="edit-slide-center" style="${BTN}">Centro</button></span></div>
<div style="display:flex;align-items:center;justify-content:space-between;margin:6px 0">Ordem do slide
  <span><button id="edit-move-left" style="${BTN}">◀</button>
  <button id="edit-move-right" style="${BTN}">▶</button></span></div>
<button id="edit-export" style="width:100%;margin-top:10px;font:inherit;cursor:pointer;border:none;border-radius:8px;padding:8px;background:#1f1e1c;color:#fff">Exportar overrides.json</button>
<button id="edit-reset" style="width:100%;margin-top:6px;font:inherit;cursor:pointer;border:1px solid rgba(0,0,0,.2);border-radius:8px;padding:6px;background:#fff">Resetar edições</button>
<p style="margin:10px 0 0;font-size:11px;color:#6f6d64">Clique num texto/bloco para selecionar · <b>E</b> sai</p>
`;

function buildPanel(): HTMLElement {
  const panel = document.createElement("div");
  panel.id = "edit-panel";
  panel.style.cssText =
    "position:fixed;top:16px;right:16px;z-index:60;width:250px;max-height:calc(100vh - 32px);overflow:auto;" +
    "background:#fff;color:#1f1e1c;border:1px solid rgba(0,0,0,.15);border-radius:12px;" +
    "box-shadow:0 10px 40px rgba(0,0,0,.18);padding:14px 16px;" +
    'font:13px/1.4 ui-sans-serif,system-ui,sans-serif;display:none';
  panel.innerHTML = PANEL_HTML;
  return panel;
}

export function initEditMode(ctx: EditCtx) {
  const panel = buildPanel();
  document.body.appendChild(panel);
  let on = false;
  let selText: HTMLElement | null = null;
  let selBlock: HTMLElement | null = null;

  const theme = getMerged().theme ?? {};
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

  // alinhamento e tamanho agem sobre o texto selecionado
  function setAlign(val: Align) {
    if (!selText) return;
    alignEl(selText, val);
    patchAlign(selText.dataset.edit!, val);
  }
  function bumpSize(delta: number) {
    if (!selText) return;
    const key = selText.dataset.edit!;
    const step = Math.max(-4, Math.min(6, ((getMerged().size ?? {})[key] ?? 0) + delta));
    sizeEl(selText, step);
    patchSize(key, step);
  }
  $("edit-align-left").addEventListener("click", () => setAlign("left"));
  $("edit-align-center").addEventListener("click", () => setAlign("center"));
  $("edit-align-right").addEventListener("click", () => setAlign("right"));
  $("edit-size-down").addEventListener("click", () => bumpSize(-1));
  $("edit-size-up").addEventListener("click", () => bumpSize(1));

  // mover bloco selecionado dentro do slide
  function moveBlock(delta: number) {
    if (!selBlock) return;
    const wrap = selBlock.parentElement!;
    const kids = Array.from(wrap.children) as HTMLElement[];
    const to = kids.indexOf(selBlock) + delta;
    if (to < 0 || to >= kids.length) return;
    if (delta > 0) kids[to].after(selBlock);
    else kids[to].before(selBlock);
    const slideId = wrap.closest<HTMLElement>(".slide")!.dataset.slideId!;
    saveBlocks(slideId, (Array.from(wrap.children) as HTMLElement[]).map((c) => c.dataset.blockId!));
  }
  $("edit-block-up").addEventListener("click", () => moveBlock(-1));
  $("edit-block-down").addEventListener("click", () => moveBlock(1));

  // alinhamento vertical do slide ativo
  function setSlideAlign(val: SlideAlign) {
    const s = document.querySelector<HTMLElement>(".slide.is-active");
    if (!s) return;
    slideAlignEl(s, val);
    patchSlideAlign(s.dataset.slideId!, val);
  }
  $("edit-slide-top").addEventListener("click", () => setSlideAlign("top"));
  $("edit-slide-center").addEventListener("click", () => setSlideAlign("center"));

  $("edit-move-left").addEventListener("click", () => ctx.moveCurrent(-1));
  $("edit-move-right").addEventListener("click", () => ctx.moveCurrent(1));
  $("edit-export").addEventListener("click", exportOverrides);
  $("edit-reset").addEventListener("click", () => {
    localStorage.removeItem(LS_KEY);
    location.reload();
  });

  const editables = () => Array.from(document.querySelectorAll<HTMLElement>("[data-edit]"));

  // seleção por clique (fora do painel): define alvo de texto e de bloco
  document.addEventListener(
    "click",
    (e) => {
      if (!on) return;
      const t = e.target as HTMLElement;
      if (panel.contains(t)) return;
      document.querySelectorAll(".is-selected").forEach((x) => x.classList.remove("is-selected"));
      selText = t.closest<HTMLElement>("[data-edit]");
      selBlock = t.closest<HTMLElement>("[data-blocks] > *");
      selText?.classList.add("is-selected");
    },
    true,
  );

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
    if (!state) {
      document.querySelectorAll(".is-selected").forEach((x) => x.classList.remove("is-selected"));
      selText = selBlock = null;
    }
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
