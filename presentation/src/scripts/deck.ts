import { clampIndex, nextIndex } from "./deck.mjs";
import { applyStoredOverrides, initEditMode, saveOrder } from "./edit";

function initDeck() {
  // aplica overrides (ordem/texto/tema) antes de indexar os slides
  applyStoredOverrides();

  const deck = document.querySelector<HTMLElement>(".deck");
  let slides = Array.from(document.querySelectorAll<HTMLElement>(".slide"));
  const total = slides.length;
  if (!deck || total === 0) return;

  const dotsWrap = document.getElementById("deck-dots");
  const counter = document.getElementById("deck-counter");
  const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  let current = 0;
  let notesOn = false;

  // bolinhas de progresso
  if (dotsWrap) {
    slides.forEach((_, i) => {
      const dot = document.createElement("button");
      dot.className = "deck-dot h-2 w-2 rounded-full bg-black/20 transition";
      dot.setAttribute("aria-label", `Ir ao slide ${i + 1}`);
      dot.addEventListener("click", () => go(i));
      dotsWrap.appendChild(dot);
    });
  }

  const idOf = (i: number) => Number(slides[i].dataset.slideId);
  const indexOfId = (id: number) => slides.findIndex((s) => Number(s.dataset.slideId) === id);

  function show(i: number) {
    slides.forEach((s, idx) => s.classList.toggle("is-active", idx === i));
  }

  function paint() {
    if (counter) counter.textContent = `${current + 1} / ${total}`;
    dotsWrap?.querySelectorAll(".deck-dot").forEach((d, i) => {
      d.classList.toggle("bg-black/20", i !== current);
      d.classList.toggle("bg-[var(--accent)]", i === current);
    });
    // hash por ID do slide (robusto a reordenação; casa com as âncoras do mapa Vertex)
    const hash = `#slide-${idOf(current)}`;
    if (location.hash !== hash) history.replaceState(null, "", hash);
  }

  // troca de slide com crossfade: fade-out -> troca -> fade-in
  function go(i: number) {
    const target = clampIndex(i, total);
    if (target === current && slides[target].classList.contains("is-active")) return;
    if (reduce) {
      current = target;
      show(current);
      paint();
      return;
    }
    deck!.classList.add("is-fading");
    window.setTimeout(() => {
      current = target;
      show(current);
      paint();
      requestAnimationFrame(() => deck!.classList.remove("is-fading"));
    }, 160);
  }

  // move o slide atual na ordem (modo edição) e persiste
  function moveCurrent(delta: number) {
    const to = clampIndex(current + delta, total);
    if (to === current) return;
    const moving = slides[current];
    const ref = slides[to];
    if (delta > 0) ref.after(moving);
    else ref.before(moving);
    slides = Array.from(deck!.querySelectorAll<HTMLElement>(".slide"));
    current = slides.indexOf(moving);
    paint();
    saveOrder(slides.map((s) => Number(s.dataset.slideId)));
  }

  function toggleNotes() {
    notesOn = !notesOn;
    slides.forEach((s) => {
      const tpl = s.querySelector<HTMLTemplateElement>("template[data-notes]");
      let box = s.querySelector<HTMLElement>(".deck-notes");
      if (notesOn && tpl && !box) {
        box = document.createElement("div");
        box.className =
          "deck-notes absolute bottom-24 left-1/2 -translate-x-1/2 max-w-2xl rounded-lg bg-black/85 px-5 py-3 text-sm text-white";
        box.innerHTML = tpl.innerHTML;
        s.appendChild(box);
      } else if (!notesOn && box) {
        box.remove();
      }
    });
  }

  function toggleBlackout() {
    document.body.classList.toggle("deck-blackout");
  }

  // não sequestrar teclas enquanto o usuário digita num texto/campo
  function isField(t: EventTarget | null) {
    const el = t as HTMLElement | null;
    return !!el && (el.isContentEditable || /^(INPUT|SELECT|TEXTAREA)$/.test(el.tagName));
  }

  document.addEventListener("keydown", (e) => {
    if (isField(e.target)) return;
    if (e.key === "ArrowRight" || e.key === " ") {
      e.preventDefault();
      go(nextIndex(current, 1, total));
    } else if (e.key === "ArrowLeft") {
      e.preventDefault();
      go(nextIndex(current, -1, total));
    } else if (e.key.toLowerCase() === "s") {
      toggleNotes();
    } else if (e.key.toLowerCase() === "b") {
      toggleBlackout();
    } else if (e.key.toLowerCase() === "d") {
      window.location.href = import.meta.env.BASE_URL + "dashboard";
    }
  });

  // âncoras internas (#slide-N do mapa Vertex) navegam via fade, por ID do slide
  window.addEventListener("hashchange", () => {
    const m = location.hash.match(/slide-(\d+)/);
    if (!m) return;
    const idx = indexOfId(Number(m[1]));
    if (idx >= 0) go(idx);
  });

  // deep-link inicial (#slide-7) por ID — render instantâneo, sem fade
  const m = location.hash.match(/slide-(\d+)/);
  const startIdx = m ? indexOfId(Number(m[1])) : 0;
  current = startIdx >= 0 ? startIdx : 0;
  show(current);
  paint();

  initEditMode({ moveCurrent });
}

if (document.readyState !== "loading") initDeck();
else document.addEventListener("DOMContentLoaded", initDeck);
