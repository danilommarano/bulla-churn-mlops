import { clampIndex, nextIndex } from "./deck.mjs";

function initDeck() {
  const deck = document.querySelector<HTMLElement>(".deck");
  const slides = Array.from(document.querySelectorAll<HTMLElement>(".slide"));
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

  function show(i: number) {
    slides.forEach((s, idx) => s.classList.toggle("is-active", idx === i));
  }

  function paint() {
    if (counter) counter.textContent = `${current + 1} / ${total}`;
    dotsWrap?.querySelectorAll(".deck-dot").forEach((d, i) => {
      d.classList.toggle("bg-black/20", i !== current);
      d.classList.toggle("bg-[var(--accent)]", i === current);
    });
    // hash sincronizado sem empurrar histórico nem disparar hashchange
    if (location.hash !== `#slide-${current + 1}`) {
      history.replaceState(null, "", `#slide-${current + 1}`);
    }
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

  document.addEventListener("keydown", (e) => {
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

  // âncoras internas (#slide-7 do mapa Vertex) navegam via fade
  window.addEventListener("hashchange", () => {
    const m = location.hash.match(/slide-(\d+)/);
    if (m) go(clampIndex(Number(m[1]) - 1, total));
  });

  // deep-link inicial (#slide-7) — render instantâneo, sem fade
  const m = location.hash.match(/slide-(\d+)/);
  current = m ? clampIndex(Number(m[1]) - 1, total) : 0;
  show(current);
  paint();
}

if (document.readyState !== "loading") initDeck();
else document.addEventListener("DOMContentLoaded", initDeck);
