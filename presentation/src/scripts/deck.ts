import { clampIndex, nextIndex } from "./deck.mjs";

function initDeck() {
  const slides = Array.from(document.querySelectorAll<HTMLElement>(".slide"));
  const total = slides.length;
  if (total === 0) return;

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

  function paint() {
    if (counter) counter.textContent = `${current + 1} / ${total}`;
    dotsWrap?.querySelectorAll(".deck-dot").forEach((d, i) => {
      d.classList.toggle("bg-black/20", i !== current);
      d.classList.toggle("bg-[var(--accent)]", i === current);
    });
    location.hash = `slide-${current + 1}`;
  }

  function go(i: number) {
    current = clampIndex(i, total);
    slides[current].scrollIntoView({ behavior: reduce ? "auto" : "smooth" });
    paint();
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

  // deep-link inicial (#slide-7)
  const m = location.hash.match(/slide-(\d+)/);
  if (m) current = clampIndex(Number(m[1]) - 1, total);
  go(current);
}

if (document.readyState !== "loading") initDeck();
else document.addEventListener("DOMContentLoaded", initDeck);
