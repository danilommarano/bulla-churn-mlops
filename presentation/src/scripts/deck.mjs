export function clampIndex(i, total) {
  if (i < 0) return 0;
  if (i > total - 1) return total - 1;
  return i;
}

export function nextIndex(current, delta, total) {
  return clampIndex(current + delta, total);
}
