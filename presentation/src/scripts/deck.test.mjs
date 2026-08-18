import { test } from "node:test";
import assert from "node:assert/strict";
import { clampIndex, nextIndex } from "./deck.mjs";

test("clampIndex mantém dentro dos limites", () => {
  assert.equal(clampIndex(-1, 13), 0);
  assert.equal(clampIndex(99, 13), 12);
  assert.equal(clampIndex(5, 13), 5);
});

test("nextIndex anda e satura nas pontas", () => {
  assert.equal(nextIndex(0, 1, 13), 1);
  assert.equal(nextIndex(12, 1, 13), 12);
  assert.equal(nextIndex(0, -1, 13), 0);
});
