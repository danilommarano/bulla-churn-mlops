import { test } from "node:test";
import assert from "node:assert/strict";
import { clampIndex, nextIndex } from "./deck.mjs";

test("clampIndex mantém dentro dos limites", () => {
  assert.equal(clampIndex(-1, 12), 0);
  assert.equal(clampIndex(99, 12), 11);
  assert.equal(clampIndex(5, 12), 5);
});

test("nextIndex anda e satura nas pontas", () => {
  assert.equal(nextIndex(0, 1, 12), 1);
  assert.equal(nextIndex(11, 1, 12), 11);
  assert.equal(nextIndex(0, -1, 12), 0);
});
