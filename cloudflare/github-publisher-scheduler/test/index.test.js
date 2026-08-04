import test from "node:test";
import assert from "node:assert/strict";
import { shouldDispatch } from "../src/index.js";

const now = Date.parse("2026-08-03T20:00:00Z");

test("dispatches when there are no recent runs", () => {
  assert.equal(shouldDispatch([], now), true);
  assert.equal(shouldDispatch([{ status: "completed", created_at: "2026-08-03T19:50:00Z" }], now), true);
});

test("skips while a run is queued or active", () => {
  assert.equal(shouldDispatch([{ status: "queued", created_at: "2026-08-03T19:00:00Z" }], now), false);
  assert.equal(shouldDispatch([{ status: "in_progress", created_at: "2026-08-03T19:00:00Z" }], now), false);
});

test("skips when another trigger created a recent run", () => {
  assert.equal(shouldDispatch([{ status: "completed", created_at: "2026-08-03T19:58:00Z" }], now), false);
});

