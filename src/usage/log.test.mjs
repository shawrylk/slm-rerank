import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { MAX_LOGGED_QUERY, appendUsage, readUsage, usageLogPath } from "./log.mjs";

const tempDir = () => fs.mkdtempSync(path.join(os.tmpdir(), "slm-usage-"));

test("usage log: one call appends one JSON line with the time first", () => {
  const dir = tempDir();
  const now = new Date("2026-01-02T03:04:05.000Z");
  assert.equal(appendUsage({ caller: "cli", query: "where is x", mode: "fast", latencyMs: 12, top: ["a.ts:1-100"] }, { dir, now }), true);
  appendUsage({ caller: "hook", query: "ok", isCodeQuestion: false, queried: false, outcome: "not-code", latencyMs: 1, top: [] }, { dir, now });
  const lines = fs.readFileSync(usageLogPath(dir), "utf8").trim().split("\n");
  assert.equal(lines.length, 2);
  assert.ok(lines[0].startsWith('{"time":"2026-01-02T03:04:05.000Z"'));
  assert.deepEqual(readUsage({ dir }).map(e => e.caller), ["cli", "hook"]);
});

test("usage log: a long prompt keeps its head only", () => {
  const dir = tempDir();
  appendUsage({ caller: "hook", query: "x".repeat(MAX_LOGGED_QUERY * 3) }, { dir });
  assert.equal(readUsage({ dir })[0].query.length, MAX_LOGGED_QUERY);
});

test("usage log: a write that fails returns false and never throws", () => {
  const dir = tempDir();
  const file = path.join(dir, "not-a-dir");
  fs.writeFileSync(file, "");
  assert.equal(appendUsage({ caller: "cli", query: "q" }, { dir: file }), false);
});

test("usage log: a torn line is skipped, and a missing log reads as empty", () => {
  const dir = tempDir();
  fs.writeFileSync(usageLogPath(dir), '{"caller":"cli","query":"a"}\n{"caller":"cl');
  assert.equal(readUsage({ dir }).length, 1);
  assert.deepEqual(readUsage({ dir: path.join(dir, "none") }), []);
});
