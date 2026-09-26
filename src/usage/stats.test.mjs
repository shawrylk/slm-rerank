import test from "node:test";
import assert from "node:assert/strict";
import { formatStats, summarizeUsage } from "./stats.mjs";

const fixture = [
  { caller: "cli", query: "a", mode: "fast", latencyMs: 400 },
  { caller: "cli", query: "b", mode: "rerank", latencyMs: 6000 },
  { caller: "cli", query: "c", mode: "rerank", latencyMs: 8000 },
  { caller: "hook", query: "where is x", isCodeQuestion: true, queried: true, outcome: "cited", latencyMs: 3000 },
  { caller: "hook", query: "how is y encoded", isCodeQuestion: true, queried: true, outcome: "no-hits", latencyMs: 5000 },
  { caller: "hook", query: "commit it", isCodeQuestion: false, queried: false, outcome: "not-code", latencyMs: 2 },
  { caller: "hook", query: "ok thanks", isCodeQuestion: false, queried: false, outcome: "not-code", latencyMs: 1 },
  { caller: "hook", query: "<task-notification>", isCodeQuestion: false, queried: false, outcome: "machine-text", latencyMs: 1 }
];

test("stats: query count, median latency by caller, and the share of hook prompts that queried", () => {
  const s = summarizeUsage(fixture);
  assert.equal(s.queryCount, 5);
  assert.deepEqual(s.latency, { cli: { queries: 3, medianMs: 6000 }, hook: { queries: 2, medianMs: 4000 } });
  assert.equal(s.hookPrompts, 4, "machine text is not a prompt a person asked");
  assert.equal(s.hookQueried, 2);
  assert.equal(s.hookShare, 0.5);
});

test("stats: an empty log prints zero queries and no share", () => {
  const s = summarizeUsage([]);
  assert.equal(s.queryCount, 0);
  assert.equal(s.hookShare, null);
  assert.match(formatStats(s, "/c/usage.jsonl"), /Queries: 0/);
});

test("stats: the printout names each caller and the hook share", () => {
  const text = formatStats(summarizeUsage(fixture), "/c/usage.jsonl");
  assert.match(text, /cli\s+3 queries\s+median 6\.0 s/);
  assert.match(text, /hook\s+2 queries\s+median 4\.0 s/);
  assert.match(text, /Hook prompts that queried: 2 of 4 \(50\.0%\)/);
});
