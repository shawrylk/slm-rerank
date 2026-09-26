import test from "node:test";
import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { handlePrompt } from "./claude-prompt-context.mjs";
import { readUsage } from "../usage/log.mjs";

const HOOK = fileURLToPath(new URL("./claude-prompt-context.mjs", import.meta.url));
const noServer = async () => { throw new Error("a prompt that is not about code must not probe for a server"); };

function capture() {
  const lines = [];
  return { lines, log: entry => { lines.push(entry); return true; } };
}

test("hook: a process prompt probes nothing, adds nothing, and logs the decision", async () => {
  const { lines, log } = capture();
  const context = await handlePrompt({ prompt: "how should we improve your efficiency?", cwd: "." }, { resolve: noServer, log });
  assert.equal(context, null);
  assert.equal(lines.length, 1);
  assert.equal(lines[0].caller, "hook");
  assert.equal(lines[0].isCodeQuestion, false);
  assert.equal(lines[0].queried, false);
  assert.equal(lines[0].outcome, "not-code");
});

test("hook: machine text is logged apart, so it does not dilute the share of prompts", async () => {
  const { lines, log } = capture();
  await handlePrompt({ prompt: "<task-notification>done</task-notification>", cwd: "." }, { resolve: noServer, log });
  assert.equal(lines[0].outcome, "machine-text");
});

test("hook: a code prompt with no model server logs that it queried and found no server", async () => {
  const { lines, log } = capture();
  const context = await handlePrompt({ prompt: "where does the router set app.tenant_id", cwd: "." }, { resolve: async () => null, log });
  assert.equal(context, null);
  assert.equal(lines[0].isCodeQuestion, true);
  assert.equal(lines[0].queried, true);
  assert.equal(lines[0].outcome, "no-server");
  assert.ok(Number.isFinite(lines[0].latencyMs));
});

test("hook: the process exits 0 with no output, and appends one usage line", () => {
  const cacheDir = fs.mkdtempSync(path.join(os.tmpdir(), "slm-hook-"));
  const env = { ...process.env, SLM_RERANK_CACHE_DIR: cacheDir, SLM_ENDPOINT: "http://127.0.0.1:9/v1" };
  const run = prompt => spawnSync(process.execPath, ["--input-type=module", "-e",
    `import { runPromptHook } from ${JSON.stringify(new URL("./claude-prompt-context.mjs", import.meta.url).href)}; await runPromptHook();`],
  { input: JSON.stringify({ prompt, cwd: path.dirname(HOOK) }), env, encoding: "utf8" });

  for (const prompt of ["commit and push, then open the PR", "where does the router set app.tenant_id"]) {
    const res = run(prompt);
    assert.equal(res.status, 0, res.stderr);
    assert.equal(res.stdout, "");
  }
  assert.deepEqual(readUsage({ dir: cacheDir }).map(e => e.outcome), ["not-code", "no-server"]);
});
