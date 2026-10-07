import test from "node:test";
import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { discoverCandidateFiles, isTestPath } from "./discovery.mjs";

const hasRipgrep = spawnSync("rg", ["--version"]).status === 0;

function fixture(t, files) {
  const cwd = fs.mkdtempSync(path.join(os.tmpdir(), "slm-discovery-"));
  t.after(() => fs.rmSync(cwd, { recursive: true, force: true }));
  for (const [file, content] of Object.entries(files)) {
    const target = path.join(cwd, file);
    fs.mkdirSync(path.dirname(target), { recursive: true });
    fs.writeFileSync(target, content);
  }
  return cwd;
}

test("discovery: a body-only match survives a competing filename match", { skip: !hasRipgrep }, t => {
  const cwd = fixture(t, {
    "src/needle.ts": "export const unrelated = 1;\n",
    "src/engine.ts": "export function findNeedle() { return 1; }\n",
    "node_modules/vendor.ts": "export const needle = 1;\n"
  });
  assert.deepEqual(discoverCandidateFiles("needle", { cwd }), ["src/needle.ts", "src/engine.ts"]);
});

test("discovery: path and body hits share one file identity", { skip: !hasRipgrep }, t => {
  const cwd = fixture(t, {
    "src/needle.ts": "export const needle = 1;\n",
    "src/engine.ts": "export const needle = 2;\n"
  });
  assert.deepEqual(discoverCandidateFiles("needle", { cwd }), ["src/needle.ts", "src/engine.ts"]);
});

test("discovery: implementations outrank nested tests with the same body terms", { skip: !hasRipgrep }, t => {
  const cwd = fixture(t, {
    "src/worker.ts": "export const needle = 1;\n",
    "tests/unit/worker.ts": "export const needle = 2;\n"
  });
  assert.deepEqual(discoverCandidateFiles("needle", { cwd, limit: 1 }), ["src/worker.ts"]);
});

test("discovery: test paths use the same rules for Windows separators", () => {
  assert.equal(isTestPath("tests\\unit\\worker.ts"), true);
  assert.equal(isTestPath("src\\__tests__\\worker.ts"), true);
  assert.equal(isTestPath("src\\test_worker.py"), true);
  assert.equal(isTestPath("src\\fixtures\\worker.ts"), false);
});
