import test from "node:test";
import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { readUsage } from "./usage/log.mjs";

const BIN = fileURLToPath(new URL("../bin/slm-rerank.mjs", import.meta.url));

function tempRepo() {
  const repo = fs.mkdtempSync(path.join(os.tmpdir(), "slm-cli-repo-"));
  fs.mkdirSync(path.join(repo, "src"));
  fs.writeFileSync(path.join(repo, "src", "tenant-session.ts"), "export function openTenantSession() {\n  return setTenant();\n}\n");
  fs.writeFileSync(path.join(repo, "src", "pool.ts"), "export const pool = createPool();\n");
  spawnSync("git", ["init", "-q"], { cwd: repo });
  spawnSync("git", ["add", "."], { cwd: repo });
  return repo;
}

// A dead pinned endpoint proves --fast never asks for a model server.
const run = (repo, cacheDir, args) => spawnSync(process.execPath, [BIN, ...args], {
  cwd: repo,
  encoding: "utf8",
  env: { ...process.env, SLM_RERANK_CACHE_DIR: cacheDir, SLM_ENDPOINT: "http://127.0.0.1:9/v1" }
});

test("CLI: --fast ranks with no model server, and the query appends one usage line", () => {
  const repo = tempRepo();
  const cacheDir = fs.mkdtempSync(path.join(os.tmpdir(), "slm-cli-cache-"));
  const res = run(repo, cacheDir, ["-q", "where is the tenant session opened", "--fast", "--json"]);
  assert.equal(res.status, 0, res.stderr);
  const out = JSON.parse(res.stdout);
  assert.equal(out.mode, "fast");
  assert.equal(out.results[0].chunk.filePath.replace(/\\/g, "/"), "src/tenant-session.ts");

  const [entry] = readUsage({ dir: cacheDir });
  assert.equal(entry.caller, "cli");
  assert.equal(entry.mode, "fast");
  assert.ok(entry.latencyMs > 0);
  assert.match(entry.top[0], /tenant-session\.ts:1-4$/);
  assert.equal(fs.readdirSync(repo).includes(".slm-rerank"), false);
});

test("CLI: stats summarizes the usage log", () => {
  const repo = tempRepo();
  const cacheDir = fs.mkdtempSync(path.join(os.tmpdir(), "slm-cli-cache-"));
  run(repo, cacheDir, ["-q", "tenant session", "--fast", "--json"]);
  run(repo, cacheDir, ["-q", "pool", "--fast", "--json"]);
  const res = run(repo, cacheDir, ["stats", "--json"]);
  assert.equal(res.status, 0, res.stderr);
  const summary = JSON.parse(res.stdout);
  assert.equal(summary.queryCount, 2);
  assert.equal(summary.latency.cli.queries, 2);
  assert.match(run(repo, cacheDir, ["stats"]).stdout, /Queries: 2/);
});
