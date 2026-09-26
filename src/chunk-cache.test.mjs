import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { prepareCandidates } from "./chunker.mjs";
import { prepareCandidatesCached } from "./chunk-cache.mjs";

const tempDir = prefix => fs.mkdtempSync(path.join(os.tmpdir(), prefix));
const source = Array.from({ length: 150 }, (_, i) => (i === 3 ? "export function openSession() {" : `  line${i}();`)).join("\n");
const cachedFiles = dir => fs.readdirSync(dir, { recursive: true }).filter(f => f.endsWith(".json"));

test("chunk cache: a miss and a hit both return what prepareCandidates returns", () => {
  const repo = tempDir("slm-repo-");
  const cacheDir = tempDir("slm-cache-");
  const file = path.join(repo, "session.ts");
  fs.writeFileSync(file, source);
  const plain = prepareCandidates([file]);
  assert.deepEqual(prepareCandidatesCached([file], { cacheDir }), plain);
  assert.equal(cachedFiles(cacheDir).length, 1);
  assert.deepEqual(prepareCandidatesCached([file], { cacheDir }), plain);
});

test("chunk cache: the key is the content, so a copy at another path reuses the entry under its own path", () => {
  const repo = tempDir("slm-repo-");
  const cacheDir = tempDir("slm-cache-");
  const a = path.join(repo, "a.ts");
  const b = path.join(repo, "b.ts");
  fs.writeFileSync(a, source);
  fs.writeFileSync(b, source);
  prepareCandidatesCached([a], { cacheDir });
  const fromCache = prepareCandidatesCached([b], { cacheDir });
  assert.equal(cachedFiles(cacheDir).length, 1);
  assert.deepEqual(fromCache, prepareCandidates([b]));
});

test("chunk cache: changed content misses, and a torn entry is rebuilt", () => {
  const repo = tempDir("slm-repo-");
  const cacheDir = tempDir("slm-cache-");
  const file = path.join(repo, "s.ts");
  fs.writeFileSync(file, source);
  prepareCandidatesCached([file], { cacheDir });
  fs.writeFileSync(file, `${source}\nexport function closeSession() {}`);
  assert.deepEqual(prepareCandidatesCached([file], { cacheDir }), prepareCandidates([file]));
  assert.equal(cachedFiles(cacheDir).length, 2);
  for (const f of cachedFiles(cacheDir)) fs.writeFileSync(path.join(cacheDir, f), "{torn");
  assert.deepEqual(prepareCandidatesCached([file], { cacheDir }), prepareCandidates([file]));
});

test("chunk cache: a missing path and a directory are skipped", () => {
  const repo = tempDir("slm-repo-");
  assert.deepEqual(prepareCandidatesCached([path.join(repo, "none.ts"), repo], { cacheDir: tempDir("slm-cache-") }), []);
});
