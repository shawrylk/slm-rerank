import test from "node:test";
import assert from "node:assert/strict";
import path from "node:path";
import { resolveCacheDir } from "./cache-dir.mjs";

test("cache dir: the override wins on every platform", () => {
  assert.equal(resolveCacheDir({ env: { SLM_RERANK_CACHE_DIR: "/tmp/x", LOCALAPPDATA: "C:/L" }, platform: "win32" }), "/tmp/x");
});

test("cache dir: each platform's user cache folder, never a working tree", () => {
  const home = "/home/u";
  assert.equal(resolveCacheDir({ env: { LOCALAPPDATA: "C:/L" }, platform: "win32", home }), path.join("C:/L", "slm-rerank"));
  assert.equal(resolveCacheDir({ env: {}, platform: "darwin", home }), path.join(home, "Library", "Caches", "slm-rerank"));
  assert.equal(resolveCacheDir({ env: { XDG_CACHE_HOME: "/xdg" }, platform: "linux", home }), path.join("/xdg", "slm-rerank"));
  assert.equal(resolveCacheDir({ env: {}, platform: "linux", home }), path.join(home, ".cache", "slm-rerank"));
});
