// resource: the slices of a file, stored by the SHA-1 of its content, so each version of a file is sliced once.
import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { placeSlice, sliceContent } from "./chunker.mjs";
import { resolveCacheDir } from "./cache-dir.mjs";

// Bump it when sliceContent changes its output, so no stale slice is read.
const SLICE_FORMAT = 1;

function entryPath(cacheDir, hash) {
  return path.join(cacheDir, "chunks", `v${SLICE_FORMAT}`, hash.slice(0, 2), `${hash}.json`);
}

function readEntry(file) {
  try {
    return JSON.parse(fs.readFileSync(file, "utf-8"));
  } catch {
    return null;
  }
}

// A write that fails costs the next query one slicing, so it is not an error.
function writeEntry(file, slices) {
  const temp = `${file}.${process.pid}.tmp`;
  try {
    fs.mkdirSync(path.dirname(file), { recursive: true });
    fs.writeFileSync(temp, JSON.stringify(slices));
    fs.renameSync(temp, file);
  } catch {
    fs.rmSync(temp, { force: true });
  }
}

/** prepareCandidates (chunker.mjs), with the slicing read from the cache when the content is unchanged. */
export function prepareCandidatesCached(filePaths, { cacheDir = resolveCacheDir() } = {}) {
  const chunks = [];
  for (const filePath of filePaths) {
    let content;
    try {
      content = fs.readFileSync(filePath, "utf-8");
    } catch {
      continue;
    }
    const file = entryPath(cacheDir, crypto.createHash("sha1").update(content).digest("hex"));
    let slices = readEntry(file);
    if (!Array.isArray(slices)) {
      slices = sliceContent(content);
      writeEntry(file, slices);
    }
    for (const slice of slices) chunks.push(placeSlice(filePath, slice));
  }
  return chunks;
}
