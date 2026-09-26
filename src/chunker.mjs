// Chunker module for Node.js: slices files into candidate chunks with line numbers

import fs from "node:fs";
import path from "node:path";

const SYMBOL_DEF_RE = /^(?:export\s+|public\s+|static\s+|async\s+)*(?:class|interface|struct|function|def|fn|func|type)\s+([A-Za-z0-9_]+)/m;

/** Content -> its slices, which carry no path: the chunk cache (chunk-cache.mjs) stores these by content hash. */
export function sliceContent(content, maxLines = 100, overlap = 20) {
  const lines = content.split("\n");
  if (lines.length === 0) return [];

  const slices = [];
  let start = 0;

  while (start < lines.length) {
    const end = Math.min(lines.length, start + maxLines);
    const slice = lines.slice(start, end);
    const numbered = slice.map((line, idx) => `${start + idx + 1}: ${line}`).join("\n");

    // Detect symbol in slice
    let symbol = null;
    for (const l of slice) {
      const m = l.match(SYMBOL_DEF_RE);
      if (m && m[1]) {
        symbol = m[1];
        break;
      }
    }

    slices.push({ startLine: start + 1, endLine: end, symbol, content: numbered });

    if (end >= lines.length) break;
    start += (maxLines - overlap);
  }

  return slices;
}

/** A slice at a path -> a candidate chunk. */
export function placeSlice(filePath, { startLine, endLine, symbol, content }) {
  return {
    id: `${filePath}:${startLine}-${endLine}${symbol ? `:${symbol}` : ""}`,
    filePath,
    startLine,
    endLine,
    symbol: symbol || path.basename(filePath),
    content
  };
}

export function chunkFile(filePath, content, maxLines = 100, overlap = 20) {
  return sliceContent(content, maxLines, overlap).map(slice => placeSlice(filePath, slice));
}

export function prepareCandidates(filePaths) {
  const chunks = [];
  for (const fp of filePaths) {
    try {
      if (!fs.existsSync(fp)) continue;
      const stat = fs.statSync(fp);
      if (!stat.isFile()) continue;
      const content = fs.readFileSync(fp, "utf-8");
      const fileChunks = chunkFile(fp, content);
      chunks.push(...fileChunks);
    } catch {
      // Ignore unreadable files
    }
  }
  return chunks;
}
