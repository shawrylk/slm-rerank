import path from "node:path";

/**
 * Detect the architectural vertical slice or module for a file path.
 * Recognizes standard enterprise monorepo & bounded-slice conventions:
 * - features/<slice>/
 * - modules/<slice>/
 * - packages/<slice>/
 * - apps/<slice>/
 * - workers/<slice>/
 * - src/<slice>/
 *
 * @param {string} filePath 
 * @returns {string} Architectural slice name (e.g. 'billing', 'auth', 'qc-harness')
 */
export function detectSlice(filePath) {
  if (!filePath) return "root";

  const normalized = filePath.replace(/\\/g, "/");
  const parts = normalized.split("/").filter(Boolean);

  for (let i = 0; i < parts.length - 1; i++) {
    const segment = parts[i].toLowerCase();
    if (["features", "modules", "packages", "apps", "workers", "services", "domains"].includes(segment)) {
      return parts[i + 1];
    }
  }

  // Fallback: Check if under src/<name>/
  const srcIdx = parts.indexOf("src");
  if (srcIdx !== -1 && srcIdx < parts.length - 1) {
    // If followed by another directory before file
    if (srcIdx < parts.length - 2) {
      return parts[srcIdx + 1];
    }
  }

  // Top-level directory or root
  if (parts.length > 1) {
    return parts[0];
  }
  return "root";
}

/**
 * Group reranking results by architectural vertical slice.
 *
 * @param {Array<{score: number, chunk: {filePath: string}}>} results 
 * @returns {Record<string, {slice: string, maxScore: number, items: typeof results}>}
 */
export function groupBySlice(results) {
  const groups = {};

  for (const item of results) {
    const sliceName = item.slice || detectSlice(item.chunk?.filePath);
    if (!groups[sliceName]) {
      groups[sliceName] = {
        slice: sliceName,
        maxScore: item.score,
        items: []
      };
    }

    if (item.score > groups[sliceName].maxScore) {
      groups[sliceName].maxScore = item.score;
    }
    groups[sliceName].items.push(item);
  }

  // Sort slices by highest score descending
  const sortedKeys = Object.keys(groups).sort((a, b) => groups[b].maxScore - groups[a].maxScore);
  const sortedGroups = {};
  for (const k of sortedKeys) {
    sortedGroups[k] = groups[k];
  }

  return sortedGroups;
}
