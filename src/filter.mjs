// Two-Tier Hybrid Search pre-filter with Git-Diff Biasing in pure Node.js ESM
import { spawnSync } from "node:child_process";
import path from "node:path";
import { containsTerm, countTerm, declaredNames, termKeys, tokenCounts, tokenKeys } from "./lexical.mjs";

export const TIER1_BYPASS_MAX_CANDIDATES = 60;
export const TIER1_SELECT_TOP_N = 80;

const GENERIC_STOPWORDS = new Set([
  "a", "an", "the", "and", "or", "but", "not", "in", "on", "at", "to", "of", "for",
  "with", "from", "into", "by", "as", "is", "are", "was", "were", "be", "been",
  "this", "that", "these", "those", "it", "its", "there", "here", "then", "than",
  "how", "what", "where", "which", "who", "when", "why", "do", "does", "did", "done",
  "can", "could", "should", "would", "may", "might", "must", "will", "shall",
  "code", "codes", "file", "files", "line", "lines", "function", "method", "class",
  "type", "types", "common", "refactor", "implement", "find", "search", "locate"
]);

export function extractTerms(text) {
  if (!text) return [];
  const words = text.toLowerCase().match(/\b[a-z0-9_]+\b/g) || [];
  return words.filter(w => w.length >= 2 && !GENERIC_STOPWORDS.has(w));
}

/**
 * Retrieve files modified in git working tree (uncommitted or recent commit).
 * @param {string} cwd 
 * @returns {Set<string>}
 */
export function getGitDiffFiles(cwd = process.cwd()) {
  const dirtyFiles = new Set();
  try {
    const status = spawnSync("git", ["status", "--porcelain"], { cwd, encoding: "utf-8" });
    if (status.status === 0 && status.stdout) {
      for (const line of status.stdout.split("\n")) {
        const file = line.slice(3).trim();
        if (file) dirtyFiles.add(path.normalize(file));
      }
    }
    const diff = spawnSync("git", ["diff", "--name-only", "HEAD~1"], { cwd, encoding: "utf-8" });
    if (diff.status === 0 && diff.stdout) {
      for (const file of diff.stdout.split("\n")) {
        const trimmed = file.trim();
        if (trimmed) dirtyFiles.add(path.normalize(trimmed));
      }
    }
  } catch {
    // Ignore git errors if outside repo
  }
  return dirtyFiles;
}

/** A chunk's names (file name, declared symbols) and its path, as token keys. */
function chunkKeys(chunk) {
  const segments = String(chunk.filePath || "").split(/[\x5c/]/);
  const base = (segments.pop() || "").replace(/\.[^.]+$/, "");
  const names = [base, chunk.symbol || "", ...declaredNames(chunk.content)].join(" ");
  return {
    name: tokenKeys(names),
    path: tokenKeys([...segments, base].join(" ")),
    body: tokenCounts(chunk.content)
  };
}

export function computeLexicalScore(chunk, queryTerms, gitDiffFiles = null) {
  if (!queryTerms.length && !gitDiffFiles) return 0.5;
  let score = 0;
  const pathNorm = path.normalize(chunk.filePath || "");
  const { name, path: pathKeys, body } = chunkKeys(chunk);

  for (const term of queryTerms) {
    const keys = termKeys(term);
    // The file or a symbol it declares is named after the term
    if (containsTerm(name, keys)) {
      score += 10.0;
    }
    // Path match
    if (containsTerm(pathKeys, keys)) {
      score += 3.0;
    }
    // Content occurrence (BM25 saturating term frequency)
    const matches = countTerm(body, keys);
    if (matches > 0) {
      score += 2.0 * (matches / (matches + 1.2));
    }
  }

  // Recency / Git-diff bias: boost recently touched or modified files
  if (gitDiffFiles && gitDiffFiles.has(pathNorm)) {
    score += 6.0;
  }

  return score;
}

export function applyTwoTierFilter(chunks, query, options = {}) {
  const full = typeof options === "boolean" ? options : (options.full ?? false);
  const gitDiff = options.gitDiff ?? false;
  const dirtyOnly = options.dirtyOnly ?? false;
  const cwd = options.cwd ?? process.cwd();

  const gitDiffFiles = (gitDiff || dirtyOnly) ? getGitDiffFiles(cwd) : null;

  let candidates = chunks;
  if (dirtyOnly && gitDiffFiles && gitDiffFiles.size > 0) {
    const filtered = chunks.filter(c => gitDiffFiles.has(path.normalize(c.filePath)));
    if (filtered.length > 0) {
      candidates = filtered;
    }
  }

  // Order every candidate set, small ones too: a prompt hook scores only the head.
  const queryTerms = extractTerms(query);
  const scored = candidates.map(chunk => ({
    chunk,
    score: computeLexicalScore(chunk, queryTerms, gitDiffFiles)
  }));
  scored.sort((a, b) => b.score - a.score);

  if (full || candidates.length <= TIER1_BYPASS_MAX_CANDIDATES) {
    return {
      retained: scored.map(s => s.chunk),
      lexicalScores: scored.map(s => s.score),
      tier1Applied: false,
      reason: full ? "forced_full_scan" : "small_candidate_set"
    };
  }

  const kept = scored.slice(0, TIER1_SELECT_TOP_N);

  return {
    retained: kept.map(s => s.chunk),
    lexicalScores: kept.map(s => s.score),
    tier1Applied: true,
    reason: `filtered_${candidates.length}_to_${kept.length}`
  };
}
