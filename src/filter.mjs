// Two-Tier Hybrid Search pre-filter in pure Node.js ESM

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

export function computeLexicalScore(chunk, queryTerms) {
  if (!queryTerms.length) return 0.5;
  let score = 0;
  const symLower = (chunk.symbol || "").toLowerCase();
  const pathLower = (chunk.filePath || "").toLowerCase();
  const contentLower = (chunk.content || "").toLowerCase();

  for (const term of queryTerms) {
    // Exact symbol match
    if (symLower.includes(term)) {
      score += 10.0;
    }
    // Path match
    if (pathLower.includes(term)) {
      score += 3.0;
    }
    // Content occurrence (BM25 saturating term frequency)
    const matches = (contentLower.match(new RegExp(term, "g")) || []).length;
    if (matches > 0) {
      score += 2.0 * (matches / (matches + 1.2));
    }
  }

  return score;
}

export function applyTwoTierFilter(chunks, query, full = false) {
  if (full || chunks.length <= TIER1_BYPASS_MAX_CANDIDATES) {
    return {
      retained: chunks,
      tier1Applied: false,
      reason: full ? "forced_full_scan" : "small_candidate_set"
    };
  }

  const queryTerms = extractTerms(query);
  const scored = chunks.map(chunk => ({
    chunk,
    score: computeLexicalScore(chunk, queryTerms)
  }));

  scored.sort((a, b) => b.score - a.score);
  const retained = scored.slice(0, TIER1_SELECT_TOP_N).map(s => s.chunk);

  return {
    retained,
    tier1Applied: true,
    reason: `filtered_${chunks.length}_to_${retained.length}`
  };
}
