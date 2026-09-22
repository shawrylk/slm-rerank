import { spawnSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";

export const SLM_PORT_RANGE = [8033, 8034, 8035, 8036, 8037, 8038, 8039, 8040];

/**
 * Endpoint and host environment variables, in precedence order.
 * SLM_ENDPOINT/SLM_HOST are the canonical spellings; RERANKER_BASE_URL,
 * LFM_ENDPOINT and RERANKER_HOST are the names the Python implementation has
 * always read, and are honoured here so one variable configures either side.
 */
export const ENDPOINT_ENV_VARS = ["SLM_ENDPOINT", "RERANKER_BASE_URL", "LFM_ENDPOINT"];
export const HOST_ENV_VARS = ["SLM_HOST", "RERANKER_HOST"];
export const DEFAULT_HOST = "127.0.0.1";

function firstEnvValue(names, env) {
  for (const name of names) {
    const value = env[name];
    if (typeof value === "string" && value.trim()) return value.trim();
  }
  return null;
}

/** Full endpoint URL pinned via env, or null when none is set. */
export function resolveEndpointEnv(env = process.env) {
  return firstEnvValue(ENDPOINT_ENV_VARS, env);
}

/** Host to scan for model servers. Defaults to loopback. */
export function resolveHostEnv(env = process.env) {
  return firstEnvValue(HOST_ENV_VARS, env) || DEFAULT_HOST;
}

/**
 * Probe a single HTTP endpoint for model availability.
 * @param {string} host 
 * @param {number} port 
 * @param {number} timeoutMs 
 * @returns {Promise<{port: number, host: string, url: string, modelId: string, ok: boolean}|null>}
 */
export async function probePort(host, port, timeoutMs = 250) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  const url = `http://${host}:${port}/v1`;

  try {
    const res = await fetch(`${url}/models`, {
      method: "GET",
      signal: controller.signal,
      headers: { "Accept": "application/json" }
    });
    clearTimeout(timer);

    if (!res.ok) return null;
    const body = await res.json();
    const modelId = body?.data?.[0]?.id || body?.models?.[0]?.name || body?.models?.[0]?.model || "unknown";
    return { port, host, url, modelId, ok: true };
  } catch {
    clearTimeout(timer);
    return null;
  }
}

/**
 * Concurrently discover the active SLM model across ports 8033-8040.
 * If requestedModel is provided, matches against the model ID.
 * Otherwise returns the first healthy port, preferring 8034 (LFM) and 8033 (Qwen).
 */
export async function autoDiscoverEndpoint({
  env = process.env,
  host = resolveHostEnv(env),
  requestedModel = null,
  ports = SLM_PORT_RANGE,
  timeoutMs = 300
} = {}) {
  // An explicitly pinned URL wins outright, including when a model was requested:
  // the caller named the server, so we do not second-guess it by scanning loopback.
  const pinned = resolveEndpointEnv(env);
  if (pinned) {
    return { url: pinned, modelId: "pinned-via-env", port: null, host, ok: true };
  }

  const probes = ports.map(port => probePort(host, port, timeoutMs));
  const results = (await Promise.all(probes)).filter(Boolean);

  if (!results.length) {
    // No phantom endpoint: returning a URL nothing answered on turns "no server
    // here" into a connection error much later, which reads like a transport bug.
    const range = `${ports[0]}-${ports[ports.length - 1]}`;
    return {
      url: null,
      modelId: null,
      port: null,
      host,
      ok: false,
      reason: `No SLM model server answered on ${host} (ports ${range}). ` +
        `Discovery only scans ports on a single host, never the network. ` +
        `If the model runs on another machine, set SLM_ENDPOINT=http://<host>:8034/v1 ` +
        `or SLM_HOST=<host>.`
    };
  }

  if (requestedModel) {
    const reqLower = requestedModel.toLowerCase();
    const matched = results.find(r => r.modelId.toLowerCase().includes(reqLower));
    if (matched) return matched;
  }

  // Prioritize port 8034 (LFM) or 8033 (Qwen) if alive, otherwise first responding port
  const preferred = results.find(r => r.port === 8034) || results.find(r => r.port === 8033) || results[0];
  return preferred;
}

/**
 * Smart candidate file auto-discovery via ripgrep (rg), git grep, or git ls-files.
 * Allows running `slm-rerank -q "query"` without passing manual file globs.
 */
/**
 * Smart candidate file auto-discovery via ripgrep (rg), git grep, or git ls-files.
 * Allows running `slm-rerank -q "query"` without passing manual file globs.
 *
 * Recall is a union of two signals, always both consulted and then ranked:
 * where a term appears in a file's *path* and where it appears in its *body*.
 * Gating path matching behind "content search found nothing" used to make a file
 * named after the thing you asked for unreachable as soon as any other file
 * mentioned the words -- which is precisely what test files do.
 */
const CODE_EXTS = new Set([".py", ".ts", ".tsx", ".js", ".jsx", ".mjs", ".rs", ".go", ".c", ".cpp", ".h"]);

const QUERY_STOP_WORDS = new Set([
  "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "of", "for",
  "with", "from", "into", "by", "as", "is", "are", "was", "were", "be",
  "code", "file", "function", "method", "class", "find", "search", "how", "what", "where"
]);

const IGNORE_GLOBS = ["!node_modules", "!.git", "!dist", "!build", "!coverage", "!__pycache__", "!.venv"];
const MAX_SEARCH_TERMS = 6;   // one rg spawn each, so this is the cost knob
const PATH_HIT_WEIGHT = 2;    // a term in the path is a stronger signal than one body mention
const TEST_FILE_FACTOR = 0.5; // tests restate domain vocabulary; rank them below implementations
const MIN_STEM_LENGTH = 4;
const NO_MATCH_SAMPLE = 20;

function globArgs() {
  return IGNORE_GLOBS.flatMap(g => ["--glob", g]);
}

/** Suffix stems, emitted next to their root so a term cap never severs them. */
function stemVariants(word) {
  const out = [];
  const add = s => { if (s.length >= MIN_STEM_LENGTH && !out.includes(s)) out.push(s); };

  if (word.endsWith("ing") && word.length > 6) {
    add(word.slice(0, -3));
  } else if (word.endsWith("ion") && word.length > 6) {
    const base = word.slice(0, -3);
    add(base);          // migration -> migrat, which substring-matches migrate too
    add(`${base}e`);    // migration -> migrate
  } else if (word.endsWith("ate") && word.length > 5) {
    add(word.slice(0, -1));
  } else if (word.endsWith("ed") && word.length > 5) {
    add(word.slice(0, -2));
  } else if (/(?:s|x|z|ch|sh)es$/.test(word) && word.length > 4) {
    add(word.slice(0, -2));   // classes -> class, boxes -> box
  } else if (word.endsWith("s") && !word.endsWith("ss") && word.length > 4) {
    add(word.slice(0, -1));   // exports -> export, but harness stays harness
  }
  return out;
}

/** Query -> ordered search terms, stop words dropped, each stem following its root. */
export function extractQueryTerms(query) {
  if (!query || typeof query !== "string") return [];

  const rawWords = query
    .toLowerCase()
    .match(/\b[a-z0-9_]{2,}\b/g)
    ?.filter(w => !QUERY_STOP_WORDS.has(w)) || [];

  if (!rawWords.length) {
    const first = query.trim().split(/\s+/)[0];
    if (first) rawWords.push(first.toLowerCase());
  }

  const terms = [];
  const push = t => { if (t && !terms.includes(t)) terms.push(t); };
  for (const word of rawWords) {
    push(word);
    for (const stem of stemVariants(word)) push(stem);
  }
  return terms;
}

/** Tests mention domain vocabulary constantly; this keeps them from crowding the budget. */
export function isTestPath(filePath) {
  const lower = filePath.toLowerCase();
  const base = lower.split("/").pop() || lower;
  return /(^|\/)(tests?|__tests__|specs?)(\/|$)/.test(lower)
    || /\.(test|spec)\.[a-z0-9]+$/.test(base)
    || /^test_/.test(base)
    || /_test\.[a-z0-9]+$/.test(base);
}

function splitFileList(stdout, cwd, checkExists) {
  const files = [];
  for (const line of stdout.split("\n")) {
    const trimmed = line.trim();
    if (!trimmed || !CODE_EXTS.has(path.extname(trimmed))) continue;
    if (checkExists && !fs.existsSync(path.resolve(cwd, trimmed))) continue;
    files.push(trimmed);
  }
  return files;
}

/** Every code file in the tree, for path matching. Empty when neither tool is available. */
function listRepoFiles(cwd) {
  try {
    const rg = spawnSync("rg", ["--files", ...globArgs()], { cwd, encoding: "utf-8", maxBuffer: 10 * 1024 * 1024 });
    if (rg.status === 0 && rg.stdout) return splitFileList(rg.stdout, cwd, false);
  } catch {
    // ripgrep not available
  }
  try {
    const git = spawnSync("git", ["ls-files"], { cwd, encoding: "utf-8", maxBuffer: 10 * 1024 * 1024 });
    if (git.status === 0 && git.stdout) return splitFileList(git.stdout, cwd, true);
  } catch {
    // not a git repository
  }
  return [];
}

/** file -> set of terms matched in its contents. */
function findBodyMatches(cwd, terms) {
  const hits = new Map();
  const record = (file, term) => {
    if (!hits.has(file)) hits.set(file, new Set());
    hits.get(file).add(term);
  };

  let ripgrepUsable = false;
  for (const term of terms) {
    try {
      const res = spawnSync("rg", [
        "--files-with-matches", "--ignore-case", "--max-count", "1", ...globArgs(), term
      ], { cwd, encoding: "utf-8", maxBuffer: 10 * 1024 * 1024 });
      if (res.error) break;
      ripgrepUsable = true;
      if (res.status === 0 && res.stdout) {
        for (const f of splitFileList(res.stdout, cwd, false)) record(f, term);
      }
    } catch {
      break;
    }
  }

  if (ripgrepUsable) return hits;

  for (const term of terms) {
    try {
      const res = spawnSync("git", ["grep", "-l", "-i", term], { cwd, encoding: "utf-8", maxBuffer: 10 * 1024 * 1024 });
      if (res.status === 0 && res.stdout) {
        for (const f of splitFileList(res.stdout, cwd, true)) record(f, term);
      }
    } catch {
      break;
    }
  }
  return hits;
}

export function discoverCandidateFiles(query, { cwd = process.cwd(), limit = 60 } = {}) {
  const terms = extractQueryTerms(query);
  if (!terms.length) return [];

  const repoFiles = listRepoFiles(cwd);
  // Path matching is in-memory, so every term is used; only content search is capped.
  const bodyHits = findBodyMatches(cwd, terms.slice(0, MAX_SEARCH_TERMS));

  const scored = new Map();
  const entryFor = file => {
    if (!scored.has(file)) scored.set(file, { pathTerms: new Set(), bodyTerms: new Set() });
    return scored.get(file);
  };

  for (const file of repoFiles) {
    const lower = file.toLowerCase();
    for (const term of terms) {
      if (lower.includes(term)) entryFor(file).pathTerms.add(term);
    }
  }
  for (const [file, matchedTerms] of bodyHits) {
    const entry = entryFor(file);
    for (const term of matchedTerms) entry.bodyTerms.add(term);
  }

  const ranked = [];
  for (const [file, hit] of scored) {
    if (!hit.pathTerms.size && !hit.bodyTerms.size) continue;
    let score = PATH_HIT_WEIGHT * hit.pathTerms.size + hit.bodyTerms.size;
    if (isTestPath(file)) score *= TEST_FILE_FACTOR;
    ranked.push({ file, score, depth: file.split("/").length });
  }

  if (!ranked.length) return repoFiles.slice(0, NO_MATCH_SAMPLE);

  ranked.sort((a, b) => b.score - a.score || a.depth - b.depth || (a.file < b.file ? -1 : 1));
  return ranked.slice(0, limit).map(r => r.file);
}
