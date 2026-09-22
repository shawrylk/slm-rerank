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
export function discoverCandidateFiles(query, { cwd = process.cwd(), limit = 60 } = {}) {
  if (!query || typeof query !== "string") return [];

  // Extract key search terms (excluding short / stop words)
  const stopWords = new Set([
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "of", "for",
    "with", "from", "into", "by", "as", "is", "are", "was", "were", "be",
    "code", "file", "function", "method", "class", "find", "search", "how", "what", "where"
  ]);

  const rawWords = query
    .toLowerCase()
    .match(/\b[a-z0-9_]{2,}\b/g)
    ?.filter(w => !stopWords.has(w)) || [];

  if (!rawWords.length) {
    rawWords.push(query.trim().split(/\s+/)[0]);
  }

  // Stemming & term expansion (e.g. chunking -> chunk)
  const expandedTerms = [...rawWords];
  for (const t of rawWords) {
    if (t.endsWith("ing") && t.length > 4) {
      const stem = t.slice(0, -3);
      if (!expandedTerms.includes(stem)) expandedTerms.push(stem);
    } else if (t.endsWith("ed") && t.length > 3) {
      const stem = t.slice(0, -2);
      if (!expandedTerms.includes(stem)) expandedTerms.push(stem);
    } else if (t.endsWith("s") && t.length > 3) {
      const stem = t.slice(0, -1);
      if (!expandedTerms.includes(stem)) expandedTerms.push(stem);
    }
  }

  const codeExts = new Set([".py", ".ts", ".tsx", ".js", ".jsx", ".mjs", ".rs", ".go", ".c", ".cpp", ".h"]);
  const collectedFiles = new Set();

  // 1. Try ripgrep first
  try {
    for (const term of expandedTerms.slice(0, 3)) {
      if (collectedFiles.size >= limit) break;
      const res = spawnSync("rg", [
        "--files-with-matches",
        "--ignore-case",
        "--max-count", "1",
        "--glob", "!node_modules",
        "--glob", "!.git",
        "--glob", "!dist",
        "--glob", "!build",
        "--glob", "!coverage",
        "--glob", "!__pycache__",
        "--glob", "!.venv",
        term
      ], { cwd, encoding: "utf-8", maxBuffer: 10 * 1024 * 1024 });

      if (res.status === 0 && res.stdout) {
        for (const line of res.stdout.split("\n")) {
          const trimmed = line.trim();
          if (trimmed && fs.existsSync(path.resolve(cwd, trimmed)) && codeExts.has(path.extname(trimmed))) {
            collectedFiles.add(trimmed);
            if (collectedFiles.size >= limit) break;
          }
        }
      }
    }
  } catch {
    // ripgrep not available
  }

  // 2. Fallback to git grep if ripgrep not available or found nothing
  if (collectedFiles.size === 0) {
    try {
      for (const term of expandedTerms.slice(0, 3)) {
        if (collectedFiles.size >= limit) break;
        const gitGrep = spawnSync("git", ["grep", "-l", "-i", term], { cwd, encoding: "utf-8" });
        if (gitGrep.status === 0 && gitGrep.stdout) {
          for (const line of gitGrep.stdout.split("\n")) {
            const trimmed = line.trim();
            if (trimmed && fs.existsSync(path.resolve(cwd, trimmed)) && codeExts.has(path.extname(trimmed))) {
              collectedFiles.add(trimmed);
              if (collectedFiles.size >= limit) break;
            }
          }
        }
      }
    } catch {
      // git grep failed
    }
  }

  // 3. Fallback to git ls-files path matching
  if (collectedFiles.size === 0) {
    try {
      const gitRes = spawnSync("git", ["ls-files"], { cwd, encoding: "utf-8" });
      if (gitRes.status === 0 && gitRes.stdout) {
        const allFiles = gitRes.stdout.split("\n").filter(f => codeExts.has(path.extname(f)));

        for (const f of allFiles) {
          const lower = f.toLowerCase();
          if (expandedTerms.some(t => lower.includes(t))) {
            collectedFiles.add(f);
            if (collectedFiles.size >= limit) break;
          }
        }

        if (collectedFiles.size === 0) {
          for (const f of allFiles.slice(0, 20)) {
            collectedFiles.add(f);
          }
        }
      }
    } catch {
      // Not a git repository
    }
  }

  return Array.from(collectedFiles);
}
