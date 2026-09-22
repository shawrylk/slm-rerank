// Optional SLM query expansion: turn a natural-language query into extra search
// terms that recall can match on.
//
// Deterministic recall (src/discovery.mjs) stems words — it can reach "migrate"
// from "migration", but never "bridge" from "harness". Only a model that knows
// the two are the same idea can do that, which is the whole point of this module.
//
// It is strictly additive and strictly optional. Every failure path returns an
// empty array, so a missing, slow or confused model degrades recall back to the
// deterministic terms rather than breaking it.
import crypto from "node:crypto";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { resolveCompletionUrls } from "./client.mjs";
import { extractQueryTerms, QUERY_STOP_WORDS } from "./discovery.mjs";

export const EXPANSION_CACHE_VERSION = 1;
const DEFAULT_TIMEOUT_MS = 2000;
const DEFAULT_MAX_TERMS = 12;
const CACHE_TTL_MS = 30 * 24 * 60 * 60 * 1000;
const MIN_TERM_LENGTH = 3;

// Generic words a model reaches for that match half a repository.
const USELESS_TERMS = new Set([
  "code", "file", "files", "function", "functions", "method", "methods", "class", "classes",
  "module", "modules", "source", "implementation", "logic", "handler", "handlers", "util",
  "utils", "helper", "helpers", "value", "values", "data", "object", "objects", "type", "types",
  "string", "number", "boolean", "return", "const", "let", "var", "import", "export", "src",
  // generic software vocabulary the model reaches for first; they match everything,
  // so they cost search slots and add noise without narrowing anything
  "api", "sdk", "app", "application", "framework", "library", "tool", "tools", "system",
  "service", "services", "server", "client", "integration", "documentation", "docs", "testing",
  "deployment", "configuration", "config", "environment", "development", "production",
  "feature", "features", "component", "components", "interface", "pattern", "patterns",
  "structure", "process", "processing", "operation", "operations", "management", "support"
]);

function cacheFile() {
  const base = process.env.XDG_CACHE_HOME || path.join(os.homedir(), ".cache");
  return path.join(base, "slm-rerank", "expansions.json");
}

function cacheKey(query, model) {
  return crypto.createHash("sha256").update(`${EXPANSION_CACHE_VERSION}|${model}|${query}`).digest("hex");
}

function readCache(key, now) {
  try {
    const raw = JSON.parse(fs.readFileSync(cacheFile(), "utf-8"));
    const entry = raw?.[key];
    if (!entry || !Array.isArray(entry.terms)) return null;
    if (now - (entry.ts || 0) > CACHE_TTL_MS) return null;
    return entry.terms;
  } catch {
    return null;   // no cache, unreadable cache, corrupt JSON — all mean "ask the model"
  }
}

function writeCache(key, terms, now) {
  try {
    const file = cacheFile();
    fs.mkdirSync(path.dirname(file), { recursive: true });
    let store = {};
    try {
      store = JSON.parse(fs.readFileSync(file, "utf-8")) || {};
    } catch {
      store = {};
    }
    store[key] = { terms, ts: now };
    fs.writeFileSync(file, JSON.stringify(store));
  } catch {
    // a cache that cannot be written must not fail the query
  }
}

/** ChatML with the reasoning bypass, matching the scorer's prompt shape. */
export function formatExpansionPrompt(query) {
  return `<|startoftext|><|im_start|>system\nYou expand code search queries for a codebase search tool.\nGiven a query, list the identifiers, domain nouns and synonyms that would plausibly appear in the source file that answers it, especially words the query does not already contain.\nReply with one line of lowercase comma-separated terms. No prose, no explanation.\n<|im_end|>\n<|im_start|>user\nQuery: ${query}\n<|im_end|>\n<|im_start|>assistant\n<think>\n</think>\n`;
}

/** Pull terms out of whatever shape the endpoint returned. */
export function parseExpansionText(text, query, maxTerms = DEFAULT_MAX_TERMS) {
  if (typeof text !== "string" || !text.trim()) return [];

  const already = new Set(extractQueryTerms(query));
  const terms = [];
  for (const piece of text.toLowerCase().split(/[^a-z0-9_]+/)) {
    const term = piece.trim();
    if (term.length < MIN_TERM_LENGTH) continue;
    if (USELESS_TERMS.has(term) || QUERY_STOP_WORDS.has(term)) continue;
    if (already.has(term) || terms.includes(term)) continue;
    terms.push(term);
    if (terms.length >= maxTerms) break;
  }
  return terms;
}

function extractText(data) {
  if (!data || typeof data !== "object") return "";
  if (typeof data.content === "string") return data.content;               // llama.cpp /completion
  const choice = Array.isArray(data.choices) ? data.choices[0] : null;
  if (!choice) return "";
  if (typeof choice.text === "string") return choice.text;                 // legacy completions
  if (typeof choice.message?.content === "string") return choice.message.content;
  return "";
}

/**
 * Ask the local model for extra search terms. Returns [] rather than throwing on
 * any failure: no server, timeout, bad JSON, empty answer.
 */
export async function expandQuery(query, {
  baseUrl,
  model = "lfm",
  timeoutMs = DEFAULT_TIMEOUT_MS,
  maxTerms = DEFAULT_MAX_TERMS,
  useCache = true,
  fetchImpl = fetch,
  now = Date.now()
} = {}) {
  if (!query || typeof query !== "string" || !baseUrl) return [];

  const key = cacheKey(query, model);
  if (useCache) {
    const cached = readCache(key, now);
    if (cached) return cached.slice(0, maxTerms);
  }

  const { completionUrl } = resolveCompletionUrls(baseUrl);
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const resp = await fetchImpl(completionUrl, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      signal: controller.signal,
      body: JSON.stringify({
        prompt: formatExpansionPrompt(query),
        n_predict: 64,
        temperature: 0,      // greedy: same query -> same terms, so the cache is meaningful
        top_p: 0.9,
        // Unlike scoring, this call reads generated text rather than logprobs, so
        // newline stop tokens are safe here — they are not in adapters.py.
        stop: ["<|im_end|>", "\n"],
        cache_prompt: true
      })
    });
    if (!resp.ok) return [];

    const terms = parseExpansionText(extractText(await resp.json()), query, maxTerms);
    if (terms.length && useCache) writeCache(key, terms, now);
    return terms;
  } catch {
    return [];   // aborted, unreachable, malformed — recall carries on without us
  } finally {
    clearTimeout(timer);
  }
}
