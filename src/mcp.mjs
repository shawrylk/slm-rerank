// Zero-dependency Model Context Protocol (MCP) server over JSON-RPC 2.0 / stdio.
//
// Launch via `npx slm-rerank --mcp`, or register with Claude Code:
//   claude mcp add slm-reranker -- npx -y slm-rerank --mcp
//
// Framing is newline-delimited JSON: one JSON-RPC message per line on stdin,
// one response per line on stdout. Every diagnostic goes to stderr so stdout
// stays a clean protocol channel.

import fs from "node:fs";
import path from "node:path";
import { Reranker } from "./client.mjs";
import { prepareCandidates } from "./chunker.mjs";
import { autoDiscoverEndpoint, discoverCandidateFiles, SLM_PORT_RANGE } from "./discovery.mjs";
import { groupBySlice } from "./boundary.mjs";

export const MCP_PROTOCOL_VERSION = "2024-11-05";
export const MCP_SERVER_NAME = "slm-reranker";
export const MCP_SERVER_VERSION = "0.6.4";

const IGNORE_DIRS = new Set([
  ".git", "node_modules", "dist", "build", ".cache", ".next", "__pycache__",
  ".turbo", ".pytest_cache", "venv", ".venv", "coverage", ".worktrees"
]);

const MAX_RESOLVED_FILES = 200;

export const RERANK_TOOL = {
  name: "rerank_codebase",
  description:
    "Semantically rerank codebase files and AST chunks against a query using a local small " +
    "language model (LFM/Qwen/Gemma). Use this to locate exact symbols, definitions and logic " +
    "across many files without ingesting thousands of unnecessary tokens. Returns ranked " +
    "file:line citations plus a candidate manifest you can read selectively.",
  inputSchema: {
    type: "object",
    properties: {
      query: {
        type: "string",
        description: "Natural language query or code task (e.g. 'handle database pool connection')."
      },
      paths_or_globs: {
        type: "array",
        items: { type: "string" },
        description:
          "File paths, directories, or glob patterns (e.g. ['src/**/*.ts']). " +
          "If omitted, candidate files are auto-discovered via ripgrep/git."
      },
      threshold: {
        type: "number",
        description: "Relevance score threshold between 0.0 and 1.0 (default: 0.65).",
        default: 0.65
      },
      top_k: {
        type: "integer",
        description: "Maximum number of high-confidence results to return (default: 5).",
        default: 5
      },
      stub: {
        type: "boolean",
        description: "Attach AST Ghost Stubs (folded skeletons) to top results to save tokens.",
        default: false
      },
      dirty: {
        type: "boolean",
        description: "Bias or filter candidates toward git uncommitted/modified files.",
        default: false
      },
      by_slice: {
        type: "boolean",
        description: "Group results by architectural vertical slice instead of a flat ranking.",
        default: false
      }
    },
    required: ["query"]
  }
};

const TOOLS = [RERANK_TOOL];

// JSON-RPC 2.0 error codes
const PARSE_ERROR = -32700;
const INVALID_REQUEST = -32600;
const METHOD_NOT_FOUND = -32601;
const INVALID_PARAMS = -32602;
const INTERNAL_ERROR = -32603;

class McpError extends Error {
  constructor(code, message) {
    super(message);
    this.code = code;
  }
}

function ok(id, result) {
  return { jsonrpc: "2.0", id, result };
}

function fail(id, code, message) {
  return { jsonrpc: "2.0", id, error: { code, message } };
}

/** Convert a glob pattern into an anchored RegExp (supports **, * and ?). */
function globToRegExp(pattern) {
  let out = "";
  for (let i = 0; i < pattern.length; i += 1) {
    const ch = pattern[i];
    if (ch === "*") {
      if (pattern[i + 1] === "*") {
        // `**/` crosses directory boundaries and may also match zero segments
        if (pattern[i + 2] === "/") {
          out += "(?:.*/)?";
          i += 2;
        } else {
          out += ".*";
          i += 1;
        }
      } else {
        out += "[^/]*";
      }
    } else if (ch === "?") {
      out += "[^/]";
    } else {
      out += ch.replace(/[.+^${}()|[\]\\]/g, "\\$&");
    }
  }
  return new RegExp(`^${out}$`);
}

function walkDir(dir, acc, limit) {
  let entries;
  try {
    entries = fs.readdirSync(dir, { withFileTypes: true });
  } catch {
    return;
  }
  for (const entry of entries) {
    if (acc.length >= limit) return;
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      if (IGNORE_DIRS.has(entry.name)) continue;
      walkDir(full, acc, limit);
    } else if (entry.isFile()) {
      acc.push(full);
    }
  }
}

/**
 * Expand file paths, directory walks and glob patterns into a de-duplicated
 * candidate file list, skipping build/dependency directories.
 */
export function resolveCandidatePaths(patterns, { cwd = process.cwd(), limit = MAX_RESOLVED_FILES } = {}) {
  const resolved = [];
  const seen = new Set();

  const push = (fp) => {
    const rel = path.isAbsolute(fp) ? path.relative(cwd, fp) || fp : fp;
    if (seen.has(rel) || resolved.length >= limit) return;
    seen.add(rel);
    resolved.push(rel);
  };

  for (const raw of patterns || []) {
    if (resolved.length >= limit) break;
    const pattern = String(raw || "").trim();
    if (!pattern) continue;

    const abs = path.isAbsolute(pattern) ? pattern : path.join(cwd, pattern);

    if (!/[*?]/.test(pattern)) {
      let stat = null;
      try {
        stat = fs.statSync(abs);
      } catch {
        continue;
      }
      if (stat.isFile()) {
        push(abs);
      } else if (stat.isDirectory()) {
        const acc = [];
        walkDir(abs, acc, limit - resolved.length);
        acc.forEach(push);
      }
      continue;
    }

    // Glob: walk the deepest literal prefix, then match the full pattern.
    const segments = pattern.split("/");
    const literal = [];
    for (const seg of segments) {
      if (/[*?]/.test(seg)) break;
      literal.push(seg);
    }
    const root = path.isAbsolute(pattern)
      ? (literal.join("/") || "/")
      : path.join(cwd, literal.join("/"));
    const re = globToRegExp(path.isAbsolute(pattern) ? pattern : path.join(cwd, pattern));

    const acc = [];
    walkDir(root, acc, limit * 4);
    for (const fp of acc) {
      if (re.test(fp)) push(fp);
    }
  }

  return resolved;
}

function formatRerankText(query, result, results, { endpoint, bySlice, stub, autoDiscovered, fileCount }) {
  const lines = [];
  const where = endpoint.port ? `port :${endpoint.port} (${endpoint.modelId})` : endpoint.url;
  lines.push(`🎯 SLM Semantic Reranker via ${where}`);
  lines.push(`Query: "${query}"`);
  lines.push(
    `Scanned ${fileCount} file(s)${autoDiscovered ? " (auto-discovered)" : ""} | ` +
    `evaluated ${result.totalEvaluated} chunk(s) | ${results.length} match(es)`
  );
  lines.push("");

  if (!results.length) {
    lines.push("No chunks scored above the relevance threshold.");
    return lines.join("\n");
  }

  if (bySlice) {
    const groups = groupBySlice(results);
    for (const [sliceName, group] of Object.entries(groups)) {
      lines.push(`📦 [Slice: ${sliceName}] (${group.items.length} matches, max ${(group.maxScore * 100).toFixed(1)}%)`);
      group.items.forEach((item, idx) => {
        const sym = item.chunk.symbol ? ` (${item.chunk.symbol})` : "";
        lines.push(`    #${idx + 1} | ${(item.score * 100).toFixed(1)}% | ${item.chunk.filePath}:${item.chunk.startLine}-${item.chunk.endLine}${sym}`);
      });
      lines.push("");
    }
  } else {
    results.forEach((item, idx) => {
      const sym = item.chunk.symbol ? ` (${item.chunk.symbol})` : "";
      const sliceTag = item.slice ? `[${item.slice}] ` : "";
      lines.push(` #${idx + 1} | Score: ${(item.score * 100).toFixed(1)}% | ${sliceTag}${item.chunk.filePath}:${item.chunk.startLine}-${item.chunk.endLine}${sym}`);
      if (stub && item.ghostStub) {
        lines.push(`--- 👻 Ghost Stub (${item.foldedLines} lines folded) ---`);
        lines.push(item.ghostStub.slice(0, 800) + (item.ghostStub.length > 800 ? "\n..." : ""));
        lines.push("--------------------------------------------------");
      }
    });
    lines.push("");
  }

  return lines.join("\n");
}

function buildManifest(query, result, results, endpoint) {
  return {
    query,
    endpoint: { url: endpoint.url, port: endpoint.port ?? null, model: endpoint.modelId ?? null },
    total_evaluated: result.totalEvaluated,
    tier1_applied: result.tier1Applied,
    filter_reason: result.filterReason ?? null,
    candidates: results.map(item => ({
      file: item.chunk.filePath,
      start_line: item.chunk.startLine,
      end_line: item.chunk.endLine,
      symbol: item.chunk.symbol ?? null,
      slice: item.slice ?? null,
      score: Number(item.score.toFixed(4))
    }))
  };
}

function textResult(text, isError = false) {
  return { content: [{ type: "text", text }], isError };
}

async function callRerankTool(args = {}, context = {}) {
  const query = typeof args.query === "string" ? args.query.trim() : "";
  if (!query) {
    throw new McpError(INVALID_PARAMS, "Missing required parameter: query");
  }

  const cwd = context.cwd || process.cwd();
  const host = context.host || process.env.SLM_HOST || "127.0.0.1";
  const threshold = typeof args.threshold === "number" && Number.isFinite(args.threshold)
    ? args.threshold
    : 0.65;
  const topK = Number.isInteger(args.top_k) && args.top_k > 0 ? args.top_k : 5;
  const stub = args.stub === true;
  const dirty = args.dirty === true;
  const bySlice = args.by_slice === true;

  const discover = context.autoDiscoverEndpoint || autoDiscoverEndpoint;
  const resolvePaths = context.resolveCandidatePaths || resolveCandidatePaths;
  const discoverFiles = context.discoverCandidateFiles || discoverCandidateFiles;
  const buildChunks = context.prepareCandidates || prepareCandidates;
  const RerankerImpl = context.Reranker || Reranker;

  const patterns = Array.isArray(args.paths_or_globs)
    ? args.paths_or_globs
    : (typeof args.paths_or_globs === "string" ? [args.paths_or_globs] : []);

  let files = resolvePaths(patterns, { cwd });
  let autoDiscovered = false;
  if (!files.length) {
    files = discoverFiles(query, { cwd });
    autoDiscovered = true;
  }
  if (!files.length) {
    return textResult(
      `No candidate files found for "${query}". Pass explicit paths_or_globs (e.g. ["src/**/*.ts"]).`,
      true
    );
  }

  const chunks = buildChunks(files);
  if (!chunks.length) {
    return textResult(`No readable code chunks found across ${files.length} candidate file(s).`, true);
  }

  const endpoint = await discover({ host, ports: context.ports || SLM_PORT_RANGE });

  const reranker = new RerankerImpl({ baseUrl: endpoint.url, model: context.model || "lfm", threshold });
  const result = await reranker.rerank(query, chunks, {
    threshold,
    stub,
    gitDiff: dirty,
    dirtyOnly: dirty
  });

  const results = result.results.slice(0, topK);
  const text = formatRerankText(query, result, results, {
    endpoint,
    bySlice,
    stub,
    autoDiscovered,
    fileCount: files.length
  });
  const manifest = buildManifest(query, result, results, endpoint);

  return {
    content: [
      { type: "text", text },
      { type: "text", text: `Candidate manifest (JSON):\n${JSON.stringify(manifest, null, 2)}` }
    ],
    isError: false
  };
}

/**
 * Handle a single JSON-RPC 2.0 message.
 *
 * Returns a response object, or `null` for notifications (which get no reply).
 * `context` allows injecting collaborators (Reranker, autoDiscoverEndpoint, cwd,
 * host) so the dispatcher can be unit tested without a live model server.
 */
export async function handleMcpMessage(request, context = {}) {
  if (!request || typeof request !== "object" || Array.isArray(request)) {
    return fail(null, INVALID_REQUEST, "Request must be a JSON-RPC 2.0 object");
  }

  const { id = null, method } = request;
  const isNotification = request.id === undefined || request.id === null;

  if (typeof method !== "string") {
    return isNotification ? null : fail(id, INVALID_REQUEST, "Missing method");
  }

  try {
    switch (method) {
      case "initialize":
        return ok(id, {
          protocolVersion: MCP_PROTOCOL_VERSION,
          capabilities: { tools: {} },
          serverInfo: { name: MCP_SERVER_NAME, version: MCP_SERVER_VERSION }
        });

      case "notifications/initialized":
      case "initialized":
        // Acknowledged silently: notifications never receive a response.
        return null;

      case "ping":
        return ok(id, {});

      case "tools/list":
        return ok(id, { tools: TOOLS });

      case "tools/call": {
        const params = request.params || {};
        if (params.name !== RERANK_TOOL.name) {
          return fail(id, METHOD_NOT_FOUND, `Unknown tool: ${params.name}`);
        }
        const result = await callRerankTool(params.arguments || {}, context);
        return ok(id, result);
      }

      default:
        if (method.startsWith("notifications/")) return null;
        return isNotification ? null : fail(id, METHOD_NOT_FOUND, `Method not found: ${method}`);
    }
  } catch (err) {
    if (isNotification) return null;
    const code = err instanceof McpError ? err.code : INTERNAL_ERROR;
    return fail(id, code, err?.message || String(err));
  }
}

/**
 * Start the stdio MCP server. Reads newline-delimited JSON-RPC from stdin and
 * writes responses to stdout. Never writes anything else to stdout.
 */
export function startMcpServer({ host = process.env.SLM_HOST || "127.0.0.1" } = {}) {
  const context = { host };
  let buffer = "";

  const send = (message) => {
    if (!message) return;
    process.stdout.write(`${JSON.stringify(message)}\n`);
  };

  const dispatch = async (line) => {
    let request;
    try {
      request = JSON.parse(line);
    } catch {
      send(fail(null, PARSE_ERROR, "Invalid JSON"));
      return;
    }

    if (Array.isArray(request)) {
      const responses = [];
      for (const item of request) {
        const res = await handleMcpMessage(item, context);
        if (res) responses.push(res);
      }
      if (responses.length) send(responses);
      return;
    }

    send(await handleMcpMessage(request, context));
  };

  // Serialize dispatch so responses are emitted in request order.
  let queue = Promise.resolve();

  process.stdin.setEncoding("utf-8");
  process.stdin.on("data", (chunk) => {
    buffer += chunk;
    let idx;
    while ((idx = buffer.indexOf("\n")) !== -1) {
      const line = buffer.slice(0, idx).trim();
      buffer = buffer.slice(idx + 1);
      if (!line) continue;
      queue = queue.then(() => dispatch(line)).catch((err) => {
        process.stderr.write(`[slm-reranker] dispatch error: ${err?.message || err}\n`);
      });
    }
  });

  process.stdin.on("end", () => {
    queue.finally(() => process.exit(0));
  });

  process.stderr.write(`[slm-reranker] MCP server ready (host=${host}, protocol=${MCP_PROTOCOL_VERSION})\n`);
  process.stdin.resume();

  return { context, handle: (req) => handleMcpMessage(req, context) };
}
