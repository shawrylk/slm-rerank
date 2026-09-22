import test from "node:test";
import assert from "node:assert/strict";
import { applyTwoTierFilter, computeLexicalScore } from "./filter.mjs";
import { chunkFile } from "./chunker.mjs";
import { stitchChunkContext } from "./stitcher.mjs";
import { generateGhostStub } from "./stubber.mjs";
import { detectSlice, groupBySlice } from "./boundary.mjs";
import { autoDiscoverEndpoint, discoverCandidateFiles } from "./discovery.mjs";
import { handleMcpMessage } from "./mcp.mjs";

test("Two-Tier Filter: bypass under 60 candidates", () => {
  const chunks = Array.from({ length: 40 }, (_, i) => ({
    id: `chunk_${i}`,
    symbol: `func_${i}`,
    filePath: `file_${i}.ts`,
    content: `function func_${i}() {}`
  }));

  const res = applyTwoTierFilter(chunks, "find func_5", false);
  assert.equal(res.retained.length, 40);
  assert.equal(res.tier1Applied, false);
});

test("Two-Tier Filter: pre-filter over 60 candidates with recency bias", () => {
  const chunks = Array.from({ length: 100 }, (_, i) => ({
    id: `chunk_${i}`,
    symbol: `func_${i}`,
    filePath: `file_${i}.ts`,
    content: `function func_${i}() {}`
  }));
  chunks[42].symbol = "critical_auth_handler";

  const res = applyTwoTierFilter(chunks, "critical_auth_handler", false);
  assert.equal(res.retained.length, 80);
  assert.equal(res.tier1Applied, true);
  assert.ok(res.retained.some(c => c.symbol === "critical_auth_handler"));
});

test("Lexical Score: git-diff recency boost", () => {
  const chunk = { filePath: "src/modified.ts", symbol: "calc", content: "calc()" };
  const scoreNormal = computeLexicalScore(chunk, ["calc"]);
  const scoreBoosted = computeLexicalScore(chunk, ["calc"], new Set(["src/modified.ts"]));
  assert.ok(scoreBoosted > scoreNormal, "Dirty file should receive recency score boost");
});

test("Chunker: splits lines into chunks with symbol detection", () => {
  const sample = `
export class PaymentService {
  process() {
    return true;
  }
}
`;
  const chunks = chunkFile("payment.ts", sample, 50, 10);
  assert.ok(chunks.length >= 1);
  assert.equal(chunks[0].symbol, "PaymentService");
  assert.equal(chunks[0].startLine, 1);
});

test("Ghost Stub: preserves imports, target chunk, folds sibling methods", () => {
  const code = `import { logger } from "./logger.mjs";
export interface User { id: string; }

function oldUnrelatedHandler() {
  const a = 1;
  const b = 2;
  return a + b;
}

export function targetPaymentFlow() {
  return "paid";
}
`;

  const { stub, foldedLines } = generateGhostStub("service.ts", {
    startLine: 10,
    endLine: 12,
    symbol: "targetPaymentFlow"
  }, { content: code });

  assert.ok(stub.includes("import { logger }"));
  assert.ok(stub.includes("export interface User"));
  assert.ok(stub.includes("folded"));
  assert.ok(stub.includes("export function targetPaymentFlow()"));
  assert.ok(foldedLines > 0);
});

test("Boundary: detects enterprise slice conventions", () => {
  assert.equal(detectSlice("features/billing/src/checkout.ts"), "billing");
  assert.equal(detectSlice("modules/auth/tokens.ts"), "auth");
  assert.equal(detectSlice("packages/qc-harness/index.mjs"), "qc-harness");
  assert.equal(detectSlice("workers/rasterizer/worker.ts"), "rasterizer");
  assert.equal(detectSlice("src/common/utils.ts"), "common");
});

test("Boundary: groups results by slice with max scores", () => {
  const results = [
    { score: 0.92, chunk: { filePath: "features/billing/charge.ts" } },
    { score: 0.65, chunk: { filePath: "features/billing/invoice.ts" } },
    { score: 0.88, chunk: { filePath: "features/auth/login.ts" } }
  ];

  const grouped = groupBySlice(results);
  assert.ok(grouped.billing);
  assert.ok(grouped.auth);
  assert.equal(grouped.billing.items.length, 2);
  assert.equal(grouped.billing.maxScore, 0.92);
  assert.equal(grouped.auth.items.length, 1);
});

test("Discovery: discovers live ports in 8033-8040 range", async () => {
  const http = await import("node:http");
  const server = http.createServer((req, res) => {
    res.writeHead(200, { "Content-Type": "application/json" });
    res.end(JSON.stringify({ data: [{ id: "mock-lfm-model" }] }));
  });

  await new Promise(resolve => server.listen(8039, "127.0.0.1", resolve));

  try {
    const ep = await autoDiscoverEndpoint({ host: "127.0.0.1", ports: [8039] });
    assert.ok(ep.url);
    assert.equal(ep.ok, true);
    assert.equal(ep.port, 8039);
    assert.equal(ep.modelId, "mock-lfm-model");
  } finally {
    await new Promise(resolve => server.close(resolve));
  }

  // Offline fallback
  const fallback = await autoDiscoverEndpoint({ host: "127.0.0.1", ports: [8099] });
  assert.equal(fallback.port, 8034);
  assert.ok(fallback.url.includes("8034"));
});

test("Discovery: ripgrep file discovery extracts valid files", () => {
  const files = discoverCandidateFiles("chunker and symbols");
  assert.ok(Array.isArray(files));
  assert.ok(files.length > 0);
});

test("MCP: initialize returns protocol version, server info and tool capability", async () => {
  const res = await handleMcpMessage({
    jsonrpc: "2.0",
    id: 1,
    method: "initialize",
    params: { protocolVersion: "2024-11-05", capabilities: {}, clientInfo: { name: "test", version: "0" } }
  });

  assert.equal(res.jsonrpc, "2.0");
  assert.equal(res.id, 1);
  assert.equal(res.result.protocolVersion, "2024-11-05");
  assert.deepEqual(res.result.serverInfo, { name: "slm-reranker", version: "0.6.2" });
  assert.deepEqual(res.result.capabilities, { tools: {} });
});

test("MCP: notifications/initialized is acknowledged without a response", async () => {
  const res = await handleMcpMessage({ jsonrpc: "2.0", method: "notifications/initialized" });
  assert.equal(res, null);
});

test("MCP: tools/list advertises rerank_codebase with its full parameter schema", async () => {
  const res = await handleMcpMessage({ jsonrpc: "2.0", id: 2, method: "tools/list" });

  assert.equal(res.id, 2);
  assert.equal(res.result.tools.length, 1);

  const tool = res.result.tools[0];
  assert.equal(tool.name, "rerank_codebase");
  assert.equal(tool.inputSchema.type, "object");
  assert.deepEqual(tool.inputSchema.required, ["query"]);

  for (const param of ["query", "paths_or_globs", "threshold", "top_k", "stub", "dirty", "by_slice"]) {
    assert.ok(tool.inputSchema.properties[param], `missing parameter: ${param}`);
  }
});

test("MCP: ping returns an empty result", async () => {
  const res = await handleMcpMessage({ jsonrpc: "2.0", id: 3, method: "ping" });
  assert.equal(res.id, 3);
  assert.deepEqual(res.result, {});
});

test("MCP: unknown method yields JSON-RPC method-not-found error", async () => {
  const res = await handleMcpMessage({ jsonrpc: "2.0", id: 4, method: "resources/list" });
  assert.equal(res.error.code, -32601);
});

test("MCP: tools/call reranks injected candidates and returns a manifest", async () => {
  class FakeReranker {
    constructor(opts) { this.opts = opts; }
    async rerank(query, chunks) {
      return {
        query,
        results: chunks.map((chunk, i) => ({ score: 0.9 - i * 0.1, chunk, slice: "core" })),
        bySlice: {},
        totalEvaluated: chunks.length,
        tier1Applied: false,
        filterReason: "bypass"
      };
    }
  }

  const res = await handleMcpMessage(
    {
      jsonrpc: "2.0",
      id: 5,
      method: "tools/call",
      params: { name: "rerank_codebase", arguments: { query: "payment flow", paths_or_globs: ["x.ts"], top_k: 1 } }
    },
    {
      Reranker: FakeReranker,
      resolveCandidatePaths: () => ["x.ts"],
      prepareCandidates: () => [
        { id: "x.ts:1-4", filePath: "x.ts", startLine: 1, endLine: 4, symbol: "charge", content: "charge()" }
      ],
      autoDiscoverEndpoint: async () => ({ url: "http://127.0.0.1:8034/v1", port: 8034, modelId: "mock-lfm" })
    }
  );

  assert.equal(res.result.isError, false);
  assert.equal(res.result.content.length, 2);
  assert.ok(res.result.content[0].text.includes("x.ts:1-4"));

  const manifest = JSON.parse(res.result.content[1].text.replace("Candidate manifest (JSON):\n", ""));
  assert.equal(manifest.candidates.length, 1);
  assert.equal(manifest.candidates[0].file, "x.ts");
  assert.equal(manifest.endpoint.port, 8034);
});

test("MCP: tools/call without a query returns invalid-params", async () => {
  const res = await handleMcpMessage({
    jsonrpc: "2.0",
    id: 6,
    method: "tools/call",
    params: { name: "rerank_codebase", arguments: {} }
  });
  assert.equal(res.error.code, -32602);
});
