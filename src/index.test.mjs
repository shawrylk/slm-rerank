import test from "node:test";
import assert from "node:assert/strict";
import { applyTwoTierFilter, computeLexicalScore } from "./filter.mjs";
import { chunkFile } from "./chunker.mjs";
import { stitchChunkContext } from "./stitcher.mjs";
import { generateGhostStub } from "./stubber.mjs";
import { detectSlice, groupBySlice } from "./boundary.mjs";
import { spawnSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import {
  autoDiscoverEndpoint,
  discoverCandidateFiles,
  extractQueryTerms,
  isTestPath,
  resolveEndpointEnv,
  resolveHostEnv
} from "./discovery.mjs";
import { handleMcpMessage } from "./mcp.mjs";
import { Reranker } from "./client.mjs";
import { expandQuery, formatExpansionPrompt, parseExpansionText } from "./expander.mjs";
import { containsTerm, termKeys, tokenKeys } from "./lexical.mjs";
import { fuseLexicalPrior } from "./fusion.mjs";
import { VERSION } from "./index.mjs";
import { createRequire } from "node:module";

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
  const scoreBoosted = computeLexicalScore(chunk, ["calc"], new Set([path.normalize("src/modified.ts")]));
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

  // Nothing answering must not yield a phantom endpoint
  const missing = await autoDiscoverEndpoint({ host: "127.0.0.1", ports: [8099], env: {} });
  assert.equal(missing.ok, false);
  assert.equal(missing.url, null);
  assert.equal(missing.port, null);
  assert.match(missing.reason, /SLM_ENDPOINT/);
  assert.match(missing.reason, /8099/);
});

test("Discovery: env resolution is shared and ordered", () => {
  assert.equal(resolveHostEnv({}), "127.0.0.1");
  assert.equal(resolveHostEnv({ RERANKER_HOST: "10.2.99.1" }), "10.2.99.1");
  assert.equal(resolveHostEnv({ SLM_HOST: "a", RERANKER_HOST: "b" }), "a");
  assert.equal(resolveHostEnv({ SLM_HOST: "   " }), "127.0.0.1");

  assert.equal(resolveEndpointEnv({}), null);
  assert.equal(resolveEndpointEnv({ LFM_ENDPOINT: "http://x:8034/v1" }), "http://x:8034/v1");
  assert.equal(resolveEndpointEnv({ RERANKER_BASE_URL: "http://b/v1", LFM_ENDPOINT: "http://c/v1" }), "http://b/v1");
  assert.equal(
    resolveEndpointEnv({ SLM_ENDPOINT: " http://a/v1 ", RERANKER_BASE_URL: "http://b/v1" }),
    "http://a/v1"
  );
});

test("Discovery: a pinned endpoint wins even when a model is requested", async () => {
  const ep = await autoDiscoverEndpoint({
    env: { SLM_ENDPOINT: "http://192.168.1.220:8034/v1" },
    requestedModel: "qwen",
    ports: [8099]
  });
  assert.equal(ep.url, "http://192.168.1.220:8034/v1");
  assert.equal(ep.ok, true);
  assert.equal(ep.modelId, "pinned-via-env");
});

test("Reranker: baseUrl falls back to any endpoint env var", () => {
  const saved = { ...process.env };
  try {
    delete process.env.SLM_ENDPOINT;
    delete process.env.LFM_ENDPOINT;
    process.env.RERANKER_BASE_URL = "http://192.168.1.220:8034/v1/";
    assert.equal(new Reranker().baseUrl, "http://192.168.1.220:8034/v1");
  } finally {
    process.env = saved;
  }
});

test("Expander: prompt carries the query and the reasoning bypass", () => {
  const prompt = formatExpansionPrompt("interop harness");
  assert.ok(prompt.includes("Query: interop harness"));
  assert.ok(prompt.endsWith("<|im_start|>assistant\n<think>\n</think>\n"));
});

test("Expander: parsing drops prose, generic nouns and words already in the query", () => {
  const terms = parseExpansionText(
    "communication, protocol, the, code, function, interop, bridge, adapter, bridge",
    "interop harness"
  );
  assert.ok(terms.includes("bridge"));
  assert.ok(terms.includes("adapter"));
  assert.ok(!terms.includes("the"), "stop word leaked through");
  assert.ok(!terms.includes("code"), "generic noun leaked through");
  assert.ok(!terms.includes("function"), "generic noun leaked through");
  assert.ok(!terms.includes("interop"), "term already in the query was re-added");
  assert.equal(terms.filter(t => t === "bridge").length, 1, "duplicate term");

  assert.deepEqual(parseExpansionText("", "q"), []);
  assert.deepEqual(parseExpansionText(null, "q"), []);
});

test("Expander: reads llama.cpp content and caps the term count", async () => {
  const calls = [];
  const fetchImpl = async (url, opts) => {
    calls.push({ url, body: JSON.parse(opts.body) });
    return { ok: true, json: async () => ({ content: "bridge, adapter, wrapper, connector, marshal" }) };
  };

  const terms = await expandQuery("interop harness", {
    baseUrl: "http://127.0.0.1:8034/v1", useCache: false, maxTerms: 3, fetchImpl
  });

  assert.deepEqual(terms, ["bridge", "adapter", "wrapper"]);
  assert.equal(calls[0].url, "http://127.0.0.1:8034/completion", "must use the native endpoint");
  assert.equal(calls[0].body.temperature, 0, "greedy decoding keeps the cache meaningful");
  assert.equal(calls[0].body.n_predict, 64);
});

test("Expander: every failure degrades to no terms, never an exception", async () => {
  const reject = async () => { throw new Error("unreachable"); };
  const notOk = async () => ({ ok: false, json: async () => ({}) });
  const empty = async () => ({ ok: true, json: async () => ({}) });

  assert.deepEqual(await expandQuery("q", { baseUrl: "http://x/v1", useCache: false, fetchImpl: reject }), []);
  assert.deepEqual(await expandQuery("q", { baseUrl: "http://x/v1", useCache: false, fetchImpl: notOk }), []);
  assert.deepEqual(await expandQuery("q", { baseUrl: "http://x/v1", useCache: false, fetchImpl: empty }), []);
  assert.deepEqual(await expandQuery("q", { useCache: false, fetchImpl: reject }), [], "no baseUrl");
  assert.deepEqual(await expandQuery("", { baseUrl: "http://x/v1", useCache: false, fetchImpl: reject }), []);
});

test("Discovery: an expanded term widens recall without outranking a literal hit", () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "slm-expand-"));
  try {
    fs.mkdirSync(path.join(dir, "src"), { recursive: true });
    fs.writeFileSync(path.join(dir, "src", "harness.ts"), "export const x = 1;\n");
    fs.writeFileSync(path.join(dir, "src", "bridge.ts"), "export const y = 2;\n");
    const git = (...args) => spawnSync("git", args, { cwd: dir, encoding: "utf-8" });
    git("init", "-q", ".");
    git("add", "-A");
    git("-c", "user.email=t@example.com", "-c", "user.name=test", "commit", "-qm", "fixture");

    const base = discoverCandidateFiles("harness", { cwd: dir });
    assert.ok(!base.includes("src/bridge.ts"), "bridge.ts should not match literally");

    const expanded = discoverCandidateFiles("harness", { cwd: dir, extraTerms: ["bridge"] });
    assert.ok(expanded.includes("src/bridge.ts"), "expansion did not widen recall");
    assert.equal(expanded[0], "src/harness.ts", "a synonym outranked the literal match");
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

test("Discovery: query terms keep stems beside their roots", () => {
  const terms = extractQueryTerms("interop harness migration fixture");
  assert.deepEqual(terms, ["interop", "harness", "migration", "migrat", "migrate", "fixture"]);

  // stems must follow their own root, or a term cap severs them
  assert.ok(terms.indexOf("migrat") > terms.indexOf("migration"));
  assert.ok(terms.indexOf("fixture") > terms.indexOf("migrate"));

  // -ss is not a plural, -es is
  assert.deepEqual(extractQueryTerms("harness"), ["harness"]);
  assert.deepEqual(extractQueryTerms("classes"), ["classes", "class"]);
  assert.deepEqual(extractQueryTerms("exports"), ["exports", "export"]);
  assert.deepEqual(extractQueryTerms("chunking"), ["chunking", "chunk"]);

  // stop words only -> falls back to the first raw word
  assert.deepEqual(extractQueryTerms("the and of"), ["the"]);
  assert.deepEqual(extractQueryTerms(""), []);
});

test("Discovery: test paths are recognised, fixtures are not", () => {
  assert.equal(isTestPath("tests/unit-1.test.ts"), true);
  assert.equal(isTestPath("src/__tests__/thing.ts"), true);
  assert.equal(isTestPath("tests/test_client.py"), true);
  assert.equal(isTestPath("pkg/client_test.go"), true);
  assert.equal(isTestPath("src/interop/migrate-fixture.ts"), false);
  assert.equal(isTestPath("src/fixtures/rows.ts"), false);
  assert.equal(isTestPath("src/latest.ts"), false);
});

test("Discovery: a filename match outranks tests that merely mention the words", () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "slm-recall-"));
  try {
    fs.mkdirSync(path.join(dir, "src", "interop"), { recursive: true });
    fs.mkdirSync(path.join(dir, "tests"), { recursive: true });
    // the file that holds the harness: its name carries the vocabulary, its body does not
    fs.writeFileSync(
      path.join(dir, "src", "interop", "migrate-fixture.ts"),
      "export function buildLegacyBridge(rows) { return rows.map(normalizeRow); }\n"
    );
    // tests that restate the query vocabulary over and over
    for (let i = 1; i <= 5; i++) {
      fs.writeFileSync(
        path.join(dir, "tests", `unit-${i}.test.ts`),
        'describe("interop harness", () => { it("runs the interop harness migration", () => {}); });\n'
      );
    }

    // discovery shells out to rg or git; make the fixture tree visible to either
    const git = (...args) => spawnSync("git", args, { cwd: dir, encoding: "utf-8" });
    git("init", "-q", ".");
    git("add", "-A");
    git("-c", "user.email=t@example.com", "-c", "user.name=test", "commit", "-qm", "fixture");

    for (const query of ["fixture", "interop fixture", "interop harness migration fixture"]) {
      const files = discoverCandidateFiles(query, { cwd: dir });
      const hit = files.find(f => f.includes("migrate-fixture"));
      assert.ok(hit, `"${query}" lost the fixture file: ${files.join(", ") || "(none)"}`);
      assert.ok(
        files[0].includes("migrate-fixture"),
        `"${query}" ranked ${files[0]} above the fixture file`
      );
    }
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
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
  assert.deepEqual(res.result.serverInfo, { name: "slm-reranker", version: "0.7.0" });
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

  for (const param of ["query", "paths_or_globs", "threshold", "top_k", "stub", "dirty", "by_slice", "expand"]) {
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

test("MCP: tools/call reports why no model server was found", async () => {
  const reason = "No SLM model server answered on 127.0.0.1 (ports 8033-8040). Set SLM_ENDPOINT=http://<host>:8034/v1.";
  const res = await handleMcpMessage(
    {
      jsonrpc: "2.0",
      id: 7,
      method: "tools/call",
      params: { name: "rerank_codebase", arguments: { query: "payment flow", paths_or_globs: ["x.ts"] } }
    },
    {
      Reranker: class { constructor() { throw new Error("must not score against a dead endpoint"); } },
      resolveCandidatePaths: () => ["x.ts"],
      prepareCandidates: () => [
        { id: "x.ts:1-4", filePath: "x.ts", startLine: 1, endLine: 4, symbol: "charge", content: "charge()" }
      ],
      autoDiscoverEndpoint: async () => ({ url: null, port: null, modelId: null, ok: false, reason })
    }
  );

  assert.equal(res.result.isError, true);
  assert.equal(res.result.content[0].text, reason);
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

test("Reranker: resolves /completion and /chat/completions from a /v1 base URL", () => {
  const r = new Reranker({ baseUrl: "http://127.0.0.1:8034/v1/" });
  assert.equal(r.baseUrl, "http://127.0.0.1:8034/v1");
  assert.equal(r.completionUrl, "http://127.0.0.1:8034/completion");
  assert.equal(r.chatUrl, "http://127.0.0.1:8034/v1/chat/completions");
});

test("Reranker: resolves both URLs from a bare host and from a /completion base URL", () => {
  const bare = new Reranker({ baseUrl: "http://127.0.0.1:8034" });
  assert.equal(bare.completionUrl, "http://127.0.0.1:8034/completion");
  assert.equal(bare.chatUrl, "http://127.0.0.1:8034/v1/chat/completions");

  const direct = new Reranker({ baseUrl: "http://127.0.0.1:8034/completion" });
  assert.equal(direct.completionUrl, "http://127.0.0.1:8034/completion");
  assert.equal(direct.chatUrl, "http://127.0.0.1:8034/v1/chat/completions");
});

test("extractLogprobs: parses llama.cpp completion_probabilities into a confident yes", () => {
  const r = new Reranker();
  const payload = {
    completion_probabilities: [
      {
        content: "yes",
        top_logprobs: [
          { token: "yes", logprob: -0.02 },
          { token: "no", logprob: -3.9 },
          { token: "maybe", logprob: -7.1 }
        ]
      }
    ]
  };

  const score = r.extractLogprobs(payload, { content: "function chargeCard() { return gateway.charge(); }" });
  assert.ok(score > 0.7, `expected > 0.70, got ${score}`);
  assert.ok(score <= 1.0);
});

test("extractLogprobs: llama.cpp completion_probabilities scores a confident no low", () => {
  const r = new Reranker();
  const payload = {
    completion_probabilities: [
      {
        content: "no",
        top_logprobs: [
          { token: "no", logprob: -0.01 },
          { token: "yes", logprob: -4.5 }
        ]
      }
    ]
  };

  const score = r.extractLogprobs(payload, { content: "export const PI = 3.14;" });
  assert.ok(score < 0.3, `expected < 0.30, got ${score}`);
});

test("extractLogprobs: parses OpenAI chat choices[0].logprobs.content[0].top_logprobs", () => {
  const r = new Reranker();
  const payload = {
    choices: [
      {
        logprobs: {
          content: [
            {
              token: "Yes",
              logprob: -0.05,
              top_logprobs: [
                { token: "Yes", logprob: -0.05 },
                { token: "No", logprob: -3.2 }
              ]
            }
          ]
        }
      }
    ]
  };

  const score = r.extractLogprobs(payload, { content: "async function handlePayment() {}" });
  assert.ok(score > 0.7, `expected > 0.70, got ${score}`);
});

test("extractLogprobs: parses legacy completions top_logprobs token maps", () => {
  const r = new Reranker();
  const payload = {
    choices: [{ logprobs: { top_logprobs: [{ " yes": -0.1, " no": -2.8, ".": -6.0 }] } }]
  };

  const score = r.extractLogprobs(payload, { content: "function route() {}" });
  assert.ok(score > 0.7, `expected > 0.70, got ${score}`);
});

test("extractLogprobs: single-sided and absent yes/no signals", () => {
  const r = new Reranker();
  const chunk = { content: "function noop() {}" };

  const yesOnly = r.extractLogprobs({ top_logprobs: [{ token: "yes", logprob: -0.3 }] }, chunk);
  assert.ok(yesOnly > 0.9, `expected ~0.95, got ${yesOnly}`);

  const noOnly = r.extractLogprobs({ top_logprobs: [{ token: "no", logprob: -0.3 }] }, chunk);
  assert.ok(noOnly < 0.1, `expected ~0.05, got ${noOnly}`);

  const neither = r.extractLogprobs({ top_logprobs: [{ token: "{", logprob: -0.1 }] }, chunk);
  assert.equal(neither, 0.05);

  assert.equal(r.extractLogprobs({}, chunk), 0.05);
  assert.equal(r.extractLogprobs(null, chunk), 0.05);
});

test("scoreChunk: scores against llama.cpp /completion without touching the chat endpoint", async () => {
  const r = new Reranker({ baseUrl: "http://127.0.0.1:8034/v1" });
  const calls = [];
  const originalFetch = globalThis.fetch;

  globalThis.fetch = async (url, init) => {
    calls.push({ url, body: JSON.parse(init.body) });
    return {
      ok: true,
      status: 200,
      json: async () => ({
        completion_probabilities: [
          { content: "yes", top_logprobs: [{ token: "yes", logprob: -0.01 }, { token: "no", logprob: -4.0 }] }
        ]
      })
    };
  };

  try {
    const res = await r.scoreChunk("payment flow", {
      filePath: "src/pay.ts",
      symbol: "charge",
      startLine: 1,
      endLine: 4,
      content: "function charge() {}"
    });

    assert.equal(calls.length, 1);
    assert.equal(calls[0].url, "http://127.0.0.1:8034/completion");
    assert.equal(calls[0].body.n_predict, 1);
    assert.equal(calls[0].body.n_probs, 10);
    assert.deepEqual(calls[0].body.stop, ["<|im_end|>"]);
    assert.ok(calls[0].body.prompt.includes("Respond only with yes or no."));
    assert.equal(res.error, undefined);
    assert.ok(res.score > 0.7, `expected > 0.70, got ${res.score}`);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("scoreChunk: falls back from /completion to /chat/completions on non-200", async () => {
  const r = new Reranker({ baseUrl: "http://127.0.0.1:8034/v1" });
  const calls = [];
  const originalFetch = globalThis.fetch;

  globalThis.fetch = async (url, init) => {
    calls.push({ url, body: JSON.parse(init.body) });
    if (url.endsWith("/completion")) {
      return { ok: false, status: 404, json: async () => ({}) };
    }
    return {
      ok: true,
      status: 200,
      json: async () => ({
        choices: [
          {
            logprobs: {
              content: [
                { token: "yes", logprob: -0.02, top_logprobs: [{ token: "yes", logprob: -0.02 }, { token: "no", logprob: -3.5 }] }
              ]
            }
          }
        ]
      })
    };
  };

  try {
    const res = await r.scoreChunk("payment flow", { filePath: "src/pay.ts", content: "function charge() {}" });

    assert.equal(calls.length, 2);
    assert.equal(calls[0].url, "http://127.0.0.1:8034/completion");
    assert.equal(calls[1].url, "http://127.0.0.1:8034/v1/chat/completions");
    assert.equal(calls[1].body.max_tokens, 1);
    assert.equal(calls[1].body.logprobs, true);
    assert.equal(calls[1].body.top_logprobs, 10);
    assert.equal(res.error, undefined);
    assert.ok(res.score > 0.7, `expected > 0.70, got ${res.score}`);
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("scoreChunk: falls back to chat on a /completion network error, and reports both failures", async () => {
  const r = new Reranker({ baseUrl: "http://127.0.0.1:8034/v1" });
  const originalFetch = globalThis.fetch;
  const calls = [];

  globalThis.fetch = async url => {
    calls.push(url);
    if (url.endsWith("/completion")) throw new Error("ECONNREFUSED");
    return {
      ok: true,
      status: 200,
      json: async () => ({ choices: [{ logprobs: { content: [{ top_logprobs: [{ token: "no", logprob: -0.01 }, { token: "yes", logprob: -5.0 }] }] } }] })
    };
  };

  try {
    const res = await r.scoreChunk("payment flow", { filePath: "src/pay.ts", content: "const x = 1;" });
    assert.deepEqual(calls, ["http://127.0.0.1:8034/completion", "http://127.0.0.1:8034/v1/chat/completions"]);
    assert.equal(res.error, undefined);
    assert.ok(res.score < 0.3, `expected < 0.30, got ${res.score}`);
  } finally {
    globalThis.fetch = originalFetch;
  }

  globalThis.fetch = async () => {
    throw new Error("ECONNREFUSED");
  };

  try {
    const res = await r.scoreChunk("payment flow", { filePath: "src/pay.ts", content: "const x = 1;" });
    assert.equal(res.score, 0.0);
    assert.equal(res.error, "ECONNREFUSED");
  } finally {
    globalThis.fetch = originalFetch;
  }
});

test("formatPrompt: emits the ChatML binary-evaluator prompt with file, line and symbol context", () => {
  const r = new Reranker();
  const prompt = r.formatPrompt("find the charge handler", {
    filePath: "src/pay.ts",
    symbol: "charge",
    startLine: 10,
    endLine: 20,
    content: "function charge() {}"
  });

  assert.ok(prompt.startsWith("<|startoftext|><|im_start|>system\n"));
  assert.ok(prompt.includes("You are a binary code retrieval evaluator."));
  assert.ok(prompt.includes("Query: find the charge handler"));
  assert.ok(prompt.includes("File: src/pay.ts\n"));
  assert.ok(prompt.includes("Lines: 10-20\n"));
  assert.ok(prompt.includes("Symbol: charge\n"));
  assert.ok(prompt.includes("Respond only with yes or no.<|im_end|>"));
  assert.ok(prompt.endsWith("<|im_start|>assistant\n<think>\n</think>\n"));
});

test("Lexical: a term matches whole tokens, never a substring of another word", () => {
  const has = (text, word) => containsTerm(tokenKeys(text), termKeys(word));

  // every feature has a shared/ directory; it is not a share
  assert.equal(has("src/features/photos/shared/queries", "share"), false);
  assert.equal(has("src/features/sharing/slices/create-share", "share"), true);
  assert.equal(has("SharePhotosDialog", "share"), true);
  assert.equal(has("shares", "share"), true);

  // the verb form in a query reaches the base form in code
  assert.equal(has("createShare", "created"), true);
  assert.equal(has("deleteFolder", "delete"), true);
  assert.equal(has("blackboard/slices/folders", "folder"), true);
  assert.equal(has("runMigrate", "migration"), true);

  // short words stay short
  assert.equal(has("mobile-sync/slices/run-replay", "lay"), false);
  assert.equal(has("patterns/layouts/overlay-placement", "lay"), false);
  assert.equal(has("application/ports/storage", "app"), false);
  assert.equal(has("shell/chrome/app-header", "app"), true);

  // a query word written as one word still meets a camelCase identifier
  assert.equal(has("const store = openIndexedDB();", "indexeddb"), true);
  assert.equal(has("<ToolBar />", "toolbar"), true);
});

test("Discovery: a path term is a whole token, so a shared/ directory is not a share", () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "slm-tokens-"));
  try {
    for (const feature of ["audit", "photos", "pins"]) {
      fs.mkdirSync(path.join(dir, "src", "features", feature, "shared"), { recursive: true });
      fs.writeFileSync(path.join(dir, "src", "features", feature, "shared", "queries.ts"), "export const rows = [];\n");
    }
    fs.mkdirSync(path.join(dir, "src", "features", "sharing", "slices"), { recursive: true });
    fs.writeFileSync(
      path.join(dir, "src", "features", "sharing", "slices", "create-share.ts"),
      "export function mint(input) { return input; }\n"
    );
    const git = (...args) => spawnSync("git", args, { cwd: dir, encoding: "utf-8" });
    git("init", "-q", ".");
    git("add", "-A");
    git("-c", "user.email=t@example.com", "-c", "user.name=test", "commit", "-qm", "fixture");

    const files = discoverCandidateFiles("share", { cwd: dir });
    assert.equal(files[0], "src/features/sharing/slices/create-share.ts", `ranked: ${files.join(", ")}`);
    assert.ok(!files.some(f => f.includes("/shared/")), `a shared/ file matched "share": ${files.join(", ")}`);
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

test("Two-Tier Filter: a small candidate set still comes back in lexical order", () => {
  const frame = { id: "frame", filePath: "src/ui/layout/content-frame.tsx", symbol: "ContentFrame", content: "export function ContentFrame() { return null; }" };
  const header = { id: "header", filePath: "src/shell/chrome/app-header.tsx", symbol: "AppHeader", content: "export function AppHeader() { return null; }" };

  const res = applyTwoTierFilter([frame, header], "mobile app header", {});
  assert.equal(res.tier1Applied, false);
  assert.equal(res.retained.length, 2);
  assert.equal(res.retained[0].id, "header", "a prompt hook scores only the head, so the head must be the best lexical match");
});

test("Two-Tier Filter: a shared/ chunk does not outrank the chunk named after the share", () => {
  const filler = Array.from({ length: 70 }, (_, i) => ({
    id: `filler_${i}`, filePath: `src/misc/file_${i}.ts`, symbol: `f${i}`, content: "const x = 1;"
  }));
  const shared = {
    id: "shared", filePath: "src/features/photos/shared/types.ts", symbol: "SharedRow",
    content: "export interface SharedRow {}\n// shared by the shared queries in shared/"
  };
  const share = {
    id: "share", filePath: "src/features/sharing/slices/create-share.ts", symbol: "createShare",
    content: "export function createShare() {}"
  };

  const res = applyTwoTierFilter([...filler, shared, share], "where a share is created", {});
  assert.equal(res.tier1Applied, true);
  assert.equal(res.retained[0].id, "share");
  assert.ok(res.retained.indexOf(shared) === -1 || res.retained.indexOf(shared) > 0);
});

test("Two-Tier Filter: every name a chunk declares counts, not only its first", () => {
  const panels = {
    id: "panels", filePath: "src/ui/explorer/explorer-panels.tsx", symbol: "Menu",
    content: "type Menu = string;\nexport function ExplorerHeader() { return null; }"
  };
  const workspace = {
    id: "workspace", filePath: "src/ui/explorer/workspace-explorer.tsx", symbol: "WorkspaceExplorer",
    content: "export function WorkspaceExplorer() { /* header, header, header */ return null; }"
  };

  const res = applyTwoTierFilter([workspace, panels], "explorer header", {});
  assert.equal(res.retained[0].id, "panels");
});

// A fake llama.cpp /completion endpoint whose yes/no logprobs depend on the file in the prompt.
function withModelVerdicts(verdicts, run) {
  const originalFetch = globalThis.fetch;
  globalThis.fetch = async (url, init) => {
    const { prompt } = JSON.parse(init.body);
    const file = Object.keys(verdicts).find(f => prompt.includes(`File: ${f}\n`));
    const verdict = verdicts[file];
    if (!verdict) throw new Error("ECONNREFUSED");
    return {
      ok: true,
      status: 200,
      json: async () => ({
        completion_probabilities: [{ top_logprobs: [{ token: "yes", logprob: verdict.yes }, { token: "no", logprob: verdict.no }] }]
      })
    };
  };
  return run().finally(() => { globalThis.fetch = originalFetch; });
}

test("Reranker: a chunk the query names overtakes a distractor the model scores higher", async () => {
  const answer = {
    id: "answer", filePath: "src/features/sharing/slices/create-share.ts", startLine: 1, endLine: 3,
    symbol: "createShare", content: "1: export function createShare() { return mint(); }"
  };
  const distractor = {
    id: "distractor", filePath: "src/features/blueprints/slices/create-blueprint.ts", startLine: 1, endLine: 3,
    symbol: "createBlueprint", content: "1: export function createBlueprint() { return draft(); }"
  };
  const verdicts = {
    [answer.filePath]: { yes: -1.0, no: -0.5 },
    [distractor.filePath]: { yes: -0.5, no: -1.0 }
  };

  await withModelVerdicts(verdicts, async () => {
    const r = new Reranker({ baseUrl: "http://127.0.0.1:8034/v1", threshold: 0 });
    const res = await r.rerank("where a share is created", [distractor, answer]);
    const top = res.results[0];
    assert.equal(top.chunk.id, "answer", `ranked: ${res.results.map(x => `${x.chunk.id} ${x.score}`).join(", ")}`);
    assert.ok(top.rawScore < 0.5, "rawScore must stay the model's own score");
    assert.ok(top.score > top.rawScore, "the name match must raise the fused score");
    const other = res.results.find(x => x.chunk.id === "distractor");
    assert.ok(other.rawScore > 0.5 && other.score < other.rawScore);
  });
});

test("Reranker: a chunk the model could not score stays at zero", async () => {
  const named = {
    id: "named", filePath: "src/features/sharing/slices/create-share.ts", startLine: 1, endLine: 3,
    symbol: "createShare", content: "1: export function createShare() { return mint(); }"
  };
  const plain = {
    id: "plain", filePath: "src/misc/util.ts", startLine: 1, endLine: 1, symbol: "util", content: "1: const x = 1;"
  };

  await withModelVerdicts({ [plain.filePath]: { yes: -2.0, no: -0.2 } }, async () => {
    const r = new Reranker({ baseUrl: "http://127.0.0.1:8034/v1", threshold: 0 });
    const res = await r.rerank("where a share is created", [named, plain]);
    const failed = res.results.find(x => x.chunk.id === "named");
    assert.ok(failed.error);
    assert.equal(failed.score, 0);
    assert.equal(failed.rawScore, 0);
  });
});

test("Reranker: a lexical difference inside the noise keeps the model order", async () => {
  const a = { id: "a", filePath: "src/a/share.ts", startLine: 1, endLine: 1, symbol: "share.ts", content: "1: // share share" };
  const b = { id: "b", filePath: "src/b/share.ts", startLine: 1, endLine: 1, symbol: "share.ts", content: "1: // share" };
  const verdicts = {
    [a.filePath]: { yes: -1.0, no: -0.6 },
    [b.filePath]: { yes: -0.6, no: -1.0 }
  };

  await withModelVerdicts(verdicts, async () => {
    const r = new Reranker({ baseUrl: "http://127.0.0.1:8034/v1", threshold: 0 });
    const res = await r.rerank("share", [a, b]);
    assert.deepEqual(res.results.map(x => x.chunk.id), ["b", "a"]);
  });
});

test("Fusion: a caller that scores only the head can judge it against every candidate", () => {
  const head = [{ rawScore: 0.56, score: 0.56 }, { rawScore: 0.2, score: 0.2 }];
  const headLexical = [32, 41];
  const everyCandidate = [41, 32, 5, 4, 3, 3, 2, 2, 1, 1];

  const [againstHead] = fuseLexicalPrior(head, headLexical);
  const [againstAll] = fuseLexicalPrior(head, headLexical, everyCandidate);

  assert.ok(againstAll.score > 0.65, `a strong match among all candidates should pass: ${againstAll.score}`);
  assert.ok(againstHead.score < head[0].rawScore, `among the head alone it is below average: ${againstHead.score}`);
});

test("Fusion: the default population is the scored set, as rerank() uses it", () => {
  const results = [{ rawScore: 0.4, score: 0.4 }, { rawScore: 0.7, score: 0.7 }, { rawScore: 0.1, score: 0.1 }];
  const lexical = [20, 5, 1];
  assert.deepEqual(fuseLexicalPrior(results, lexical), fuseLexicalPrior(results, lexical, lexical));
});

test("VERSION is the version in package.json", () => {
  const { version } = createRequire(import.meta.url)("../package.json");
  assert.equal(VERSION, version);
});
