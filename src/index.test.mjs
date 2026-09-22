import test from "node:test";
import assert from "node:assert/strict";
import { applyTwoTierFilter, computeLexicalScore } from "./filter.mjs";
import { chunkFile } from "./chunker.mjs";
import { stitchChunkContext } from "./stitcher.mjs";
import { generateGhostStub } from "./stubber.mjs";
import { detectSlice, groupBySlice } from "./boundary.mjs";
import { autoDiscoverEndpoint, discoverCandidateFiles } from "./discovery.mjs";

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
  const ep = await autoDiscoverEndpoint({ host: "127.0.0.1" });
  assert.ok(ep.url);
  assert.equal(ep.ok, true);
  assert.equal(ep.port, 8034); // Active port on this workstation
});

test("Discovery: ripgrep file discovery extracts valid files", () => {
  const files = discoverCandidateFiles("chunker and symbols");
  assert.ok(Array.isArray(files));
  assert.ok(files.length > 0);
});
