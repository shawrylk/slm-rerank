import test from "node:test";
import assert from "node:assert/strict";
import { applyTwoTierFilter, computeLexicalScore } from "./filter.mjs";
import { chunkFile } from "./chunker.mjs";
import { stitchChunkContext } from "./stitcher.mjs";

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

test("Two-Tier Filter: pre-filter over 60 candidates", () => {
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
