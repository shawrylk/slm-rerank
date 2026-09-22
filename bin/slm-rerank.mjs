#!/usr/bin/env node

import { spawn } from "node:child_process";
import fs from "node:fs";
import { parseArgs } from "node:util";
import { Reranker } from "../src/client.mjs";
import { prepareCandidates } from "../src/chunker.mjs";

const options = {
  query: { type: "string", short: "q" },
  "base-url": { type: "string", short: "e" },
  model: { type: "string", short: "m" },
  threshold: { type: "string", short: "t" },
  top: { type: "string", short: "k" },
  full: { type: "boolean", default: false },
  "with-context": { type: "boolean", default: false },
  json: { type: "boolean", default: false },
  help: { type: "boolean", short: "h" }
};

function printHelp() {
  console.log(`
Usage: slm-rerank --query <query> [options] [files...]

Options:
  -q, --query <string>     Search query (required)
  -e, --base-url <url>     Endpoint URL (default: http://localhost:8034/v1)
  -m, --model <string>     Target model name (default: lfm)
  -t, --threshold <float>  Relevance threshold [0.0, 1.0] (default: 0.65)
  -k, --top <int>          Maximum top candidates to return
      --full               Force full GPU evaluation (bypass Tier-1 filter)
      --with-context       Stitch 1-hop type and call context (<= 150 tokens)
      --json               Output raw JSON
  -h, --help               Show help
`);
}

async function runNative(query, files, parsed) {
  const baseUrl = parsed.values["base-url"] || process.env.SLM_ENDPOINT || "http://localhost:8034/v1";
  const model = parsed.values.model || "lfm";
  const threshold = parsed.values.threshold ? parseFloat(parsed.values.threshold) : 0.65;
  const withContext = parsed.values["with-context"];
  const full = parsed.values.full;

  const chunks = prepareCandidates(files);
  if (!chunks.length) {
    console.error("Error: No candidate chunks found in provided paths.");
    process.exit(1);
  }

  const reranker = new Reranker({ baseUrl, model, threshold });
  const result = await reranker.rerank(query, chunks, { withContext, full });

  if (parsed.values.json) {
    console.log(JSON.stringify(result, null, 2));
    return;
  }

  console.log(`\n🎯 SLM Semantic Reranker (Evaluated ${result.totalEvaluated} chunks)\n`);
  result.results.forEach((item, idx) => {
    const sym = item.chunk.symbol ? `(${item.chunk.symbol})` : "";
    console.log(` #${idx + 1} | Score: ${(item.score * 100).toFixed(1)}% | ${item.chunk.filePath}:${item.chunk.startLine}-${item.chunk.endLine} ${sym}`);
  });
  console.log("");
}

async function main() {
  let parsed;
  try {
    parsed = parseArgs({ options, allowPositionals: true });
  } catch (err) {
    console.error(err.message);
    printHelp();
    process.exit(1);
  }

  if (parsed.values.help) {
    printHelp();
    process.exit(0);
  }

  const query = parsed.values.query;
  if (!query) {
    console.error("Error: --query is required.");
    printHelp();
    process.exit(1);
  }

  const files = parsed.positionals;
  if (!files.length) {
    console.error("Error: At least one file or directory path is required.");
    process.exit(1);
  }

  // Check if python slm-rerank is installed in PATH
  let hasPythonSlm = false;
  try {
    const check = spawn("slm-rerank", ["--help"], { stdio: "ignore" });
    await new Promise((resolve) => {
      check.on("close", (code) => {
        hasPythonSlm = code === 0;
        resolve();
      });
      check.on("error", () => resolve());
    });
  } catch {
    hasPythonSlm = false;
  }

  if (hasPythonSlm) {
    // Delegate to Python CLI with streaming stdio
    const args = process.argv.slice(2);
    const child = spawn("slm-rerank", args, { stdio: "inherit" });
    child.on("close", (code) => process.exit(code || 0));
  } else {
    // Run pure Node.js implementation
    await runNative(query, files, parsed);
  }
}

main().catch((err) => {
  console.error("Fatal:", err);
  process.exit(1);
});
