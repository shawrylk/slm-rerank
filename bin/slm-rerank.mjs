#!/usr/bin/env node

import { spawn } from "node:child_process";
import fs from "node:fs";
import { parseArgs } from "node:util";
import { Reranker } from "../src/client.mjs";
import { prepareCandidates } from "../src/chunker.mjs";
import { autoDiscoverEndpoint, discoverCandidateFiles, resolveHostEnv } from "../src/discovery.mjs";
import { groupBySlice } from "../src/boundary.mjs";
import { startMcpServer } from "../src/mcp.mjs";

const options = {
  query: { type: "string", short: "q" },
  "base-url": { type: "string", short: "e" },
  host: { type: "string" },
  model: { type: "string", short: "m" },
  threshold: { type: "string", short: "t" },
  top: { type: "string", short: "k" },
  full: { type: "boolean", default: false },
  "with-context": { type: "boolean", default: false },
  stub: { type: "boolean", default: false },
  slice: { type: "boolean", default: false },
  dirty: { type: "boolean", default: false },
  "git-diff": { type: "boolean", default: false },
  "by-slice": { type: "boolean", default: false },
  json: { type: "boolean", default: false },
  mcp: { type: "boolean", default: false },
  help: { type: "boolean", short: "h" }
};

function printHelp() {
  console.log(`
Usage: slm-rerank --query <query> [options] [files...]

Options:
  -q, --query <string>     Search query (required)
  -e, --base-url <url>     Endpoint URL (probes ports 8033-8040 if omitted)
      --host <string>      Target server host (default: 127.0.0.1 or SLM_HOST)
  -m, --model <string>     Target model name/profile (e.g. lfm, qwen, gemma)
  -t, --threshold <float>  Relevance threshold [0.0, 1.0] (default: 0.65)
  -k, --top <int>          Maximum top candidates to return
      --stub, --slice      Generate AST Ghost Stubs for top results (token-saver)
      --dirty              Bias or filter by git uncommitted/modified files
      --git-diff           Boost candidates recently touched in git history
      --by-slice           Group results by architectural vertical slice
      --full               Force full GPU evaluation (bypass Tier-1 filter)
      --with-context       Stitch 1-hop type and call context (<= 150 tokens)
      --json               Output raw JSON
      --mcp                Run as an MCP (Model Context Protocol) stdio server
  -h, --help               Show help

Note: If no files or globs are passed, slm-rerank automatically runs smart
ripgrep/git auto-discovery to locate the top candidate files across the repo.
`);
}

async function runNative(query, files, parsed) {
  const host = parsed.values.host || resolveHostEnv();
  const requestedModel = parsed.values.model || null;
  const withStub = parsed.values.stub || parsed.values.slice || false;
  const withContext = parsed.values["with-context"];
  const full = parsed.values.full;
  const dirtyOnly = parsed.values.dirty;
  const gitDiff = parsed.values["git-diff"] || parsed.values.dirty;
  const showBySlice = parsed.values["by-slice"];
  const topK = parsed.values.top ? parseInt(parsed.values.top, 10) : undefined;
  const threshold = parsed.values.threshold ? parseFloat(parsed.values.threshold) : 0.65;

  let baseUrl = parsed.values["base-url"];
  let detectedPort = null;
  let detectedModel = null;

  if (!baseUrl) {
    const discovered = await autoDiscoverEndpoint({ host, requestedModel });
    if (!discovered.url) {
      console.error(`Error: ${discovered.reason}`);
      process.exit(1);
    }
    baseUrl = discovered.url;
    detectedPort = discovered.port;
    detectedModel = discovered.modelId;
  }

  const chunks = prepareCandidates(files);
  if (!chunks.length) {
    console.error("Error: No candidate code chunks found in target paths.");
    process.exit(1);
  }

  const reranker = new Reranker({
    baseUrl,
    model: requestedModel || "lfm",
    threshold
  });

  const result = await reranker.rerank(query, chunks, {
    withContext,
    full,
    stub: withStub,
    gitDiff,
    dirtyOnly
  });

  if (topK && result.results.length > topK) {
    result.results = result.results.slice(0, topK);
  }

  if (parsed.values.json) {
    console.log(JSON.stringify(result, null, 2));
    return;
  }

  const endpointInfo = detectedPort ? `port :${detectedPort} (${detectedModel})` : baseUrl;
  console.log(`\n🎯 SLM Semantic Reranker via ${endpointInfo}`);
  console.log(`Evaluated ${result.totalEvaluated} chunks | Query: "${query}"\n`);

  if (showBySlice) {
    const groups = groupBySlice(result.results);
    for (const [sliceName, group] of Object.entries(groups)) {
      console.log(`📦 [Slice: ${sliceName}] (${group.items.length} matches, max score: ${(group.maxScore * 100).toFixed(1)}%)`);
      group.items.forEach((item, idx) => {
        const sym = item.chunk.symbol ? `(${item.chunk.symbol})` : "";
        console.log(`    #${idx + 1} | ${(item.score * 100).toFixed(1)}% | ${item.chunk.filePath}:${item.chunk.startLine}-${item.chunk.endLine} ${sym}`);
      });
      console.log("");
    }
  } else {
    result.results.forEach((item, idx) => {
      const sym = item.chunk.symbol ? `(${item.chunk.symbol})` : "";
      const sliceTag = item.slice ? `[${item.slice}] ` : "";
      console.log(` #${idx + 1} | Score: ${(item.score * 100).toFixed(1)}% | ${sliceTag}${item.chunk.filePath}:${item.chunk.startLine}-${item.chunk.endLine} ${sym}`);

      if (withStub && item.ghostStub) {
        console.log(`\n--- 👻 Ghost Stub (${item.foldedLines} lines folded) ---`);
        console.log(item.ghostStub.slice(0, 800) + (item.ghostStub.length > 800 ? "\n..." : ""));
        console.log("--------------------------------------------------\n");
      }
    });
    console.log("");
  }
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

  // MCP stdio mode: stdout is reserved for JSON-RPC framing, so never log there.
  if (parsed.values.mcp) {
    startMcpServer({ host: parsed.values.host });
    return;
  }

  const query = parsed.values.query;
  if (!query) {
    console.error("Error: --query is required.");
    printHelp();
    process.exit(1);
  }

  let files = parsed.positionals;
  if (!files.length) {
    // Feature: Smart auto-discovery via ripgrep/git
    files = discoverCandidateFiles(query);
    if (!files.length) {
      console.error("Error: No relevant candidate files found automatically. Please specify file paths.");
      process.exit(1);
    }
    if (!parsed.values.json) {
      console.log(`🔍 Auto-discovered ${files.length} candidate files via ripgrep...`);
    }
  }

  // If explicitly requested via SLM_ENGINE=python, delegate to python module
  if (process.env.SLM_ENGINE === "python") {
    const args = ["-m", "slm_rerank.cli", ...process.argv.slice(2)];
    const child = spawn("python3", args, { stdio: "inherit" });
    child.on("close", (code) => process.exit(code || 0));
    return;
  }

  // Pure Node.js implementation (lightweight, zero-dep, LAN/remote capable)
  await runNative(query, files, parsed);
}

main().catch((err) => {
  console.error("Fatal:", err);
  process.exit(1);
});
