// trigger: the Claude Code UserPromptSubmit hook. It adds file:line citations to a prompt about code when a model
// server answers. On every other path it writes nothing and exits 0, so the prompt goes through unchanged.
// A local shim imports runPromptHook from the pinned package, so this file is the one home of the hook logic.
import process from "node:process";
import { Reranker } from "../client.mjs";
import { prepareCandidatesCached } from "../chunk-cache.mjs";
import { autoDiscoverEndpoint, discoverCandidateFiles, resolveEndpointEnv } from "../discovery.mjs";
import { applyTwoTierFilter } from "../filter.mjs";
import { fuseLexicalPrior } from "../fusion.mjs";
import { appendUsage } from "../usage/log.mjs";
import { isCodeQuestion, isMachineText } from "./code-question.mjs";

const THRESHOLD = 0.65;
const TOP_K = 3;
const MAX_FILES = 30;
// The lexical order puts the target of a code question in its first few chunks, so a prompt
// hook scores only the head. A full search is the slm-rerank command.
const MAX_CHUNKS = 8;
const PROBE_MS = 300;
// The local LFM2.5-8B-A1B scores a batch of four chunks in about 1.3 s, and it needs about
// 2.6 s to wake from idle sleep. The budget fits two batches, or one batch after a wake.
// DEADLINE_MS is the hard exit, below the registration timeout of 10 s.
const BUDGET_MS = 4500;
const DEADLINE_MS = 5000;

async function answers(baseUrl) {
  try {
    const res = await fetch(`${baseUrl.replace(/\/+$/, "")}/models`, { signal: AbortSignal.timeout(PROBE_MS) });
    return res.ok;
  } catch {
    return false;
  }
}

// The package trusts a pinned SLM_ENDPOINT without a probe, and its requests carry no timeout.
// Probe here, so a dead server costs one probe and not one connect timeout per chunk.
export async function resolveEndpoint() {
  const pinned = resolveEndpointEnv();
  if (pinned) return (await answers(pinned)) ? pinned : null;
  return (await autoDiscoverEndpoint({ timeoutMs: PROBE_MS })).url;
}

// Reranker.rerank() waits for every head chunk, which a prompt hook cannot do inside its budget.
// This loop scores the head of the lexical order and keeps each chunk that finishes before the
// budget ends, then blends its lexical evidence into its model score, as rerank() does. The blend
// judges a chunk against every candidate, not only the scored head, where every chunk matches well.
// Only a chunk whose blended score reaches the threshold is a citation.
async function scoreWithinBudget(reranker, query, chunks, lexicalScores, startedAt) {
  const scored = [];
  const lexical = [];
  let passing = 0;
  const record = index => r => {
    scored.push(r);
    lexical.push(lexicalScores[index]);
    if (!r.error && r.score >= THRESHOLD) passing += 1;
  };
  for (let i = 0; i < chunks.length && passing < TOP_K; i += reranker.concurrency) {
    const left = startedAt + BUDGET_MS - Date.now();
    if (left <= 0) break;
    const batch = chunks.slice(i, i + reranker.concurrency).map((c, j) => reranker.scoreChunk(query, c).then(record(i + j)));
    const finished = await Promise.race([
      Promise.all(batch).then(() => true),
      new Promise(resolve => setTimeout(resolve, left, false))
    ]);
    if (!finished) break;
  }
  return fuseLexicalPrior(scored, lexical, lexicalScores)
    .filter(r => !r.error && r.score >= THRESHOLD)
    .sort((a, b) => b.score - a.score)
    .slice(0, TOP_K);
}

/** Fills `decision` with the outcome and the cited paths; returns the context text, or null. */
async function citations(prompt, cwd, resolve, decision, startedAt) {
  // Probe before the file search, not during it. The search is synchronous, and a blocked event
  // loop fires the probe's abort timer before it reads the answer.
  const baseUrl = await resolve();
  if (!baseUrl) {
    decision.outcome = "no-server";
    return null;
  }

  // Discovered paths are relative to cwd, and the chunk reader resolves them against the process directory.
  process.chdir(cwd);
  const files = discoverCandidateFiles(prompt, { cwd, limit: MAX_FILES });
  const chunks = files.length ? prepareCandidatesCached(files) : [];
  if (!chunks.length) {
    decision.outcome = "no-candidates";
    return null;
  }

  const { retained, lexicalScores } = applyTwoTierFilter(chunks, prompt, { cwd });
  const reranker = new Reranker({ baseUrl, threshold: THRESHOLD });
  const hits = await scoreWithinBudget(reranker, prompt, retained.slice(0, MAX_CHUNKS), lexicalScores, startedAt);
  decision.top = hits.map(({ chunk }) => `${chunk.filePath}:${chunk.startLine}-${chunk.endLine}`);
  decision.outcome = hits.length ? "cited" : "no-hits";
  if (!hits.length) return null;

  return [
    "[SLM Wide Reranker] Code citations for this prompt, ranked by a local model:",
    ...hits.map(({ chunk, score, rawScore }) =>
      `- ${chunk.filePath}:${chunk.startLine}-${chunk.endLine} (${chunk.symbol || "module scope"}), ` +
      `confidence ${(score * 100).toFixed(1)}% (model ${(rawScore * 100).toFixed(1)}%)`)
  ].join("\n");
}

/**
 * One prompt -> the context to add, or null. Logs one usage line per decision.
 * `decision` is shared with the caller, so a deadline exit can still log what was decided.
 */
export async function handlePrompt(input, {
  resolve = resolveEndpoint,
  log = appendUsage,
  startedAt = Date.now(),
  decision = {}
} = {}) {
  const prompt = typeof input?.prompt === "string" ? input.prompt : "";
  const code = isCodeQuestion(prompt);
  Object.assign(decision, {
    caller: "hook",
    query: prompt,
    isCodeQuestion: code,
    queried: code,
    outcome: code ? "deadline" : isMachineText(prompt) ? "machine-text" : "not-code",
    top: []
  });
  const context = code ? await citations(prompt, input.cwd || process.cwd(), resolve, decision, startedAt) : null;
  log({ ...decision, latencyMs: Date.now() - startedAt });
  decision.logged = true;
  return context;
}

async function readStdin(stdin) {
  let data = "";
  for await (const chunk of stdin) data += chunk;
  return data;
}

/** The hook process: stdin JSON in, the hook output JSON out, exit 0 on every path. */
export async function runPromptHook({ stdin = process.stdin, stdout = process.stdout } = {}) {
  const startedAt = Date.now();
  const decision = {};
  setTimeout(() => {
    if (decision.queried && !decision.logged) appendUsage({ ...decision, latencyMs: Date.now() - startedAt });
    process.exit(0);
  }, DEADLINE_MS);
  try {
    const input = JSON.parse(await readStdin(stdin));
    const context = await handlePrompt(input, { startedAt, decision });
    if (!context) process.exit(0);
    const output = { hookSpecificOutput: { hookEventName: "UserPromptSubmit", additionalContext: context } };
    stdout.write(JSON.stringify(output), () => process.exit(0));
  } catch {
    process.exit(0);
  }
}
