// Reranker Client for Node.js: model-agnostic binary logprob evaluator with Ghost Stubs & Slice Boundaries
import { detectSlice, groupBySlice } from "./boundary.mjs";
import { generateGhostStub } from "./stubber.mjs";

// Length-normalized sigmoid calibration (mirrors slm_rerank.adapters.ModelProfile.calibrate_score).
const LENGTH_NORM_EXPONENT = 0.15;
const LENGTH_NORM_REFERENCE = 120.0;
const LENGTH_NORM_FLOOR = 40.0;
const SINGLE_SIDED_LOGIT = 3.0;
const NO_SIGNAL_SCORE = 0.05;
const DEFAULT_TOKEN_EST = 100;

const YES_TOKENS = new Set(["yes", "y", "true"]);
const NO_TOKENS = new Set(["no", "n", "false"]);

// Top-logprob containers, most specific first: llama.cpp /completion, OpenAI chat
// completions, OpenAI legacy completions (per-position token->logprob map), bare payload.
const LOGPROB_SOURCES = [
  payload => payload?.completion_probabilities?.[0]?.top_logprobs,
  payload => payload?.choices?.[0]?.logprobs?.content?.[0]?.top_logprobs,
  payload => payload?.choices?.[0]?.logprobs?.top_logprobs?.[0],
  payload => payload?.choices?.[0]?.logprobs?.top_logprobs,
  payload => payload?.top_logprobs
];

function normalizeEntries(raw) {
  if (!raw) return [];

  if (Array.isArray(raw)) {
    return raw
      .filter(item => item && typeof item === "object")
      .map(item => ({
        token: typeof item.token === "string" ? item.token : typeof item.tok_str === "string" ? item.tok_str : "",
        logprob:
          typeof item.logprob === "number"
            ? item.logprob
            : typeof item.prob === "number"
              ? Math.log(Math.max(item.prob, 1e-9))
              : null
      }))
      .filter(entry => entry.logprob !== null);
  }

  if (typeof raw === "object") {
    return Object.entries(raw)
      .filter(([, logprob]) => typeof logprob === "number")
      .map(([token, logprob]) => ({ token, logprob }));
  }

  return [];
}

function scanYesNo(entries) {
  let yesLp = null;
  let noLp = null;

  for (const entry of entries) {
    // Strip BPE/SentencePiece word-boundary markers before matching.
    const token = entry.token.toLowerCase().trim().replace(/^[Ġ▁_\s]+/, "");
    if (YES_TOKENS.has(token)) {
      if (yesLp === null || entry.logprob > yesLp) yesLp = entry.logprob;
    } else if (NO_TOKENS.has(token)) {
      if (noLp === null || entry.logprob > noLp) noLp = entry.logprob;
    }
  }

  return { yesLp, noLp };
}

export class Reranker {
  constructor(options = {}) {
    this.baseUrl = (options.baseUrl || process.env.SLM_ENDPOINT || "http://localhost:8034/v1").replace(/\/+$/, "");

    // llama.cpp serves native scoring on /completion at the server root, while the
    // OpenAI-compatible shim lives under /v1 — resolve both from whatever was supplied.
    if (this.baseUrl.endsWith("/v1")) {
      const root = this.baseUrl.slice(0, -3);
      this.completionUrl = `${root}/completion`;
      this.chatUrl = `${this.baseUrl}/chat/completions`;
    } else if (this.baseUrl.endsWith("/completion")) {
      this.completionUrl = this.baseUrl;
      this.chatUrl = `${this.baseUrl.replace(/\/completion$/, "")}/v1/chat/completions`;
    } else {
      this.completionUrl = `${this.baseUrl}/completion`;
      this.chatUrl = `${this.baseUrl}/v1/chat/completions`;
    }

    this.model = options.model || "lfm";
    this.concurrency = options.concurrency || 4;
    this.threshold = options.threshold ?? 0.65;
  }

  formatPrompt(query, chunk, stitchedContext = "") {
    const contextPrefix = stitchedContext ? `${stitchedContext}\n` : "";
    const fileInfo = chunk.filePath ? `File: ${chunk.filePath}\n` : "";
    let lineInfo = "";
    if (chunk.filePath) {
      lineInfo =
        chunk.startLine != null && chunk.endLine != null
          ? `Lines: ${chunk.startLine}-${chunk.endLine}\n`
          : "Lines: 1-1\n";
    }
    const symInfo = chunk.symbol ? `Symbol: ${chunk.symbol}\n` : "";

    return `<|startoftext|><|im_start|>system\nYou are a binary code retrieval evaluator. For the given search query and code snippet, evaluate if the snippet contains the relevant implementation, specification, definition, or answer requested.\nRespond with exactly "yes" if relevant, or "no" if not relevant.\n<|im_end|>\n<|im_start|>user\nQuery: ${query}\n\n${fileInfo}${lineInfo}${symInfo}Code:\n${contextPrefix}${chunk.content}\n\nDoes this snippet contain the relevant code for the query? Respond only with yes or no.<|im_end|>\n<|im_start|>assistant\n<think>\n</think>\n`;
  }

  calibrateScore(yesLp, noLp, chunk = {}) {
    let rawLogit;
    if (yesLp !== null && noLp === null) {
      rawLogit = SINGLE_SIDED_LOGIT;
    } else if (noLp !== null && yesLp === null) {
      rawLogit = -SINGLE_SIDED_LOGIT;
    } else if (yesLp !== null && noLp !== null) {
      rawLogit = yesLp - noLp;
    } else {
      return NO_SIGNAL_SCORE;
    }

    const tokenEst = chunk?.content ? Math.ceil(chunk.content.length / 4) : DEFAULT_TOKEN_EST;
    const effLength = Math.max(LENGTH_NORM_FLOOR, tokenEst);
    const lengthScale = Math.pow(LENGTH_NORM_REFERENCE / effLength, LENGTH_NORM_EXPONENT);

    return 1 / (1 + Math.exp(-(rawLogit * lengthScale)));
  }

  extractLogprobs(payload, chunk = {}) {
    try {
      for (const pick of LOGPROB_SOURCES) {
        const { yesLp, noLp } = scanYesNo(normalizeEntries(pick(payload)));
        if (yesLp !== null || noLp !== null) return this.calibrateScore(yesLp, noLp, chunk);
      }
      return NO_SIGNAL_SCORE;
    } catch {
      return NO_SIGNAL_SCORE;
    }
  }

  async scoreChunk(query, chunk, withContext = false) {
    let stitched = "";
    if (withContext) {
      const { stitchChunkContext } = await import("./stitcher.mjs");
      const ctx = stitchChunkContext(chunk);
      stitched = ctx.text;
    }

    const prompt = this.formatPrompt(query, chunk, stitched);

    let payload = null;
    let lastError = null;

    // Preferred path: llama.cpp native /completion returns completion_probabilities.
    try {
      const resp = await fetch(this.completionUrl, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt,
          n_predict: 1,
          n_probs: 10,
          temperature: 0.0,
          stop: ["<|im_end|>"]
        })
      });

      if (resp.ok) {
        payload = await resp.json();
      } else {
        lastError = `HTTP ${resp.status}`;
      }
    } catch (err) {
      lastError = err.message;
    }

    // Fallback: OpenAI-compatible chat completions with top_logprobs.
    if (payload === null) {
      try {
        const resp = await fetch(this.chatUrl, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            model: this.model,
            messages: [{ role: "user", content: prompt }],
            max_tokens: 1,
            temperature: 0.0,
            logprobs: true,
            top_logprobs: 10
          })
        });

        if (resp.ok) {
          payload = await resp.json();
        } else {
          lastError = `HTTP ${resp.status}`;
        }
      } catch (err) {
        lastError = err.message;
      }
    }

    if (payload === null) {
      return {
        chunk,
        score: 0.0,
        rawScore: 0.0,
        error: lastError || "no response",
        slice: detectSlice(chunk.filePath)
      };
    }

    const rawScore = this.extractLogprobs(payload, chunk);

    return {
      chunk,
      score: Math.round(rawScore * 10000) / 10000,
      rawScore: Math.round(rawScore * 10000) / 10000,
      slice: detectSlice(chunk.filePath)
    };
  }

  async rerank(query, chunks, options = {}) {
    const withContext = options.withContext ?? false;
    const full = options.full ?? false;
    const threshold = options.threshold ?? this.threshold;
    const stub = options.stub ?? false;
    const gitDiff = options.gitDiff ?? false;
    const dirtyOnly = options.dirtyOnly ?? false;

    // Apply Tier-1 hybrid pre-filter (with git-diff biasing support)
    const { applyTwoTierFilter } = await import("./filter.mjs");
    const { retained, tier1Applied, reason } = applyTwoTierFilter(chunks, query, { full, gitDiff, dirtyOnly });

    // Concurrently score chunks in slots
    const results = [];
    for (let i = 0; i < retained.length; i += this.concurrency) {
      const batch = retained.slice(i, i + this.concurrency);
      const batchResults = await Promise.all(
        batch.map(chunk => this.scoreChunk(query, chunk, withContext))
      );
      results.push(...batchResults);
    }

    results.sort((a, b) => b.score - a.score);

    const filtered = results.filter(r => r.score >= threshold);
    const finalResults = filtered.length ? filtered : results.slice(0, 3);

    // If stub / skeleton requested, attach Ghost Stub to top candidate results
    if (stub) {
      for (const item of finalResults) {
        const ghost = generateGhostStub(item.chunk.filePath, item.chunk);
        item.ghostStub = ghost.stub;
        item.foldedLines = ghost.foldedLines;
      }
    }

    const bySlice = groupBySlice(finalResults);

    return {
      query,
      results: finalResults,
      bySlice,
      totalEvaluated: retained.length,
      tier1Applied,
      filterReason: reason
    };
  }
}
