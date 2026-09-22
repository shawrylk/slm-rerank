// Reranker Client for Node.js: model-agnostic binary logprob evaluator

export class Reranker {
  constructor(options = {}) {
    this.baseUrl = options.baseUrl || process.env.SLM_ENDPOINT || "http://localhost:8034/v1";
    this.model = options.model || "lfm";
    this.concurrency = options.concurrency || 4;
    this.threshold = options.threshold ?? 0.65;
  }

  formatPrompt(query, chunk, stitchedContext = "") {
    const contextPrefix = stitchedContext ? `${stitchedContext}\n` : "";
    return `<|im_start|>user\nYou are an expert code search evaluator. Is the following code candidate relevant to the search query?\n\nQuery: ${query}\n\nFile: ${chunk.filePath || "unknown"}\nSymbol: ${chunk.symbol || "unknown"}\nLines: ${chunk.startLine}-${chunk.endLine}\n\nCandidate Code:\n${contextPrefix}${chunk.content}\n\nRespond with only 'Yes' or 'No'.<|im_end|>\n<|im_start|>assistant\n<think>\n</think>\n`;
  }

  extractLogprobs(payload) {
    try {
      const choice = payload.choices?.[0];
      const logprobsData = choice?.logprobs?.content?.[0] || choice?.logprobs?.top_logprobs?.[0];
      const topLogprobs = logprobsData?.top_logprobs || [];

      let yesLp = null;
      let noLp = null;

      for (const item of topLogprobs) {
        const token = (item.token || "").toLowerCase().trim().replace(/^[Ġ_]/, "");
        if (token === "yes" || token === "true" || token === "y") {
          if (yesLp === null || item.logprob > yesLp) yesLp = item.logprob;
        } else if (token === "no" || token === "false" || token === "n") {
          if (noLp === null || item.logprob > noLp) noLp = item.logprob;
        }
      }

      if (yesLp === null && noLp === null) {
        return 0.5;
      }
      if (yesLp !== null && noLp === null) return 0.95;
      if (yesLp === null && noLp !== null) return 0.05;

      const pYes = Math.exp(yesLp);
      const pNo = Math.exp(noLp);
      return pYes / (pYes + pNo);
    } catch {
      return 0.5;
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

    try {
      const resp = await fetch(`${this.baseUrl}/chat/completions`, {
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

      if (!resp.ok) {
        return { chunk, score: 0.0, rawScore: 0.0, error: `HTTP ${resp.status}` };
      }

      const data = await resp.json();
      const rawScore = this.extractLogprobs(data);

      return {
        chunk,
        score: Math.round(rawScore * 10000) / 10000,
        rawScore: Math.round(rawScore * 10000) / 10000
      };
    } catch (err) {
      return { chunk, score: 0.0, rawScore: 0.0, error: err.message };
    }
  }

  async rerank(query, chunks, options = {}) {
    const withContext = options.withContext ?? false;
    const full = options.full ?? false;
    const threshold = options.threshold ?? this.threshold;

    // Apply Tier-1 hybrid pre-filter
    const { applyTwoTierFilter } = await import("./filter.mjs");
    const { retained, tier1Applied, reason } = applyTwoTierFilter(chunks, query, full);

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
    return {
      query,
      results: filtered.length ? filtered : results.slice(0, 3),
      totalEvaluated: retained.length,
      tier1Applied,
      filterReason: reason
    };
  }
}
