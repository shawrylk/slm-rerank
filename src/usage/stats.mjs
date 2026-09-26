// pipeline: usage lines (log.mjs) -> the numbers that `slm-rerank stats` prints.

function median(values) {
  const sorted = values.filter(Number.isFinite).sort((a, b) => a - b);
  if (!sorted.length) return null;
  const mid = sorted.length >> 1;
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

export function summarizeUsage(entries) {
  // A hook line with queried: false is a prompt the hook let through, not a query.
  const queries = entries.filter(e => e.queried !== false);
  const byCaller = new Map();
  for (const e of queries) {
    const caller = e.caller || "unknown";
    if (!byCaller.has(caller)) byCaller.set(caller, []);
    byCaller.get(caller).push(e.latencyMs);
  }
  const latency = Object.fromEntries([...byCaller].map(([caller, ms]) => [caller, { queries: ms.length, medianMs: median(ms) }]));

  // Task notifications and slash commands are machine text, not a prompt a person asked.
  const prompts = entries.filter(e => e.caller === "hook" && e.outcome !== "machine-text");
  const hookQueried = prompts.filter(e => e.queried).length;
  return {
    queryCount: queries.length,
    latency,
    hookPrompts: prompts.length,
    hookQueried,
    hookShare: prompts.length ? hookQueried / prompts.length : null
  };
}

const seconds = ms => (ms == null ? "n/a" : `${(ms / 1000).toFixed(1)} s`);

export function formatStats(summary, logPath) {
  const lines = [`slm-rerank usage (${logPath})`, `Queries: ${summary.queryCount}`];
  const callers = Object.entries(summary.latency);
  if (callers.length) {
    lines.push("Median latency by caller:");
    for (const [caller, { queries, medianMs }] of callers) {
      lines.push(`  ${caller.padEnd(6)} ${String(queries).padStart(5)} queries  median ${seconds(medianMs)}`);
    }
  }
  const share = summary.hookShare == null ? "n/a" : `${(summary.hookShare * 100).toFixed(1)}%`;
  lines.push(`Hook prompts that queried: ${summary.hookQueried} of ${summary.hookPrompts} (${share})`);
  return lines.join("\n");
}
