// ledger: one JSON line per query, appended to usage.jsonl in the cache folder. A failed write never fails the query.
import fs from "node:fs";
import path from "node:path";
import { resolveCacheDir } from "../cache-dir.mjs";

export const USAGE_FILE = "usage.jsonl";
// A pasted log or a work order can be many KB, and its head says what was asked.
export const MAX_LOGGED_QUERY = 500;

export function usageLogPath(dir = resolveCacheDir()) {
  return path.join(dir, USAGE_FILE);
}

// Synchronous, because the hook leaves through process.exit, which drops a pending write.
export function appendUsage(entry, { dir = resolveCacheDir(), now = new Date() } = {}) {
  try {
    const line = JSON.stringify({ time: now.toISOString(), ...entry, query: String(entry.query ?? "").slice(0, MAX_LOGGED_QUERY) });
    fs.mkdirSync(dir, { recursive: true });
    fs.appendFileSync(usageLogPath(dir), `${line}\n`);
    return true;
  } catch {
    return false;
  }
}

export function readUsage({ dir = resolveCacheDir() } = {}) {
  let text;
  try {
    text = fs.readFileSync(usageLogPath(dir), "utf-8");
  } catch {
    return [];
  }
  const entries = [];
  for (const line of text.split("\n")) {
    if (!line.trim()) continue;
    try {
      entries.push(JSON.parse(line));
    } catch {
      // A process ended mid-write.
    }
  }
  return entries;
}
