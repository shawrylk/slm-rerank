// Whole-token term matching for discovery, Tier-1 and fusion: a substring test finds "share" in "shared/".

const WORD_RE = /[A-Za-z0-9]+/g;
// A nested const is a local, so only a top-level binding names the chunk. "N: " is the chunker's line prefix.
const UNIT_RE = /^(?:\d+: )?[ \t]*(?:(?:export|default|declare|abstract|async|public|private|protected|static|pub)\s+)*(?:function\*?|class|interface|type|enum|def|fn|func|struct|trait)\s+([A-Za-z_$][\w$]*)/gm;
const BINDING_RE = /^(?:\d+: )?(?:export\s+)?(?:const|let|var)\s+([A-Za-z_$][\w$]*)/gm;

/** "openIndexedDB" -> ["open", "indexed", "db"]. */
function splitWord(word) {
  return word
    .replace(/([a-z0-9])([A-Z])/g, "$1 $2")
    .replace(/([A-Z]+)([A-Z][a-z])/g, "$1 $2")
    .toLowerCase()
    .split(" ")
    .filter(Boolean);
}

/**
 * Reduce a code token by number and gerund only: shares -> share, sharing -> share.
 * An -ed token in code names a state (shared/, createdAt), not the action, so it keeps its form.
 */
export function tokenKey(token) {
  let w = token;
  if (w.length > 4 && w.endsWith("ies")) w = `${w.slice(0, -3)}y`;
  else if (w.length > 4 && /(?:s|x|z|ch|sh)es$/.test(w)) w = w.slice(0, -2);
  else if (w.length > 3 && w.endsWith("s") && !w.endsWith("ss")) w = w.slice(0, -1);
  if (w.length > 5 && w.endsWith("ing")) w = w.slice(0, -3);
  if (w.length > 3 && w.endsWith("e")) w = w.slice(0, -1);
  return w;
}

/** A query word -> every token key that counts as that word. "created" -> ["created", "creat"]. */
export function termKeys(word) {
  const w = String(word || "").toLowerCase();
  const keys = [tokenKey(w)];
  // The query names the action; code names it in the base form (created -> create, migration -> migrate).
  const base = w.length > 4 && w.endsWith("ed") ? w.slice(0, -2)
    : w.length > 6 && w.endsWith("ion") ? w.slice(0, -3)
      : null;
  if (base && !keys.includes(tokenKey(base))) keys.push(tokenKey(base));
  return keys;
}

// Tier-1 splits every word of every candidate; identifiers repeat, so each word is split once.
const wordKeyCache = new Map();
const WORD_KEY_CACHE_LIMIT = 100_000;

/** A word's token keys, plus each adjacent pair joined (ToolBar -> toolbar). */
function wordKeys(word) {
  let keys = wordKeyCache.get(word);
  if (keys) return keys;
  const parts = splitWord(word);
  keys = [];
  for (let i = 0; i < parts.length; i += 1) {
    keys.push(tokenKey(parts[i]));
    if (i > 0) keys.push(tokenKey(parts[i - 1] + parts[i]));
  }
  if (wordKeyCache.size >= WORD_KEY_CACHE_LIMIT) wordKeyCache.clear();
  wordKeyCache.set(word, keys);
  return keys;
}

function eachKey(text, visit) {
  for (const word of String(text || "").match(WORD_RE) || []) {
    for (const key of wordKeys(word)) visit(key);
  }
}

/** Path, identifier or code -> the set of its token keys. */
export function tokenKeys(text) {
  const keys = new Set();
  eachKey(text, key => keys.add(key));
  return keys;
}

/** Code -> how often each wanted token key occurs. Other keys are not counted. */
export function tokenCounts(text, wanted) {
  const counts = new Map();
  eachKey(text, key => { if (wanted.has(key)) counts.set(key, (counts.get(key) || 0) + 1); });
  return counts;
}

export function containsTerm(keySet, keys) {
  return keys.some(key => keySet.has(key));
}

export function countTerm(counts, keys) {
  return keys.reduce((sum, key) => sum + (counts.get(key) || 0), 0);
}

/** Names a chunk declares: functions, classes and types, and top-level bindings. */
export function declaredNames(content) {
  const text = String(content || "");
  return [...text.matchAll(UNIT_RE), ...text.matchAll(BINDING_RE)].map(m => m[1]);
}
