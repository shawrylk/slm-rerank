// branch: does a prompt ask about code? Deterministic and cheap, with no model call.
// It errs toward false: a missed prompt costs one context, and a wrong one costs its tokens on every later turn.

const SOURCE_EXTENSIONS = "py|ts|tsx|js|jsx|mjs|cjs|rs|go|c|cpp|h|java|kt|cs|rb|php|swift|sql|sh|ps1";

// Signals that decide alone: only code has them.
const STRONG = [
  /```/,
  new RegExp(`(?:^|[\\s\`'"(])[\\w./\\\\-]*\\.(?:${SOURCE_EXTENSIONS})\\b`, "i"),
  /\b[a-z][a-z0-9]+[A-Z][A-Za-z0-9]*\b/,
  /\b[a-z0-9]+_[a-z0-9_]+\b/i,
  /\b[A-Za-z_$][\w$]*\(\)/,
  /\b[A-Za-z_$][\w$]*\.[A-Za-z_$][\w$]*\(/,
  /\b[A-Z][a-z]+(?:Error|Exception)\b/,
  /\bE(?:CONNREFUSED|NOENT|ACCES|ADDRINUSE|TIMEDOUT|PERM)\b/,
  /\b(?:stack trace|traceback|segfault|is not a function|cannot read propert(?:y|ies)|undefined is not)\b/i
];

// Brand names that look like camelCase.
const NOT_CODE_WORDS = /\b(?:macOS|iOS|iPadOS|npm|pnpm|eBay|iPhone|iPad)\b/g;

// A prompt that starts with these is an instruction about the workflow, whatever it names.
const PROCESS_LEAD = /^(?:ok|okay|yes|no|thanks|thank you|cool|great|got it|sure|yep|nope)[.!?, ]*$|^(?:commit|push|merge|rebase|cherry-pick|stash|pull|checkout|deploy|release|publish|tag|bump|close|stop|continue|proceed|wait|go ahead|open (?:a |the )?pr)\b/;

// Words of the workflow in quality-control-mono. They veto the weak signals below, not the strong ones above.
const PROCESS_NOUNS = /\b(?:prs?|pull requests?|issues?|ci|branch(?:es)?|worktrees?|commits?|merges?|agents?|work orders?|briefs?|plans?|handoffs?|reports?|status|sprints?|reviews?|hooks?|gates?|releases?|deploys?|lfm|docs?|efficiency)\b/;
const ASKS_US = /\bhow (?:do|should|can|could|would) (?:we|you|i)\b/;

const CODE_VERBS = [
  "define", "declare", "implement", "handle", "compute", "calculate", "encode", "decode", "parse", "serialize",
  "validate", "sign", "hash", "store", "persist", "load", "call", "invoke", "register", "render", "generate",
  "initialize", "configure", "wire", "map", "resolve", "route", "dispatch", "cache", "reduce", "filter", "sort",
  "rank", "score", "chunk", "emit", "raise", "import", "export", "mount", "fetch", "insert", "delete", "upload",
  "redirect", "authenticate", "authorize", "verify", "sanitize", "escape", "convert", "transform", "coarsen"
];
const CODE_VERB = new RegExp(
  `\\b(?:(?:${CODE_VERBS.join("|")})(?:s|es|ed|d|ing)?|set(?! up)|read|written|wrote|built|thrown|caught|sent)\\b`
);

const LOCATOR = /^(?:so |and |but )?(?:where|in which file|which (?:file|function|method|class|module|component|type|interface|line|query|endpoint|route|handler|table|column)s?)\b/;
const LOCATOR_UNIT = /^(?:so |and |but )?(?:in which file|which (?:file|function|method|class|module|component|type|interface|line|query|endpoint|route|handler|table|column)s?)\b/;
const PARTICIPLE = /\b[a-z]{4,}ed\b/;
const MECHANISM = /\bhow (?:is|are|does|do)\b/;
const DEBUG = /\b(?:fix|debug|diagnose)\b.*\b(?:bug|error|crash|exception|regression|leak|null|undefined|nan)\b/;

export function isMachineText(prompt) {
  return /^\s*[</]/.test(String(prompt || ""));
}

export function isCodeQuestion(prompt) {
  const text = String(prompt || "").trim();
  if (text.length < 8 || isMachineText(text)) return false;
  const lower = text.toLowerCase();
  if (PROCESS_LEAD.test(lower)) return false;

  if (STRONG.some(re => re.test(text.replace(NOT_CODE_WORDS, "")))) return true;

  if (PROCESS_NOUNS.test(lower) || ASKS_US.test(lower)) return false;
  if (LOCATOR_UNIT.test(lower)) return true;
  if (LOCATOR.test(lower) && (CODE_VERB.test(lower) || PARTICIPLE.test(lower))) return true;
  if (MECHANISM.test(lower) && CODE_VERB.test(lower)) return true;
  return DEBUG.test(lower);
}
