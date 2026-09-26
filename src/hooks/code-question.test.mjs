import test from "node:test";
import assert from "node:assert/strict";
import { isCodeQuestion, isMachineText } from "./code-question.mjs";

// Real-shaped prompts from the quality-control-mono sessions. The two issue examples are verbatim.
const CODE = [
  "where does the router set app.tenant_id",
  "why does `resolveShare` return 404 when the grant is expired?",
  "fix TypeError: Cannot read properties of undefined (reading 'tenantId') in photo upload",
  "what does backend/src/infrastructure/db/tenant-session.ts do when a client is checked out",
  "where is the storage key prefix for a project thumbnail built",
  "how is the keyset pagination cursor encoded",
  "```ts\nconst rows = await listAuditLog(tx, principal)\n```\nwhy this query is slow",
  "rename coarsenGps to coarsenPoint in all callers",
  "where is the session_lifetimes registry read",
  "which function computes the photo chain hash",
  "how does discoverCandidateFiles rank a path hit over a body hit?",
  "the api boots with ECONNREFUSED 127.0.0.1:5432, where is the pool created",
  "record.test.ts fails with AssertionError after my change, can you look",
  "where is photo GPS coarsened before storage",
  "explain the retry in apiClient.request()",
  'every request fails with "code": "upstream_unavailable"'
];

const PROCESS = [
  "how should we improve your efficiency?",
  "what is the status of the S1 work order?",
  "commit and push, then open the PR",
  "ok",
  "plan the next sprint and split it into three work orders",
  "can you check whether CI is green on #394",
  "merge it when the checks pass",
  "<task-notification>\n<task-id>b6dkuwvgy</task-id>\n<status>completed</status>\n</task-notification>",
  "/code-review 394",
  "which agent is still running, and how long until it finishes?",
  "how do we set up the lfm server on a new machine?",
  "write a handoff note to the scratchpad and stop",
  "update the issue with the measurements and close it",
  "what do you think about moving the docs to Docusaurus?",
  "how is the migration of the dev database going?",
  "read the brief at plans/s1.md and follow it exactly",
  "why did the hook go quiet yesterday? check lfm status",
  "where are we on the pagination work?",
  "push the fix for src/hooks/claude-prompt-context.mjs",
  "set SLM_HOST to the LAN box and restart lfm",
  "upgrade Node.js to 24 and rerun it"
];

test("isCodeQuestion: the fixture holds at least 20 prompts of both kinds", () => {
  assert.ok(CODE.length + PROCESS.length >= 20);
  assert.ok(CODE.length >= 10 && PROCESS.length >= 10);
});

test("isCodeQuestion: no process prompt is a code question (zero false positives)", () => {
  const falsePositives = PROCESS.filter(isCodeQuestion);
  assert.deepEqual(falsePositives, []);
});

test("isCodeQuestion: every code prompt in the fixture is a code question", () => {
  const falseNegatives = CODE.filter(p => !isCodeQuestion(p));
  assert.deepEqual(falseNegatives, []);
});

test("isCodeQuestion: a brand name in PascalCase or a docs path is no code signal", () => {
  assert.equal(isCodeQuestion("is GitHub or PowerShell faster on this machine?"), false);
  assert.equal(isCodeQuestion("summarize docs/architecture.md for me"), false);
});

test("isMachineText: task notifications and slash commands are machine text", () => {
  assert.equal(isMachineText("<task-notification>x</task-notification>"), true);
  assert.equal(isMachineText("  /loop 5m"), true);
  assert.equal(isMachineText("where is the router"), false);
});
