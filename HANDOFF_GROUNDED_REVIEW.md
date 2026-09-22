# Handoff — Grounded code review (v0.8.0)

> **Date:** 2026-09-22 · **Branch:** `feat/grounded-review` · **Repo:** [github.com/shawrylk/slm-rerank](https://github.com/shawrylk/slm-rerank)
> **Worktree:** `/home/shawry/Documents/GitHub/slm-rerank-grounded`
> **Base:** `main` at `b4c3e2d` (v0.7.0)

---

## 1. Why this exists

An ad-hoc review swarm built on this repository produced hallucinated findings. It called
`/v1/chat/completions`, which `HANDOFF.md` already marks "DO NOT USE for scoring" because the chat
template emits `thinking` as token 1. The model invented findings about code that was not there
("onSkip is called twice", "the close button has no accessible name").

slm-rerank itself did not hallucinate: it is a logprob scorer and its ranked chunks were accurate.
The gap was that nothing checked the generative output against the code. This branch adds that check.

**Goal:** a code review is trustworthy when every reported finding points at a line that exists on
disk and names only symbols the cited chunk contains.

---

## 2. What was built

| File | Change |
| --- | --- |
| `slm_rerank/review.py` | New. Retrieval + generation + deterministic verification + optional support judge. |
| `slm_rerank/review_bench.py` | New. Trust benchmark with the adversarial corpus. |
| `tests/test_review.py` | New. 20 tests. |
| `slm_rerank/__init__.py` | Exports the review API; version `0.8.0`. |
| `pyproject.toml` | Console scripts `slm-rerank-review`, `slm-rerank-review-bench`; version `0.8.0`. |
| `package.json` | Version `0.8.0`. |
| `README.md` | New "Grounded code review" section. |

### Pipeline

1. **Retrieve.** `LFMReranker.rerank` selects chunks; only `VERIFIED` citations are reviewed.
2. **Generate.** One call per chunk to the native `/completion` endpoint. The prompt asks for
   `FINDING: <claim> | EVIDENCE: <line copied verbatim>`.
3. **Evidence gate.** `verify_claim` reads the file at the citation's line range and requires the
   evidence to appear, matching across a collapsed multi-line quote.
4. **Identifier gate.** Every code identifier the claim names must appear in the chunk.
5. **Support gate.** The calibrated binary scorer answers whether the evidence supports the claim.
   Optional; `--no-support` disables it.

A candidate that fails steps 3-5 is returned in `report.dropped` with a stable reason, never
silently.

### Key functions

- `review(query, candidates, reranker=..., support_threshold=...)` and `review_sync(...)`.
- `verify_claim(claim, lines)` → `(ReviewFinding | None, DroppedClaim | None)`.
- `extract_claim_identifiers(text)` → identifiers a claim names.
- `parse_review_claims`, `format_review_prompt`, `read_citation_lines`, `render_review_report`.
- Reasons: `EMPTY_EVIDENCE`, `EVIDENCE_TOO_SHORT`, `EVIDENCE_NOT_FOUND`,
  `CLAIM_SYMBOL_NOT_IN_CHUNK`, `CLAIM_NOT_SUPPORTED`, `SUPPORT_UNVERIFIED`, `CHUNK_UNREADABLE`.

---

## 3. Verification (fresh evidence, 2026-09-22)

- `python3 -m pytest tests/` → **174 passed, 1 skipped**.
- `npm test` (Node) → **39 passed**.
- `python3 -m slm_rerank.review_bench` → deterministic benchmark **PASS**:
  - supported recalled `2/2`
  - fabricated rejected `6/6`
  - symbol-invented rejected `3/3`
  - reported precision `100.00%` grounded vs `18.18%` with no verifier
- Live run (`--live`) against LFM2.5 on `:8034`: claims generated, evidence and identifier gates
  applied, unsupported claims dropped. One run verified a true defect (stale-closure `onClick`);
  another dropped a false claim at the support gate.

---

## 4. Commands

```bash
# Tests
python3 -m pytest tests/ -q
npm test

# Trust benchmark (deterministic, no model)
python3 -m slm_rerank.review_bench
python3 -m slm_rerank.review_bench --json

# Trust benchmark (live)
python3 -m slm_rerank.review_bench --live --top-k 1

# Review files
python3 -m slm_rerank.review "<path>" --query "find real defects" --top 5
python3 -m slm_rerank.review "<path>" --no-support --json
```

The model server must be running on `:8034` for the live paths (`llama-server ... --port 8034`).

---

## 5. Design decisions

- **Native `/completion` only for generation.** `HANDOFF.md` documents why `/v1/chat/completions`
  is unusable here. `review.py` reuses `expander._completion_url` and `_extract_text`.
- **Evidence comes from disk, not the chunk string.** The chunk renders lines as `N: code`; a model
  quotes code without the prefix. Verification reads the citation range from the file, which also
  matches the repository's physical ground-truth philosophy.
- **Failure means fewer findings, never a raise.** Matches `expander.py`.
- **The support judge is honest about its limits.** It is a probabilistic filter, and the benchmark
  measures the deterministic gates separately. A false claim scored `0.77` on the judge once; it is
  not a guarantee. Do not present it as one.
- **`support_threshold=None` disables the judge** so offline tests and callers without a prompt
  profile still work.

---

## 6. Known limitations and next steps

1. **The support judge is the weak link.** An 8B model accepts some misinterpretations of real
   lines. Options: a stricter judge prompt, an ensemble of judges, or a per-claim-type rule set.
2. **Live recall is model-bounded.** A weak model may quote no real lines; the pipeline then reports
   nothing. That is correct behaviour, not a bug, but it limits usefulness until generation quality
   improves.
3. **No Node port.** The review layer is Python-only. `src/` has no equivalent.
4. **No MCP tool.** `mcp.mjs` / `mcp_server.py` expose `rerank_codebase`; a `review_codebase` tool
   would let Claude Code call this directly.
5. **The `python -m slm_rerank.review` warning** ("found in sys.modules") is cosmetic; the console
   script entry point does not emit it.
6. **Not merged.** Open a PR against `main`; the branch is not pushed.

---

## 7. First action for the next agent

Run `python3 -m pytest tests/` and `python3 -m slm_rerank.review_bench`. Both must pass before any
change. Then decide whether to attack the support judge (limitation 1) or port the layer to Node
(limitation 3).