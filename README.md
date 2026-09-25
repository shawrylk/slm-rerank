# Model-Agnostic Wide Reranker (`slm-rerank`)

The **Wide Reranker** is an ultra-high-throughput, prefill-dominant semantic filter and reranker for codebases and document collections. Originally designed for Liquid Foundation Models (**LFM 2.5 8B-A1B**), it is now **completely AI model agnostic**, supporting:
- **LFM 2.5** (ChatML with `<think>\n</think>\n` bypass)
- **Qwen** (2.5 / 3 / 3.8 / QwQ)
- **Gemma** (2 / 3 turn-based formats)
- **RWKV** (v5 / v6 / v7 raw completion RNN format)
- **Any OpenAI-compatible provider** (vLLM, Ollama, llama.cpp, TabbyAPI)

---

## Key Architectural Highlights

### 1. Model / Provider Adapter Architecture (`slm_rerank/adapters.py`)
- Base `ModelProfile` / `ProviderAdapter` interface:
  - `name`: identifier (e.g. `'lfm'`, `'qwen'`, `'gemma'`, `'rwkv'`, `'openai'`)
  - `format_prompt(query, chunk_content, file_path, symbol, intent) -> str`
  - `extract_yes_no_logprobs(response_payload) -> Tuple[float, float, bool]`
  - `length_normalization_exponent`: model-specific tuning (LFM 0.15, Qwen 0.12, Gemma 0.12, RWKV 0.10)
  - `default_base_url`: default local port (LFM :8034, Qwen :8033, Ollama :11434, RWKV :8000)
- Built-in profiles:
  - `LFMProfile`: ChatML with `<think>\n</think>\n` bypass.
  - `QwenProfile`: Qwen ChatML `<|im_start|>` without think tags (unless QwQ).
  - `GemmaProfile`: Gemma format (`<start_of_turn>user\n...<start_of_turn>model\n`).
  - `RWKVProfile`: Clean raw completion format optimized for RNN/linear state attention.
  - `GenericOpenAIProfile`: Compatible with vLLM, Ollama, llama.cpp, TabbyAPI.

### 2. Dynamic Token String & Vocabulary Discovery
- Hardcoded integer token IDs are eliminated; token text matching works directly from `top_logprobs` (e.g. `token.strip().lower() in ('yes', 'no')`).
- Automatically probes `/tokenize` at startup to dynamically discover model-specific Yes/No token IDs, falling back to normalized text matching.

### 3. Model Isolation in Persistent SQLite Cache
- SQLite cache key is isolated by active model profile:
  $$\text{cache\_key} = \text{sha256}(f"\{\text{model\_id}\}:\{\text{normalized\_query}\}:\{\text{chunk\_hash}\}")$$
- Stored locally at `~/.cache/slm-rerank/cache.db`.
- Switching between LFM, Qwen, Gemma, or RWKV guarantees zero score cross-contamination.

### 5. Multi-Port Auto-Discovery (8033–8040)
- Scans active models across dedicated ports `8033..8040` (e.g. 8033 Qwen, 8034 LFM 2.5, 8035-8040 others) in parallel (< 100ms).
- Seamlessly resolves target port based on requested model profile or active hardware.
- **Scans ports on a single host, never the network.** The default host is loopback, so a
  model server on *another* machine is found only when you name it — see
  [Remote & Multi-Machine Setup](#remote--multi-machine-setup). When nothing answers,
  discovery reports why instead of returning an endpoint that is not there.

### 6. Smart Ripgrep Candidate Discovery
- Runs when no paths or globs are passed: `slm-rerank -q "auth token"` finds its own candidates.
- Recall is a **ranked union of two signals** — terms matching a file's *path* and terms
  matching its *body* — never one gated behind the other. A file named after what you asked
  for stays reachable even when a dozen test files mention the same words.
- Candidates are ranked by how many distinct terms hit (a path hit counts double), with test
  files ranked below implementations, then truncated to the limit. Ordering is by relevance,
  not directory traversal.
- A path term matches whole tokens, split on separators and camelCase. `share` does not match
  `shared/`, and `lay` does not match `replay`.
- Query terms are stemmed with each stem kept beside its root (`migration` → `migrat`,
  `migrate`; `chunking` → `chunk`; `classes` → `class`), so a term cap never severs a stem
  from the word it came from.
- Requires `rg` (ripgrep) or `git` on PATH; ripgrep is preferred and also reaches untracked files.

### 7. AST "Ghost Stubs" / Context Skeletons (`--stub` / `--slice`)
- Drastically reduces frontier model context window consumption: preserves imports, types, and the target chunk while collapsing non-relevant sibling functions into 1-line stubs (`folded N lines`).

### 8. Architecture-Aware Boundary Slicing (`--by-slice`)
- Automatically clusters reranked code by architectural domain slice (`features/<slice>`, `modules/<slice>`, `packages/<slice>`).

### 9. Lexical Prior on the Final Score
- Tier-1 scores every candidate lexically. A query term that names the file or a symbol it
  declares counts most, then a directory match, then mentions in the body.
- The final `score` adds that evidence to the model's verdict in logit space:
  `logit(score) = logit(rawScore) + z`. Here `z` is the chunk's lexical score, standardized
  over the scored chunks and clipped to ±3. A spread below 3 points counts as noise.
- `rawScore` stays the model's own score, and a chunk the model could not score stays at 0.
- The prior costs no model call.

---

## Installation & Quickstart

### Option A: npm / npx (Zero Python, Zero PyTorch — Ideal for Laptops & Remote LAN)

You can run `slm-rerank` directly on any machine with Node.js >= 20 without installing Python:

```bash
# Direct execution via npx (points to local or LAN GPU endpoint)
npx slm-rerank -q "user authentication token" -e "http://<gpu-host-ip>:8034/v1" src/**/*.ts

# Or install globally
npm install -g slm-rerank
```

### Option B: Python (Rich Terminal Table & Local Hardware Server)

```bash
cd /home/shawry/Documents/GitHub/slm-rerank
pip install -e . --break-system-packages
```

---

## MCP Server (Claude Code & other MCP clients)

`slm-rerank` ships a zero-dependency, pure Node.js MCP server speaking JSON-RPC 2.0 over
stdio — no Python, no `mcp` SDK, no PyTorch required.

```bash
# Run the server directly (stdin/stdout JSON-RPC)
npx slm-rerank --mcp

# Register it with Claude Code
claude mcp add slm-reranker -- npx -y slm-rerank --mcp

# Point it at a remote LAN GPU box
claude mcp add slm-reranker -- npx -y slm-rerank --mcp --host 192.168.1.50
```

It exposes one tool, `rerank_codebase`, which auto-discovers a live SLM endpoint across
ports 8033–8040 and returns ranked `file:line` citations plus a JSON candidate manifest:

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `query` | string | *(required)* | Natural language query or code task |
| `paths_or_globs` | string[] | auto-discovery | Files, directories or globs (e.g. `["src/**/*.ts"]`) |
| `threshold` | number | `0.65` | Relevance score cutoff |
| `top_k` | integer | `5` | Maximum results returned |
| `stub` | boolean | `false` | Attach AST Ghost Stubs to top results |
| `dirty` | boolean | `false` | Bias toward git uncommitted/modified files |
| `by_slice` | boolean | `false` | Group results by architectural vertical slice |
| `expand` | boolean | `false` | Ask the model for synonyms before auto-discovery (see below) |

---

## Query Expansion (opt-in, `--expand`)

Deterministic recall matches words and their stems. It reaches `migrate` from `migration`,
but it will never reach `bridge` from `harness` — and a file whose code says `buildBridge`
is invisible to a query that says "interop harness".

`--expand` (CLI) or `"expand": true` (MCP) asks the local model for the vocabulary the query
is missing, then feeds those terms into candidate discovery alongside the literal ones:

```bash
slm-rerank --query "interop harness" --expand
# 🧠 Expanded query with: communication, protocol, serialization, wrapper, bridge, adapter, ...
# → finds src/zz-adapter.ts, whose body says buildBridge and marshalRow
```

It is **strictly additive and strictly optional**:

- Expanded terms are weighted at half a literal term, so a synonym can add a file to the
  candidate set but never displace one the query named outright.
- Every failure — no server, timeout, malformed answer — returns no terms, and recall
  proceeds on the deterministic list. Expansion can widen recall; it can't break it.
- Decoding is greedy (`temperature: 0`), so a query always yields the same terms. Results
  are cached for 30 days in `~/.cache/slm-rerank/expansions.json`, shared between the Node
  and Python implementations.

### Measured effect

On ten queries against this repository where the deterministic path already finds the
target file: **1 improved, 9 unchanged, 0 worse**. Expansion is not a general win — it is a
rescue for vocabulary mismatch. On a fixture repo where the target shares no words with the
query, the same query goes from **not found at all** to **rank #1**:

| Query | Target | Without `--expand` | With `--expand` |
| --- | --- | --- | --- |
| `interop harness` | `src/zz-adapter.ts` (`buildBridge`) | missing | **#1** |

Cost is one model call, roughly 280ms on an 8B LFM, plus a few extra searches (~40ms).

**Known limitation:** the model sometimes answers with prose or reads a word differently than
you meant ("harness" as *test* harness). Generic and stop words are filtered out, and the
half-weight keeps the rest from doing damage, but expansion quality is bounded by the model.

---

## Grounded code review (v0.8.0)

The reranker answers *which chunk is relevant* with a calibrated probability. It writes no prose,
so it cannot invent a finding. A generative model can, and a review that reports an invented
finding is worse than no review.

`slm-rerank-review` closes that gap. It retrieves with the same calibrated scorer, asks the model
for candidate findings through the native `/completion` endpoint only (`/v1/chat/completions`
applies a chat template and is not usable here), then checks every candidate against the physical
file before reporting it.

### What is guaranteed, and what is not

Three checks run, in order. The first two are deterministic:

1. **Evidence gate.** The quoted evidence must be a line of the cited chunk. A fabricated quote is
   dropped. A multi-line quote collapsed onto one line still matches.
2. **Identifier gate.** Every code identifier the finding names must exist in the cited chunk. A
   finding about `onSkip` in a file that has no `onSkip` is dropped.
3. **Support gate (best effort).** The calibrated binary scorer judges whether the evidence
   proves the claim, through a dedicated judge prompt rather than the retrieval prompt.
4. **Contradiction gate (best effort veto).** A second judge asks whether the code contradicts
   the claim. A high score vetoes the finding. This catches a real case the support gate missed:
   "the state variable is not initialized" against `useState(0)`.

Every failure path returns fewer findings rather than raising, so a missing or confused model
degrades the review; it never fabricates one.

### Measured on the local 8B model

The semantic gates are a filter, not a guarantee. On a five-case labeled set (`SEMANTIC_CASES` in
`review_bench.py`), the local LFM2.5 judge rejected every case: precision stayed `100%` but recall
fell to `0%`. Support probabilities were near `0.02` for true and false claims alike, so the score
cannot separate them. The contradiction judge shows weak separation only (false `0.56` / `0.31` /
`0.22` against true `0.36` / `0.15`).

Treat the semantic gates as advisory until a stronger judge model is available. The evidence and
identifier gates are the deterministic guarantee.

### Usage

```bash
# Review files or globs, with all three gates
slm-rerank-review "frontend/src/**/*.tsx" --query "find real defects" --top 5

# Evidence and identifier gates only, no semantic judge
slm-rerank-review src/app.ts --no-support --no-contradiction

# Deterministic gates plus the semantic judges (default)
slm-rerank-review src/app.ts --top 5

# Machine-readable report
slm-rerank-review src/app.ts --json
```

Reasons a candidate is not reported are stable strings: `EMPTY_EVIDENCE`, `EVIDENCE_TOO_SHORT`,
`EVIDENCE_NOT_FOUND`, `CLAIM_SYMBOL_NOT_IN_CHUNK`, `CLAIM_NOT_SUPPORTED`, `SUPPORT_UNVERIFIED`,
`CHUNK_UNREADABLE`.

### Trust benchmark

```bash
slm-rerank-review-bench          # deterministic, no model needed
slm-rerank-review-bench --live   # also run retrieval and generation against the local model
```

The offline corpus carries findings the local model actually produced during the
quality-control-mono guide-tour review, each with evidence that is absent from the target file. The
benchmark proves the deterministic gates report every supported finding and drop every fabricated
one:

| Corpus | Claims | Reported | Dropped |
| --- | ---: | ---: | ---: |
| Supported (evidence on disk) | 2 | 2 | 0 |
| Fabricated (evidence absent) | 6 | 0 | 6 |
| Symbol-invented (symbol absent) | 3 | 0 | 3 |

Reported precision is `100%` with the gates and `18.18%` without them.

---

## Remote & Multi-Machine Setup

Discovery probes **ports on one host**. Running the reranker on a laptop while the GPU box
runs `llama-server` elsewhere means pointing it at that box explicitly — nothing is
auto-detected across the network.

On the machine serving the model, bind to all interfaces (not just loopback):

```bash
llama-server -m LFM2.5-8B-A1B-Q8_0.gguf --host 0.0.0.0 --port 8034
```

On the client machine, set one environment variable:

```bash
# Full endpoint — skips port scanning entirely
export SLM_ENDPOINT=http://192.168.1.220:8034/v1

# Or just the host — ports 8033-8040 are then scanned on that box
export SLM_HOST=192.168.1.220
```

For Claude Code, put it in the MCP registration itself:

```bash
claude mcp add slm-reranker -e SLM_ENDPOINT=http://192.168.1.220:8034/v1 -- npx -y slm-rerank --mcp

# Equivalent, scanning the remote port range instead of pinning one endpoint
claude mcp add slm-reranker -- npx -y slm-rerank --mcp --host 192.168.1.220
```

### Environment Variables

Both the Node and Python implementations read the same variables, in this order:

| Purpose | Variables (highest precedence first) | Default |
| --- | --- | --- |
| Full endpoint URL | `SLM_ENDPOINT`, `RERANKER_BASE_URL`, `LFM_ENDPOINT` | *(unset)* |
| Host to scan | `SLM_HOST`, `RERANKER_HOST` | `127.0.0.1` |

A `--base-url` / `-e` argument beats both. A pinned endpoint URL beats `--host`, and is used
as-is even when a specific `--model` is requested, on the assumption that you named the
server deliberately.

---

## CLI Usage

### Model Selection & Auto-Detection
```bash
# Auto-detect model from default port 8034 (LFM)
slm-rerank --query "auth middleware" src/**/*.ts

# Explicitly target Qwen on port 8033
slm-rerank --query "auth middleware" src/**/*.ts --model qwen --base-url http://localhost:8033/v1

# Target Gemma on Ollama
slm-rerank --query "find cache key" src/**/*.py --model gemma --base-url http://localhost:11434/v1

# Target RWKV on port 8000
slm-rerank --query "linear attention state" src/**/*.py --model rwkv -e http://localhost:8000/v1
```

### Piping from Git / Find
```bash
git ls-files "*.py" | slm-rerank --query "brier score calibration" --threshold 0.65
```

### Cache Management
```bash
# Bypass cache for fresh live evaluation
slm-rerank --query "auth" src/*.py --no-cache

# Clear cache entries
slm-rerank --clear-cache --query "auth" src/*.py
```

---

## Configuration File (`~/.config/reranker/config.yaml`)

```yaml
model: qwen
base_url: http://localhost:8033/v1
endpoints:
  lfm: http://localhost:8034/v1
  qwen: http://localhost:8033/v1
  gemma: http://localhost:11434/v1
  rwkv: http://localhost:8000/v1
  openai: http://localhost:11434/v1
concurrency: 4
timeout: 45.0
```

---

## Python API

```python
import asyncio
from slm_rerank import Reranker, QwenProfile

async def main():
    # Model-agnostic initialization (auto-detects or uses explicit profile)
    reranker = Reranker(model="qwen", base_url="http://localhost:8033/v1")

    response = await reranker.rerank(
        query="progress trigger handler",
        candidates=["backend/src/features/progress/trigger.ts"],
        threshold=0.65,
    )

    for item in response.results:
        print(f"[{item.score*100:.1f}%] {item.file_path}:{item.citation.start_line} ({item.symbol})")

if __name__ == "__main__":
    asyncio.run(main())
```
