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
- Query terms are stemmed with each stem kept beside its root (`migration` → `migrat`,
  `migrate`; `chunking` → `chunk`; `classes` → `class`), so a term cap never severs a stem
  from the word it came from.
- Requires `rg` (ripgrep) or `git` on PATH; ripgrep is preferred and also reaches untracked files.

### 7. AST "Ghost Stubs" / Context Skeletons (`--stub` / `--slice`)
- Drastically reduces frontier model context window consumption: preserves imports, types, and the target chunk while collapsing non-relevant sibling functions into 1-line stubs (`folded N lines`).

### 8. Architecture-Aware Boundary Slicing (`--by-slice`)
- Automatically clusters reranked code by architectural domain slice (`features/<slice>`, `modules/<slice>`, `packages/<slice>`).

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
