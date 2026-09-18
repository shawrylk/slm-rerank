# Model-Agnostic Wide Reranker (`lfm-rerank`)

The **Wide Reranker** is an ultra-high-throughput, prefill-dominant semantic filter and reranker for codebases and document collections. Originally designed for Liquid Foundation Models (**LFM 2.5 8B-A1B**), it is now **completely AI model agnostic**, supporting:
- **LFM 2.5** (ChatML with `<think>\n</think>\n` bypass)
- **Qwen** (2.5 / 3 / 3.8 / QwQ)
- **Gemma** (2 / 3 turn-based formats)
- **RWKV** (v5 / v6 / v7 raw completion RNN format)
- **Any OpenAI-compatible provider** (vLLM, Ollama, llama.cpp, TabbyAPI)

---

## Key Architectural Highlights

### 1. Model / Provider Adapter Architecture (`lfm_rerank/adapters.py`)
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
- Stored locally at `~/.cache/lfm-rerank/cache.db`.
- Switching between LFM, Qwen, Gemma, or RWKV guarantees zero score cross-contamination.

### 4. Auto-Detection & CLI Selection
- Probe `GET /v1/models` at startup to auto-select `QwenProfile`, `LFMProfile`, `GemmaProfile`, or `RWKVProfile`.
- Optional config file at `~/.config/reranker/config.yaml` to set default providers and endpoint mappings.

---

## Installation

```bash
cd /home/shawry/lfm-kit/reranker
pip install -e . --break-system-packages
```

---

## CLI Usage

### Model Selection & Auto-Detection
```bash
# Auto-detect model from default port 8034 (LFM)
lfm-rerank --query "auth middleware" src/**/*.ts

# Explicitly target Qwen on port 8033
lfm-rerank --query "auth middleware" src/**/*.ts --model qwen --base-url http://localhost:8033/v1

# Target Gemma on Ollama
lfm-rerank --query "find cache key" src/**/*.py --model gemma --base-url http://localhost:11434/v1

# Target RWKV on port 8000
lfm-rerank --query "linear attention state" src/**/*.py --model rwkv -e http://localhost:8000/v1
```

### Piping from Git / Find
```bash
git ls-files "*.py" | lfm-rerank --query "brier score calibration" --threshold 0.65
```

### Cache Management
```bash
# Bypass cache for fresh live evaluation
lfm-rerank --query "auth" src/*.py --no-cache

# Clear cache entries
lfm-rerank --clear-cache --query "auth" src/*.py
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
from lfm_rerank import Reranker, QwenProfile

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
