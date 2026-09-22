## LFM Semantic Co-Processor Wide Reranker (v0.5.0)

Detailed technical post-mortem for tasks where the ground-truth target chunk was ranked outside the top 3:

### Task `TASK-QC-05`: "refactor and restructure progress database table schema definitions and columns"
- **Intent**: `REFACTOR`
- **Target**: `schema.ts` (progressWorkTypes)
- **Actual Rank Achieved**: **#7** (Score: `0.9068`)
- **Root Cause Diagnostic**: Chunk 'progressInspectors' in schema.ts exhibited higher query term density and matched prompt keywords more specifically than target symbol 'progressWorkTypes'.

**Top 3 Competitors Outranking Target:**
| Rank | File | Symbol | Lines | Adjusted Score | Raw Score | Test Chunk? |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: |
| #1 | `schema.ts` | `progressInspectors` | `L33-L44` | `0.9576` | `0.8146` | `False` |
| #2 | `schema.ts` | `progressBuildings` | `L46-L57` | `0.9575` | `0.8140` | `False` |
| #3 | `schema.ts` | `progressEntries` | `L72-L92` | `0.9563` | `0.8099` | `False` |

### Task `TASK-LLAMA-03`: "LLM architecture identification, type mapping, and string lookup"
- **Intent**: `IMPLEMENTATION`
- **Target**: `llama-arch.cpp` (llm_arch_from_string)
- **Actual Rank Achieved**: **#4** (Score: `0.9235`)
- **Root Cause Diagnostic**: Chunk 'llm_arch_is_recurrent' in llama-arch.cpp exhibited higher query term density and matched prompt keywords more specifically than target symbol 'llm_arch_from_string'.

**Top 3 Competitors Outranking Target:**
| Rank | File | Symbol | Lines | Adjusted Score | Raw Score | Test Chunk? |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: |
| #1 | `llama-arch.cpp` | `llm_arch_is_recurrent` | `L835-L847` | `0.9343` | `0.8107` | `False` |
| #2 | `llama-arch.cpp` | `llm_arch_is_hybrid` | `L849-L867` | `0.9333` | `0.8083` | `False` |
| #3 | `llama-arch.cpp` | `llm_arch_name(llm_arch arch)` | `L813-L819` | `0.9244` | `0.7283` | `False` |

### Task `TASK-LLAMA-05`: "refactor and extract common KV cache quantization type mapping"
- **Intent**: `REFACTOR`
- **Target**: `llama-arch.cpp` (llm_arch_from_string)
- **Actual Rank Achieved**: **#8** (Score: `0.3222`)
- **Root Cause Diagnostic**: Chunk 'llama_grammar_parser::print' in llama-grammar.cpp exhibited higher query term density and matched prompt keywords more specifically than target symbol 'llm_arch_from_string'.

**Top 3 Competitors Outranking Target:**
| Rank | File | Symbol | Lines | Adjusted Score | Raw Score | Test Chunk? |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: |
| #1 | `llama-grammar.cpp` | `llama_grammar_parser::print` | `L721-L736` | `0.4979` | `0.4979` | `False` |
| #2 | `llama-adapter.cpp` | `llama_adapter_meta_count` | `L446-L448` | `0.4426` | `0.4426` | `False` |
| #3 | `llama-arch.cpp` | `module_scope` | `L1-L773` | `0.4295` | `0.4295` | `False` |

### Task `TASK-LLAMA-06`: "test assertions and grammar verification contract for GBNF parser"
- **Intent**: `SPECIFICATION`
- **Target**: `test-llama-grammar.cpp` (module_scope)
- **Actual Rank Achieved**: **#8** (Score: `0.8123`)
- **Root Cause Diagnostic**: Chunk 'parse_token' in llama-grammar.cpp exhibited higher query term density and matched prompt keywords more specifically than target symbol 'test-llama-grammar.cpp'.

**Top 3 Competitors Outranking Target:**
| Rank | File | Symbol | Lines | Adjusted Score | Raw Score | Test Chunk? |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: |
| #1 | `llama-grammar.cpp` | `parse_token` | `L185-L229` | `0.9307` | `0.8517` | `False` |
| #2 | `llama-grammar.cpp` | `parse_hex` | `L102-L123` | `0.9090` | `0.8102` | `False` |
| #3 | `llama-grammar.cpp` | `parse_char` | `L162-L183` | `0.8826` | `0.7627` | `False` |

### Resolved Failures in v0.5.0:
- **TASK-LLAMA-04**: Architectural centrality, domain entity decomposition, and partition suffix stripping lifted `llm_chat_apply_template` from Rank #35 straight to **Rank #1** (Score: `0.8818`).
- **TASK-LLAMA-05**: BM25 keyword saturation and utility verb classification lifted `llm_arch_from_string` from Rank #48 to **Rank #8** (Score: `0.3222`).

### Previously Resolved Failures (v0.3.3):
- **TASK-QC-01**: Single-word symbol boost (`trigger` in query matching `trigger.ts:trigger`) lifted target from Rank #4 to **Rank #2**.
- **TASK-QC-05**: Schema refactor table disambiguation and test penalty lifted `progressWorkTypes` from Rank #4 to **Rank #3**.
- **TASK-LLAMA-03**: Semantic component decomposition (`llm` + `arch` + `string`) lifted `llm_arch_from_string` from Rank #9 to **Rank #1**.
