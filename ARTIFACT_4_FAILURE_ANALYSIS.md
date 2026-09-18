# Validation Artifact 4: Failure Analysis (Rank > 3)
## LFM Semantic Co-Processor Wide Reranker (v0.3.3)

Detailed technical post-mortem for tasks where the ground-truth target chunk was ranked outside the top 3 (Total Failures: 3 / 11):

### Task `TASK-LLAMA-04`: "diagnose crash exception and syntax error in chat template formatting"
- **Intent**: `BUG_DIAGNOSIS`
- **Target**: `llama-chat.cpp` (llm_chat_apply_template)
- **Actual Rank Achieved**: **#35** (Score: `0.5568`)
- **Root Cause Diagnostic**: Chunk 'parse_token' in llama-grammar.cpp exhibited higher query term density and matched prompt keywords more specifically than target symbol 'llm_chat_apply_template'.

**Top 3 Competitors Outranking Target:**
| Rank | File | Symbol | Lines | Adjusted Score | Raw Score | Test Chunk? |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: |
| #1 | `llama-grammar.cpp` | `parse_token` | `L185-L229` | `0.8400` | `0.8400` | `False` |
| #2 | `llama-grammar.cpp` | `llama_grammar_match_partial_char` | `L788-L834` | `0.7968` | `0.8200` | `False` |
| #3 | `llama-grammar.cpp` | `llama_grammar_match_char` | `L758-L783` | `0.7747` | `0.7998` | `False` |

### Task `TASK-LLAMA-05`: "refactor and extract common KV cache quantization type mapping"
- **Intent**: `REFACTOR`
- **Target**: `llama-arch.cpp` (llm_arch_from_string)
- **Actual Rank Achieved**: **#48** (Score: `0.3222`)
- **Root Cause Diagnostic**: Chunk 'llama_grammar_match_partial_char' in llama-grammar.cpp exhibited higher query term density and matched prompt keywords more specifically than target symbol 'llm_arch_from_string'.

**Top 3 Competitors Outranking Target:**
| Rank | File | Symbol | Lines | Adjusted Score | Raw Score | Test Chunk? |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: |
| #1 | `llama-grammar.cpp` | `llama_grammar_match_partial_char` | `L788-L834` | `0.7952` | `0.7952` | `False` |
| #2 | `llama-grammar.cpp` | `decode_utf8` | `L34-L92` | `0.7495` | `0.7766` | `False` |
| #3 | `llama-grammar.cpp` | `print_rule_binary` | `L251-L294` | `0.7323` | `0.7607` | `False` |

### Task `TASK-LLAMA-06`: "test assertions and grammar verification contract for GBNF parser"
- **Intent**: `SPECIFICATION`
- **Target**: `test-llama-grammar.cpp` (module_scope)
- **Actual Rank Achieved**: **#5** (Score: `0.8125`)
- **Root Cause Diagnostic**: Chunk 'parse_token' in llama-grammar.cpp exhibited higher query term density and matched prompt keywords more specifically than target symbol 'test-llama-grammar.cpp'.

**Top 3 Competitors Outranking Target:**
| Rank | File | Symbol | Lines | Adjusted Score | Raw Score | Test Chunk? |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: |
| #1 | `llama-grammar.cpp` | `parse_token` | `L185-L229` | `0.8999` | `0.8577` | `False` |
| #2 | `llama-grammar.cpp` | `parse_hex` | `L102-L123` | `0.8599` | `0.8270` | `False` |
| #3 | `llama-grammar.cpp` | `llama_grammar_match_partial_char` | `L788-L834` | `0.8310` | `0.8510` | `False` |

### Resolved Failures in v0.3.3:
- **TASK-QC-01**: Single-word symbol boost (`trigger` in query matching `trigger.ts:trigger`) lifted target from Rank #4 to **Rank #2**.
- **TASK-QC-05**: Schema refactor table disambiguation and test penalty lifted `progressWorkTypes` from Rank #4 to **Rank #3**.
- **TASK-LLAMA-03**: Semantic component decomposition (`llm` + `arch` + `string`) lifted `llm_arch_from_string` from Rank #9 to **Rank #1**.
