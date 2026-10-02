# Engineering Report: GenAI Job Application Assistant

This report documents the architectural design, exercise implementations, empirical findings, and limitations for the GenAI Job Application Assistant.

---

## Exercise 1: Token Accounting & Multilingual Text (`tokens.py`)

### Goal
Analyze token overhead across different writing systems (English, Devanagari Hindi, and Hinglish transliteration) using BPE tokenizers, and evaluate Unicode normalization impact (NFC vs NFD) and screening cost scaling.

### Approach
- Compare token counts and character/token ratios using `tiktoken` (`cl100k_base` and `gpt2`).
- Evaluate Unicode normalization (`unicodedata.normalize("NFC")` vs `"NFD"`) on tokenization counts.
- Model screening costs with parametrized Hindi token multipliers (1.5x - 3.0x).

### Results & Empirical Findings
*(Populated during execution)*

### Limitations
- `tiktoken` byte-pair encodings were optimized primarily for English text; Devanagari script incurs a higher token-to-word ratio due to UTF-8 multi-byte sequence split across subwords.

---

## Exercise 2: Next-Token Prediction & Sampling (`sampling.py`)

### Goal
Implement numerical sampling utilities for autoregressive token generation: Softmax with Temperature, Top-K filtering, Top-P (nucleus) filtering, and next-token sampling.

### Approach
- Implement `softmax(logits, T)` with temperature scaling and numerical stability guards.
- Implement `top_k_filter` retaining top K logits.
- Implement `top_p_filter` keeping the smallest set of descending-sorted tokens whose cumulative probability $\ge p$ using `np.searchsorted(cum, p, side="left")` and `sorted[:cutoff+1]`.
- Enforce execution order: Temperature $\to$ Top-K $\to$ Top-P $\to$ Renormalize $\to$ Sample.

### Results & Empirical Findings
*(Populated during execution)*

### Limitations
- Toy vocabulary simulation demonstrates probability redistribution; real models compute logits over vocabulary sizes exceeding 32k-128k tokens.

---

## Exercise 3: Attention & Context Window (`attention.py`)

### Goal
Implement scaled dot-product causal self-attention in NumPy and demonstrate positional encoding necessity for sequence sensitivity ("dog bites man" vs "man bites dog").

### Approach
- Scaled dot-product attention formula: $\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}} + M\right) V$ with upper-triangular causal mask $M$.
- Demonstrate permutation variance by adding Sinusoidal/RoPE positional encodings to token embeddings.

### Results & Empirical Findings
*(Populated during execution)*

### Limitations
- Full self-attention exhibits $O(N^2)$ time and memory complexity relative to sequence length $N$.

---

## Exercise 4: Context Budgeting & Lost-in-the-Middle (`context.py`)

### Goal
Manage prompt context windows under token limits and investigate the "Lost in the Middle" retrieval degradation phenomenon.

### Approach
- Synthetic long document generator placing target facts at beginning, middle, and end positions.
- Relevance-based chunk packing (BM25/TF-IDF) prioritizing top-k relevant context chunks into the context budget.
- Simulation of U-shaped recall curves with `MockLLM`.

### Results & Empirical Findings
*(Populated during execution)*

### Limitations
- The 20-30% recall drop in middle context positions cited in literature ("Lost in the Middle", Liu et al., 2023) depends on model architecture, instruction fine-tuning, and attention head mechanisms.

---

## Exercise 5: Training Stages (`training_stages.py`)

### Goal
Categorize LLM behaviors into pretraining, supervised fine-tuning (SFT), and preference alignment (RLHF/DPO), with strict error validation.

### Approach
- `TrainingStage` Enum mapping behaviors.
- Explicit `ValueError` on unknown/unrecognized behaviors.

### Results & Empirical Findings
*(Populated during execution)*

### Limitations
- Behavior alignment is non-binary; preference tuning reduces hallucinations but does not eliminate them.

---

## Exercise 6: Model-Tier Selection (`tiers.py`)

### Goal
Design an explainable rule-based tier selector and dynamic router escalating tasks from Small $\to$ Medium $\to$ Large based on confidence thresholds and schema validation.

### Approach
- Config-driven tier definition (`config/pricing.yaml`).
- Router simulating an 80% Small / 15% Medium / 5% Large distribution vs an all-Large baseline.

### Results & Empirical Findings
*(Populated during execution)*

### Limitations
- Simulated routing distribution assumes independence between prompt complexity and validation failure rate.

---

## Exercise 7: Hallucination Detection (`hallucination.py`)

### Goal
Build a deterministic programmatic grounding checker verifying candidate assessments against original resume context.

### Approach
- Exact normalized substring search and fuzzy matching (`difflib.SequenceMatcher` $\ge 0.9$).
- Grounding verification: evidence strings must exist in resume, claimed skills must exist in resume text, absent JD requirements must be `NOT SPECIFIED` or `Does Not Meet`.

### Results & Empirical Findings
*(Populated during execution)*

### Limitations
- Fuzzy substring matching may misclassify heavily paraphrased text or complex semantic equivalences.

---

## Exercise 8: Cost Estimation & Prompt Caching (`cost.py`)

### Goal
Model token costs and quantify savings from static prefix prompt caching and asynchronous batch processing.

### Approach
- Prefix caching discount formula applying only to static system instructions (Doc 1 full price/cache write, Docs 2..N cached price on prefix + standard price on variable JD/resume).
- Separate modeling of batch processing discounts.

### Results & Empirical Findings
*(Populated during execution)*

### Limitations
- Real provider cache lifetimes and minimum prefix length constraints vary.

---

## Exercise 9: Structured Output & Validation (`schema.py`)

### Goal
Construct Pydantic v2 schemas (`CandidateAssessment`) with field validators, Markdown code fence stripping, structured error parsing, and automated retry loops.

### Approach
- Pydantic v2 validation enforcing numerical bounds (experience 0-50, score 0-100), non-empty skills, evidence length $\ge 10$, and score/recommendation consistency.
- `validate_json_output` returning `ValidationResult` with error field paths.

### Results & Empirical Findings
*(Populated during execution)*

### Limitations
- Re-prompting adds latency and token consumption; retry limits (max 2) cap cost overhead.
