# AGENTS.md — GenAI Job Application Assistant

> Author's note (senior ML review): This spec turns the roadmap PDF into an
> implementable, testable project. It is written for a coding agent. Follow it
> literally; where it says **DECISION**, the choice is already made.

## 0. Mission

Build a Python package + CLI + notebook-free demo that:
1. Analyzes a resume against a job description (JD).
2. Answers questions about the JD/resume, grounded only in the provided text.
3. Returns a **validated structured assessment** (JSON, Pydantic).
4. Demonstrates, with runnable code and tests, 9 exercises: token accounting,
   sampling, attention/position, lost-in-the-middle, training stages,
   model-tier selection, hallucination checks, cost/caching, structured-output validation.

## 1. Hard rules (non-negotiable)

- **Offline-first.** Everything must run and pass tests with NO API key and NO
  network. LLM calls go through an `LLMClient` interface with two backends:
  `MockLLM` (deterministic, default) and `OpenAICompatLLM` (optional, reads
  `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL` from env). Never hardcode keys.
- **Python 3.11+, Pydantic v2** (`field_validator`, `model_validate_json`,
  `model_dump`). Do NOT use v1 `@validator` / `.dict()` (the PDF's snippets use
  them; they are deprecated).
- **No fabricated numbers.** Prices, latencies, accuracy percentages in the PDF
  are illustrative. Put them in `config/pricing.yaml`, label them
  `illustrative`, and never present them as measured facts.
- Type hints everywhere, `ruff` + `pytest` clean, seeded randomness
  (`np.random.default_rng(seed)`), no global state.
- Small, readable modules. Each exercise = one module + one test file + one
  section in `REPORT.md`.

## 2. Repository layout

```
job-assistant/
├── AGENTS.md
├── README.md                # how to run, 1 page
├── REPORT.md                # design decisions + results (auto-filled by scripts)
├── pyproject.toml
├── config/pricing.yaml      # illustrative tier prices, cache discounts
├── data/
│   ├── resumes/*.txt        # >=5 synthetic resumes (1 Hindi/Devanagari, 1 Hinglish)
│   └── jds/*.txt            # >=3 synthetic JDs
├── src/jobassist/
│   ├── llm.py               # LLMClient, MockLLM, OpenAICompatLLM
│   ├── tokens.py            # Ex1
│   ├── sampling.py          # Ex2
│   ├── attention.py         # Ex3
│   ├── context.py           # Ex4
│   ├── training_stages.py   # Ex5
│   ├── tiers.py             # Ex6 + router
│   ├── hallucination.py     # Ex7
│   ├── cost.py              # Ex8
│   ├── schema.py            # Ex9 Pydantic models
│   ├── prompts.py           # prompt template(s)
│   └── cli.py               # `python -m jobassist ...`
├── tests/test_*.py
└── scripts/run_all_exercises.py   # prints a table per exercise, writes results/*.json
```

## 3. Exercise specs

### Ex1 — Token accounting & multilingual text (`tokens.py`)
- Function `compare_token_counts(texts: dict[str,str], tokenizers) -> list[Row]`.
- Tokenizers: `tiktoken` `cl100k_base` and `gpt2` (local, no download issues).
  Optional HF tokenizer behind a try/except, skipped with a clear message if
  unavailable or gated (**the PDF's `meta-llama/Llama-3-8b` id is wrong and
  gated; do not depend on it**).
- Test texts: the same sentence in (a) English, (b) **real Devanagari Hindi**,
  (c) Hinglish (Latin transliteration). The PDF's "Hindi" sample is only
  transliterated, which does not demonstrate Devanagari tokenization — fix that.
- Report: tokens, words, tokens/word, chars/token, and ratio vs English.
  Also apply `unicodedata.normalize("NFC"/"NFD")` and show whether token counts change.
- `calculate_screening_cost(n=1000, avg_tokens=500, price_per_m)` exactly as in
  the PDF; plus a Hindi-multiplier variant (parametrize 1.5x–3x, label as assumption).
- Tests: ratios computed correctly; Devanagari tokens/word > English tokens/word
  for at least one tokenizer.

### Ex2 — Next-token prediction & sampling (`sampling.py`)
- Implement `softmax(logits, T)`, `top_k_filter`, `top_p_filter`, `sample_next_token`.
- **Correct top-p**: keep the smallest prefix of the descending-sorted tokens
  whose cumulative prob >= p. Implementation:
  `cutoff = np.searchsorted(cum, p, side="left")`; keep `sorted[:cutoff+1]`.
  (The PDF uses `side="right"`, which can keep one extra token — do not copy it.)
- T=0 → argmax (greedy). Guard against T<0. Use a passed-in `rng`.
- Order of operations: temperature → top-k → top-p → renormalize → sample.
- Demo: fixed toy vocabulary of ~10 tokens; print distributions for
  T ∈ {0, 0.3, 0.7, 1.0, 1.5}; empirical sampling histograms over 5,000 draws.
- Tests: probs sum to 1; greedy deterministic; top-k keeps exactly k nonzero;
  top-p mass >= p and minimal; higher T raises entropy.

### Ex3 — Attention & context window (`attention.py`)
- `create_prompt(system, jd, resume, question) -> dict`.
- Implement a tiny **scaled dot-product causal attention** in NumPy
  (`softmax(QK^T/√d + mask)V`) and a **sinusoidal or RoPE** positional demo
  showing "dog bites man" ≠ "man bites dog" only when positions are added.
- Return an explanation string (short, accurate: O(n²) compute, causal mask,
  positional info needed since attention is permutation-equivariant).
- Tests: causal mask zeroes future attention; attention rows sum to 1;
  permuting tokens changes output only with positional encoding.

### Ex4 — Context budgeting & lost-in-the-middle (`context.py`)
- `generate_long_document(n_sections, fact_positions)` — place KEY FACTs at
  beginning/middle/end (fix the PDF's missing comma bug in `filler_texts`
  and the never-appended `document_sections`).
- `chunk_document(text, chunk_size, overlap)`; guard `overlap < chunk_size`.
- `ContextBudget` class: given max tokens, reserve for system + output, pack
  chunks by relevance score (simple BM25/TF-IDF, no network).
- Strategy: **critical info first/last**, retrieve top-k chunks instead of
  stuffing the whole document.
- Experiment (offline): with `MockLLM` simulating a U-shaped recall curve
  (documented as a *simulation*, not a measurement), show recall by position
  and the improvement from retrieval. If a real LLM backend is configured,
  run the same probe for real and save results separately.
- Note in REPORT: the "20–30% middle drop" figure is from the literature
  ("Lost in the Middle", Liu et al.) and is model-dependent — cite, don't assert.

### Ex5 — Training stages (`training_stages.py`)
- `TrainingStage` enum; `identify_training_stage(behavior)` using the PDF's map.
- **Fix**: unknown behavior must raise `ValueError` (the PDF silently defaults
  to PRETRAINING, which hides bugs).
- Add `explain(stage)` returning data, objective, and one example I/O pair.
- Nuance to state in REPORT: behaviors like "avoid hallucination" are
  *improved* by preference tuning but not fully solved by it.

### Ex6 — Model-tier selection (`tiers.py`)
- Dataclasses `TaskRequirements`, `ModelTier` (load from `pricing.yaml`).
- `select_tier(req) -> (tier, reasons: list[str])` — rule-based, explainable.
- Required three cases with justification text:
  1. Bulk resume classification → small tier
  2. Complex technical interview feedback → large tier
  3. High-volume FAQ → small tier
- Router: small → medium → large escalation on `confidence < threshold` or
  schema-validation failure. Simulate the 80/15/5 mix and print blended cost
  vs all-large baseline (computed, not hardcoded).
- Do not name specific vendor models as facts; use config labels.

### Ex7 — Hallucination detection (`hallucination.py`)
- Sample resume (Sarah Chen) + JD from the PDF as fixtures.
- Prompt enforcing: only stated info, cite evidence, `NOT SPECIFIED`, confidence.
- **Programmatic grounding checker** (the real deliverable):
  - every `evidence` string must appear in the resume (normalized substring or
    fuzzy ratio >= 0.9 via `difflib`);
  - every claimed skill must be in the resume text;
  - requirements absent from the resume (e.g. Kubernetes, GraphQL, ML) must be
    `NOT SPECIFIED`/`Does Not Meet`, never `Meets`.
  - Output a `GroundingReport(passed, unsupported_claims, missing_evidence)`.
- Provide a "bad" mock response that hallucinates (claims Kubernetes) and show
  the checker catches it; and a "good" response that passes.

### Ex8 — Cost estimation & prompt caching (`cost.py`)
- `TokenPricing`, `ProcessingJob`, `estimate_cost(job, pricing)`.
- **Fix the PDF's formula intent**: cached tokens apply only to the *static
  prefix* (system instruction). Model it as: doc 1 pays full price (optionally a
  cache-write premium, configurable, default 1.0); docs 2..N pay
  `cached_price` on the prefix and normal price on the variable part.
- Provide separate functions/flags for **batching** (no token reduction;
  throughput only; some providers offer a batch discount — make it a config
  parameter, default 0) vs **caching**.
- Output a comparison table: no-cache, cache, cache+batch for N = 100/1,000/10,000.
- Tests: cached cost <= uncached cost; savings % monotonic in prefix share;
  N=1 equals uncached.

### Ex9 — Structured output & validation (`schema.py`)
- `CandidateAssessment` (Pydantic v2) with fields: `candidate_name`, `skills`,
  `experience_years` (0–50), `score` (0–100), `evidence`,
  `requirement_assessments[]`, `missing_information`,
  `recommendation` (Literal: Strong Yes|Yes|Maybe|No|Strong No), `reasoning`.
- Validators: non-empty skills; evidence items >= 10 chars; score/recommendation
  consistency (e.g. score < 30 cannot be "Strong Yes").
- `validate_json_output(raw) -> ValidationResult` that: strips markdown code
  fences, parses, validates, and returns typed errors
  (`malformed_json` vs `schema_validation`) with field paths.
- **Retry loop**: on validation failure, re-prompt (max 2) with the error
  message; surface final failure explicitly.
- The 6 test cases from the PDF are mandatory:
  valid; score 150 rejected; malformed JSON rejected; missing `score` rejected;
  empty skills rejected; valid-with-code-fences accepted.

## 4. CLI

```
python -m jobassist analyze --resume data/resumes/a.txt --jd data/jds/x.txt [--backend mock|api]
python -m jobassist ask --resume ... --jd ... --q "Does the candidate have Kubernetes experience?"
python -m jobassist run-exercises          # runs scripts/run_all_exercises.py
```
`analyze` prints validated JSON + a grounding report + estimated cost.

## 5. Prompt template (`prompts.py`)

XML-sectioned: `<system_role>`, `<task>`, `<instructions>`, `<output_schema>`,
`<job_description>`, `<resume>`, `<assessment>`. Static parts first (cacheable
prefix), variable JD/resume last. Instruct "JSON only". Keep instructions
numbered. Provide one few-shot example in a separate constant.

## 6. Testing & acceptance criteria

Done means ALL true:
- [ ] `pip install -e .[dev]` then `pytest -q` passes, no network.
- [ ] `ruff check .` clean.
- [ ] `python scripts/run_all_exercises.py` prints a result table for each of
      the 9 exercises and writes `results/*.json`.
- [ ] `analyze` works end-to-end on every sample resume with `MockLLM`.
- [ ] Hallucination checker flags the planted bad response.
- [ ] `REPORT.md` has per-exercise: goal, approach, result, limitations.
- [ ] No API keys, no TODO stubs, no unused modules.

## 7. Known issues in the source roadmap (already corrected above)

1. Wrong/gated Llama tokenizer id → use tiktoken, HF optional.
2. "Hindi" sample was transliterated Latin → use real Devanagari + Hinglish.
3. `top_p` used `side="right"` → use `side="left"` and keep `cutoff+1`.
4. Pydantic v1 API (`validator`, `.dict`) → v2 (`field_validator`, `model_dump`).
5. Missing comma in `filler_texts` list; `document_sections` never appended.
6. Unknown behavior defaulted to PRETRAINING → raise `ValueError`.
7. Caching math must apply to the static prefix only.
8. Prices/latencies/accuracy numbers are illustrative → config, labeled.

## 8. Working agreement for the agent

- Plan briefly, then implement exercise by exercise; run tests after each.
- Prefer the simplest correct implementation. Ask before adding dependencies
  beyond: `numpy`, `pydantic>=2`, `tiktoken`, `pyyaml`, `pytest`, `ruff`,
  optional `transformers`, `httpx`.
- If something in this file is ambiguous, choose the simplest option, state the
  assumption in `REPORT.md`, and continue.
