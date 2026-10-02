# GenAI Job Application Assistant

An offline-first Python CLI and toolkit for resume analysis, JD question answering, hallucination checking, cost estimation, and LLM engineering exercise demos.

## Quick Start

### Installation
```bash
pip install -e .[dev]
```

### Run Exercises & Generate Results
```bash
python -m jobassist run-exercises
# or
python scripts/run_all_exercises.py
```

### CLI Commands

1. **Analyze Resume against Job Description**
```bash
python -m jobassist analyze --resume data/resumes/sarah_chen.txt --jd data/jds/senior_fullstack.txt
```

2. **Ask Questions Grounded in Resume/JD Context**
```bash
python -m jobassist ask --resume data/resumes/sarah_chen.txt --jd data/jds/senior_fullstack.txt --q "Does the candidate have Kubernetes experience?"
```

3. **Run All Demonstrations**
```bash
python -m jobassist run-exercises
```

### Run the UI (Streamlit)
To launch the interactive web application:
```bash
# Install UI dependencies
pip install -e .[ui]

# Run the app
streamlit run app/streamlit_app.py
```
*(Screenshot placeholder: `docs/ui_screenshot.png`)*

## Features & Exercises

- **Ex1: Token Accounting & Multilingual Text** (`tokens.py`) - Compares token counts across English, Devanagari Hindi, and Hinglish transliteration using `tiktoken` (cl100k_base & gpt2) and Unicode normalization (NFC/NFD).
- **Ex2: Next-Token Prediction & Sampling** (`sampling.py`) - Implements softmax temperature scaling, top-k filtering, top-p nucleus sampling (with correct prefix cutoffs), and entropy analysis.
- **Ex3: Attention & Context Window** (`attention.py`) - Implements scaled dot-product causal attention and positional encodings (Sinusoidal/RoPE) in NumPy.
- **Ex4: Context Budgeting & Lost-in-the-Middle** (`context.py`) - Document chunking, relevance scoring (BM25/TF-IDF), budget packing, and U-shaped recall curve simulation.
- **Ex5: Training Stages** (`training_stages.py`) - Classifies model behaviors across pretraining, SFT, and preference alignment (RLHF/DPO) with strict error handling.
- **Ex6: Model-Tier Selection & Router** (`tiers.py`) - Rule-based tier selection (Small, Medium, Large) and confidence/schema-escalation routing with blended cost computation.
- **Ex7: Grounding & Hallucination Checking** (`hallucination.py`) - Strict programmatic grounding verification for skill claims and requirement assessments.
- **Ex8: Cost Estimation & Prompt Caching** (`cost.py`) - Calculates accurate static prefix prompt caching discounts and batch processing cost models.
- **Ex9: Structured Output & Schema Validation** (`schema.py`) - Pydantic v2 validation for `CandidateAssessment` with markdown fence handling, field path errors, and retry loops.
