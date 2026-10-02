"""Tests for Exercise 4: Context budgeting & lost-in-the-middle."""

import pytest

from jobassist.context import (
    ContextBudget,
    chunk_document,
    generate_long_document,
    simple_bm25_score,
    simulate_lost_in_the_middle_recall,
)
from jobassist.llm import MockLLM


def test_generate_long_document():
    doc = generate_long_document(n_sections=5, fact_positions=[2])
    assert "Section 0:" in doc
    assert "Section 4:" in doc
    
    sections = doc.split("\n\n")
    assert len(sections) == 5
    assert "KEY FACT" in sections[2]
    assert "KEY FACT" not in sections[1]


def test_chunk_document():
    text = "word1 word2 word3 word4 word5 word6"
    chunks = chunk_document(text, chunk_size=3, overlap=1)
    
    assert len(chunks) == 3
    assert chunks[0] == "word1 word2 word3"
    assert chunks[1] == "word3 word4 word5"
    assert chunks[2] == "word5 word6"
    
    with pytest.raises(ValueError):
        chunk_document(text, chunk_size=3, overlap=3)


def test_simple_bm25_score():
    doc = "The quick brown fox jumps over the lazy dog"
    score_high = simple_bm25_score("fox dog", doc)
    score_low = simple_bm25_score("cat", doc)
    
    assert score_high > score_low
    assert score_low == 0.0


def test_context_budget():
    budget = ContextBudget(max_total_tokens=100, system_tokens=20, max_output_tokens=30)
    assert budget.available_context == 50
    
    chunks = [
        "This is a low relevance chunk with some words.",
        "This is a highly relevant chunk about the secret key.",
        "Another medium relevance chunk here."
    ]
    
    packed = budget.pack_chunks("secret key", chunks, chars_per_token=4)
    # The highly relevant chunk should be included
    assert any("highly relevant" in c for c in packed)


def test_simulate_lost_in_the_middle():
    llm = MockLLM()
    results = simulate_lost_in_the_middle_recall(llm, n_sections=10, chunk_size=20)
    
    stuffed = results["stuffed_context_recall"]
    retrieved = results["retrieved_context_recall"]
    
    # In MockLLM simulation, edges should be True, middle (pos 5) should be False for stuffed
    assert stuffed["pos_0"] is True
    assert stuffed["pos_5"] is False
    assert stuffed["pos_9"] is True
    
    # Retrieved RAG should succeed on all
    assert retrieved["pos_0"] is True
    assert retrieved["pos_5"] is True
    assert retrieved["pos_9"] is True
