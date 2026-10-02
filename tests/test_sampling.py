"""Tests for Exercise 2: Next-token prediction & sampling."""

import numpy as np
import pytest

from jobassist.sampling import (
    run_sampling_demo,
    sample_next_token,
    softmax,
    top_k_filter,
    top_p_filter,
)


def test_softmax_properties():
    logits = np.array([2.0, 1.0, 0.1, -1.0])
    
    # Probs sum to 1
    probs = softmax(logits, temperature=1.0)
    assert np.isclose(np.sum(probs), 1.0)
    assert np.all(probs >= 0)

    # Greedy T=0
    probs_greedy = softmax(logits, temperature=0.0)
    assert probs_greedy[0] == 1.0
    assert np.sum(probs_greedy[1:]) == 0.0

    # Negative T raises ValueError
    with pytest.raises(ValueError):
        softmax(logits, temperature=-0.5)


def test_top_k_filter():
    logits = np.array([10.0, 5.0, 2.0, 1.0, 0.5])
    filtered = top_k_filter(logits, k=2)

    assert filtered[0] == 10.0
    assert filtered[1] == 5.0
    assert np.isneginf(filtered[2])
    assert np.isneginf(filtered[3])
    assert np.isneginf(filtered[4])


def test_top_p_filter_left_side():
    # Sorted probs: [0.4, 0.3, 0.2, 0.1]
    probs = np.array([0.4, 0.3, 0.2, 0.1])
    
    # With p=0.6: cumsum is [0.4, 0.7]. 0.7 >= 0.6. Side left searchsorted index is 1.
    # Keep prefix indices 0 and 1 (mass 0.7).
    filtered = top_p_filter(probs, p=0.6)
    assert np.isclose(np.sum(filtered), 0.7)
    assert filtered[0] == 0.4
    assert filtered[1] == 0.3
    assert filtered[2] == 0.0
    assert filtered[3] == 0.0


def test_sampling_order_and_entropy():
    vocab = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]
    logits = np.array([5.0, 4.0, 3.0, 2.0, 1.0, 0.5, 0.1, -0.5, -1.0, -2.0])

    rng = np.random.default_rng(42)

    # Greedy check
    token_idx = sample_next_token(logits, temperature=0.0, rng=rng)
    assert token_idx == 0

    # Demo run checking entropy monotonically increases with higher temperature
    demo_results = run_sampling_demo(vocab, logits, temperatures=[0.1, 0.7, 1.5], n_samples=3000, seed=42)
    entropies = [res["entropy"] for res in demo_results]
    assert entropies[0] < entropies[1] < entropies[2]
