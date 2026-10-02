"""Tests for Exercise 3: Attention & context window."""

import numpy as np

from jobassist.attention import (
    create_prompt,
    demonstrate_positional_sensitivity,
    get_attention_explanation,
    get_sinusoidal_positional_encoding,
    scaled_dot_product_attention,
)


def test_create_prompt():
    p = create_prompt(
        system="System role",
        jd="Senior Fullstack JD",
        resume="Sarah Chen resume",
        question="Does candidate know Python?",
    )
    assert "<system>System role</system>" in p["full_prompt"]
    assert "<job_description>Senior Fullstack JD</job_description>" in p["full_prompt"]
    assert "<resume>Sarah Chen resume</resume>" in p["full_prompt"]
    assert "<question>Does candidate know Python?</question>" in p["full_prompt"]


def test_scaled_dot_product_attention_causal_mask():
    seq_len = 4
    d_k = 8
    rng = np.random.default_rng(42)

    Q = rng.standard_normal((seq_len, d_k))
    K = rng.standard_normal((seq_len, d_k))
    V = rng.standard_normal((seq_len, d_k))

    _, attn_weights = scaled_dot_product_attention(Q, K, V, causal=True)

    # 1. Attention rows sum to 1
    row_sums = np.sum(attn_weights, axis=-1)
    assert np.allclose(row_sums, 1.0)

    # 2. Upper triangular elements (future tokens) are zero
    upper_tri = np.triu(attn_weights, k=1)
    assert np.allclose(upper_tri, 0.0)

    # 3. First token attends only to itself
    assert np.isclose(attn_weights[0, 0], 1.0)


def test_sinusoidal_positional_encoding_shape():
    pe = get_sinusoidal_positional_encoding(seq_len=10, d_model=16)
    assert pe.shape == (10, 16)


def test_positional_sensitivity_demo():
    demo = demonstrate_positional_sensitivity()

    # Without positional encoding, token 'bites' (index 1) has identical output representation
    assert demo["bites_equal_without_pos"] is True or np.bool_(demo["bites_equal_without_pos"])

    # With positional encoding, token 'bites' (index 1) output representation differs because context positions changed
    assert demo["bites_equal_with_pos"] is False or not np.bool_(demo["bites_equal_with_pos"])


def test_explanation_string():
    exp = get_attention_explanation()
    assert "O(N^2)" in exp
    assert "causal" in exp.lower()
