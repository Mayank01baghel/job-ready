"""Tests for Exercise 8: Cost estimation."""

from jobassist.cost import (
    ProcessingJob,
    TokenPricing,
    estimate_cost,
    generate_cost_comparison_table,
)


def test_estimate_cost_caching():
    pricing = TokenPricing(
        input_price_per_m=2.0, 
        output_price_per_m=0.0, # Ignore output for this test
        cached_input_discount=0.5, # 50% discount
        cache_write_premium=1.0
    )
    
    # N = 1 (No savings from cache since Doc 1 pays full)
    job_1_no_cache = ProcessingJob(1, 1000, 500, 0, use_caching=False)
    job_1_cache = ProcessingJob(1, 1000, 500, 0, use_caching=True)
    
    cost_1_no_cache = estimate_cost(job_1_no_cache, pricing)["total_cost"]
    cost_1_cache = estimate_cost(job_1_cache, pricing)["total_cost"]
    
    assert cost_1_no_cache == cost_1_cache
    
    # N = 10 (Cache should be cheaper)
    job_10_no_cache = ProcessingJob(10, 1000, 500, 0, use_caching=False)
    job_10_cache = ProcessingJob(10, 1000, 500, 0, use_caching=True)
    
    cost_10_no_cache = estimate_cost(job_10_no_cache, pricing)["total_cost"]
    cost_10_cache = estimate_cost(job_10_cache, pricing)["total_cost"]
    
    assert cost_10_cache < cost_10_no_cache


def test_savings_monotonic_with_prefix_share():
    pricing = TokenPricing(input_price_per_m=1.0, output_price_per_m=0.0, cached_input_discount=0.5)
    
    # Low prefix share (100 static, 900 variable)
    job_low = ProcessingJob(100, 100, 900, 0, use_caching=True)
    base_low = estimate_cost(ProcessingJob(100, 100, 900, 0, use_caching=False), pricing)["total_cost"]
    cost_low = estimate_cost(job_low, pricing)["total_cost"]
    savings_low = (base_low - cost_low) / base_low
    
    # High prefix share (900 static, 100 variable)
    job_high = ProcessingJob(100, 900, 100, 0, use_caching=True)
    base_high = estimate_cost(ProcessingJob(100, 900, 100, 0, use_caching=False), pricing)["total_cost"]
    cost_high = estimate_cost(job_high, pricing)["total_cost"]
    savings_high = (base_high - cost_high) / base_high
    
    # Higher static prefix share should yield higher percentage savings
    assert savings_high > savings_low


def test_batching_discount():
    pricing = TokenPricing(input_price_per_m=1.0, output_price_per_m=1.0, batch_discount=0.5)
    
    job_no_batch = ProcessingJob(10, 100, 100, 100, use_batching=False)
    job_batch = ProcessingJob(10, 100, 100, 100, use_batching=True)
    
    cost_no = estimate_cost(job_no_batch, pricing)["total_cost"]
    cost_yes = estimate_cost(job_batch, pricing)["total_cost"]
    
    assert cost_yes == cost_no * 0.5


def test_generate_cost_comparison_table():
    pricing = TokenPricing(input_price_per_m=1.0, output_price_per_m=1.0, batch_discount=0.2, cached_input_discount=0.5)
    results = generate_cost_comparison_table(pricing, n_values=[100])
    
    assert len(results) == 1
    row = results[0]
    assert row["N"] == 100
    assert row["cache"] < row["no_cache"]
    assert row["cache_batch"] < row["cache"]
    assert row["savings_pct"] > 0
