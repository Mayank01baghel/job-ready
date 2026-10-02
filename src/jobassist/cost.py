"""Ex8: Cost estimation & prompt caching."""

from dataclasses import dataclass


@dataclass
class TokenPricing:
    input_price_per_m: float
    output_price_per_m: float
    cached_input_discount: float = 0.50  # 50% discount for cached input
    batch_discount: float = 0.0          # Additional discount for async batch processing
    cache_write_premium: float = 1.0     # Multiplier for first write to cache (1.0 = no premium)


@dataclass
class ProcessingJob:
    n_documents: int
    static_prefix_tokens: int
    variable_input_tokens: int
    expected_output_tokens: int
    use_caching: bool = False
    use_batching: bool = False


def estimate_cost(job: ProcessingJob, pricing: TokenPricing) -> dict[str, float]:
    """Estimate cost modeling static prefix caching and batching discounts.
    
    Caching math: Applies only to the static prefix.
    Doc 1: Pays full price (or cache_write_premium) on prefix + full price on variable.
    Docs 2..N: Pays cached price on prefix + full price on variable.
    """
    total_output_tokens = job.n_documents * job.expected_output_tokens
    
    if job.use_caching and job.n_documents > 0:
        # Doc 1: full prefix (potentially with premium) + full variable
        first_doc_input = (job.static_prefix_tokens * pricing.cache_write_premium) + job.variable_input_tokens
        
        # Docs 2..N: discounted prefix + full variable
        subsequent_docs_input = 0.0
        if job.n_documents > 1:
            discounted_prefix = job.static_prefix_tokens * (1.0 - pricing.cached_input_discount)
            subsequent_docs_input = (discounted_prefix + job.variable_input_tokens) * (job.n_documents - 1)
            
        total_input_cost = ((first_doc_input + subsequent_docs_input) / 1_000_000.0) * pricing.input_price_per_m
    else:
        # No caching: full price on all tokens
        total_input_tokens = job.n_documents * (job.static_prefix_tokens + job.variable_input_tokens)
        total_input_cost = (total_input_tokens / 1_000_000.0) * pricing.input_price_per_m

    # Output cost
    total_output_cost = (total_output_tokens / 1_000_000.0) * pricing.output_price_per_m
    
    base_total = total_input_cost + total_output_cost
    
    # Apply batch discount (applies to total cost, no token reduction)
    if job.use_batching:
        final_cost = base_total * (1.0 - pricing.batch_discount)
    else:
        final_cost = base_total
        
    return {
        "input_cost": round(total_input_cost, 6),
        "output_cost": round(total_output_cost, 6),
        "total_cost": round(final_cost, 6)
    }


def generate_cost_comparison_table(
    pricing: TokenPricing, 
    static_tokens: int = 1500, 
    variable_tokens: int = 500,
    output_tokens: int = 300,
    n_values: list[int] | None = None
) -> list[dict[str, float]]:
    """Generate cost comparisons for N=100, 1000, 10000 across strategies."""
    if n_values is None:
        n_values = [100, 1000, 10000]
        
    results = []
    
    for n in n_values:
        # 1. No Cache, No Batch
        job_base = ProcessingJob(n, static_tokens, variable_tokens, output_tokens, use_caching=False, use_batching=False)
        cost_base = estimate_cost(job_base, pricing)["total_cost"]
        
        # 2. Cache, No Batch
        job_cache = ProcessingJob(n, static_tokens, variable_tokens, output_tokens, use_caching=True, use_batching=False)
        cost_cache = estimate_cost(job_cache, pricing)["total_cost"]
        
        # 3. Cache and Batch
        job_cache_batch = ProcessingJob(n, static_tokens, variable_tokens, output_tokens, use_caching=True, use_batching=True)
        cost_cache_batch = estimate_cost(job_cache_batch, pricing)["total_cost"]
        
        results.append({
            "N": n,
            "no_cache": cost_base,
            "cache": cost_cache,
            "cache_batch": cost_cache_batch,
            "savings_pct": round(((cost_base - cost_cache_batch) / cost_base) * 100, 2)
        })
        
    return results
