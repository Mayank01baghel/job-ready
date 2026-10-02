"""Ex6: Model-tier selection & dynamic routing."""

import os
from dataclasses import dataclass
from typing import Any

import yaml


@dataclass
class ModelTier:
    name: str
    input_price_per_m: float
    output_price_per_m: float
    cached_input_discount: float
    batch_discount: float
    cache_write_premium: float
    latency_ms_per_doc: int
    max_context_window: int
    description: str


@dataclass
class TaskRequirements:
    task_type: str  # e.g., "bulk_resume_classification", "complex_technical_feedback", "high_volume_faq"
    is_bulk: bool
    requires_complex_reasoning: bool
    requires_strict_schema: bool
    expected_context_size: int


def load_tiers_config(config_path: str = "config/pricing.yaml") -> dict[str, ModelTier]:
    """Load model tier specifications from pricing.yaml."""
    if not os.path.exists(config_path):
        # Fallback if running from a different directory (e.g. tests)
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config", "pricing.yaml")
        
    if not os.path.exists(config_path):
        # Hard fallback for isolated testing
        return {
            "small": ModelTier("Tier-Small", 0.15, 0.60, 0.5, 0.5, 1.0, 200, 128000, "Small tier"),
            "medium": ModelTier("Tier-Medium", 0.80, 3.20, 0.5, 0.5, 1.0, 600, 128000, "Medium tier"),
            "large": ModelTier("Tier-Large", 3.00, 12.00, 0.5, 0.5, 1.0, 1800, 200000, "Large tier"),
        }
        
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
        
    tiers = {}
    for key, spec in data.get("tiers", {}).items():
        tiers[key] = ModelTier(**spec)
    return tiers


def select_tier(req: TaskRequirements, tiers: dict[str, ModelTier] | None = None) -> tuple[ModelTier, list[str]]:
    """Select the optimal model tier based on task requirements and return justifications."""
    if tiers is None:
        tiers = load_tiers_config()
        
    reasons = []
    
    # 1. Complex technical interview feedback -> large tier
    if req.task_type == "complex_technical_feedback" or req.requires_complex_reasoning:
        reasons.append("Task requires complex reasoning and deep multi-step synthesis.")
        # Check context window
        if req.expected_context_size > tiers["medium"].max_context_window:
            reasons.append(f"Expected context size ({req.expected_context_size}) exceeds medium tier limits.")
        return tiers["large"], reasons
        
    # 2. Bulk resume classification -> small tier
    if req.task_type == "bulk_resume_classification":
        reasons.append("Bulk classification task requires high throughput and low cost.")
        if req.expected_context_size <= tiers["small"].max_context_window:
            reasons.append("Context fits well within the small tier limits.")
            return tiers["small"], reasons
        else:
            reasons.append("Context exceeds small tier limits, upgrading to medium.")
            return tiers["medium"], reasons
            
    # 3. High-volume FAQ -> small tier
    if req.task_type == "high_volume_faq":
        reasons.append("High-volume FAQ task prioritizes fast latency and low cost.")
        return tiers["small"], reasons
        
    # Fallback default
    if req.requires_strict_schema and not req.is_bulk:
        reasons.append("Standard task with strict schema requirements maps to medium tier.")
        return tiers["medium"], reasons
        
    reasons.append("Defaulting to small tier for standard routine tasks.")
    return tiers["small"], reasons


def simulate_dynamic_routing(total_queries: int = 1000) -> dict[str, Any]:
    """Simulate router escalating from small -> medium -> large based on confidence/schema failure.
    
    Target mix: 80% small, 15% medium, 5% large.
    """
    tiers = load_tiers_config()
    
    # Simulate counts
    n_small = int(total_queries * 0.80)
    n_medium = int(total_queries * 0.15)
    n_large = total_queries - n_small - n_medium
    
    # Assume average tokens: 500 in, 200 out
    avg_in = 500
    avg_out = 200
    
    def cost(tier: ModelTier, n: int) -> float:
        in_cost = (avg_in * n / 1_000_000) * tier.input_price_per_m
        out_cost = (avg_out * n / 1_000_000) * tier.output_price_per_m
        return in_cost + out_cost
        
    cost_small = cost(tiers["small"], n_small)
    # The escalated tasks STILL paid for the failed small/medium attempts
    # For medium, they failed small first
    cost_medium_attempts = cost(tiers["small"], n_medium)
    cost_medium_success = cost(tiers["medium"], n_medium)
    
    # For large, they failed small AND medium first
    cost_large_attempts_1 = cost(tiers["small"], n_large)
    cost_large_attempts_2 = cost(tiers["medium"], n_large)
    cost_large_success = cost(tiers["large"], n_large)
    
    blended_total_cost = (
        cost_small + 
        cost_medium_attempts + cost_medium_success + 
        cost_large_attempts_1 + cost_large_attempts_2 + cost_large_success
    )
    
    # Baseline: route everything to large
    all_large_cost = cost(tiers["large"], total_queries)
    
    savings = all_large_cost - blended_total_cost
    savings_pct = (savings / all_large_cost) * 100 if all_large_cost > 0 else 0
    
    return {
        "total_queries": total_queries,
        "distribution": {"small": n_small, "medium": n_medium, "large": n_large},
        "blended_cost": round(blended_total_cost, 4),
        "all_large_baseline_cost": round(all_large_cost, 4),
        "savings_usd": round(savings, 4),
        "savings_percent": round(savings_pct, 2)
    }
