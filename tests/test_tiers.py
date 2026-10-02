"""Tests for Exercise 6: Model-tier selection."""

from jobassist.tiers import (
    ModelTier,
    TaskRequirements,
    load_tiers_config,
    select_tier,
    simulate_dynamic_routing,
)


def test_load_tiers_config():
    tiers = load_tiers_config()
    assert "small" in tiers
    assert "medium" in tiers
    assert "large" in tiers
    assert isinstance(tiers["small"], ModelTier)
    assert tiers["small"].input_price_per_m < tiers["large"].input_price_per_m


def test_select_tier():
    tiers = load_tiers_config()
    
    # 1. Bulk resume classification -> small tier
    req1 = TaskRequirements(
        task_type="bulk_resume_classification",
        is_bulk=True,
        requires_complex_reasoning=False,
        requires_strict_schema=True,
        expected_context_size=1000
    )
    tier1, reasons1 = select_tier(req1, tiers)
    assert tier1.name == tiers["small"].name
    assert any("Bulk classification" in r for r in reasons1)
    
    # 2. Complex technical interview feedback -> large tier
    req2 = TaskRequirements(
        task_type="complex_technical_feedback",
        is_bulk=False,
        requires_complex_reasoning=True,
        requires_strict_schema=True,
        expected_context_size=5000
    )
    tier2, reasons2 = select_tier(req2, tiers)
    assert tier2.name == tiers["large"].name
    assert any("complex reasoning" in r for r in reasons2)
    
    # 3. High-volume FAQ -> small tier
    req3 = TaskRequirements(
        task_type="high_volume_faq",
        is_bulk=True,
        requires_complex_reasoning=False,
        requires_strict_schema=False,
        expected_context_size=500
    )
    tier3, reasons3 = select_tier(req3, tiers)
    assert tier3.name == tiers["small"].name
    assert any("High-volume FAQ" in r for r in reasons3)


def test_simulate_dynamic_routing():
    results = simulate_dynamic_routing(total_queries=1000)
    assert results["distribution"]["small"] == 800
    assert results["distribution"]["medium"] == 150
    assert results["distribution"]["large"] == 50
    
    assert results["blended_cost"] < results["all_large_baseline_cost"]
    assert results["savings_percent"] > 0
