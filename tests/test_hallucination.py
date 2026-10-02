"""Tests for Exercise 7: Hallucination detection."""

from jobassist.hallucination import (
    BAD_HALLUCINATED_ASSESSMENT,
    GOOD_ASSESSMENT,
    SARAH_CHEN_RESUME,
    check_string_presence,
    verify_grounding,
)


def test_check_string_presence():
    doc = "Architected and implemented high-throughput web APIs using Node.js and PostgreSQL."
    
    # Exact match
    assert check_string_presence("implemented high-throughput web APIs", doc) is True
    
    # Fuzzy match (minor typo or case difference)
    assert check_string_presence("implemented high throughput web apis", doc) is True
    
    # Not found
    assert check_string_presence("Managed Kubernetes clusters", doc) is False


def test_verify_grounding_good_assessment():
    report = verify_grounding(GOOD_ASSESSMENT, SARAH_CHEN_RESUME)
    
    assert report.passed is True
    assert len(report.unsupported_claims) == 0
    assert len(report.missing_evidence) == 0


def test_verify_grounding_bad_assessment():
    report = verify_grounding(BAD_HALLUCINATED_ASSESSMENT, SARAH_CHEN_RESUME)
    
    assert report.passed is False
    
    # Should catch hallucinated skills
    assert any("Kubernetes" in claim for claim in report.unsupported_claims)
    assert any("GraphQL" in claim for claim in report.unsupported_claims)
    
    # Should catch hallucinated evidence
    assert any("Managed Kubernetes clusters" in ev for ev in report.missing_evidence)
    
    # Should catch hallucinated requirement fulfillment
    assert any("Hallucinated meeting requirement" in claim for claim in report.unsupported_claims)
