"""Tests for Exercise 9: Structured output validation."""

from jobassist.schema import validate_json_output

# The 6 mandatory test cases

def test_valid_json():
    raw = """
    {
      "candidate_name": "Jane Doe",
      "skills": ["Python"],
      "experience_years": 4,
      "score": 80,
      "evidence": ["Has 4 years experience building backends"],
      "requirement_assessments": [],
      "missing_information": [],
      "recommendation": "Yes",
      "reasoning": "Solid fit."
    }
    """
    res = validate_json_output(raw)
    assert res.success is True
    assert res.data.candidate_name == "Jane Doe"


def test_score_150_rejected():
    raw = """
    {
      "candidate_name": "Jane Doe",
      "skills": ["Python"],
      "experience_years": 4,
      "score": 150,
      "evidence": ["Has 4 years experience building backends"],
      "requirement_assessments": [],
      "missing_information": [],
      "recommendation": "Yes",
      "reasoning": "Solid fit."
    }
    """
    res = validate_json_output(raw)
    assert res.success is False
    assert res.error_type == "schema_validation"
    assert any("score" in str(e["loc"]) for e in res.errors)


def test_malformed_json_rejected():
    raw = """{ "candidate_name": "Jane Doe", "skills": ["Python", ] """
    res = validate_json_output(raw)
    assert res.success is False
    assert res.error_type == "malformed_json"


def test_missing_score_rejected():
    raw = """
    {
      "candidate_name": "Jane Doe",
      "skills": ["Python"],
      "experience_years": 4,
      "evidence": ["Has 4 years experience building backends"],
      "requirement_assessments": [],
      "missing_information": [],
      "recommendation": "Yes",
      "reasoning": "Solid fit."
    }
    """
    res = validate_json_output(raw)
    assert res.success is False
    assert res.error_type == "schema_validation"
    assert any("score" in str(e["loc"]) for e in res.errors)


def test_empty_skills_rejected():
    raw = """
    {
      "candidate_name": "Jane Doe",
      "skills": [],
      "experience_years": 4,
      "score": 80,
      "evidence": ["Has 4 years experience building backends"],
      "requirement_assessments": [],
      "missing_information": [],
      "recommendation": "Yes",
      "reasoning": "Solid fit."
    }
    """
    res = validate_json_output(raw)
    assert res.success is False
    assert res.error_type == "schema_validation"
    assert any("skills" in str(e["loc"]) for e in res.errors)


def test_valid_with_code_fences_accepted():
    raw = """```json
    {
      "candidate_name": "Jane Doe",
      "skills": ["Python"],
      "experience_years": 4,
      "score": 80,
      "evidence": ["Has 4 years experience building backends"],
      "requirement_assessments": [],
      "missing_information": [],
      "recommendation": "Yes",
      "reasoning": "Solid fit."
    }
    ```"""
    res = validate_json_output(raw)
    assert res.success is True
    assert res.data.candidate_name == "Jane Doe"
