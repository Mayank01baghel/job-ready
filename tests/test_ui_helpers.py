"""Tests for UI helpers."""

import os
import sys
from unittest import mock

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import pytest

from app.ui_helpers import extract_text_from_bytes, run_analysis
from jobassist.tiers import load_tiers_config


def test_extract_text_from_txt():
    content = b"Hello world."
    text = extract_text_from_bytes(content, "test.txt")
    assert text == "Hello world."


def test_extract_text_fallback():
    content = b"Hello world."
    with pytest.raises(ValueError, match="Unsupported file format"):
        extract_text_from_bytes(content, "test.unknown")


@mock.patch("pypdf.PdfReader")
def test_extract_text_from_pdf(mock_pdf_reader):
    mock_page = mock.Mock()
    mock_page.extract_text.return_value = "PDF text."
    mock_pdf_reader.return_value.pages = [mock_page]
    
    text = extract_text_from_bytes(b"fake_pdf_bytes", "test.pdf")
    assert text == "PDF text."


@mock.patch("docx.Document")
def test_extract_text_from_docx(mock_docx):
    mock_paragraph = mock.Mock()
    mock_paragraph.text = "DOCX text."
    mock_docx.return_value.paragraphs = [mock_paragraph]
    
    text = extract_text_from_bytes(b"fake_docx_bytes", "test.docx")
    assert text == "DOCX text."


def test_run_analysis_mock_backend():
    tiers = load_tiers_config()
    tier = tiers["small"]
    
    # We use very simple text that the MockLLM handles by default
    resume = "Sarah Chen\nSkills: TypeScript, JavaScript, React, Node.js, PostgreSQL, Docker, AWS"
    jd = "Requirement: 5+ years of experience with modern web application development"
    
    result = run_analysis(resume_text=resume, jd_text=jd, backend="mock", tier=tier)
    
    assert result["success"] is True
    assert "assessment" in result
    assert result["assessment"].candidate_name == "Sarah Chen"
    assert "grounding_report" in result
    assert "cost_info" in result
    assert result["cost_info"]["cost_no_cache"] > 0
