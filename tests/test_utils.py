"""Tests for utility functions."""

import pytest

from llm_benchmarks.utils.text import (
    normalize_text,
    extract_json,
    truncate_text,
    count_words,
    extract_code_block,
    extract_numbered_list,
)
from llm_benchmarks.utils.metrics import (
    exact_match,
    contains_match,
    fuzzy_match,
    levenshtein_similarity,
    f1_score,
    rouge_l,
)


class TestTextUtils:
    """Tests for text utilities."""
    
    def test_normalize_text(self):
        """Test text normalization."""
        assert normalize_text("  Hello World  ") == "hello world"
        assert normalize_text("Hello", lowercase=False) == "Hello"
        # strip=False keeps leading/trailing space, but internal whitespace is still normalized
        assert normalize_text("  spaces  ", strip=False) == " spaces "
    
    def test_extract_json(self):
        """Test JSON extraction."""
        # Plain JSON
        assert extract_json('{"key": "value"}') == {"key": "value"}
        
        # JSON in code block
        text = '''```json
{"key": "value"}
```'''
        assert extract_json(text) == {"key": "value"}
        
        # Invalid JSON
        assert extract_json("not json") is None
    
    def test_truncate_text(self):
        """Test text truncation."""
        text = "Hello world, this is a test"
        assert truncate_text(text, 15) == "Hello world,..."
        assert len(truncate_text(text, 10)) <= 10
    
    def test_count_words(self):
        """Test word counting."""
        assert count_words("hello world") == 2
        assert count_words("one") == 1
        assert count_words("") == 0  # Empty string split gives []
    
    def test_extract_code_block(self):
        """Test code block extraction."""
        text = '''```python
def hello():
    pass
```'''
        code = extract_code_block(text, "python")
        assert "def hello():" in code
    
    def test_extract_numbered_list(self):
        """Test numbered list extraction."""
        text = """1. First item
2. Second item
3. Third item"""
        items = extract_numbered_list(text)
        assert len(items) == 3
        assert items[0] == "First item"


class TestMetrics:
    """Tests for evaluation metrics."""
    
    def test_exact_match(self):
        """Test exact matching."""
        assert exact_match("hello", "hello") == 1.0
        assert exact_match("Hello", "hello") == 1.0  # Normalized
        assert exact_match("hello", "world") == 0.0
    
    def test_contains_match(self):
        """Test contains matching."""
        assert contains_match("The answer is Paris", "Paris") == 1.0
        assert contains_match("The answer is Paris", ["London", "Paris"]) == 1.0
        assert contains_match("The answer is Berlin", "Paris") == 0.0
    
    def test_fuzzy_match(self):
        """Test fuzzy matching."""
        assert fuzzy_match("hello", "hello") == 1.0
        assert fuzzy_match("hello", "helo") >= 0.8
        assert fuzzy_match("hello", "world") < 0.5
    
    def test_levenshtein_similarity(self):
        """Test Levenshtein similarity."""
        assert levenshtein_similarity("hello", "hello") == 1.0
        assert levenshtein_similarity("", "") == 1.0
        assert levenshtein_similarity("a", "b") == 0.0
        assert levenshtein_similarity("kitten", "sitting") > 0.5
    
    def test_f1_score(self):
        """Test F1 score calculation."""
        assert f1_score({"a", "b"}, {"a", "b"}) == 1.0
        assert f1_score(set(), set()) == 1.0
        assert f1_score({"a"}, {"b"}) == 0.0
    
    def test_rouge_l(self):
        """Test ROUGE-L score."""
        assert rouge_l("hello world", "hello world") == 1.0
        assert rouge_l("the cat sat", "the cat sat on the mat") > 0.5
        assert rouge_l("", "hello") == 0.0
