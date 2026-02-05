"""Evaluation metrics for benchmarks."""

import re
from typing import Optional, Sequence


def exact_match(
    response: str,
    expected: str,
    normalize: bool = True,
) -> float:
    """
    Check if response exactly matches expected value.
    
    Args:
        response: Model response
        expected: Expected value
        normalize: Normalize both strings before comparison
        
    Returns:
        1.0 if match, 0.0 otherwise
    """
    if normalize:
        response = response.strip().lower()
        expected = expected.strip().lower()
    
    return 1.0 if response == expected else 0.0


def contains_match(
    response: str,
    expected: str | Sequence[str],
    case_sensitive: bool = False,
) -> float:
    """
    Check if response contains expected value(s).
    
    Args:
        response: Model response
        expected: Expected value or list of values (any match counts)
        case_sensitive: Whether to do case-sensitive matching
        
    Returns:
        1.0 if any expected value is found, 0.0 otherwise
    """
    if not case_sensitive:
        response = response.lower()
    
    if isinstance(expected, str):
        expected = [expected]
    
    for exp in expected:
        check = exp if case_sensitive else exp.lower()
        if check in response:
            return 1.0
    
    return 0.0


def fuzzy_match(
    response: str,
    expected: str,
    threshold: float = 0.8,
) -> float:
    """
    Check if response fuzzy matches expected value using similarity.
    
    Args:
        response: Model response
        expected: Expected value
        threshold: Minimum similarity for a match
        
    Returns:
        Similarity score (0.0 to 1.0)
    """
    similarity = levenshtein_similarity(response.strip(), expected.strip())
    return similarity


def levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate the Levenshtein distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


def levenshtein_similarity(s1: str, s2: str) -> float:
    """
    Calculate similarity between two strings using Levenshtein distance.
    
    Returns:
        Similarity score from 0.0 to 1.0
    """
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    
    distance = levenshtein_distance(s1.lower(), s2.lower())
    max_len = max(len(s1), len(s2))
    return 1.0 - (distance / max_len)


def f1_score(
    predicted: set[str],
    expected: set[str],
) -> float:
    """
    Calculate F1 score between predicted and expected sets.
    
    Args:
        predicted: Set of predicted items
        expected: Set of expected items
        
    Returns:
        F1 score from 0.0 to 1.0
    """
    if not predicted and not expected:
        return 1.0
    if not predicted or not expected:
        return 0.0
    
    intersection = predicted & expected
    precision = len(intersection) / len(predicted)
    recall = len(intersection) / len(expected)
    
    if precision + recall == 0:
        return 0.0
    
    return 2 * (precision * recall) / (precision + recall)


def rouge_l(response: str, reference: str) -> float:
    """
    Calculate ROUGE-L score (longest common subsequence).
    
    Args:
        response: Model response
        reference: Reference text
        
    Returns:
        ROUGE-L F1 score from 0.0 to 1.0
    """
    response_tokens = response.lower().split()
    reference_tokens = reference.lower().split()
    
    if not response_tokens or not reference_tokens:
        return 0.0
    
    # Calculate LCS length
    lcs_length = _lcs_length(response_tokens, reference_tokens)
    
    precision = lcs_length / len(response_tokens)
    recall = lcs_length / len(reference_tokens)
    
    if precision + recall == 0:
        return 0.0
    
    return 2 * (precision * recall) / (precision + recall)


def _lcs_length(x: Sequence[str], y: Sequence[str]) -> int:
    """Calculate the length of the longest common subsequence."""
    m, n = len(x), len(y)
    
    # Create a table to store lengths of LCS
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    
    return dp[m][n]


def bleu_score(
    response: str,
    reference: str,
    max_n: int = 4,
) -> float:
    """
    Calculate a simplified BLEU score.
    
    Args:
        response: Model response
        reference: Reference text
        max_n: Maximum n-gram size
        
    Returns:
        BLEU score from 0.0 to 1.0
    """
    import math
    
    response_tokens = response.lower().split()
    reference_tokens = reference.lower().split()
    
    if not response_tokens:
        return 0.0
    
    # Calculate n-gram precisions
    precisions = []
    
    for n in range(1, max_n + 1):
        response_ngrams = _get_ngrams(response_tokens, n)
        reference_ngrams = _get_ngrams(reference_tokens, n)
        
        if not response_ngrams:
            precisions.append(0.0)
            continue
        
        # Count matches
        matches = sum(
            min(response_ngrams.get(ng, 0), reference_ngrams.get(ng, 0))
            for ng in response_ngrams
        )
        total = sum(response_ngrams.values())
        
        precision = matches / total if total > 0 else 0.0
        precisions.append(precision)
    
    # Calculate geometric mean of precisions
    if all(p > 0 for p in precisions):
        log_precision = sum(math.log(p) for p in precisions) / len(precisions)
        geometric_mean = math.exp(log_precision)
    else:
        geometric_mean = 0.0
    
    # Brevity penalty
    bp = 1.0
    if len(response_tokens) < len(reference_tokens):
        bp = math.exp(1 - len(reference_tokens) / len(response_tokens))
    
    return bp * geometric_mean


def _get_ngrams(tokens: Sequence[str], n: int) -> dict[tuple, int]:
    """Get n-gram counts from tokens."""
    ngrams: dict[tuple, int] = {}
    
    for i in range(len(tokens) - n + 1):
        ngram = tuple(tokens[i : i + n])
        ngrams[ngram] = ngrams.get(ngram, 0) + 1
    
    return ngrams


def answer_relevance(
    response: str,
    question: str,
    keywords: Optional[Sequence[str]] = None,
) -> float:
    """
    Calculate a simple answer relevance score.
    
    Args:
        response: Model response
        question: Original question
        keywords: Optional keywords that should appear in response
        
    Returns:
        Relevance score from 0.0 to 1.0
    """
    if not response.strip():
        return 0.0
    
    score = 0.0
    
    # Check for keyword presence
    if keywords:
        found = sum(1 for kw in keywords if kw.lower() in response.lower())
        score = found / len(keywords)
    else:
        # Use question words as fallback
        question_words = set(question.lower().split())
        response_words = set(response.lower().split())
        
        # Remove common words
        stopwords = {"the", "a", "an", "is", "are", "was", "were", "what", "who", 
                     "where", "when", "why", "how", "do", "does", "did", "to", "of"}
        question_words -= stopwords
        
        if question_words:
            overlap = question_words & response_words
            score = len(overlap) / len(question_words)
    
    return min(score, 1.0)
