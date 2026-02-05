"""Text processing utilities."""

import json
import re
from typing import Any, Optional


def normalize_text(text: str, lowercase: bool = True, strip: bool = True) -> str:
    """
    Normalize text for comparison.
    
    Args:
        text: Input text
        lowercase: Convert to lowercase
        strip: Strip leading/trailing whitespace
        
    Returns:
        Normalized text
    """
    if strip:
        text = text.strip()
    if lowercase:
        text = text.lower()
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)
    return text


def extract_json(text: str) -> Optional[dict[str, Any]]:
    """
    Extract JSON from text, handling markdown code blocks.
    
    Args:
        text: Text possibly containing JSON
        
    Returns:
        Parsed JSON dict or None if no valid JSON found
    """
    # Try to find JSON in code blocks first
    patterns = [
        r"```json\s*([\s\S]*?)\s*```",
        r"```\s*([\s\S]*?)\s*```",
        r"\{[\s\S]*\}",
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            try:
                # Handle both string match and full text
                json_str = match if isinstance(match, str) else text
                return json.loads(json_str)
            except json.JSONDecodeError:
                continue
    
    # Try parsing the entire text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def extract_code_block(text: str, language: Optional[str] = None) -> Optional[str]:
    """
    Extract code from a markdown code block.
    
    Args:
        text: Text containing code blocks
        language: Optional language specifier to match
        
    Returns:
        Code content or None if not found
    """
    if language:
        pattern = rf"```{language}\s*([\s\S]*?)\s*```"
    else:
        pattern = r"```(?:\w+)?\s*([\s\S]*?)\s*```"
    
    match = re.search(pattern, text)
    if match:
        return match.group(1).strip()
    return None


def truncate_text(
    text: str,
    max_length: int,
    suffix: str = "...",
    word_boundary: bool = True,
) -> str:
    """
    Truncate text to a maximum length.
    
    Args:
        text: Input text
        max_length: Maximum length (including suffix)
        suffix: Suffix to add if truncated
        word_boundary: Try to break at word boundary
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    
    target_length = max_length - len(suffix)
    
    if word_boundary:
        # Find last space before target length
        truncated = text[:target_length]
        last_space = truncated.rfind(" ")
        if last_space > target_length * 0.5:  # Only if reasonable
            truncated = truncated[:last_space]
        return truncated + suffix
    
    return text[:target_length] + suffix


def count_words(text: str) -> int:
    """Count the number of words in text."""
    return len(text.split())


def count_sentences(text: str) -> int:
    """Count the number of sentences in text."""
    # Simple sentence counting based on common terminators
    sentences = re.split(r"[.!?]+", text)
    return len([s for s in sentences if s.strip()])


def extract_numbered_list(text: str) -> list[str]:
    """
    Extract items from a numbered list.
    
    Args:
        text: Text containing a numbered list
        
    Returns:
        List of extracted items
    """
    pattern = r"^\s*\d+[\.\)]\s*(.+)$"
    items = []
    
    for line in text.split("\n"):
        match = re.match(pattern, line)
        if match:
            items.append(match.group(1).strip())
    
    return items


def extract_bullet_list(text: str) -> list[str]:
    """
    Extract items from a bullet list.
    
    Args:
        text: Text containing a bullet list
        
    Returns:
        List of extracted items
    """
    pattern = r"^\s*[-*•]\s*(.+)$"
    items = []
    
    for line in text.split("\n"):
        match = re.match(pattern, line)
        if match:
            items.append(match.group(1).strip())
    
    return items
