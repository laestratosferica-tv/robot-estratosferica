from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher


def normalize_editorial_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(value or "").casefold())
    without_marks = "".join(
        character
        for character in normalized
        if not unicodedata.combining(character)
    )
    return " ".join(re.findall(r"[a-z0-9]+", without_marks))


def text_is_equivalent(left: str, right: str) -> bool:
    """Detect lexical equivalence without models or external calls."""
    normalized_left = normalize_editorial_text(left)
    normalized_right = normalize_editorial_text(right)
    if not normalized_left or not normalized_right:
        return False
    if normalized_left == normalized_right:
        return True

    left_tokens = normalized_left.split()
    right_tokens = normalized_right.split()
    if min(len(left_tokens), len(right_tokens)) < 4:
        return False

    sequence_similarity = SequenceMatcher(
        None,
        normalized_left,
        normalized_right,
    ).ratio()
    shared_tokens = set(left_tokens) & set(right_tokens)
    token_coverage = len(shared_tokens) / max(
        len(set(left_tokens)),
        len(set(right_tokens)),
    )
    length_ratio = min(len(left_tokens), len(right_tokens)) / max(
        len(left_tokens),
        len(right_tokens),
    )
    return (
        sequence_similarity >= 0.92
        or (token_coverage >= 0.90 and length_ratio >= 0.80)
    )


def substantive_summary_issue(title: str, summary: str) -> str | None:
    if not normalize_editorial_text(summary):
        return "missing_substantive_summary"
    if text_is_equivalent(title, summary):
        return "summary_equivalent_to_title"
    return None
