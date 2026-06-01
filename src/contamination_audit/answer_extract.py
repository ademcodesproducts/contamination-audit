"""Final-answer extraction from chain-of-thought traces.

A trace's ``\\boxed{...}`` is the canonical model answer per the paper. We fall
back to common natural-language patterns when no boxed expression is present.
The optional LLM extractor is used by ``scripts/08_score_answers.py`` when both
regexes fail.
"""

from __future__ import annotations

import re


BOXED_PATTERN = re.compile(r"\\boxed\{([^{}]+)\}")

NATURAL_LANGUAGE_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"(?:the\s+)?(?:answer|value|result)\s+is[:\s]+([^\n\.]+)",
        r"(?:therefore|thus|so)[,\s]+(?:the\s+)?(?:answer|value|result)\s+is[:\s]+([^\n\.]+)",
    )
]


def extract_boxed(text: str) -> str | None:
    """Return the contents of the final ``\\boxed{...}``, or ``None`` if absent."""
    matches = BOXED_PATTERN.findall(text)
    return matches[-1].strip() if matches else None


def extract_natural_language(text: str) -> str | None:
    """Fall back to common 'the answer is X' phrasings."""
    for pattern in NATURAL_LANGUAGE_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(1).strip()
    return None


def extract_from_trace(text: str) -> str | None:
    """Best-effort extraction. Prefers ``\\boxed{}``, then natural language."""
    return extract_boxed(text) or extract_natural_language(text)


_EQUIVALENCE_RULES = {
    # Map verbose forms to their canonical numeric form for cheap pre-checks.
    "1/2": "0.5",
    "0.5": "0.5",
    "1/4": "0.25",
    "0.25": "0.25",
    "3/4": "0.75",
    "0.75": "0.75",
}


_FRAC_PATTERN = re.compile(r"\\(?:t?frac|d?frac)\{([^{}]+)\}\{([^{}]+)\}")
_DEGREE_PATTERN = re.compile(r"\^\{?\\circ\}?|°|deg(?:rees)?", re.IGNORECASE)


def normalize(answer: str) -> str:
    """Strip whitespace, LaTeX commands, and degree/percent markers for cheap equality.

    Heavy lifting (e.g. ``3\\sqrt{13} == 3*sqrt(13)``) belongs to the LLM
    equivalence judge in ``judge.py``; this helper just catches the easy wins.
    """
    s = answer.strip()
    s = re.sub(r"\\(?:text|mathrm|operatorname)\{([^}]*)\}", r"\1", s)
    s = _FRAC_PATTERN.sub(r"\1/\2", s)  # \frac{a}{b} -> a/b
    s = _DEGREE_PATTERN.sub("", s)
    s = re.sub(r"\s+", "", s)
    return _EQUIVALENCE_RULES.get(s, s)


def equivalent(a: str | None, b: str | None) -> bool:
    """Cheap structural equivalence; returns False for either-None."""
    if a is None or b is None:
        return False
    return normalize(a) == normalize(b)
