"""LLM judge response parsing.

Exercises the three layers of ``parse_response``:
  1. Direct JSON parse.
  2. Markdown-fenced JSON / single ``{...}`` block extraction.
  3. Regex fallback on the ``classification`` field alone.

These tests block the most common silent-failure mode: a malformed judge
response getting silently labelled as ERROR and disappearing from C_sem.
"""

from contamination_audit.judge import parse_response


def test_valid_json_parses():
    raw = (
        '{"classification":"CONTAMINATED",'
        '"confidence":"HIGH",'
        '"reasoning":"same arctan identity used",'
        '"shared_insight":"arctan addition"}'
    )
    result = parse_response(raw)
    assert result is not None
    assert result.classification == "CONTAMINATED"
    assert result.confidence == "HIGH"
    assert result.shared_insight == "arctan addition"


def test_markdown_fenced_json():
    raw = (
        "```json\n"
        '{"classification":"RELATED","confidence":"MEDIUM",'
        '"reasoning":"shares topic","shared_insight":null}\n'
        "```"
    )
    result = parse_response(raw)
    assert result is not None
    assert result.classification == "RELATED"


def test_unknown_classification_normalizes_to_error():
    raw = '{"classification":"FOOBAR","confidence":"HIGH","reasoning":"bad label","shared_insight":null}'
    result = parse_response(raw)
    assert result is not None
    assert result.classification == "ERROR"


def test_regex_fallback_on_garbage_prose():
    """Even with trailing prose, the regex layer extracts a usable classification."""
    raw = (
        'Sure, here is my response:\n'
        '"classification": "CLEAN" — the problems share only superficial vocabulary.'
    )
    result = parse_response(raw)
    assert result is not None
    assert result.classification == "CLEAN"
    assert result.confidence == "LOW"


def test_unparseable_returns_none():
    """Truly empty / garbage strings return None so the caller can retry."""
    assert parse_response("") is None
    assert parse_response("totally unparseable") is None
