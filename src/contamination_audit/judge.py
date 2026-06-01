"""LLM-as-judge (Stage 3 of the detection pipeline).

A single ``Judge`` class handles both the default and strict prompts used in
the paper, with a layered JSON parser that survives the typical Open AI failure
modes: markdown fences, unescaped LaTeX backslashes, and trailing prose.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from .config import JudgeConfig

_log = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).parent / "prompts"

VALID_CLASSES = ("CONTAMINATED", "RELATED", "CLEAN")
VALID_CONFIDENCES = ("HIGH", "MEDIUM", "LOW")


@dataclass
class Judgment:
    classification: str  # CONTAMINATED | RELATED | CLEAN | ERROR
    confidence: str
    reasoning: str
    shared_insight: str | None

    def to_dict(self) -> dict:
        return {
            "classification": self.classification,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "shared_insight": self.shared_insight,
        }


def load_prompt(name: str = "judge_default") -> str:
    """Load a prompt template by file stem (e.g. ``judge_default``)."""
    path = PROMPTS_DIR / f"{name}.txt"
    return path.read_text(encoding="utf-8")


def parse_response(raw: str) -> Judgment | None:
    """Layered JSON parse with regex fallbacks. Returns ``None`` if all attempts fail."""
    cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip()

    try:
        payload = json.loads(cleaned)
        return _build_judgment(payload)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        candidate = re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", match.group())
        try:
            payload = json.loads(candidate)
            return _build_judgment(payload)
        except json.JSONDecodeError:
            pass

    cls = re.search(r'"classification"\s*:\s*"(CONTAMINATED|RELATED|CLEAN)"', raw, re.IGNORECASE)
    if cls:
        conf = re.search(r'"confidence"\s*:\s*"(HIGH|MEDIUM|LOW)"', raw, re.IGNORECASE)
        reasoning = re.search(r'"reasoning"\s*:\s*"([^"]*)"', raw)
        return Judgment(
            classification=cls.group(1).upper(),
            confidence=conf.group(1).upper() if conf else "LOW",
            reasoning=reasoning.group(1) if reasoning else "extracted via regex",
            shared_insight=None,
        )
    return None


def _build_judgment(payload: dict) -> Judgment:
    cls = str(payload.get("classification", "ERROR")).upper()
    if cls not in VALID_CLASSES:
        cls = "ERROR"
    conf = str(payload.get("confidence", "LOW")).upper()
    if conf not in VALID_CONFIDENCES:
        conf = "LOW"
    return Judgment(
        classification=cls,
        confidence=conf,
        reasoning=str(payload.get("reasoning", "")),
        shared_insight=payload.get("shared_insight"),
    )


class Judge:
    """Wraps the OpenAI client with retry + structured parsing."""

    def __init__(self, config: JudgeConfig, prompt_name: str = "judge_default"):
        load_dotenv()
        from openai import OpenAI  # local import

        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY not set. Add it to .env or export it in your shell."
            )
        self._client = OpenAI(api_key=api_key)
        self.config = config
        self._template = load_prompt(prompt_name)

    def judge(self, train_problem: str, math500_problem: str) -> Judgment:
        prompt = self._template.format(
            train_problem=train_problem[: self.config.truncate_chars],
            math500_problem=math500_problem[: self.config.truncate_chars],
        )

        last_err = "unknown"
        for attempt in range(self.config.max_retries):
            try:
                resp = self._client.chat.completions.create(
                    model=self.config.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    timeout=self.config.timeout_seconds,
                )
                raw = resp.choices[0].message.content or ""
                parsed = parse_response(raw)
                if parsed:
                    return parsed
                last_err = f"no parseable JSON in: {raw[:100]}"
            except Exception as e:  # network / API errors
                last_err = str(e)
                if attempt < self.config.max_retries - 1:
                    time.sleep(2 ** attempt)

        _log.warning("judge failed after %d attempts: %s", self.config.max_retries, last_err)
        return Judgment(
            classification="ERROR",
            confidence="LOW",
            reasoning=last_err[:200],
            shared_insight=None,
        )
