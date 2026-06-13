"""LLM-as-judge (Stage 3 of the detection pipeline).

Supports multiple judge providers (paper §5.1 used GPT-4o-mini for s1 and
Gemini 2.5 Flash for Tülu 3):

  - ``openai``        — GPT-4o-mini / GPT-4o (default)
  - ``google``        — Gemini via Google AI Studio (free tier)
  - ``vertex``        — Gemini via Vertex AI (GCP billing)
  - ``openai_compat`` — any OpenAI-compatible endpoint (Groq, Cerebras,
                        DeepInfra, NVIDIA NIM — see configs/models.yaml)

Three prompt templates ship with the package:
  - ``judge_default``  — CONTAMINATED / RELATED / CLEAN  (paper s1 + OpenThoughts)
  - ``judge_strict``   — step-for-step CONTAMINATED  (paper Tulu re-judge)
  - ``judge_tulu``     — instance / template / clean  (teammate Tulu pipeline)
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

from .config import JudgeConfig

_log = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).parent / "prompts"

VALID_CLASSES = (
    "CONTAMINATED", "RELATED", "CLEAN",
    "INSTANCE_CONTAMINATED", "TEMPLATE_CONTAMINATED",
)
VALID_CONFIDENCES = ("HIGH", "MEDIUM", "LOW")

Provider = Literal["openai", "google", "vertex", "openai_compat"]


@dataclass
class Judgment:
    classification: str  # one of VALID_CLASSES, or ERROR
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
        return _build_judgment(json.loads(cleaned))
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        # Fix bare LaTeX backslashes that aren't valid JSON escapes.
        candidate = re.sub(r'\\(?!["\\/bfnrtu0-9])', r"\\\\", match.group())
        try:
            return _build_judgment(json.loads(candidate))
        except json.JSONDecodeError:
            pass

    cls_pattern = "|".join(VALID_CLASSES) + "|instance_contaminated|template_contaminated|clean"
    cls = re.search(rf'"(?:classification|label)"\s*:\s*"({cls_pattern})"', raw, re.IGNORECASE)
    if cls:
        conf = re.search(r'"confidence"\s*:\s*"(HIGH|MEDIUM|LOW)"', raw, re.IGNORECASE)
        reasoning = re.search(r'"reasoning"\s*:\s*"([^"]*)"', raw)
        return Judgment(
            classification=_normalize_class(cls.group(1)),
            confidence=conf.group(1).upper() if conf else "LOW",
            reasoning=reasoning.group(1) if reasoning else "extracted via regex",
            shared_insight=None,
        )
    return None


def _normalize_class(value: str) -> str:
    """Accept either ``label`` or ``classification`` field; normalize casing."""
    upper = value.upper()
    if upper in VALID_CLASSES:
        return upper
    return "ERROR"


def _build_judgment(payload: dict) -> Judgment:
    # Teammate Tulu pipeline uses "label" instead of "classification"; accept both.
    raw_cls = payload.get("classification", payload.get("label", "ERROR"))
    cls = _normalize_class(str(raw_cls))
    conf = str(payload.get("confidence", "LOW")).upper()
    if conf not in VALID_CONFIDENCES:
        conf = "LOW"
    return Judgment(
        classification=cls,
        confidence=conf,
        reasoning=str(payload.get("reasoning", "")),
        shared_insight=payload.get("shared_insight"),
    )


# ── Provider backends ────────────────────────────────────────────────────────


class _OpenAIBackend:
    """Standard OpenAI chat completions API (default for paper §5.1 s1/OT)."""

    def __init__(self, model: str, timeout: int, max_tokens: int, temperature: float):
        from openai import OpenAI
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set")
        self._client = OpenAI(api_key=api_key)
        self.model, self.timeout, self.max_tokens, self.temperature = (
            model, timeout, max_tokens, temperature,
        )

    def call(self, prompt: str) -> str:
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout=self.timeout,
        )
        return resp.choices[0].message.content or ""


class _OpenAICompatBackend(_OpenAIBackend):
    """Groq / Cerebras / DeepInfra / NVIDIA / any OpenAI-compatible base_url."""

    def __init__(self, model: str, base_url: str, api_key_env: str,
                 timeout: int, max_tokens: int, temperature: float):
        from openai import OpenAI
        api_key = os.environ.get(api_key_env, "")
        if not api_key:
            raise RuntimeError(f"{api_key_env} not set")
        self._client = OpenAI(base_url=base_url, api_key=api_key)
        self.model, self.timeout, self.max_tokens, self.temperature = (
            model, timeout, max_tokens, temperature,
        )


class _GoogleAIBackend:
    """Google AI Studio (Gemini, free tier — paper Tulu pipeline used 2.5 Flash)."""

    def __init__(self, model: str, max_tokens: int, temperature: float):
        from google import genai
        from google.genai import types as gtypes
        api_key = os.environ.get("GOOGLE_API_KEY", "")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY not set")
        self._client = genai.Client(api_key=api_key)
        self._types = gtypes
        self.model, self.max_tokens, self.temperature = model, max_tokens, temperature

    def call(self, prompt: str) -> str:
        resp = self._client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=self._types.GenerateContentConfig(
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
            ),
        )
        return (resp.text or "").strip()


class _VertexBackend:
    """Vertex AI (Gemini via GCP — paper Tulu used 2.5 Flash here for production)."""

    def __init__(self, model: str, max_tokens: int, temperature: float,
                 project_env: str = "GOOGLE_CLOUD_PROJECT",
                 location_env: str = "GOOGLE_CLOUD_LOCATION"):
        from google import genai
        from google.genai import types as gtypes
        project = os.environ.get(project_env)
        location = os.environ.get(location_env, "us-central1")
        if not project:
            raise RuntimeError(f"{project_env} not set")
        self._client = genai.Client(vertexai=True, project=project, location=location)
        self._types = gtypes
        self.model, self.max_tokens, self.temperature = model, max_tokens, temperature

    def call(self, prompt: str) -> str:
        resp = self._client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=self._types.GenerateContentConfig(
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
                # Disable Gemini 2.5 thinking — produces reliable JSON faster.
                thinking_config=self._types.ThinkingConfig(thinking_budget=0),
            ),
        )
        return (resp.text or "").strip()


class Judge:
    """Provider-agnostic LLM judge with retry + structured parsing.

    The provider is selected from ``config.provider`` (defaults to ``openai``).
    Prompts live in ``src/contamination_audit/prompts/`` and are loaded by name.
    """

    def __init__(self, config: JudgeConfig, prompt_name: str = "judge_default"):
        load_dotenv()
        self.config = config
        self._template = load_prompt(prompt_name)
        self._backend = self._make_backend(config)

    @staticmethod
    def _make_backend(config: JudgeConfig):
        provider = getattr(config, "provider", "openai")
        kwargs = dict(model=config.model, max_tokens=config.max_tokens,
                      temperature=config.temperature)
        if provider == "openai":
            return _OpenAIBackend(timeout=config.timeout_seconds, **kwargs)
        if provider == "google":
            return _GoogleAIBackend(**kwargs)
        if provider == "vertex":
            return _VertexBackend(**kwargs)
        if provider == "openai_compat":
            base_url = getattr(config, "base_url", "")
            api_key_env = getattr(config, "api_key_env", "OPENAI_API_KEY")
            if not base_url:
                raise ValueError("openai_compat provider requires config.base_url")
            return _OpenAICompatBackend(base_url=base_url, api_key_env=api_key_env,
                                        timeout=config.timeout_seconds, **kwargs)
        raise ValueError(f"Unknown judge provider: {provider!r}")

    def judge(self, train_problem: str, math500_problem: str) -> Judgment:
        prompt = self._template.format(
            train_problem=train_problem[: self.config.truncate_chars],
            math500_problem=math500_problem[: self.config.truncate_chars],
        )

        last_err = "unknown"
        for attempt in range(self.config.max_retries):
            try:
                raw = self._backend.call(prompt)
                parsed = parse_response(raw)
                if parsed:
                    return parsed
                last_err = f"no parseable JSON in: {raw[:100]}"
            except Exception as e:  # noqa: BLE001 — network / API errors
                last_err = str(e)
                if attempt < self.config.max_retries - 1:
                    sleep_for = _retry_after_seconds(last_err) or 2 ** attempt
                    time.sleep(sleep_for)

        _log.warning("judge failed after %d attempts: %s", self.config.max_retries, last_err)
        return Judgment(
            classification="ERROR",
            confidence="LOW",
            reasoning=last_err[:200],
            shared_insight=None,
        )


def _retry_after_seconds(error_message: str) -> float | None:
    """Extract a retry-after hint from common provider error strings."""
    if "429" not in error_message and "quota" not in error_message.lower():
        return None
    match = re.search(r"try again in (?:(\d+)m)?(?:([\d.]+)s)?", error_message)
    if not match:
        return 60.0
    return float(match.group(1) or 0) * 60 + float(match.group(2) or 0) + 5
