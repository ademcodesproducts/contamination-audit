"""Offline tooling for the frozen LLM-judge experiment in results/llm_judge/.

Nothing here calls an API. The script that made the calls is preserved
unchanged as results/llm_judge/run_judge.py.
"""

from .pairs import CandidatePairBuilder, pair_id
from .record import JudgeRecord

__all__ = ["CandidatePairBuilder", "JudgeRecord", "pair_id"]
