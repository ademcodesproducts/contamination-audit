"""Paraphrases of the control items, for planting as disguised contamination.

n-gram filters are built to catch copies. Contamination is not always a copy: a
problem restated in other words carries the same content with few shared
strings. This asks a language model to restate each control item faithfully,
and the planted-recall experiment plants the restatements.

There are two styles, because the instruction shapes the result. "faithful"
tells the model to keep the LaTeX, so long formulas survive and a filter can
match on them. "free" tells it not to copy any formula, which is harder on
every filter and shows how much of the recall on "faithful" came from notation.

Makes paid API calls and needs OPENAI_API_KEY. The output is committed, so the
experiment itself never calls an API.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

from .data import DOLCI, BenchmarkLibrary

log = logging.getLogger(__name__)

PROMPTS = {
    "faithful": """Rewrite the mathematics problem you are given as a different author posing the
same question would. Keep every quantity, condition, and the requested result
exactly the same, so that any correct answer to the original is a correct answer
to yours. Change the wording and the sentence structure as much as you can while
staying faithful. Keep mathematical notation in LaTeX. Reply with the rewritten
problem only.""",
    "free": """Rewrite the mathematics problem you are given as a different author posing the
same question in their own words and notation would. Keep every quantity,
condition, and the requested result exactly the same, so that any correct answer
to the original is a correct answer to yours. Do not copy any formula from the
original: rename the variables, write expressions in an equivalent form or in
words, and reorder the conditions, wherever that stays faithful. Reply with the
rewritten problem only.""",
}


def paraphrase_controls(
    library: BenchmarkLibrary, output: Path, model: str, style: str = "faithful"
) -> None:
    """Restate every Dolci control item, resuming if the output already has some."""
    from openai import OpenAI

    done = set()
    if output.exists():
        done = {json.loads(line)["id"] for line in output.read_text(encoding="utf-8").splitlines()}
    client = OpenAI()
    prompt = PROMPTS[style]
    prompt_sha256 = hashlib.sha256(prompt.encode()).hexdigest()
    with open(output, "a", encoding="utf-8", newline="\n") as out:
        for item in DOLCI.split(library).control:
            if item.id in done:
                continue
            response = client.responses.create(
                model=model, instructions=prompt, input=item.problem, store=False
            )
            record = {
                "id": item.id,
                "paraphrase": response.output_text.strip(),
                "model": response.model,
                "response_id": response.id,
                "prompt_sha256": prompt_sha256,
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            log.info("%s paraphrased", item.id)
