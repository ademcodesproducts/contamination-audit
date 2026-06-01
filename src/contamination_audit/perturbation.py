"""Perturbation generation (number-swap + surface-noise).

STATUS: stub. The original perturbations used in the paper were generated in
``notebooks/generate_perturbations.ipynb`` (deferred to follow-up integration).
The resulting CSV lives at ``data/processed/perturbations_sample.csv``.

When you linearize the notebook into a script, implement these two functions
against the prompt specifications in Appendix A of the paper:

  - ``number_swap(problem)`` calls GPT-4o with a prompt that:
      * changes numerical values to different but similarly-scaled numbers
      * preserves mathematical structure and operation type
      * keeps the problem well-defined with a clean answer
      * solves the modified problem and rates confidence (high/medium/low)
      * returns JSON with keys: perturbed_problem, perturbed_answer, confidence

  - ``surface_noise(problem)`` calls GPT-4o with a prompt that:
      * adds exactly ONE non-mathematical context sentence (setting / motivation)
      * does not alter any existing wording, values, or mathematical structure
      * returns JSON with key: perturbed_problem
      * the answer is unchanged by construction (do not re-solve)

Inputs:  data/processed/contaminated_and_perturbations.csv (problem column)
Outputs: data/processed/contaminated_and_perturbations.csv (filled-in columns)
"""

from __future__ import annotations


def number_swap(problem: str) -> dict:
    raise NotImplementedError("Pending notebook integration (Appendix A.1).")


def surface_noise(problem: str) -> dict:
    raise NotImplementedError("Pending notebook integration (Appendix A.2).")
