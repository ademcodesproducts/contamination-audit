"""Model inference driver for the DiD experiment.

STATUS: stub. Inference traces in ``results/traces/{model}_traces.jsonl`` were
generated via ``notebooks/run_inference.ipynb`` and are checked in as data.

The notebook ran OpenThinker-7B and Tulu-3-8B-SFT on the contaminated+clean
splits across all three perturbation types (original, number_swap, surface_noise).
Sampling params per the paper: bfloat16, T=0.8, N=1, max_new_tokens=4096.
A single NVIDIA RTX PRO 6000 Blackwell (102 GB VRAM) is enough for both models.

Expected output schema per record (jsonl):
  math500_id        — benchmark item id
  split             — "contaminated" | "clean"
  perturbation_type — "original" | "number_swap" | "surface_noise"
  prompt            — exact text sent to the model
  full_trace        — raw model output (includes \\boxed{} if present)
  ground_truth      — expected answer for the perturbed problem
  final_answer      — extracted via answer_extract.extract_boxed (may be None)
  correct           — bool from answer_extract.equivalent(final_answer, ground_truth)
"""

from __future__ import annotations


def run(model_name: str, prompts_path: str, out_path: str) -> None:
    raise NotImplementedError("Pending notebook integration.")
