"""Label-free evaluation of decontamination filters.

Reimplements the decontamination filters released with open post-training
datasets and runs them against benchmark questions published after each corpus
was assembled. Those questions cannot be in the corpus, so any filter that
flags one is wrong, which makes filter error measurable without annotation.

Start with the `contamination-audit` command; see the README.
"""

__version__ = "1.0.0"
