# Reproduce paper/arr_decontamination.tex end-to-end.
#
# The `paper` target is the current pipeline and runs entirely on CPU. The
# `legacy-*` targets belong to the superseded ANLP draft (inference, DiD,
# chain-of-thought) and are kept only so its outputs remain regenerable; none of
# it feeds the current paper. See README.

PY := python

# decon is a Rust binary and is not vendored. Build it and point this at the
# result:  git clone https://github.com/allenai/decon && cd decon && cargo build --release
DECON_BIN ?= decon/target/release/decon

.PHONY: help paper paper-data paper-results paper-decon paper-figures \
        legacy-all legacy-data legacy-detect legacy-inference legacy-behavioral \
        legacy-report test clean

help:
	@echo "Current paper:"
	@echo "  paper           data + results + figures (CPU only)"
	@echo "  paper-data      fetch Dolci + OlymMATH + MathArena benchmark groups"
	@echo "  paper-results   regenerate every number in the paper"
	@echo "  paper-decon     run Ai2 decon (needs DECON_BIN=<path to binary>)"
	@echo "  paper-figures   redraw Figures 1 and 2"
	@echo ""
	@echo "Superseded ANLP draft:"
	@echo "  legacy-all      the old detect + inference + behavioral pipeline"
	@echo ""
	@echo "  test            pytest tests/"

# ----------------------------------------------------------------- current paper

paper: paper-data paper-results paper-figures

paper-data:
	$(PY) scripts/19_prepare_paper_data.py

paper-results:
	$(PY) scripts/20_paper_results.py

paper-decon:
	$(PY) scripts/21_run_decon.py --decon-bin "$(DECON_BIN)"

paper-figures:
	$(PY) scripts/98_make_paper_figures.py

# ----------------------------------------------------------------- superseded

legacy-all: legacy-data legacy-detect legacy-inference legacy-behavioral legacy-report

legacy-data:
	$(PY) scripts/00_load_datasets.py

legacy-detect: legacy-data
	$(PY) scripts/01_ngram_filter.py
	$(PY) scripts/02_semantic_retrieval.py
	$(PY) scripts/03_llm_judge.py
	$(PY) scripts/04_build_clean_set.py
	$(PY) scripts/05_validate_and_report.py --crosscheck

legacy-inference:
	$(PY) scripts/07_run_inference.py
	-$(PY) scripts/08_score_answers.py results/traces/openthoughts_traces.jsonl
	-$(PY) scripts/08_score_answers.py results/traces/tulu_traces.jsonl
	-$(PY) scripts/08_score_answers.py results/traces/s1_traces.jsonl

legacy-behavioral:
	$(PY) scripts/09_compute_did.py
	$(PY) scripts/10_compute_null_rate.py
	$(PY) scripts/11_cot_features.py
	$(PY) scripts/14_recitation_analysis.py

legacy-report:
	$(PY) scripts/12_robustness_checks.py
	$(PY) scripts/13_build_annotation_csv.py
	$(PY) scripts/99_make_pipeline_figure.py

test:
	$(PY) -m pytest tests/

clean:
	rm -rf results/ngram/*.jsonl results/embeddings/*.npy \
	       results/embeddings/*.jsonl results/judge/*.jsonl
	@echo "kept: data/raw, data/processed, results/paper, results/tables, results/figures"
