# Reproduce the paper end-to-end. Stub stages exit with status 2 ([stub] notice);
# the remaining stages produce the audit numbers in Table 2 and the pipeline figure.

PY := python

.PHONY: all data detect inference behavioral report clean test help

help:
	@echo "Targets:"
	@echo "  data         download HF datasets to data/raw/"
	@echo "  detect       n-gram + embedding + judge (stages 1-3)"
	@echo "  report       assemble C_lex/C_sem + Table 2 + robustness (stages 4-5,12-13,99)"
	@echo "  inference    [stubs] perturbation gen + inference (stages 6-8)"
	@echo "  behavioral   [stubs] DiD + null rate + CoT features (stages 9-11)"
	@echo "  all          everything in dependency order"
	@echo "  test         pytest tests/"

all: data detect report inference behavioral

data:
	$(PY) scripts/00_load_datasets.py

detect: data
	$(PY) scripts/01_ngram_filter.py
	$(PY) scripts/02_embedding_retrieval.py
	$(PY) scripts/03_llm_judge.py
	$(PY) scripts/04_build_clean_set.py

report: detect
	$(PY) scripts/05_validate_and_report.py --crosscheck --no-spotcheck 2>/dev/null || \
	    $(PY) scripts/05_validate_and_report.py --crosscheck
	$(PY) scripts/12_robustness_checks.py
	$(PY) scripts/13_build_annotation_csv.py
	$(PY) scripts/99_make_pipeline_figure.py

inference: report
	-$(PY) scripts/06_generate_perturbations.py
	-$(PY) scripts/07_run_inference.py
	-$(PY) scripts/08_score_answers.py

behavioral: inference
	-$(PY) scripts/09_compute_did.py
	-$(PY) scripts/10_compute_null_rate.py
	-$(PY) scripts/11_cot_features.py

test:
	$(PY) -m pytest tests/

clean:
	rm -rf results/ngram/*.jsonl results/embeddings/*.npy results/embeddings/*.jsonl results/judge/*.jsonl
	@echo "kept: data/raw, data/processed, results/tables, results/figures, results/traces, results/annotations"
