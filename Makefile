# Reproduce the paper end-to-end. The inference target requires GPU; everything
# else runs on CPU. Behavioral analysis depends on results/traces/ being populated.

PY := python

.PHONY: all data detect inference score behavioral report clean test help

help:
	@echo "Targets:"
	@echo "  data         download HF datasets to data/raw/"
	@echo "  detect       n-gram + retrieval + judge + clean-set + report (stages 1-5)"
	@echo "  inference    run OpenThinker / Tulu / s1 on the DiD prompt set (stages 7-8)"
	@echo "  behavioral   compute DiD / null rate / CoT features / recitation (stages 9-11, 14)"
	@echo "  report       robustness + annotations + pipeline figure (stages 12, 13, 99)"
	@echo "  all          everything in dependency order"
	@echo "  test         pytest tests/"

all: data detect inference behavioral report

data:
	$(PY) scripts/00_load_datasets.py

detect: data
	$(PY) scripts/01_ngram_filter.py
	$(PY) scripts/02_semantic_retrieval.py
	$(PY) scripts/03_llm_judge.py
	$(PY) scripts/04_build_clean_set.py
	$(PY) scripts/05_validate_and_report.py --crosscheck

inference:
	$(PY) scripts/07_run_inference.py
	-$(PY) scripts/08_score_answers.py results/traces/openthoughts_traces.jsonl
	-$(PY) scripts/08_score_answers.py results/traces/tulu_traces.jsonl
	-$(PY) scripts/08_score_answers.py results/traces/s1_traces.jsonl

behavioral:
	$(PY) scripts/09_compute_did.py
	$(PY) scripts/10_compute_null_rate.py
	$(PY) scripts/11_cot_features.py
	$(PY) scripts/14_recitation_analysis.py

report:
	$(PY) scripts/12_robustness_checks.py
	$(PY) scripts/13_build_annotation_csv.py
	$(PY) scripts/99_make_pipeline_figure.py

test:
	$(PY) -m pytest tests/

clean:
	rm -rf results/ngram/*.jsonl results/embeddings/*.npy results/embeddings/*.jsonl results/judge/*.jsonl
	@echo "kept: data/raw, data/processed, results/tables, results/figures, results/traces, results/annotations"
