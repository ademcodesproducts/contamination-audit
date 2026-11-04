"""Reading decon's output, with the binary replaced by a stand-in."""

import json
import subprocess
from pathlib import Path

from contamination_audit.decon import DeconBinary, DolciDecon
from contamination_audit.experiments.scaling import CORPUS_SIZES

SUMMARY = "│ Contaminated documents                  2 │\n"


def fake_decon(pairs, seen):
    """A subprocess.run that writes a report of (document, item) pairs as decon does."""

    def run(command, **_):
        seen.append(command)
        report = Path(command[command.index("--report-output-dir") + 1])
        report.mkdir(parents=True, exist_ok=True)
        lines = [json.dumps({"training_line": d, "eval_instance_index": i}) for d, i in pairs]
        (report / "results.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout=SUMMARY, stderr="")

    return run


def test_findings_name_the_items_not_just_the_documents(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(subprocess, "run", fake_decon([(4, 7), (9, 7), (9, 2)], seen))
    findings = DeconBinary(tmp_path / "decon").detect(
        tmp_path / "training", tmp_path / "evals", tmp_path / "report"
    )
    assert (findings.documents, findings.items) == (2, [2, 7])
    assert "--index-answers" not in seen[0]


def test_a_stale_report_is_cleared_and_options_are_passed(tmp_path, monkeypatch):
    report = tmp_path / "report"
    report.mkdir()
    (report / "old.jsonl").write_text('{"eval_instance_index": 99}\n', encoding="utf-8")
    seen = []
    monkeypatch.setattr(subprocess, "run", fake_decon([(1, 3)], seen))
    findings = DeconBinary(tmp_path / "decon").detect(
        tmp_path / "training", tmp_path / "evals", report, threshold=0.4, index_answers=False
    )
    assert findings.items == [3]
    assert seen[0][-4:] == ["--contamination-score-threshold", "0.4", "--index-answers", "false"]


def test_the_threshold_sweep_covers_the_controls_without_rerunning_the_default(
    tiny_study, monkeypatch
):
    seen = []
    monkeypatch.setattr(subprocess, "run", fake_decon([(1, 0)], seen))
    result = tiny_study.run(DolciDecon, binary=Path("decon"), scaling=True)
    assert set(result["threshold_sweep"]) == {"measure", "control"}
    assert result["threshold_sweep"]["control"][0.8] == {"documents": 2, "items": result["control"]}
    assert not any("0.8" in command for command in seen)
    # defaults 2, three lower thresholds on each group 6, without answers 2,
    # and every corpus size but the whole corpus, which is the default run
    assert len(seen) == 2 + 6 + 2 + len(CORPUS_SIZES) - 1
    whole = result["corpus_rows"]
    assert result["scaling"][whole] == result["threshold_sweep"]["measure"][0.8]
