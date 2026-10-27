"""The command-line interface."""

from contamination_audit.cli import main

from .conftest import REPO


def test_run_list_names_every_experiment(capsys):
    assert main(["run", "--list"]) == 0
    assert "flag_counts" in capsys.readouterr().out


def test_unknown_experiment_is_rejected(capsys):
    assert main(["run", "nonsense"]) == 2


def test_missing_inputs_are_reported_not_raised(tmp_path, capsys):
    assert main(["--root", str(tmp_path), "run", "flag_counts"]) == 1
    assert "contamination-audit fetch" in capsys.readouterr().err


def test_judge_verify_on_the_published_record(capsys):
    assert main(["--root", str(REPO), "judge-verify"]) == 0
    assert "verified 565 judgments" in capsys.readouterr().out
