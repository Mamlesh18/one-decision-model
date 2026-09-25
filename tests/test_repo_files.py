"""Checks on this repo's own files. No model needed."""
import ast
import json
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
from tasks import TASKS  # noqa: E402

SCRIPTS = sorted((ROOT / "use-cases").glob("*.py")) + sorted((ROOT / "examples").glob("*.py")) + \
    sorted((ROOT / "benchmarks").glob("*.py"))


@pytest.mark.parametrize("path", SCRIPTS, ids=lambda p: f"{p.parent.name}/{p.name}")
def test_script_parses(path):
    ast.parse(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("name", list(TASKS))
def test_dataset_labels_match_question(name):
    spec = TASKS[name]
    q = spec["question"]
    rows = [json.loads(line) for line in (ROOT / "benchmarks" / spec["file"]).read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) >= 20
    for row in rows:
        assert "state" in row and "label" in row
        if q["type"] == "choice":
            assert row["label"] in q["criteria"], row
        elif q["type"] == "noul":
            assert isinstance(row["label"], bool), row


@pytest.mark.parametrize("name", list(TASKS))
def test_task_question_is_well_formed(name):
    q = TASKS[name]["question"]
    assert q["type"] in ("choice", "score", "noul")
    assert q["instructions"].strip()
    if q["type"] == "choice":
        assert len(q["criteria"]) >= 2
        # Laya's own advice: avoid boolean-word labels as choice keys.
        assert not set(map(str.lower, q["criteria"])) & {"yes", "no", "true", "false"}
    if q["type"] == "noul" and "criteria" in q:
        assert set(q["criteria"]) == {"true", "false"}
