"""Checks on the fine-tuning kit: mock data, questions and the Colab notebook. No model needed."""
import ast
import json
import pathlib
import subprocess
import sys

import pytest

FT = pathlib.Path(__file__).resolve().parents[1] / "finetune"
QDEFS = json.loads((FT / "questions.json").read_text(encoding="utf-8"))


def rows(split):
    return [json.loads(line) for line in (FT / "data" / f"{split}.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]


@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_labels_match_questions(split):
    for r in rows(split):
        assert r["state"][-1]["role"] == "user" and all(set(t) == {"role", "content"} for t in r["state"])
        for qid, q in QDEFS.items():
            v = r["labels"][qid]
            if q["type"] == "choice":
                assert v in q["criteria"], (qid, r)
            elif q["type"] == "score":
                assert isinstance(v, int) and 0 <= v < len(q["criteria"]), (qid, r)
            else:
                assert isinstance(v, bool), (qid, r)


def test_splits_do_not_overlap():
    key = lambda r: json.dumps(r["state"])
    train = {key(r) for r in rows("train")}
    assert not train & {key(r) for r in rows("val")}
    assert not train & {key(r) for r in rows("test")}


def test_every_label_appears_in_training():
    tr = rows("train")
    for qid, q in QDEFS.items():
        seen = {r["labels"][qid] for r in tr}
        expected = set(q["criteria"]) if q["type"] == "choice" else set(range(len(q["criteria"]))) if q["type"] == "score" else {True, False}
        assert seen == expected, qid


def test_generator_is_deterministic(tmp_path):
    subprocess.run([sys.executable, str(FT / "make_dataset.py"), "--out", str(tmp_path)], check=True, capture_output=True)
    for split in ("train", "val", "test"):
        assert (tmp_path / f"{split}.jsonl").read_text(encoding="utf-8") == (FT / "data" / f"{split}.jsonl").read_text(encoding="utf-8")


def test_notebook_code_cells_parse_and_embed_current_files():
    nb = json.loads((FT / "laya_finetune_colab.ipynb").read_text(encoding="utf-8"))
    for c in nb["cells"]:
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        if src.startswith("%%writefile"):
            name, body = src.split("\n", 1)[0].split()[1], src.split("\n", 1)[1]
            # the builder strips the trailing newline from each cell
            assert body.rstrip("\n") == (FT / name).read_text(encoding="utf-8").rstrip("\n"), \
                f"{name} is stale: run python finetune/build_notebook.py"
            continue
        ast.parse("\n".join(line for line in src.splitlines() if not line.lstrip().startswith("!")))
