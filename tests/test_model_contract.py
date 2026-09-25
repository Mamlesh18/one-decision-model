"""Output contract: the shape and invariants your code will rely on.

Run with:  LAYA_RUN_MODEL_TESTS=1 pytest tests/test_model_contract.py -v
(PowerShell: $env:LAYA_RUN_MODEL_TESTS="1"; pytest tests -v)
"""
import math

import pytest

pytestmark = pytest.mark.model

Q = {
    "dept": {"type": "choice", "instructions": "Which department?",
             "criteria": {"billing": "payments, refunds", "technical": "bugs, crashes", "other": "anything else"}},
    "urgency": {"type": "score", "instructions": "How urgent?", "criteria": ["low", "medium", "high"]},
    "refund": {"type": "noul", "instructions": "Does the customer ask for money back?"},
}
TEXT = "I was charged twice for March, please refund the duplicate today."


@pytest.fixture(scope="module")
def result(router):
    return router.predict(TEXT, Q)


def test_top_level_keys(result):
    assert {"answers", "usage", "routing"} <= set(result)
    assert set(result["answers"]) == set(Q)


def test_choice_shape(result):
    a = result["answers"]["dept"]
    assert a["type"] == "choice" and a["choice"] in Q["dept"]["criteria"]
    assert set(a["probabilities"]) == set(Q["dept"]["criteria"])
    assert math.isclose(sum(a["probabilities"].values()), 1.0, abs_tol=1e-3)
    assert a["choice"] == max(a["probabilities"], key=a["probabilities"].get)


def test_score_shape(result):
    a = result["answers"]["urgency"]
    assert a["type"] == "score" and 0.0 <= a["score"] <= 2.0
    assert a["legend"] == {"0": "low", "1": "medium", "2": "high"}


def test_noul_shape(result):
    a = result["answers"]["refund"]
    assert a["type"] == "noul" and 0.0 <= a["noul"] <= 1.0


def test_answer_confidence_in_range(result):
    for a in result["answers"].values():
        assert 0.0 <= a["answer_confidence"] <= 1.0


def test_deterministic(router):
    a = router.predict(TEXT, Q)["answers"]
    b = router.predict(TEXT, Q)["answers"]
    assert a["dept"]["probabilities"] == b["dept"]["probabilities"]


def test_empty_questions_skip_inference(router):
    r = router.predict(TEXT, {})
    assert r["answers"] == {}


def test_question_order_does_not_change_answers(router):
    reordered = dict(reversed(list(Q.items())))
    a = router.predict(TEXT, Q)["answers"]["dept"]["choice"]
    b = router.predict(TEXT, reordered)["answers"]["dept"]["choice"]
    assert a == b


def test_batch_matches_single(router):
    texts = [TEXT, "The app crashes when I open settings", "What are your opening hours?"]
    single = [router.predict(t, Q)["answers"]["dept"]["choice"] for t in texts]
    batch = [r["answers"]["dept"]["choice"] for r in router.predict_batch([{"state": t, "questions": Q} for t in texts])]
    assert single == batch


def test_invalid_question_rejected(router):
    with pytest.raises(ValueError):
        router.predict(TEXT, {"bad": {"type": "choice", "instructions": "x", "criteria": {}}})
