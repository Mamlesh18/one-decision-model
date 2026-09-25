"""Behaviour checks on obvious cases, including the known weak spots the project documents.

These are NOT an accuracy benchmark (see benchmarks/). They catch regressions after upgrading
Laya or changing a prompt. Tests marked xfail document known limits: if one starts
passing (XPASS), the limit may be fixed in a newer release.
"""
import pytest

pytestmark = pytest.mark.model

DEPT = {"dept": {"type": "choice", "instructions": "Which department should handle this?",
                 "criteria": {"billing": "invoices, payments, refunds", "technical": "bugs, outages, crashes",
                              "delivery": "shipping and parcels"}}}


@pytest.mark.parametrize("text,label", [
    ("I was billed twice, refund me", "billing"),
    ("The app crashes every time I log in", "technical"),
    ("My parcel hasn't arrived after two weeks", "delivery"),
    ("मुझसे दो बार शुल्क लिया गया, पैसे वापस करें", "billing"),
    ("La aplicación se cierra al abrirla", "technical"),
])
def test_obvious_department(router, text, label):
    assert router.predict(text, DEPT)["answers"]["dept"]["choice"] == label


TURN = {"t": {"type": "noul", "instructions": "Has the caller finished their thought, so the assistant can reply now?"}}


def _turn(router, text):
    state = [{"role": "assistant", "content": "How can I help?"}, {"role": "caller", "content": text}]
    return router.predict(state, TURN, model="english")["answers"]["t"]["noul"]


def test_complete_turn_scores_higher_than_cut_off(router):
    assert _turn(router, "I want to change my delivery address to 42 Baker Street") > _turn(router, "I want to change my")


BARGE = {"k": {"type": "choice", "instructions": "The assistant was speaking when the caller said this. What is it?",
               "criteria": {"backchannel": "uh-huh, yeah, okay, listening signal",
                            "objection": "the caller disagrees or corrects the assistant"}}}


@pytest.mark.parametrize("text,label", [("uh-huh", "backchannel"), ("no no, that's wrong", "objection")])
def test_barge_in(router, text, label):
    state = [{"role": "assistant (speaking)", "content": "Your refund of 49 dollars is on its way and"},
             {"role": "caller", "content": text}]
    assert router.predict(state, BARGE, model="english")["answers"]["k"]["choice"] == label


def test_speaker_key_blinds_english_checkpoint(router):
    """Found here, not documented upstream: a turn keyed "speaker" gets uniform output on english.

    If this starts failing, a newer checkpoint fixed it and speaker/text turns become usable.
    """
    q = {"c": {"type": "choice", "instructions": "What does the caller want?",
               "criteria": {"refund": "money back", "address": "change address", "agent": "talk to a human"}}}
    text = "I want a refund for my order"
    blind = router.predict([{"speaker": "caller", "text": text}], q, model="english")["answers"]["c"]
    ok = router.predict([{"role": "caller", "content": text}], q, model="english")["answers"]["c"]
    assert max(blind["probabilities"].values()) < 0.40      # ~uniform: text ignored
    assert ok["choice"] == "refund" and ok["answer_confidence"] > 0.8


@pytest.mark.xfail(reason="Known limit (#377): negated requests can still pick the action label", strict=False)
def test_negation_not_cancel(router):
    q = {"a": {"type": "choice", "instructions": "What should we do with the account?",
               "criteria": {"no_action": "leave the account as it is", "cancel_account": "close the account"}}}
    r = router.predict("I do NOT want to cancel my account, I just have a question", q, model="english")
    assert r["answers"]["a"]["choice"] == "no_action"
