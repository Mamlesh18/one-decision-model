"""Routing and language detection. Pure Python: no checkpoint download, runs in milliseconds."""
import pytest
from laya import Router
from laya.lang import detect_script


@pytest.fixture(scope="module")
def r():
    return Router()


@pytest.mark.parametrize("text", [
    "मुझसे दो बार शुल्क लिया गया",          # Devanagari
    "எனது ஆர்டர் இன்னும் வரவில்லை",          # Tamil
    "تم خصم المبلغ مرتين",                   # Arabic
    "荷物がまだ届いていません",                # Japanese
])
def test_non_latin_goes_to_multilingual(r, text):
    assert r.route(text).model == "multilingual"


def test_plain_english_goes_to_english(r):
    assert r.route("Please refund the duplicate charge on my invoice").model == "english"


def test_german_sentence_goes_to_multilingual(r):
    assert r.route("Der Kunde wurde zweimal belastet und ist sehr verärgert").model == "multilingual"


def test_explicit_model_wins(r):
    assert r.route("hello", model="multilingual").model == "multilingual"


def test_lang_guess_overrides_detection(r):
    assert r.route("Quero cancelar", lang_guess="pt-BR").model == "multilingual"
    assert r.route("Quero cancelar", lang_guess="en_US.UTF-8").model == "english"


@pytest.mark.xfail(strict=False, reason=(
    "PyPI laya 0.3.20 treats 'C.UTF-8' as non-English and routes to multilingual. The upstream "
    "README describes the fix on main; it is not released yet. Never pass $LANG from a Docker "
    "image as lang_guess on 0.3.20."))
def test_lang_guess_abstains_on_c_locale(r):
    # C.UTF-8 is the default $LANG in Docker images; it must not force a checkpoint.
    assert r.route("Please refund me", lang_guess="C.UTF-8").model == "english"


def test_default_multilingual_for_ambiguous_short_text():
    assert Router(default="multilingual").route("Esqueci minha senha").model == "multilingual"


def test_route_returns_reason(r):
    d = r.route("मुझसे दो बार शुल्क लिया गया")
    assert isinstance(d.reason, str) and d.reason


def test_conversation_list_state_routes(r):
    state = [{"role": "assistant", "content": "How can I help?"},
             {"role": "caller", "content": "मेरा कार्ड खो गया"}]
    assert r.route(state).model == "multilingual"


def test_detect_script():
    assert detect_script("hello world") == "latin"
    assert detect_script("नमस्ते दुनिया") == "devanagari"


def test_route_batch_keeps_order(r):
    reqs = [{"state": "refund me please", "questions": {}},
            {"state": "मेरा पैसा वापस करो", "questions": {}},
            {"state": "refund me again", "questions": {}}]
    assert [d.model for d in r.route_batch(reqs)] == ["english", "multilingual", "english"]
