"""Use case 8 - Multilingual intake without a translation step.

Replaces: detect language -> machine-translate -> English classifier (3 hops, 3 bills).
The Router detects the script/language in <1 ms and sends non-English text to laya-multilingual.
Questions stay in English; the customer's text can be in any of 100+ languages.
"""
import pathlib, sys; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from shared import run_cases, get_router

QUESTIONS = {
    "department": {"type": "choice", "instructions": "Which department should handle this?",
                   "criteria": {"billing": "invoices, payments, refunds",
                                "technical": "bugs, outages, app crashes",
                                "delivery": "shipping, late or missing parcels",
                                "other": "everything else"}},
    "angry": {"type": "noul", "instructions": "Is the customer angry?"},
}

MESSAGES = [
    "मुझसे मार्च में दो बार शुल्क लिया गया, कृपया डुप्लिकेट राशि वापस करें।",  # Hindi
    "La aplicación se cierra cada vez que abro la configuración.",        # Spanish
    "Mein Paket ist seit zwei Wochen nicht angekommen!",                  # German
    "எனது ஆர்டர் இன்னும் வரவில்லை, மிகவும் கோபமாக இருக்கிறேன்",              # Tamil
    "تم خصم المبلغ مرتين من بطاقتي",                                       # Arabic
    "I was charged twice for my order",                                   # English
]

if __name__ == "__main__":
    router = get_router()
    print("Routing only (no model load, microseconds):")
    for m in MESSAGES:
        d = router.route(m)
        print(f"  {d.model:<13} {d.reason[:70]}")
    print()
    run_cases("Multilingual intake", QUESTIONS, MESSAGES)
    print("\nTip: short Latin-script text ('Quero cancelar') may be read as English. If most")
    print("traffic is not English use Router(default='multilingual') or pass lang_guess=.")
