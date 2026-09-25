"""Example 2 - The Router: one API, any language.

route()   -> which checkpoint would be used and why (pure Python, no model load, microseconds)
predict() -> route + answer
"""
from laya import Router

router = Router()
questions = {"department": {"type": "choice", "instructions": "Which department should handle this?",
                            "criteria": {"billing": "payments, refunds", "technical": "bugs, crashes",
                                         "other": "everything else"}}}

texts = [
    "The app crashes when I open settings",
    "La aplicación se cierra cada vez que abro la configuración.",
    "मुझसे दो बार शुल्क लिया गया, कृपया पैसे वापस करें।",
    "Der Kunde wurde zweimal belastet",
    "Quero cancelar",   # very short Latin text: often looks like English
]

print("1) Routing decisions only")
for t in texts:
    d = router.route(t, questions)
    print(f"   {d.model:<13} {t[:40]:<42} {d.reason}")

print("\n2) Forcing things")
print("   lang='pt'           ->", router.route("Quero cancelar", lang="pt").model)
print("   lang_guess='pt-BR'  ->", router.route("Quero cancelar", lang_guess="pt-BR").model)
print("   model='multilingual'->", router.route("Quero cancelar", model="multilingual").model)
print("   Router(default='multilingual') ->", Router(default="multilingual").route("Quero cancelar").model)

print("\n3) Full predictions")
for t in texts[:3]:
    r = router.predict(t, questions)
    print(f"   {r['routing']['model']:<13} {r['answers']['department']['choice']:<10} {t[:50]}")
