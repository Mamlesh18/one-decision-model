"""Example 1 - The smallest useful Laya program.

    state     = the thing to judge (text, dict, or a list of conversation turns)
    questions = what you want to know, each with a type:
                  choice -> pick one label        (returns the label + a probability per label)
                  score  -> pick a level 0..N      (returns an expected level, e.g. 1.7)
                  noul   -> yes / no               (returns P(yes) between 0 and 1)
    result    = one forward pass, every question answered at once
"""
import json
from laya import Router

router = Router()  # downloads the checkpoint on first use

state = "Hi, we were billed twice for March. Please refund the duplicate today or we will cancel."
questions = {
    "department": {"type": "choice", "instructions": "Which department should handle this?",
                   "criteria": {"billing": "invoices, payments, refunds",
                                "technical": "bugs, outages, system errors",
                                "other": "everything else"}},
    "urgency": {"type": "score", "instructions": "How urgent is this?",
                "criteria": ["not urgent", "soon", "blocking"]},
    "churn_risk": {"type": "noul", "instructions": "Does the user threaten to cancel or leave?"},
}

result = router.predict(state, questions)

print("department :", result["answers"]["department"]["choice"])
print("urgency    :", result["answers"]["urgency"]["score"], "(0 = not urgent, 2 = blocking)")
print("churn risk :", result["answers"]["churn_risk"]["noul"], "= probability of yes")
print("model used :", result["routing"]["model"])
print("\nFull raw result:")
print(json.dumps(result, indent=2, ensure_ascii=False))
