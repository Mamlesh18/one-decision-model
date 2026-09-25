"""Use case 4 - Content moderation.

Replaces: a toxicity API + a spam classifier + manual rules, or an LLM moderation prompt.
Three outcomes from one probability: remove (high), human review (middle), publish (low).
"""
import pathlib, sys; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from shared import run_cases

QUESTIONS = {
    "toxic": {"type": "noul", "instructions": "Is the post insulting, hateful or abusive?"},
    "threat": {"type": "noul", "instructions": "Does the post threaten violence against someone?"},
    "spam": {"type": "noul", "instructions": "Is the post spam or an unrelated advertisement?"},
    "severity": {"type": "score", "instructions": "How harmful is the post overall?",
                 "criteria": ["harmless", "rude but acceptable", "harmful", "severe, remove immediately"]},
}

POSTS = [
    "Great article, thanks for sharing the benchmark numbers!",
    "You are an idiot and everyone here hates you.",
    "Buy cheap followers at www.fastfollowers.biz !!!",
    "I know where you live. Watch your back tonight.",
]


def decide(r):
    worst = max(r["answers"][k]["noul"] for k in ("toxic", "threat", "spam"))
    return "REMOVE" if worst >= 0.85 else "REVIEW" if worst >= 0.40 else "PUBLISH"


if __name__ == "__main__":
    results = run_cases("Content moderation", QUESTIONS, POSTS)
    print()
    for p, r in zip(POSTS, results):
        print(f"  {decide(r):<8} <- {p}")
    # Ready-made preset: import laya; laya.moderation_questions()
