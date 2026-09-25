"""Use case 6 - AI agent tool routing and model-tier selection.

Replaces: asking a big LLM "which tool should I call?" on every step, or always paying for
the most expensive model. Laya answers in tens of ms, so the expensive model only runs when needed.
"""
import pathlib, sys; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from shared import run_cases

QUESTIONS = {
    "tool": {"type": "choice", "instructions": "Which tool should the assistant use for this request?",
             "criteria": {"knowledge_base": "answer from company documents and FAQs",
                          "crm_lookup": "needs the customer's account, order or ticket data",
                          "calendar": "book, move or cancel a meeting",
                          "web_search": "needs fresh public information from the internet",
                          "no_tool": "small talk or a direct answer, no tool needed"}},
    "difficulty": {"type": "score", "instructions": "How hard is this request to answer well?",
                   "criteria": ["trivial", "simple lookup", "needs reasoning", "complex multi-step"]},
    "needs_frontier_model": {"type": "noul",
                             "instructions": "Does this need a large reasoning model rather than a small fast model?"},
}

REQUESTS = [
    "What's the status of my order 55120?",
    "Move my demo with Acme from Tuesday to Thursday 3pm",
    "What is your refund policy?",
    "Compare our last 3 quarters of churn and suggest a retention plan with cost estimates",
    "thanks, that's all!",
]

if __name__ == "__main__":
    run_cases("Agent tool routing / model tier", QUESTIONS, REQUESTS)
    # Ready-made preset for model-tier routing: import laya; laya.router_questions()
