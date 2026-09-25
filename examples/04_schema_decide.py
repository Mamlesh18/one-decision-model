"""Example 4 - Describe the answer as a JSON schema and get typed values back.

No question dictionary needed:
    string + enum   -> choice
    integer min/max -> score
    boolean         -> noul
Handy when replacing an LLM "structured output" call: keep the same schema.
"""
from laya import Router

router = Router()

schema = {
    "type": "object",
    "properties": {
        "department": {"type": "string", "enum": ["billing", "support", "sales"],
                       "description": "Which team should handle this?"},
        "urgency": {"type": "integer", "minimum": 0, "maximum": 2, "description": "How urgent, 0 to 2"},
        "needs_human": {"type": "boolean", "description": "Should a person handle this?"},
    },
}

text = "I was charged twice and nobody answers my emails, refund me now."

print("values :", router.decide(text, schema=schema))

details = router.decide(text, schema=schema, return_details=True)
print("conf   :", details.confidence)
print("probs  :", details.probabilities)

# With pydantic (pip install pydantic):
#   class Ticket(BaseModel):
#       department: Literal["billing", "support", "sales"]
#       needs_human: bool
#   router.decide(text, schema=Ticket)
