"""Benchmark task definitions: which dataset file, which question, how to read the label.

Add your own task by adding an entry. Dataset rows are JSONL: {"state": <str|dict|list>, "label": <str|bool>}
    - choice tasks: label is one of the criteria keys
    - noul tasks:   label is true / false
"""

TASKS = {
    "voice_intent": {
        "file": "data/voice_intent.jsonl",
        "qid": "intent",
        "question": {
            "type": "choice",
            "instructions": "What does the caller want to do?",
            "criteria": {
                "check_balance": "hear the account balance",
                "block_card": "block or freeze a lost or stolen card",
                "dispute_charge": "complain about a transaction they do not recognise",
                "talk_to_agent": "speak to a human",
                "update_address": "change address or contact details",
                "pay_bill": "make a payment or pay a bill",
            },
        },
    },
    "turn_complete": {
        "file": "data/turn_complete.jsonl",
        "qid": "turn_complete",
        "question": {
            "type": "noul",
            "instructions": "Has the caller finished their thought in the last `caller` line, so the assistant can reply now?",
            "criteria": {"false": "the sentence is cut off, trailing, or they are still listing or spelling something",
                         "true": "a complete request, question or answer the assistant can respond to"},
        },
        "model": "english",
    },
    "interruption": {
        "file": "data/interruption.jsonl",
        "qid": "kind",
        "question": {
            "type": "choice",
            "instructions": "The assistant was speaking when the caller said the last line. What kind of interruption is it?",
            "criteria": {
                "backchannel": "short listening signal like uh-huh, yeah, okay, right, mm",
                "disagreement": "the caller objects, corrects or says the assistant is wrong",
                "new_request": "the caller asks a new question or changes the topic",
                "stop_request": "the caller wants the assistant to stop, wait or be quiet",
            },
        },
        "model": "english",
    },
    "confirmation": {
        "file": "data/confirmation.jsonl",
        "qid": "reply",
        "question": {
            "type": "choice",
            "instructions": "How did the user answer the assistant's yes/no question?",
            "criteria": {"A": "agrees, accepts, says go ahead",
                         "B": "declines, refuses, says do not do it",
                         "D": "unsure, thinking, or asks something else"},
        },
        "model": "english",
    },
    "multilingual_dept": {
        "file": "data/multilingual_dept.jsonl",
        "qid": "department",
        "question": {
            "type": "choice",
            "instructions": "Which department should handle this?",
            "criteria": {"billing": "invoices, payments, refunds, double charges",
                         "technical": "bugs, outages, app crashes, login problems",
                         "delivery": "shipping, late or missing parcels"},
        },
    },
}
