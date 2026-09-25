"""Use case 11 - Security alert (SIEM / EDR) triage.

Replaces: a tier-1 SOC analyst's first look, or an LLM summarising every alert.
Like invoices, the project reports this needs the fine-tuned checkpoint (0.766 on security).
"""
import pathlib, sys; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from shared import run_cases

QUESTIONS = {
    "true_positive": {"type": "noul", "instructions": "Is this alert likely real malicious activity (not a false positive)?"},
    "credential_compromise": {"type": "noul", "instructions": "Does it suggest a stolen or misused account?"},
    "severity": {"type": "score", "instructions": "How severe is it?",
                 "criteria": ["info", "low", "medium", "high", "critical"]},
    "next_step": {"type": "choice", "instructions": "What should the analyst do next?",
                  "criteria": {"close": "benign, close the alert", "monitor": "watch for more signals",
                               "investigate": "open an investigation",
                               "contain": "isolate the host or disable the account now"}},
}

ALERTS = [
    {"rule": "Impossible travel", "user": "j.doe", "logins": ["Chennai 09:02", "Frankfurt 09:20"],
     "mfa": "passed after 14 push prompts"},
    {"rule": "New scheduled task", "host": "BUILD-01", "task": "nightly_backup.ps1", "signed": True,
     "created_by": "svc_backup"},
    {"rule": "Encoded PowerShell", "host": "HR-LAPTOP-7", "cmd": "powershell -enc SQBFAFgA...",
     "parent": "WINWORD.EXE"},
]

if __name__ == "__main__":
    run_cases("Security alert triage", QUESTIONS, ALERTS, model="typed-decisions")
