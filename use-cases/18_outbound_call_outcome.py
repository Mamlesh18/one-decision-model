"""Use case 18 - Outbound voice calls: what happened in the first seconds?

Outbound bots (reminders, collections, surveys) must decide quickly from the first STT text:
is this voicemail, the right person, a gatekeeper, a callback request, or "do not call me"?

Replaces: carrier AMD (answering-machine detection) alone, or post-call manual tagging.
"""
import pathlib, sys; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from shared import run_cases

QUESTIONS = {
    "who_answered": {"type": "choice", "instructions": "Who or what answered the call?",
                     "criteria": {"voicemail": "a recorded greeting or answering machine",
                                  "target_person": "the person we are calling",
                                  "someone_else": "another person, receptionist or family member",
                                  "ivr": "an automated phone menu"}},
    "callback_requested": {"type": "noul", "instructions": "Does the person ask to be called back later?"},
    "do_not_call": {"type": "noul", "instructions": "Does the person ask not to be called again?"},
}

OPENINGS = [
    "Hi you've reached Sam, I can't take your call right now, please leave a message after the beep.",
    "Yes this is Sam speaking.",
    "He's not home right now, I'm his wife, can I take a message?",
    "I'm driving, call me after 6.",
    "Stop calling this number, take me off your list.",
    "Thank you for calling Acme Corp. For sales press 1, for support press 2.",
]

if __name__ == "__main__":
    run_cases("Outbound call outcome", QUESTIONS, OPENINGS)
