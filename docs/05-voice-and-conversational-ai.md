# 5. Laya in conversational AI and voice AI

## Why voice needs a "System 1"

In a voice bot, **latency is the product**. People notice a gap of more than about 800 ms after
they stop talking. The budget has to cover:

```
caller stops -> VAD end-of-turn -> final STT -> [decisions] -> LLM first token -> TTS first audio
                  200-700 ms        50-300 ms     ???          200-800 ms          100-300 ms
```

Any decision that calls an LLM ("is this the end of the turn?", "was that an interruption?",
"what's the intent?") takes another 250 ms or more out of that budget. Laya takes about 30-100 ms
on a GPU, and can run **while the caller is still talking**, on STT partials.

## Advantages for conversational and voice AI

| Advantage | What it gives you |
|-----------|-------------------|
| **Speed** (about 33 ms on GPU) | Decisions can run on every STT partial and every turn without adding noticeable delay |
| **Probabilities** | A threshold per decision: act when sure, fall back when not. This is what makes "only override when confident" possible |
| **No hallucination / no parsing** | The answer is always one of your options. No broken JSON in the middle of a call |
| **Questions defined per request** | Change intents or the dialogue state without retraining, e.g. different questions at each step of a flow |
| **Many decisions in one pass** | intent + sentiment + topic + escalation in one call per turn |
| **Multilingual** | 100+ languages, code-mixed traffic routed automatically, no translation hop |
| **Self-hosted** | Call audio and transcripts stay on your servers; no per-call API cost |
| **Conversation-aware** | Pass the turn list; long history is cut from the left, so the latest turns are kept |

## Where it plugs into a voice pipeline

```
Mic/phone -> VAD -> streaming STT ---+--> Laya: turn_complete?     --> end-of-turn timer
                                     +--> Laya: interruption kind? --> TTS control (stop / keep)
                         final text  +--> Laya: intent, sentiment,  --> dialogue manager:
                                             topic, complexity,         prompt choice, LLM tier,
                                             wants_human                 escalation, analytics
                                                         |
                                                         v
                                                LLM (only when needed) -> TTS
after the call:  transcript -> Laya QA scoring -> dashboard / review queue
```

## The TASA assistant, explained simply

The message you shared describes these pieces (stack: Browser, FastAPI WebSocket gateway,
streaming pipeline, Laya, TTS, with Silero VAD, Whisper STT and Piper TTS):

| TASA piece | What it means | Our file |
|------------|---------------|----------|
| **Cadence-1: turn timing.** `turn_complete` + `completion_conf` shorten the endpoint; the verdict is trusted only for 2 s | Ask "is the caller done?" on the transcript. If yes and confident, reply sooner. The answer goes stale after 2 s because the caller may keep talking | `use-cases/12_voice_turn_completion.py`, `examples/07_voice_pipeline_simulation.py` |
| **Cadence-2: turn context.** intent, sentiment, topic, complexity, shadow-logged every turn, override only when confident | Compute a context vector every turn, always log it, and only change the bot's behaviour when the probability is high | `use-cases/14_voice_live_call_context.py` |
| **Interrupt semantics.** disagreement -> cancel TTS + INTERRUPT; backchannel -> keep speaking | Classify what the caller said over the bot | `use-cases/13_voice_barge_in_interruptions.py` |
| **Raised echo gate (floor x 1.5)** | Not Laya. An audio-level VAD threshold, raised while the bot speaks, so the bot's own voice coming back through the mic doesn't trigger a barge-in | (audio side) |

## Voice use cases in this repo

* `12` end of turn, `13` barge-in, `14` live context, `15` guardrail,
  `16` post-call QA, `17` yes/no confirmation, `18` outbound call outcome (voicemail etc.)
* Conversational (chat): `03` intent with "unclear" re-prompt, `06` tool / model routing,
  `09_hybrid_laya_then_llm.py` (canned vs small LLM vs big LLM)

## Design rules for real-time use

1. **Run it off the audio thread** (worker thread or a separate `laya-serve` process).
2. **Only score the newest partial.** Drop results for older partials.
3. **Use a freshness window.** A verdict is valid for about 1-2 s.
4. **Fail open.** If Laya is late or errors, fall back to plain VAD timing and the normal flow.
5. **Preload and warm up** (`Router(preload=True)` plus one dummy call) before the first call.
6. **Pin the checkpoint** for known-language lines (`model="english"`); very short utterances can be misrouted.
7. **Keep the history short** (last 2-6 turns). It's faster, and older turns get cut anyway.
8. **Combine with audio signals.** Laya reads text only. It doesn't hear rising or falling intonation or a sigh.
9. **Mind the STT.** Laya judges the transcript, so STT errors flow straight in. Test with your real STT output, not clean text.
10. **Measure p95, not the average**, on your deployment hardware: `benchmarks/bench_latency.py`.
