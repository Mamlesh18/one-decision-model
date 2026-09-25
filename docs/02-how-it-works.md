# 2. How it works

## The pipeline for one call

```
router.predict(state, questions)
   |
   1. ROUTE (pure Python, < 1 ms)
   |    look at the script and language of the state
   |    non-Latin script, or a non-English language -> multilingual
   |    English                                     -> english
   |
   2. BUILD ONE INPUT ROW PER QUESTION
   |    [CLS] choice question: Which department? [SEP]
   |          [MASK] billing: invoices...  [MASK] technical: bugs...  [MASK] other: ... [SEP]
   |          <your state text / JSON / conversation> [SEP]
   |
   3. ONE BATCHED FORWARD PASS through a BERT-style encoder (bidirectional, reads everything at once)
   |    + a small "decision head" (2 transformer layers)
   |
   4. SCORE EACH [MASK] MARKER -> one number per option -> softmax -> probabilities
   |    divided by a calibration temperature first
   |
   5. DECODE BY TYPE
        choice -> the most likely label + all probabilities
        score  -> expected level = sum(level x probability)
        noul   -> probability of the "true" option
```

## Why it can't hallucinate

It never generates tokens. Each option gets a score and the answer is always one of **your**
options, so there is no JSON to parse and no invented label. It can still be **wrong**.

## Why it's fast

* **Non-autoregressive.** An LLM produces its answer one token at a time. Laya reads the input once and scores it.
* **Small.** 322M to 421M parameters, against billions for an LLM.
* **Batched.** All questions go through the model together (one row per question).

## What the cost depends on

* **Number of questions.** Each question is its own row, so 10 questions cost more than 1. The project measured 39.5 ms for 1 question and 158.6 ms for 10 on the English checkpoint (T4). The multilingual checkpoint scales better: 32.8 ms and 72.3 ms.
* **Input length.** Longer state means longer rows. Speed follows the real length, not `max_len`.
* **CPU or GPU.** On CPU expect roughly 0.2 to 0.5 s per call.

## How the input is cut when it's too long

The question and options go first and your state fills the space left over:

* `english`: 512 tokens in total, of which 192 are for question and options, so about 320 tokens of state.
* `multilingual` / `typed-decisions`: 1024 tokens in total, about 768 for state. Pass `max_len=8192` for long documents.
* **Conversation lists are cut from the left**, so the most recent turns are kept. That suits chat and voice.
* Many options with long descriptions share the 192/256-token option budget, so each description gets truncated. Above about 20 options, accuracy drops (see limits).

## How it was trained (RLCD)

It was trained with reinforcement learning, where the reward is a **strictly proper scoring rule**
(like the Brier or log score). That reward is highest when the stated probability matches how
often the model is really right, so the probabilities are meant to be **meaningful**, not just
rankings. In practice the shipped checkpoints are still over-confident until you fit temperatures
on your own data (see limits).

## Confidence numbers in the result

```python
a = result["answers"]["department"]
a["probabilities"]      # every option's probability
a["answer_confidence"]  # probability of the chosen answer. Use this for thresholds, on every type.
a["confidence"]         # choice/score: 1 - normalised entropy (NOT a probability); noul: max(p, 1-p)
a["action"]["act_probability"]  # ignore: the project says it carries no signal yet (#185)
```
