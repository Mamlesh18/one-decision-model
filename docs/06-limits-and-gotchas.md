# 6. Limits and gotchas (read before production)

Most of these come from the project's own "Honest limits" section. The last one we found while testing.

| # | Limit | What to do |
|---|-------|-----------|
| 1 | **Zero-shot is weak on complex domain decisions.** On typed-decisions the base checkpoints score 0.36 and 0.35, against a random baseline of 0.32 and a majority-class baseline of 0.46 | Fine-tune (Kaggle notebook, about 4-5 h on 2x T4). The fine-tuned model reaches 0.77 |
| 2 | **Over-confident as shipped.** laya-multilingual has no fitted temperatures | Fit thresholds and temperatures on your own labelled data. Don't reuse the example thresholds |
| 3 | **More than about 20 options degrades** (Banking77 0.425) | Split into coarse and fine questions, use `predict_shortlist`, or raise `head_max_len` |
| 4 | **Negation** ("I do NOT want to cancel") can still pick `cancel` (#377) | Test negated phrasings; add a rule or send those to an LLM |
| 5 | **`noul` label bias** on the English checkpoint (#156) | Give `true`/`false` descriptions, set `labels`, or use a 2-option choice with keys `A`/`B` |
| 6 | **Boolean-word choice keys** (`yes`/`no`) get followed literally | Use semantic or neutral keys |
| 7 | **Score is the weakest type** (SST-5 0.372). laya-multilingual rarely picks the first level (#131) | Prefer choice or noul where you can; send English score questions to `model="english"` |
| 8 | **The English checkpoint collapses outside English** (Khmer 0.000 at 95% confidence), and its confidence gives no warning | Always use the Router, or pin `multilingual` |
| 9 | **Short Latin-script text is often misdetected** ("Quero cancelar" is routed as English) | `Router(default="multilingual")`, `lang_guess=`, or pin `model=` |
| 10 | **Text only.** No audio, prosody or emotion from the voice | Combine with VAD and prosody features |
| 11 | **No extraction and no generation** | Keep NER or an LLM for slots and replies |
| 12 | **`act_probability` carries no signal** (#185) | Ignore it; use `answer_confidence` |
| 13 | **CPU latency is 0.2-0.5 s** | Use a GPU for real-time voice, or keep it to off-critical-path decisions on CPU |
| 14 | **English context is only 512 tokens** (about 320 for state) | Send long documents to `multilingual` with `max_len=8192`; keep chat history short |
| 15 | **Found in our tests: on PyPI 0.3.20, `lang_guess="C.UTF-8"` routes English text to multilingual.** The upstream README says it abstains, but that fix is only on `main` | Don't pass `$LANG` straight through as `lang_guess` on 0.3.20 (Docker images set `C.UTF-8`). See `tests/test_routing_offline.py` |

## Published benchmark numbers: how to read them

* The Laya versus Jev comparisons use Jev figures published by third parties; the Laya team did not measure Jev itself, and prompts and sample sizes differ.
* The 0.766 typed-decisions result is from a checkpoint fine-tuned on that benchmark's own training split.
* Treat every number as a claim until `benchmarks/` reproduces it on your data.
