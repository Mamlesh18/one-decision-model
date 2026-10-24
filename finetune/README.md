# Fine-tuning Laya

Fine-tune Laya for a clinic voice/chat assistant on five decisions:

| Question | Type | Answers |
|---|---|---|
| `intent` | choice | appointment_booking, appointment_reschedule, appointment_cancel, check_slot, refund, billing_query, escalation, general_info, other |
| `guardrail` | choice | safe, prompt_injection, abusive, privacy_violation, medical_emergency, off_topic |
| `frustration` | score | 0 calm, 1 slightly annoyed, 2 frustrated, 3 angry |
| `unsafe` | noul | should the bot block this message? |
| `needs_human` | noul | should the call go to a person? |

## Files

| File | What it is |
|---|---|
| `laya_finetune_colab.ipynb` | The notebook. Upload it to Colab, pick a T4 GPU, then Run all. It already contains the generator and the questions, so there is nothing else to upload. |
| `make_dataset.py` | Mock data generator: templates, random slots, speech-to-text noise and hard negatives. The test split uses only held-out templates. |
| `questions.json` | The five questions in production wording, plus paraphrases used during training. |
| `data/` | The generated `train` (4,000), `val` (500) and `test` (500) splits. |
| `build_notebook.py` | Rebuilds the notebook after you edit `make_dataset.py` or `questions.json`. |

## Run it

1. Open https://colab.research.google.com, choose File, then Upload notebook, and pick `laya_finetune_colab.ipynb`.
2. Choose Runtime, then Change runtime type, then **T4 GPU**.
3. Choose Runtime, then **Run all**.
   - `TRAIN_MODE = "full"` trains the encoder and the head. It should take about 20–30 minutes on a T4 (an estimate; not yet timed).
   - `TRAIN_MODE = "head"` trains only the decision head and takes minutes.
   - `QUICK_TEST = True` runs a 1-minute smoke test.
4. The notebook prints a before/after table (accuracy and ECE per question) on the held-out test set. It then saves `laya-clinic/`, which loads with `laya.Agent("laya-clinic")`. You can download it as a zip or push it to the Hugging Face Hub.

## Using your own data

Use the same JSONL shape, one conversation per line, with turns as `role`/`content`:

```json
{"id": "c1", "state": [{"role": "user", "content": "move my slot to friday"}],
 "labels": {"intent": "appointment_reschedule", "guardrail": "safe", "frustration": 0, "unsafe": false, "needs_human": false}}
```

- A label can be missing; that question is then skipped for the row.
- For distillation from GPT or Jev, a label can be a probability distribution:
  - choice: `{"refund": 0.7, "billing_query": 0.3}`
  - score: `{"0": 0.1, "1": 0.9}`
  - noul: `0.85`, meaning P(true)

## What it does

- **Objective:** it maximises Laya's strictly proper scoring rule (`laya.common.proper_reward`: log score, spherical score, and RPS for score questions). This is the same reward RLCD uses, applied directly to the model's distribution.
- **Augmentation:** each epoch, choice options are shuffled and question wording is paraphrased, so the model reads the options rather than learning their positions.
- **Model selection:** it keeps the epoch with the best validation accuracy.
- **Calibration:** it then fits one temperature per question type on the validation split and removes the inherited bucket temperatures.

## Caveats

- **The data is mock data.** High scores here prove the pipeline works; they don't predict accuracy on real calls. Before deciding anything, retrain on labelled real transcripts and measure on a human-labelled test set.
- **How it was verified:** the notebook was run end to end on CPU in both modes against a tiny random checkpoint with Laya's file layout (train, calibrate, evaluate, save, reload, predict). It has not yet been run with the real 421M weights on a GPU.
