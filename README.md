# One Curb, One Case

An open model that reads a messy field note about a pedestrian **accessibility**
barrier and returns (a) a crew-ready civic report and (b) the headline output: a
list of **missing evidence**, each with the **one concrete follow-up
instruction** to go take ("missing: lip height at the bottom of the ramp — go
measure it in inches"). The fine-tune's job is **missing-evidence detection**,
not prose.

Hacktoberfest 2026 Week 1 — Best Use of Tinker. See `devSubmission.md` for the
write-up and `eval/REPORT.md` for the held-out eval.

## Stack

- Python 3
- [Tinker](https://thinkingmachines.ai/) (Thinking Machines) — LoRA SFT on an
  open Qwen model (`Qwen/Qwen3.5-4B`)
- Flask — single-file one-screen app
- `python-dotenv`, `pandas`, `requests`

## Setup

```bash
# from this project root (.../oneCurb)
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Environment

The Tinker API key is read from a local `.env` via `python-dotenv`. The key
value is never printed. Create `.env` in the project root:

```ini
# .env  (gitignored — never commit this)
TINKER_API_KEY=your-tinker-api-key
```

Optional app overrides: `ONECURB_HOST` (default `127.0.0.1`), `ONECURB_PORT`
(default `8000`).

## Run order

The pipeline is strictly serial: **setup → train → eval → app**. The data splits
are already produced and committed (`data/*.jsonl`), so a normal run starts at
training. The held-out gold (`data/heldout.labeled.jsonl`) is **hand-labeled by
a human** and must exist before eval.

```bash
# 0. (one-time, optional) re-pull + re-split the raw Cambridge Open311 data.
#    Idempotent — skip if data/ already has the splits.
python src/pull_data.py

# 1. Fine-tune with Tinker (LoRA SFT, rank 32, assistant-loss-only).
python src/train.py --smoke        # tiny end-to-end sanity run first (~8 ex)
python src/train.py                # full run -> models/train_meta.json + sampler

# 2. Three-condition held-out eval (base zero-shot / base few-shot / fine-tuned).
python src/eval.py                 # -> eval/results.json + eval/REPORT.md

# 3. One-screen app (serves the fine-tuned Tinker sampler).
python app/server.py               # http://127.0.0.1:8000
python app/smoke_request.py        # scripted POST that validates the response
```

## Layout

```
src/rubric.py        taxonomy + PROWAG-grounded evidence rubric + gold schema (source of truth)
src/pull_data.py     Cambridge Open311 pull, PII scrub, leakage-safe train/held-out split
src/label_train.py   teacher-labels the 400 train rows (disclosed as teacher-labeled)
src/label_heldout.py helper for the HUMAN to hand-label the held-out gold
src/train.py         Tinker LoRA SFT (--smoke for a sanity run)
src/eval.py          three-condition eval -> eval/results.json + eval/REPORT.md
app/server.py        single-file Flask app + app/templates/index.html
app/smoke_request.py scripted request that validates the report against the schema
data/                train/held-out splits (held-out gold is human-authored)
models/              adapter/sampler weights + train_meta.json
eval/                results.json + REPORT.md
shots/               fewshot.json (TRAIN-only exemplars) + app screenshots
```

## Notes

- `data/`, `models/`, and `eval/` artifacts come from the run order above.
- The ~30 held-out gold rows are hand-labeled by a human **after** the model's
  outputs — no model sees a held-out note before its gold exists. That is what
  keeps the eval honest. The 400 train labels are teacher-labeled (disclosed).
- `.env`, `.venv/`, and `__pycache__/` are gitignored; never commit `.env`.
