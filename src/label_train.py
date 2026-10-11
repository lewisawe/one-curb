"""Teacher-label the ~400 TRAIN rows into data/train.labeled.jsonl.

HOW THIS WORKS
--------------
Each raw TRAIN note is sent to a STRONG open model served by Tinker
(default: openai/gpt-oss-120b) together with the per-category evidence rubric
from `src/rubric.py`. The model returns a gold object conforming to GOLD_SCHEMA.
This is **teacher labeling**: the labels are model-generated, disclosed as such
in every row (`"teacher_labeled": true`, with the teacher model id) and in
data/train.labeled.README. These TRAIN labels are the fine-tune's supervision;
they are explicitly NOT held-out gold.

The held-out rows are a DIFFERENT path: they must be hand-labeled by a human
(`src/label_heldout.py`). Never run this script on held-out data.

DETERMINISM / SAFETY
--------------------
* temperature 0, fixed seed -> reproducible labels.
* `followup_instruction` is NOT trusted from the model: it is recomputed
  deterministically from (category, missing_evidence) via rubric.followup_for,
  so it is always consistent with the rubric priority.
* every produced object is validated against the schema; invalid rows are
  retried once, then written to data/train.label_errors.jsonl for inspection
  (never silently dropped, never fabricated).
* resumable: rows already present in the output are skipped.
* the API key is loaded from .env and never printed.

USAGE
-----
    python3 src/label_train.py              # label all remaining TRAIN rows
    python3 src/label_train.py --limit 8    # label only the first 8 (smoke)
    python3 src/label_train.py --selftest   # no network; render one prompt
    python3 src/label_train.py --model openai/gpt-oss-20b
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from src import rubric  # noqa: E402

TRAIN_RAW = ROOT / "data" / "train.raw.jsonl"
TRAIN_OUT = ROOT / "data" / "train.labeled.jsonl"
ERR_OUT = ROOT / "data" / "train.label_errors.jsonl"
ENV_PATH = ROOT / ".env"

DEFAULT_TEACHER = "openai/gpt-oss-120b"


# --------------------------------------------------------------------------- #
# Prompt construction                                                          #
# --------------------------------------------------------------------------- #

def _rubric_block(category: str) -> str:
    spec = rubric.RUBRIC[category]
    lines = [f"Category {category}: {rubric.TAXONOMY[category]}",
             f"PROWAG: {spec['prowag']}",
             "Required evidence fields (use these exact field names in "
             "missing_evidence):"]
    for ev in spec["required_evidence"]:
        lines.append(f"  - {ev['field']}: {ev['label']} — {ev['why']}")
    return "\n".join(lines)


def build_messages(row: dict) -> list[dict]:
    """Build the chat messages for one raw row, including the suggested category
    and that category's rubric so the teacher grades evidence correctly."""
    suggested = rubric.suggest_category(row["issue_type"], row["issue_description"])
    taxonomy_list = ", ".join(rubric.TAXONOMY.keys())
    sev_lines = "\n".join(f"  {k} = {v}" for k, v in rubric.SEVERITY_SCALE.items())

    system = (
        "You label messy civic field notes about pedestrian ACCESSIBILITY "
        "barriers for a public-works crew. You output ONE compact JSON object "
        "and nothing else — no prose, no code fences.\n\n"
        f"Taxonomy categories: {taxonomy_list}.\n"
        "Severity is the Project Sidewalk 1-5 accessibility-impact scale:\n"
        f"{sev_lines}\n\n"
        "Output JSON keys EXACTLY: category, severity, objective_description, "
        "location, missing_evidence.\n"
        "  - category: the single best taxonomy category for THIS note.\n"
        "  - severity: integer 1-5.\n"
        "  - objective_description: a neutral, factual one-sentence restatement "
        "of the barrier for a work order (no opinions, no 'please').\n"
        "  - location: the most findable location you can derive from the note + "
        "address (which corner/side/frontage). If the note gives nothing beyond "
        "the street address, use the address.\n"
        "  - missing_evidence: a JSON array of the REQUIRED-EVIDENCE field names "
        "(from the rubric below) that the note does NOT already provide. A field "
        "is 'provided' only if the note gives the concrete value (e.g. an actual "
        "measurement in inches, not just 'it's narrow'). Judge honestly — most "
        "field notes are missing a hard measurement.\n"
        "Do NOT output followup_instruction; it is computed separately.\n"
    )

    user = (
        f"CITY issue_type: {row['issue_type']}\n"
        f"SUGGESTED category: {suggested}\n"
        f"ADDRESS: {row.get('address','')}\n"
        f"NOTE: {row['issue_description']}\n\n"
        "Rubric for the suggested category (if you pick a different category, "
        "apply that category's analogous fields):\n"
        f"{_rubric_block(suggested)}\n\n"
        "Return the JSON object now."
    )
    return [{"role": "system", "content": system},
            {"role": "user", "content": user}]


# --------------------------------------------------------------------------- #
# Output parsing                                                               #
# --------------------------------------------------------------------------- #

_FINAL_RE = re.compile(r"<\|channel\|>final<\|message\|>(.*?)(?:<\|return\|>|<\|end\|>|$)",
                       re.DOTALL)
_JSON_RE = re.compile(r"\{.*?\}", re.DOTALL)


def extract_json(raw: str) -> dict | None:
    """Pull the first JSON object out of a teacher completion.

    Handles the gpt-oss harmony format (analysis + final channels) and plain
    completions. Returns a dict or None.
    """
    candidate = raw
    m = _FINAL_RE.search(raw)
    if m:
        candidate = m.group(1)
    # strip code fences if any
    candidate = candidate.replace("```json", "").replace("```", "")
    jm = _JSON_RE.search(candidate)
    if not jm:
        # fall back to scanning the whole raw output
        jm = _JSON_RE.search(raw)
        if not jm:
            return None
    try:
        return json.loads(jm.group(0))
    except json.JSONDecodeError:
        return None


def finalize(row: dict, parsed: dict, teacher_model: str) -> dict:
    """Build a validated gold row from a parsed teacher object.

    Recomputes followup_instruction deterministically from the rubric so it is
    never left to the model. Returns the gold row (may still be invalid — caller
    validates)."""
    category = parsed.get("category", "")
    missing = parsed.get("missing_evidence", [])
    if not isinstance(missing, list):
        missing = []
    # keep only valid field names for the chosen category (defensive)
    allowed = set(rubric.required_fields(category))
    missing = [m for m in missing if m in allowed]
    followup = rubric.followup_for(category, missing)
    sev = parsed.get("severity")
    try:
        sev = int(sev)
    except (TypeError, ValueError):
        sev = sev  # leave bad value for validator to reject
    gold = {
        "ticket_id": row["ticket_id"],
        "issue_type": row["issue_type"],
        "issue_description": row["issue_description"],
        "address": row.get("address", ""),
        "category": category,
        "severity": sev,
        "objective_description": str(parsed.get("objective_description", "")).strip(),
        "location": str(parsed.get("location", row.get("address", ""))).strip(),
        "missing_evidence": missing,
        "followup_instruction": followup,
        # ---- teacher-label disclosure (do not remove) ----
        "teacher_labeled": True,
        "teacher_model": teacher_model,
        "label_source": "teacher (model-generated via src/label_train.py)",
    }
    return gold


# --------------------------------------------------------------------------- #
# IO helpers                                                                   #
# --------------------------------------------------------------------------- #

def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids = set()
    for r in read_jsonl(path):
        ids.add(str(r.get("ticket_id")))
    return ids


# --------------------------------------------------------------------------- #
# Tinker sampling                                                              #
# --------------------------------------------------------------------------- #

class Teacher:
    """Lazy Tinker sampling client + tokenizer for one teacher model."""

    def __init__(self, model: str):
        self.model = model
        self._sampler = None
        self._tok = None

    def _ensure(self):
        if self._sampler is not None:
            return
        from dotenv import load_dotenv
        load_dotenv(str(ENV_PATH))
        if not os.environ.get("TINKER_API_KEY"):
            raise SystemExit("TINKER_API_KEY not set (looked in .env). Aborting.")
        import tinker
        self._tinker = tinker
        sc = tinker.ServiceClient()
        self._sampler = sc.create_sampling_client(base_model=self.model)
        self._tok = self._sampler.get_tokenizer()

    def complete(self, messages: list[dict], max_tokens: int = 900) -> str:
        self._ensure()
        text = self._tok.apply_chat_template(messages, add_generation_prompt=True,
                                             tokenize=False)
        ids = self._tok.encode(text, add_special_tokens=False)
        mi = self._tinker.ModelInput.from_ints(ids)
        # NB: for harmony-format models (gpt-oss) stop ONLY on <|return|> — the
        # end-of-turn token. <|end|> merely closes the hidden `analysis` channel
        # and would cut sampling off *before* the final JSON is emitted.
        sp = self._tinker.SamplingParams(
            max_tokens=max_tokens, temperature=0.0, seed=0,
            stop=["<|return|>"],
        )
        resp = self._sampler.sample(prompt=mi, num_samples=1, sampling_params=sp).result()
        return self._tok.decode(resp.sequences[0].tokens)


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #

def selftest() -> int:
    rubric.validate_self()
    rows = read_jsonl(TRAIN_RAW)
    print(f"[selftest] loaded {len(rows)} train rows; schema OK")
    msgs = build_messages(rows[0])
    print("[selftest] example system/user prompt for row 1:")
    print("--- SYSTEM ---")
    print(msgs[0]["content"][:600], "...")
    print("--- USER ---")
    print(msgs[1]["content"])
    # exercise the parser + finalizer on a canned teacher output
    canned = ('<|channel|>final<|message|>{"category":"NoCurbRamp","severity":4,'
              '"objective_description":"No sloped curb ramp at the crossing.",'
              '"location":"97a Broadway crosswalk","missing_evidence":'
              '["location","measurement"]}<|return|>')
    parsed = extract_json(canned)
    gold = finalize(rows[0], parsed, "SELFTEST")
    errs = rubric.validate_gold_object({k: gold[k] for k in rubric.GOLD_FIELDS})
    print("[selftest] canned parse ->", json.dumps({k: gold[k] for k in rubric.GOLD_FIELDS}))
    print("[selftest] validation errors:", errs)
    print("[selftest] followup chosen:", gold["followup_instruction"][:80], "...")
    return 0 if not errs else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_TEACHER,
                    help=f"teacher model id (default {DEFAULT_TEACHER})")
    ap.add_argument("--limit", type=int, default=0,
                    help="only label the first N remaining rows (0 = all)")
    ap.add_argument("--selftest", action="store_true",
                    help="no network; validate schema + render one prompt")
    args = ap.parse_args()

    if args.selftest:
        return selftest()

    rubric.validate_self()
    rows = read_jsonl(TRAIN_RAW)
    already = done_ids(TRAIN_OUT)
    todo = [r for r in rows if str(r["ticket_id"]) not in already]
    if args.limit:
        todo = todo[: args.limit]

    print(f"teacher model : {args.model}")
    print(f"train rows    : {len(rows)}  already labeled: {len(already)}  "
          f"to do: {len(todo)}")
    if not todo:
        print("nothing to do.")
        return 0

    teacher = Teacher(args.model)
    n_ok = n_err = 0
    with TRAIN_OUT.open("a") as out, ERR_OUT.open("a") as errf:
        for i, row in enumerate(todo, 1):
            msgs = build_messages(row)
            gold = None
            for attempt in (1, 2):
                raw = teacher.complete(msgs)
                parsed = extract_json(raw)
                if parsed is None:
                    continue
                cand = finalize(row, parsed, args.model)
                verrs = rubric.validate_gold_object({k: cand[k] for k in rubric.GOLD_FIELDS})
                if not verrs:
                    gold = cand
                    break
                last_errs, last_raw = verrs, raw
            if gold is not None:
                out.write(json.dumps(gold, ensure_ascii=False) + "\n")
                out.flush()
                n_ok += 1
            else:
                errf.write(json.dumps({"ticket_id": row["ticket_id"],
                                       "errors": locals().get("last_errs", ["no JSON parsed"]),
                                       "raw": locals().get("last_raw", "")[:2000]},
                                      ensure_ascii=False) + "\n")
                errf.flush()
                n_err += 1
            if i % 10 == 0 or i == len(todo):
                print(f"  [{i}/{len(todo)}] ok={n_ok} err={n_err}")

    print(f"done. wrote {n_ok} labels to {TRAIN_OUT.name}; {n_err} errors "
          f"to {ERR_OUT.name} (if any).")
    _write_disclosure(args.model)
    return 0


def _write_disclosure(model: str) -> None:
    note = ROOT / "data" / "train.labeled.README"
    note.write_text(
        "data/train.labeled.jsonl — TEACHER-LABELED TRAINING DATA\n"
        "========================================================\n\n"
        "These ~400 gold labels were generated by a strong open model served by\n"
        f"Tinker (teacher model: {model}) using the PROWAG-grounded rubric in\n"
        "src/rubric.py, via src/label_train.py. They are MODEL-GENERATED, not\n"
        "human gold. Every row carries \"teacher_labeled\": true and the teacher\n"
        "model id. followup_instruction is recomputed deterministically from the\n"
        "rubric (not taken from the model).\n\n"
        "This disclosure must be surfaced in the write-up. The HELD-OUT eval gold\n"
        "(data/heldout.labeled.jsonl) is DIFFERENT: it is hand-labeled by a human\n"
        "via src/label_heldout.py and must never be model-generated.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
