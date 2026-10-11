"""HAND-LABEL the ~100 HELD-OUT rows into data/heldout.labeled.jsonl.

THIS TOOL CALLS NO MODEL. A human must author every held-out gold label by
reading each note against the rubric. The held-out gold is the ONLY honest
yardstick for the eval, so it must never be model-generated and nothing is
pre-filled for `missing_evidence` / `followup_instruction` — the human decides
those.

For each held-out row the tool shows:
  * the raw note, city issue_type, and address;
  * a SUGGESTED category (from the rubric's issue_type map + text refiners) that
    the labeler can accept or override — this is a hint, not a label;
  * the chosen category's required-evidence checklist from src/rubric.py.

It then captures the gold object:
  * category   (pick from the taxonomy)
  * severity   (1-5, Project Sidewalk scale)
  * objective_description (your neutral restatement)
  * location   (the findable location you'd give a crew)
  * missing_evidence (you tick which required fields the note does NOT supply)
  * followup_instruction (computed from the rubric once you pick missing fields;
    you may edit it, but it is never auto-filled before you choose)

Resumable: rows already in data/heldout.labeled.jsonl are skipped. Progress is
written after every row so you can stop (q) and come back.

USAGE
-----
    python3 src/label_heldout.py            # label all remaining held-out rows
    python3 src/label_heldout.py --selftest # no writes; load schema + render row 1
    python3 src/label_heldout.py --review    # print a summary of labels so far

This is a plain-terminal tool (stdlib only). No network, no API key, no model.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from src import rubric  # noqa: E402

HELDOUT_RAW = ROOT / "data" / "heldout.raw.jsonl"
HELDOUT_OUT = ROOT / "data" / "heldout.labeled.jsonl"

CATS = list(rubric.TAXONOMY.keys())


# --------------------------------------------------------------------------- #
# IO                                                                           #
# --------------------------------------------------------------------------- #

def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def labeled_ids() -> set[str]:
    if not HELDOUT_OUT.exists():
        return set()
    return {str(r.get("ticket_id")) for r in read_jsonl(HELDOUT_OUT)}


def append_gold(gold: dict) -> None:
    with HELDOUT_OUT.open("a") as fh:
        fh.write(json.dumps(gold, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------- #
# Rendering                                                                    #
# --------------------------------------------------------------------------- #

def hr(ch: str = "-") -> str:
    return ch * 72


def show_note(row: dict, idx: int, total: int) -> None:
    print()
    print(hr("="))
    print(f"HELD-OUT ROW {idx}/{total}   ticket_id={row['ticket_id']}")
    print(hr("="))
    print(f"city issue_type : {row['issue_type']}")
    print(f"address         : {row.get('address','')}")
    print(f"created         : {row.get('ticket_created_date_time','')}")
    print()
    print("NOTE:")
    for line in _wrap(row["issue_description"], 68):
        print("   " + line)
    print()


def show_rubric(category: str) -> None:
    spec = rubric.RUBRIC[category]
    print(hr())
    print(f"RUBRIC — {category}")
    print(f"   {rubric.TAXONOMY[category]}")
    print(f"   PROWAG: {spec['prowag']}")
    print("   Required evidence (tick the ones the note does NOT give):")
    for i, ev in enumerate(spec["required_evidence"], 1):
        print(f"     [{i}] {ev['field']}: {ev['label']}")
        for line in _wrap("why: " + ev["why"], 60):
            print(f"         {line}")
    print(hr())


def _wrap(text: str, width: int) -> list[str]:
    words = str(text).split()
    lines, cur = [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur)
            cur = w
        else:
            cur = (cur + " " + w).strip()
    if cur:
        lines.append(cur)
    return lines or [""]


# --------------------------------------------------------------------------- #
# Prompts                                                                      #
# --------------------------------------------------------------------------- #

def ask(prompt: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default is not None else ""
    while True:
        val = input(f"{prompt}{suffix}: ").strip()
        if val:
            return val
        if default is not None:
            return default
        print("  (required)")


def ask_category(suggested: str) -> str:
    print("Choose category:")
    for i, c in enumerate(CATS, 1):
        mark = " <- suggested" if c == suggested else ""
        print(f"   {i}. {c}{mark}")
    while True:
        raw = input(f"category # or name [{suggested}]: ").strip()
        if not raw:
            return suggested
        if raw.isdigit() and 1 <= int(raw) <= len(CATS):
            return CATS[int(raw) - 1]
        if raw in CATS:
            return raw
        print("  invalid; type a number or an exact category name.")


def ask_severity() -> int:
    print("Severity (Project Sidewalk 1-5):")
    for k, v in rubric.SEVERITY_SCALE.items():
        print(f"   {k} = {v}")
    while True:
        raw = input("severity 1-5: ").strip()
        if raw.isdigit() and 1 <= int(raw) <= 5:
            return int(raw)
        print("  enter an integer 1-5.")


def ask_missing(category: str) -> list[str]:
    """Human ticks which required-evidence fields the note is MISSING. Nothing
    is pre-selected."""
    fields = rubric.required_fields(category)
    print("Which required-evidence fields does the note NOT provide?")
    for i, f in enumerate(fields, 1):
        print(f"   {i}. {f}")
    print("   Enter the numbers of the MISSING fields separated by spaces/commas")
    print("   (e.g. '1 2 4'). Enter 'none' if the note already supplies them all.")
    while True:
        raw = input("missing #s: ").strip().lower()
        if raw in ("none", "n", ""):
            if raw == "":
                print("  type field numbers, or 'none'.")
                continue
            return []
        toks = [t for t in raw.replace(",", " ").split() if t]
        try:
            picks = sorted({int(t) for t in toks})
        except ValueError:
            print("  numbers only (or 'none').")
            continue
        if all(1 <= p <= len(fields) for p in picks):
            return [fields[p - 1] for p in picks]
        print(f"  pick numbers 1..{len(fields)}.")


# --------------------------------------------------------------------------- #
# One row                                                                      #
# --------------------------------------------------------------------------- #

def label_row(row: dict, idx: int, total: int) -> dict | None:
    """Interactively capture one gold row. Returns the gold dict, or None if the
    user asked to quit (q)."""
    show_note(row, idx, total)
    suggested = rubric.suggest_category(row["issue_type"], row["issue_description"])
    print(f"(suggested category from rubric: {suggested} — a hint, override freely)")
    print("Type 'q' at the category prompt to stop and save progress.")

    first = input(f"category # or name [{suggested}] (or q to quit): ").strip()
    if first.lower() == "q":
        return None
    if not first:
        category = suggested
    elif first.isdigit() and 1 <= int(first) <= len(CATS):
        category = CATS[int(first) - 1]
    elif first in CATS:
        category = first
    else:
        # re-prompt with the full menu
        category = ask_category(suggested)

    show_rubric(category)
    severity = ask_severity()
    objective = ask("objective_description (neutral factual restatement)")
    location = ask("location (findable spot for a crew)",
                   default=row.get("address", ""))
    missing = ask_missing(category)

    followup = rubric.followup_for(category, missing)
    if followup:
        print("\nComputed follow-up instruction (highest-priority missing field):")
        for line in _wrap(followup, 66):
            print("   " + line)
        edit = input("Press Enter to accept, or type a replacement: ").strip()
        if edit:
            followup = edit
    else:
        print("\n(no missing evidence -> followup_instruction is empty)")

    gold = {
        "ticket_id": row["ticket_id"],
        "issue_type": row["issue_type"],
        "issue_description": row["issue_description"],
        "address": row.get("address", ""),
        "category": category,
        "severity": severity,
        "objective_description": objective,
        "location": location,
        "missing_evidence": missing,
        "followup_instruction": followup,
        # ---- provenance: HUMAN gold, never model-generated ----
        "teacher_labeled": False,
        "label_source": "human (hand-labeled via src/label_heldout.py)",
    }
    errs = rubric.validate_gold_object({k: gold[k] for k in rubric.GOLD_FIELDS})
    if errs:
        print("\n!! this label is invalid against the schema:")
        for e in errs:
            print("   -", e)
        again = input("re-enter this row? [Y/n]: ").strip().lower()
        if again in ("", "y", "yes"):
            return label_row(row, idx, total)
        print("   (saving anyway — fix later)")
    return gold


# --------------------------------------------------------------------------- #
# Modes                                                                        #
# --------------------------------------------------------------------------- #

def selftest() -> int:
    rubric.validate_self()
    rows = read_jsonl(HELDOUT_RAW)
    print(f"[selftest] loaded {len(rows)} held-out rows; schema OK; model calls: NONE")
    show_note(rows[0], 1, len(rows))
    cat = rubric.suggest_category(rows[0]["issue_type"], rows[0]["issue_description"])
    print(f"[selftest] suggested category for row 1: {cat}")
    show_rubric(cat)
    print("[selftest] rendered row 1 + rubric WITHOUT writing any gold.")
    return 0


def review() -> int:
    if not HELDOUT_OUT.exists():
        print("no held-out labels yet.")
        return 0
    rows = read_jsonl(HELDOUT_OUT)
    import collections
    cats = collections.Counter(r["category"] for r in rows)
    bad = 0
    for r in rows:
        if rubric.validate_gold_object({k: r.get(k) for k in rubric.GOLD_FIELDS}):
            bad += 1
    print(f"held-out labels: {len(rows)}  invalid: {bad}")
    for c, n in cats.most_common():
        print(f"   {n:>3}  {c}")
    nonhuman = [r for r in rows if r.get("teacher_labeled") is not False]
    if nonhuman:
        print(f"!! {len(nonhuman)} rows not marked human-labeled — held-out gold "
              "must be human. Investigate.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true",
                    help="no writes; load schema + render row 1")
    ap.add_argument("--review", action="store_true",
                    help="summarize labels written so far")
    args = ap.parse_args()

    if args.selftest:
        return selftest()
    if args.review:
        return review()

    rubric.validate_self()
    rows = read_jsonl(HELDOUT_RAW)
    done = labeled_ids()
    todo = [(i, r) for i, r in enumerate(rows, 1) if str(r["ticket_id"]) not in done]

    print(hr("="))
    print("ONE CURB — HELD-OUT HAND-LABELER  (no model; human gold only)")
    print(hr("="))
    print(f"held-out rows : {len(rows)}   already labeled: {len(done)}   "
          f"remaining: {len(todo)}")
    print(f"writing to    : {HELDOUT_OUT}")
    print("Quit any time with 'q' at the category prompt — progress is saved per row.")
    if not todo:
        print("\nAll held-out rows are labeled. Run with --review to summarize.")
        return 0

    labeled_now = 0
    for idx, row in todo:
        gold = label_row(row, idx, len(rows))
        if gold is None:
            print("\nStopped. Progress saved. Re-run to continue.")
            break
        append_gold(gold)
        labeled_now += 1
        print(f"  saved. ({len(done) + labeled_now}/{len(rows)} done)")

    remaining = len(rows) - (len(done) + labeled_now)
    print()
    print(hr())
    if remaining == 0:
        print(f"ALL {len(rows)} held-out rows hand-labeled. "
              f"Gold written to {HELDOUT_OUT.name}.")
        print("The eval can now use these as the honest yardstick.")
    else:
        print(f"{len(done) + labeled_now}/{len(rows)} labeled; {remaining} remaining. "
              "Re-run `python3 src/label_heldout.py` to finish.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
