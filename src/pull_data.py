#!/usr/bin/env python3
"""Pull and prepare Cambridge MA Commonwealth Connect (SeeClickFix / Open311) data.

Source: https://data.cambridgema.gov/resource/2z9k-mv9g.json  (Socrata SoQL)
License: ODC PDDL (public domain), adopted Feb 2024. Attribution: SeeClickFix.

DATA step of the One Curb, One Case pipeline. Produces the raw (pre-gold)
train/held-out splits for the Tinker fine-tune. No API key is used or printed
by this step.

Pipeline (per idea-scorecard.md / one-curb-validation.md section 3):
  1. Pull pedestrian/sidewalk issue types where issue_description is substantive
     (one verified SoQL query).
  2. Stratify-sample ~500 rows by issue_type, oversampling rows whose text
     mentions curb cut / wheelchair / stroller / cane / ramp / tactile.
  3. Split ~400 train / ~100 held-out by BOTH address and ticket_id, so that
     duplicate complaints about the same spot cannot cross the split
     (leakage guard).
  4. Scrub PII (names where detectable, phone numbers, emails, house numbers)
     from the free text.
  5. Write JSONL splits to data/.

Outputs (data/):
  train.raw.jsonl      ~400 rows for the Tinker fine-tune
  heldout.raw.jsonl    ~100 rows to HAND-LABEL for missing-evidence gold

The held-out rows are intentionally left WITHOUT gold labels. Per the brief,
missing-evidence gold must be hand-written by a human before any model output
is seen. This script stops at that boundary.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd
import requests

# --- config -----------------------------------------------------------------

DATASET_URL = "https://data.cambridgema.gov/resource/2z9k-mv9g.json"

# Pedestrian / sidewalk issue types recommended in the scorecard.
ISSUE_TYPES = [
    "Sidewalk Defect",
    "Sidewalk Obstruction",
    "A-Frame/Business Sidewalk Obstruction",
    "Botanical Obstruction",
    "Traffic/Pedestrian Signal Complaint",
]

MIN_DESC_LEN = 40          # chars; matches the verified $where length filter
SOCRATA_LIMIT = 50000      # generous; filtered set is well under this
SAMPLE_SIZE = 500
HELDOUT_SIZE = 100
RANDOM_SEED = 20261011     # reproducible split; deadline date as seed
USER_AGENT = "Mozilla/5.0"

# Terms the pivot cares about. Rows mentioning these are oversampled so the
# accessibility signal is not drowned out by generic sidewalk complaints.
PRIORITY_TERMS = ["curb cut", "wheelchair", "stroller", "cane", "ramp",
                  "tactile"]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


# --- PII scrubbing -----------------------------------------------------------
# Conservative: strip high-confidence patterns only, so we do not destroy the
# location detail a public-works crew needs (cross-streets, frontages).

_PHONE = re.compile(r"\b(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}\b")
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
# Leading house number on a street address, e.g. "123 Main St" -> "<num> Main St".
_STREET_TYPES = (r"St|Street|Ave|Avenue|Rd|Road|Dr|Drive|Ln|Lane|Blvd|"
                 r"Boulevard|Ct|Court|Pl|Place|Way|Ter|Terrace|Cir|Circle|"
                 r"Sq|Square|Hwy|Highway|Pkwy|Parkway|Row")
_HOUSE_NUM = re.compile(
    r"\b\d{1,5}[A-Za-z]?\s+(?=[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*\s+"
    r"(?:" + _STREET_TYPES + r")\b)")
# Named-reporter patterns: "my name is X", "I am X", "Reported by X", "- John Smith".
_NAME_INTRO = re.compile(
    r"\b(?:my name is|i am|this is|reported by|contact|regards,?|thanks,?|"
    r"sincerely,?)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?",
    re.I)


def scrub_pii(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = _EMAIL.sub("<email>", text)
    text = _PHONE.sub("<phone>", text)
    text = _NAME_INTRO.sub("<name>", text)
    text = _HOUSE_NUM.sub("<num> ", text)
    return re.sub(r"\s+", " ", text).strip()


# --- pull / clean ------------------------------------------------------------

def pull_raw() -> pd.DataFrame:
    """Pull the filtered dataset via one verified SoQL query."""
    in_list = ",".join(f"'{t}'" for t in ISSUE_TYPES)
    params = {
        "$select": ("ticket_id,issue_type,issue_description,address,"
                    "ticket_created_date_time"),
        "$where": (f"issue_type in({in_list}) "
                   f"AND length(issue_description)>{MIN_DESC_LEN}"),
        "$limit": SOCRATA_LIMIT,
    }
    headers = {"User-Agent": USER_AGENT}
    print(f"Pulling {DATASET_URL} ...")
    resp = requests.get(DATASET_URL, params=params, headers=headers, timeout=120)
    resp.raise_for_status()
    rows = resp.json()
    df = pd.DataFrame(rows)
    print(f"  pulled {len(df)} rows")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Ensure all expected columns exist even if a row is missing one.
    for col in ("ticket_id", "issue_type", "issue_description", "address",
                "ticket_created_date_time"):
        if col not in df.columns:
            df[col] = ""
    df["issue_description"] = df["issue_description"].map(scrub_pii)
    # Drop rows that became too short after scrubbing.
    df = df[df["issue_description"].str.len() > MIN_DESC_LEN]
    df = df.drop_duplicates(subset=["ticket_id"])
    df["address"] = df["address"].fillna("").astype(str).str.strip()
    return df.reset_index(drop=True)


def stratified_sample(df: pd.DataFrame) -> pd.DataFrame:
    """~SAMPLE_SIZE rows, proportional by issue_type, oversampling priority terms."""
    pat = re.compile("|".join(re.escape(t) for t in PRIORITY_TERMS), re.I)
    df = df.copy()
    df["is_priority"] = df["issue_description"].str.contains(pat)

    # Take priority rows up to ~40% of the sample, then fill the rest
    # proportionally by issue_type from the remainder.
    priority_cap = int(SAMPLE_SIZE * 0.4)
    prio = df[df["is_priority"]]
    if len(prio) > priority_cap:
        prio = prio.sample(priority_cap, random_state=RANDOM_SEED)

    remaining_n = SAMPLE_SIZE - len(prio)
    rest = df[~df.index.isin(prio.index)]
    picks = []
    counts = rest["issue_type"].value_counts()
    total = int(counts.sum())
    for itype, cnt in counts.items():
        share = max(1, round(remaining_n * cnt / total)) if total else 0
        sub = rest[rest["issue_type"] == itype]
        picks.append(sub.sample(min(share, len(sub)), random_state=RANDOM_SEED))
    rest_pick = pd.concat(picks) if picks else rest.head(0)
    if len(rest_pick) > remaining_n:
        rest_pick = rest_pick.sample(remaining_n, random_state=RANDOM_SEED)

    sample = pd.concat([prio, rest_pick]).drop_duplicates(subset=["ticket_id"])
    return sample.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)


def _norm_addr(series: pd.Series) -> pd.Series:
    return (series.astype(str).str.lower()
            .str.replace(r"\s+", " ", regex=True).str.strip())


def split_by_location(df: pd.DataFrame) -> pd.DataFrame:
    """Assign train/heldout so no address (or ticket) spans the split.

    Group by a normalized address key and assign whole groups to held-out
    until we reach ~HELDOUT_SIZE rows. Rows with a blank address fall back to
    their own ticket_id as the group key (each is its own group), so an empty
    address is never a shared bucket.
    """
    df = df.copy()
    norm = _norm_addr(df["address"])
    df["_group"] = norm.where(norm != "",
                              "ticket:" + df["ticket_id"].astype(str))

    group_sizes = df.groupby("_group").size().sample(frac=1,
                                                      random_state=RANDOM_SEED)
    heldout_groups, running = [], 0
    for grp, n in group_sizes.items():
        if running >= HELDOUT_SIZE:
            break
        heldout_groups.append(grp)
        running += int(n)

    df["split"] = df["_group"].isin(heldout_groups).map(
        {True: "heldout", False: "train"})
    return df.drop(columns=["_group"])


def write_jsonl(df: pd.DataFrame, path: Path) -> None:
    cols = ["ticket_id", "issue_type", "issue_description", "address",
            "ticket_created_date_time"]
    with path.open("w", encoding="utf-8") as fh:
        for _, row in df.iterrows():
            rec = {c: ("" if pd.isna(row.get(c)) else row.get(c)) for c in cols}
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


# --- main --------------------------------------------------------------------

def main() -> int:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    raw = pull_raw()
    if raw.empty:
        print("ERROR: no rows returned; aborting.", file=sys.stderr)
        return 1
    total_rows = len(raw)
    per_type_raw = raw["issue_type"].value_counts()

    cleaned = clean(raw)
    print(f"  after PII scrub + dedupe: {len(cleaned)} rows")

    sample = stratified_sample(cleaned)
    sample = split_by_location(sample)

    train = sample[sample["split"] == "train"].reset_index(drop=True)
    heldout = sample[sample["split"] == "heldout"].reset_index(drop=True)
    write_jsonl(train, DATA_DIR / "train.raw.jsonl")
    write_jsonl(heldout, DATA_DIR / "heldout.raw.jsonl")

    # Leakage check: no address (normalized) and no ticket_id across both splits.
    s = sample.copy()
    s["_na"] = _norm_addr(s["address"])
    addr_rows = s[s["_na"] != ""]
    addr_leaks = int(
        (addr_rows.groupby("_na")["split"].nunique() > 1).sum())
    ticket_leaks = int(
        (s.groupby("ticket_id")["split"].nunique() > 1).sum())

    # Console summary (no PII, no secrets).
    print("\n--- summary ---")
    print(f"total rows returned by query : {total_rows}")
    print("per-issue_type counts (raw pull):")
    for itype, cnt in per_type_raw.items():
        print(f"  {int(cnt):6d}  {itype}")
    print(f"cleaned (post-scrub/dedupe)  : {len(cleaned)}")
    print(f"sampled                      : {len(sample)}")
    print(f"train.raw.jsonl              : {len(train)}")
    print(f"heldout.raw.jsonl            : {len(heldout)}  (HAND-LABEL these)")
    prio_n = int(sample["is_priority"].sum())
    print(f"priority-term rows in sample : {prio_n} "
          f"({100 * prio_n / len(sample):.0f}%)")
    print(f"addresses spanning both splits (should be 0): {addr_leaks}")
    print(f"ticket_ids spanning both splits (should be 0): {ticket_leaks}")

    # Machine-readable stats for the README/writer step.
    stats = {
        "query_url": DATASET_URL,
        "issue_types": ISSUE_TYPES,
        "min_desc_len": MIN_DESC_LEN,
        "total_rows_returned": int(total_rows),
        "per_issue_type_counts_raw": {k: int(v) for k, v in per_type_raw.items()},
        "cleaned_rows": int(len(cleaned)),
        "sampled_rows": int(len(sample)),
        "train_rows": int(len(train)),
        "heldout_rows": int(len(heldout)),
        "priority_rows_in_sample": prio_n,
        "address_leaks": addr_leaks,
        "ticket_leaks": ticket_leaks,
        "random_seed": RANDOM_SEED,
    }
    (DATA_DIR / "pull_stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")

    return 0 if (addr_leaks == 0 and ticket_leaks == 0) else 1


if __name__ == "__main__":
    raise SystemExit(main())
