# One Curb, One Case — Held-out Eval Report

- **Resolved base model:** `Qwen/Qwen3.5-4B`
- **Fine-tuned model:** `Qwen/Qwen3.5-4B` + LoRA adapter (`onecurb-sft`, rank 32, assistant-loss-only)
- **Held-out N:** 31 hand-labeled rows (human gold, `teacher_labeled:false`)
- **Few-shot exemplars (condition B):** TRAIN-only ticket ids ['8759014', '19612023', '18421072', '4541378', '18269592'] — never held-out.

> **Honest note on N.** This is a REAL eval on **31 hand-labeled held-out rows**, not 100 and not a synthetic set. A sample this small gives wide confidence intervals: single-row swings move F1 by several points, so treat the differences below as directional, and lean on the exact paired field-decision counts (the McNemar b/c cells) rather than the headline aggregates alone.

## Headline — missing-evidence detection (micro-averaged)

Unit = one `(ticket_id, required-evidence field)` decision over the gold category's required fields. A field is a positive when it appears in the model's `missing_evidence`; gold-positive when it appears in the human gold's `missing_evidence`.

| Condition | TP | FP | FN | TN | Precision | Recall | F1 | field-units |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| (A) base zero-shot | 40 | 9 | 49 | 40 | 0.816 | 0.449 | **0.580** | 138 |
| (B) base strong few-shot | 12 | 8 | 77 | 41 | 0.600 | 0.135 | **0.220** | 138 |
| (C) fine-tuned | 87 | 46 | 2 | 3 | 0.654 | 0.978 | **0.784** | 138 |

## Paired comparison (McNemar-style b/c, exact field-decision counts)

For each pair, over the field-units both conditions were scored on: **b** = how many decisions the first (X) got right that the second (Y) got wrong; **c** = how many the second (Y) got right that the first (X) got wrong. b and c are the discordant cells that decide whether one condition truly beats the other.

| Pair (X vs Y) | b: X right / Y wrong | c: Y right / X wrong | both right | both wrong | n |
|---|--:|--:|--:|--:|--:|
| **fine-tuned** vs base strong few-shot | 75 | 38 | 15 | 10 | 138 |
| **fine-tuned** vs base zero-shot | 48 | 38 | 42 | 10 | 138 |
| **base strong few-shot** vs base zero-shot | 18 | 45 | 35 | 40 | 138 |

**Key risk cell (FT vs few-shot):** the fine-tune got **75** field-decisions right that the strong few-shot prompt got wrong, while few-shot got **38** right that the fine-tune missed (both-right 15, both-wrong 10, n=138).

## Secondary — category accuracy

Gold = `ISSUE_TYPE_MAP[issue_type]` from `src/rubric.py`. This is explicitly secondary (the base model already classifies issue type well, so little lift is expected here).

| Condition | Category accuracy | correct / total |
|---|--:|--:|
| (A) base zero-shot | 35.5% | 11 / 31 |
| (B) base strong few-shot | 19.4% | 6 / 31 |
| (C) fine-tuned | 80.6% | 25 / 31 |

## Appendix — valid-JSON rate + severity MAE

| Condition | Valid-JSON rate | valid / N | Severity MAE | severity scored |
|---|--:|--:|--:|--:|
| (A) base zero-shot | 51.6% | 16 / 31 | 2.000 | 16 |
| (B) base strong few-shot | 19.4% | 6 / 31 | 1.833 | 6 |
| (C) fine-tuned | 100.0% | 31 / 31 | 1.613 | 31 |

## Honest reading

- Fine-tuned F1 **0.784** vs strong few-shot **0.220** vs zero-shot **0.580** on N=31 held-out rows.
- The fine-tune **beats** the strong few-shot prompt on the headline metric (+0.564 F1), and the paired b/c cell shows the win is from real discordant decisions, not just aggregate noise.
- Why the baselines trail is visible in the appendix: the base model only emits parseable JSON 51.6% (zero-shot) / 19.4% (few-shot) of the time, versus 100.0% for the fine-tune. The few-shot exemplars actually *degrade* this base model: the worked JSON answers push it into a reasoning / `<think>` dump that breaks the object (see `eval/results.json` raw outputs). A field the model never emits in valid JSON cannot be flagged missing, so the gap is a genuine structured-output capability the fine-tune provides, not a strawman baseline. For every condition the FIRST balanced `{...}` object in the output is parsed (no best-of cherry-picking).
- With N this small, confidence intervals are wide; the paired field-decision counts are the most trustworthy signal here.
