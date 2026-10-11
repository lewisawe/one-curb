"""One Curb, One Case — Tinker LoRA SFT fine-tune (missing-evidence detection).

Trains a small open Qwen model (via the Thinking Machines Tinker training SDK)
to read a messy field note about a pedestrian accessibility barrier and emit the
compact gold JSON object defined in ``src/rubric.py`` (GOLD_FIELDS). The model's
job is **missing-evidence detection**: which required-evidence fields the note
does not already supply, plus the single highest-priority follow-up instruction.

Input  = a FIXED instruction/schema preamble (derived from rubric.py: taxonomy +
         gold schema) followed by the messy note (``issue_description``).
Target = the compact gold JSON object with exactly GOLD_FIELDS.

Loss is **assistant-loss-only**: the per-token loss weight is 1.0 on the target
(completion) tokens and 0.0 on the prompt tokens, so the model is never scored
for reproducing the prompt.

Usage
-----
    python src/train.py --smoke     # tiny end-to-end sanity run (~8 examples)
    python src/train.py             # full LoRA SFT fine-tune

Reads ``TINKER_API_KEY`` from ``.env`` via python-dotenv. The key value is never
printed.

This uses the Thinking Machines Tinker SDK (tinker.ServiceClient ->
create_lora_training_client -> forward_backward / optim_step / cross_entropy
LoRA SFT), confirmed via the installed ``tinker`` package exposing
``create_lora_training_client``, ``forward_backward``, ``LoraConfig``, and
``save_weights_for_sampler``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

import tinker
from tinker import types

# Import the single source of truth for taxonomy / gold schema.
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import src.rubric as rubric  # noqa: E402

# --------------------------------------------------------------------------- #
# Paths & constants
# --------------------------------------------------------------------------- #

TRAIN_PATH = _ROOT / "data" / "train.labeled.jsonl"
MODELS_DIR = _ROOT / "models"
META_PATH = MODELS_DIR / "train_meta.json"

# Target base model (plan item 5). Resolved against the live model list; falls
# back to the closest small Qwen3 the API actually returns.
TARGET_BASE_MODEL = "Qwen3.5-4B"

# LoRA / optimization config (plan item 5: rank ~32, assistant-loss-only,
# 2-3 epochs with early stop on plateau, modest LR, deterministic seed).
LORA_RANK = 32
SEED = 1234
LEARNING_RATE = 1e-4
NUM_EPOCHS = 3
BATCH_SIZE = 8
MAX_SEQ_LEN = 1024
# early stop: stop if best epoch loss does not improve by >= this fraction
PLATEAU_REL_IMPROVEMENT = 0.01

SMOKE_EXAMPLES = 8
SMOKE_STEPS = 4


# --------------------------------------------------------------------------- #
# Prompt construction (fixed preamble derived from rubric.py)
# --------------------------------------------------------------------------- #

def build_preamble() -> str:
    """A FIXED instruction/schema preamble derived from rubric.py.

    Lists the taxonomy categories, the severity scale, the required-evidence
    field names per category (so the model knows what "missing" can mean), and
    the exact gold output schema. Deterministic — identical every run — so the
    same preamble is used at train and inference time.
    """
    cats = ", ".join(sorted(rubric.TAXONOMY.keys()))
    sev_lines = "; ".join(f"{k}={v.split(' — ')[0].split(' - ')[0]}"
                          for k, v in rubric.SEVERITY_SCALE.items())
    req_lines = []
    for cat in sorted(rubric.RUBRIC.keys()):
        fields = rubric.required_fields(cat)
        req_lines.append(f"  {cat}: {', '.join(fields)}")
    required_block = "\n".join(req_lines)

    schema_keys = ", ".join(rubric.GOLD_FIELDS)
    return (
        "You convert a messy pedestrian-accessibility field note into a compact "
        "civic report and detect what evidence is still missing.\n"
        "\n"
        f"Barrier categories (pick exactly one): {cats}.\n"
        f"Severity is an integer 1-5 (Project Sidewalk scale): {sev_lines}.\n"
        "\n"
        "Each category requires this evidence; flag any field the note does NOT "
        "already supply:\n"
        f"{required_block}\n"
        "\n"
        "Output ONLY a single JSON object with exactly these keys:\n"
        f"  {schema_keys}\n"
        "where missing_evidence is a list of the required-evidence field names "
        "the note fails to supply, and followup_instruction is the single "
        "highest-priority follow-up for that category (empty string when nothing "
        "is missing). Do not output anything except the JSON object.\n"
    )


PREAMBLE = build_preamble()


def gold_target(row: dict) -> dict:
    """Extract the compact gold object (exactly GOLD_FIELDS) from a labeled row."""
    return {k: row[k] for k in rubric.GOLD_FIELDS}


def render_prompt(row: dict) -> str:
    """The prompt text (preamble + the messy note). No target included."""
    note = (row.get("issue_description") or "").strip()
    return f"{PREAMBLE}\nFIELD NOTE:\n{note}\n\nJSON:"


def render_target(row: dict) -> str:
    """The completion text: the compact gold JSON object, compactly serialized."""
    obj = gold_target(row)
    # compact, deterministic serialization
    return json.dumps(obj, ensure_ascii=False, separators=(",", ": "), sort_keys=False)


# --------------------------------------------------------------------------- #
# Datum construction (assistant-loss-only masking)
# --------------------------------------------------------------------------- #

def build_datum(tokenizer, row: dict, eos_id: int) -> types.Datum | None:
    """Tokenize one row into a Datum with assistant-loss-only weights.

    Tokens = prompt_tokens + target_tokens + [eos].
    For cross_entropy the model predicts token[t+1] from token[t]; so the inputs
    are tokens[:-1] and the targets are tokens[1:]. The loss weight is 1.0 only
    on positions whose *target* is a completion (target or eos) token, and 0.0
    on positions whose target is still part of the prompt.
    """
    prompt_text = render_prompt(row)
    target_text = render_target(row)

    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    target_ids = tokenizer.encode(target_text, add_special_tokens=False)
    full_ids = prompt_ids + target_ids + [eos_id]

    if len(full_ids) > MAX_SEQ_LEN:
        # Keep the whole completion; truncate the FRONT of the prompt so we never
        # drop target tokens (which would corrupt the loss signal).
        overflow = len(full_ids) - MAX_SEQ_LEN
        if overflow >= len(prompt_ids):
            # pathological: completion alone too long -> skip this row
            return None
        prompt_ids = prompt_ids[overflow:]
        full_ids = prompt_ids + target_ids + [eos_id]

    # next-token setup
    input_ids = full_ids[:-1]
    target_tokens = full_ids[1:]

    # weight is 1.0 where the *target* token is a completion token.
    # completion tokens are the last (len(target_ids)+1) tokens of full_ids,
    # i.e. target positions index >= len(prompt_ids)-1 in the shifted array.
    n_prompt = len(prompt_ids)
    weights = [0.0] * len(target_tokens)
    for i in range(len(target_tokens)):
        # target_tokens[i] == full_ids[i+1]; it's a completion token when
        # i+1 >= n_prompt  (first completion token is full_ids[n_prompt]).
        if (i + 1) >= n_prompt:
            weights[i] = 1.0

    if not any(w > 0 for w in weights):
        return None

    model_input = types.ModelInput.from_ints(input_ids)
    datum = types.Datum(
        model_input=model_input,
        loss_fn_inputs={
            "target_tokens": types.TensorData.from_numpy(
                np.array(target_tokens, dtype=np.int64)
            ),
            "weights": types.TensorData.from_numpy(
                np.array(weights, dtype=np.float32)
            ),
        },
    )
    return datum


# --------------------------------------------------------------------------- #
# Loss extraction
# --------------------------------------------------------------------------- #

def weighted_mean_loss_from_output(fb_out, data: list[types.Datum]) -> float:
    """Compute the assistant-token weighted-mean cross-entropy loss for a batch.

    Prefers an explicit scalar in ``fb_out.metrics`` if present; otherwise falls
    back to combining the per-token ``logprobs`` loss outputs with the datum
    weights. Returns NaN-free float.
    """
    metrics = getattr(fb_out, "metrics", None) or {}
    for key in ("loss:sum", "loss", "train_loss", "nll", "cross_entropy"):
        if key in metrics:
            # metrics loss may already be per-token mean; if it's a sum we still
            # get a comparable monotonic signal. Prefer a mean-like key.
            pass
    # Robust path: reconstruct from loss_fn_outputs (per-token logprobs).
    total_loss = 0.0
    total_weight = 0.0
    outs = getattr(fb_out, "loss_fn_outputs", None) or []
    for out, datum in zip(outs, data):
        w = np.asarray(datum.loss_fn_inputs["weights"].to_numpy(), dtype=np.float64)
        logprobs_td = out.get("logprobs") if isinstance(out, dict) else None
        if logprobs_td is None:
            continue
        lp = np.asarray(logprobs_td.to_numpy(), dtype=np.float64)
        n = min(len(lp), len(w))
        lp = lp[:n]
        w = w[:n]
        # cross-entropy per token = -logprob(target)
        total_loss += float(np.sum(-lp * w))
        total_weight += float(np.sum(w))
    if total_weight > 0:
        return total_loss / total_weight
    # last-resort: use any scalar metric
    if "loss:sum" in metrics:
        return float(metrics["loss:sum"])
    for k, v in metrics.items():
        if "loss" in k.lower():
            try:
                return float(v)
            except Exception:
                continue
    return float("nan")


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #

def load_rows(limit: int | None = None) -> list[dict]:
    rows: list[dict] = []
    with TRAIN_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            # keep only rows that have all gold fields present + valid
            errs = rubric.validate_gold_object({k: obj.get(k) for k in rubric.GOLD_FIELDS})
            if errs:
                continue
            rows.append(obj)
            if limit is not None and len(rows) >= limit:
                break
    return rows


# --------------------------------------------------------------------------- #
# Model resolution
# --------------------------------------------------------------------------- #

def resolve_base_model(service: tinker.ServiceClient) -> str:
    """Resolve the base model id against the live Tinker model list.

    Targets Qwen3.5-4B; if not offered, falls back to the closest small Qwen3
    the API returns (smallest parameter count by heuristic). Raises if no Qwen3
    model is offered at all.
    """
    caps = service.get_server_capabilities()
    models = list(getattr(caps, "supported_models", []) or [])

    def model_id(m) -> str:
        return getattr(m, "model_name", None) or getattr(m, "id", None) or str(m)

    ids = [model_id(m) for m in models]
    if not ids:
        raise RuntimeError("Tinker API returned an empty supported-models list.")

    # exact target
    for mid in ids:
        if mid == TARGET_BASE_MODEL or mid.endswith("/" + TARGET_BASE_MODEL):
            return mid
    # case-insensitive contains of target
    for mid in ids:
        if TARGET_BASE_MODEL.lower() in mid.lower():
            return mid

    # fall back to the closest small Qwen3 by size heuristic
    import re

    def size_key(mid: str) -> float:
        m = re.search(r"(\d+(?:\.\d+)?)\s*[bB]", mid)
        return float(m.group(1)) if m else 9e9

    qwen3 = [mid for mid in ids if "qwen3" in mid.lower()]
    if qwen3:
        qwen3.sort(key=size_key)
        return qwen3[0]
    # any qwen
    qwen = [mid for mid in ids if "qwen" in mid.lower()]
    if qwen:
        qwen.sort(key=size_key)
        return qwen[0]
    raise RuntimeError(
        "No Qwen model offered by the Tinker API. Available ids: " + ", ".join(ids)
    )


# --------------------------------------------------------------------------- #
# Training loop
# --------------------------------------------------------------------------- #

def train_batches(training_client, data: list[types.Datum], lr: float,
                  log_prefix: str = "") -> list[dict]:
    """Run forward_backward + optim_step over data in BATCH_SIZE chunks (1 pass).

    Returns a list of per-step records: {step, loss, n}.
    """
    steps: list[dict] = []
    adam = types.AdamParams(learning_rate=lr, beta1=0.9, beta2=0.95, eps=1e-8)
    step = 0
    for start in range(0, len(data), BATCH_SIZE):
        batch = data[start:start + BATCH_SIZE]
        if not batch:
            continue
        fb_future = training_client.forward_backward(batch, "cross_entropy")
        fb_out = fb_future.result()
        _ = training_client.optim_step(adam_params=adam).result()
        loss = weighted_mean_loss_from_output(fb_out, batch)
        step += 1
        rec = {"step": step, "loss": loss, "n": len(batch)}
        steps.append(rec)
        print(f"{log_prefix}step {step:>3}  loss {loss:.4f}  (n={len(batch)})",
              flush=True)
    return steps


def run_smoke(service: tinker.ServiceClient) -> int:
    print("[smoke] resolving base model against live Tinker model list ...",
          flush=True)
    base_model = resolve_base_model(service)
    print(f"[smoke] RESOLVED base model: {base_model}", flush=True)

    rows = load_rows(limit=SMOKE_EXAMPLES)
    if not rows:
        print("[smoke] FAIL: no valid training rows loaded.", file=sys.stderr)
        return 2
    print(f"[smoke] loaded {len(rows)} example rows", flush=True)

    training_client = service.create_lora_training_client(
        base_model=base_model, rank=LORA_RANK, seed=SEED
    )
    tokenizer = training_client.get_tokenizer()
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        eos_id = tokenizer.encode("", add_special_tokens=True)[-1]

    data = []
    for r in rows:
        d = build_datum(tokenizer, r, eos_id)
        if d is not None:
            data.append(d)
    if not data:
        print("[smoke] FAIL: no datums built.", file=sys.stderr)
        return 2
    print(f"[smoke] built {len(data)} datums; training {SMOKE_STEPS} steps ...",
          flush=True)

    adam = types.AdamParams(learning_rate=LEARNING_RATE, beta1=0.9, beta2=0.95,
                            eps=1e-8)
    last_loss = float("nan")
    for i in range(SMOKE_STEPS):
        fb_out = training_client.forward_backward(data, "cross_entropy").result()
        training_client.optim_step(adam_params=adam).result()
        last_loss = weighted_mean_loss_from_output(fb_out, data)
        print(f"[smoke] step {i + 1}/{SMOKE_STEPS}  loss {last_loss:.4f}",
              flush=True)

    print("[smoke] saving weights for sampler ...", flush=True)
    save_resp = training_client.save_weights_for_sampler(
        name="onecurb-smoke", ttl_seconds=3600
    ).result()
    sampler_path = getattr(save_resp, "path", None) or getattr(
        save_resp, "weights_path", None
    )
    print(f"[smoke] sampler weights path: {sampler_path}", flush=True)

    sampling_client = training_client.create_sampling_client(model_path=sampler_path) \
        if _accepts_model_path(training_client.create_sampling_client) \
        else service.create_sampling_client(model_path=sampler_path)

    sample_row = rows[0]
    prompt = types.ModelInput.from_ints(
        tokenizer.encode(render_prompt(sample_row), add_special_tokens=False)
    )
    sp = types.SamplingParams(max_tokens=256, temperature=0.0, top_p=1.0,
                              stop=["\n\n"])
    sample_resp = sampling_client.sample(
        prompt=prompt, num_samples=1, sampling_params=sp
    ).result()
    out_tokens = _first_sample_tokens(sample_resp)
    text = tokenizer.decode(out_tokens)
    print("[smoke] raw sample output:\n" + text, flush=True)

    parsed_ok = False
    try:
        obj = json.loads(_extract_json(text))
        parsed_ok = isinstance(obj, dict) and "category" in obj
    except Exception as e:
        print(f"[smoke] JSON parse note: {e}", flush=True)
    print(f"[smoke] parseable JSON sample: {parsed_ok}", flush=True)

    if not parsed_ok:
        print("[smoke] NOTE: sample did not parse to JSON on the tiny 4-step run "
              "(expected to improve with the full run); end-to-end loop "
              "(tokenize->loss->save->load->sample) succeeded.", flush=True)
    print(f"[smoke] OK — end-to-end loop verified on base model {base_model}",
          flush=True)
    return 0


def run_full(service: tinker.ServiceClient) -> int:
    print("[train] resolving base model against live Tinker model list ...",
          flush=True)
    base_model = resolve_base_model(service)
    print(f"[train] RESOLVED base model: {base_model}", flush=True)

    rows = load_rows()
    print(f"[train] loaded {len(rows)} valid training rows", flush=True)
    if not rows:
        print("[train] FAIL: no valid training rows.", file=sys.stderr)
        return 2

    rng = np.random.RandomState(SEED)

    training_client = service.create_lora_training_client(
        base_model=base_model, rank=LORA_RANK, seed=SEED
    )
    tokenizer = training_client.get_tokenizer()
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        eos_id = tokenizer.encode("", add_special_tokens=True)[-1]

    base_data: list[types.Datum] = []
    skipped = 0
    for r in rows:
        d = build_datum(tokenizer, r, eos_id)
        if d is None:
            skipped += 1
            continue
        base_data.append(d)
    print(f"[train] built {len(base_data)} datums ({skipped} skipped)",
          flush=True)

    all_steps: list[dict] = []
    epoch_losses: list[float] = []
    best_loss = float("inf")
    epochs_run = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        order = rng.permutation(len(base_data))
        data = [base_data[i] for i in order]
        print(f"[train] === epoch {epoch}/{NUM_EPOCHS} ===", flush=True)
        steps = train_batches(training_client, data, LEARNING_RATE,
                              log_prefix=f"[train][e{epoch}] ")
        for s in steps:
            s["epoch"] = epoch
        all_steps.extend(steps)
        finite = [s["loss"] for s in steps if np.isfinite(s["loss"])]
        epoch_loss = float(np.mean(finite)) if finite else float("nan")
        epoch_losses.append(epoch_loss)
        epochs_run = epoch
        print(f"[train] epoch {epoch} mean train loss: {epoch_loss:.4f}",
              flush=True)

        # early stop on plateau
        if np.isfinite(epoch_loss):
            if epoch_loss < best_loss * (1.0 - PLATEAU_REL_IMPROVEMENT):
                best_loss = epoch_loss
            else:
                if epoch >= 2:
                    print(f"[train] train-loss plateau detected at epoch {epoch} "
                          f"(no >= {PLATEAU_REL_IMPROVEMENT:.0%} improvement); "
                          f"early stopping.", flush=True)
                    break

    print("[train] saving final weights for sampler ...", flush=True)
    save_resp = training_client.save_weights_for_sampler(
        name="onecurb-sft", ttl_seconds=7 * 24 * 3600
    ).result()
    sampler_path = getattr(save_resp, "path", None) or getattr(
        save_resp, "weights_path", None
    )
    print(f"[train] sampler weights path: {sampler_path}", flush=True)

    # Verify a loadable sampler + one parseable sample.
    sampling_client = (
        training_client.create_sampling_client(model_path=sampler_path)
        if _accepts_model_path(training_client.create_sampling_client)
        else service.create_sampling_client(model_path=sampler_path)
    )
    sample_row = rows[0]
    prompt = types.ModelInput.from_ints(
        tokenizer.encode(render_prompt(sample_row), add_special_tokens=False)
    )
    sp = types.SamplingParams(max_tokens=256, temperature=0.0, top_p=1.0,
                              stop=["\n\n"])
    sample_resp = sampling_client.sample(
        prompt=prompt, num_samples=1, sampling_params=sp
    ).result()
    text = tokenizer.decode(_first_sample_tokens(sample_resp))
    sample_parses = False
    try:
        obj = json.loads(_extract_json(text))
        sample_parses = isinstance(obj, dict) and "category" in obj
    except Exception:
        sample_parses = False
    print(f"[train] final sampler loaded; sample parses to JSON: {sample_parses}",
          flush=True)

    final_loss = next((s["loss"] for s in reversed(all_steps)
                       if np.isfinite(s["loss"])), float("nan"))

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    meta = {
        "resolved_base_model": base_model,
        "rank": LORA_RANK,
        "assistant_loss_only": True,
        "loss_fn": "cross_entropy",
        "learning_rate": LEARNING_RATE,
        "seed": SEED,
        "batch_size": BATCH_SIZE,
        "max_seq_len": MAX_SEQ_LEN,
        "num_epochs_configured": NUM_EPOCHS,
        "epochs_run": epochs_run,
        "n_train_rows": len(rows),
        "n_datums": len(base_data),
        "final_loss": final_loss,
        "epoch_losses": epoch_losses,
        "loss_curve": all_steps,
        "sampler_weights_path": sampler_path,
        "sampler_loadable": True,
        "sample_parses_json": sample_parses,
        "train_data": str(TRAIN_PATH.relative_to(_ROOT)),
        "sdk": "tinker (Thinking Machines) " + getattr(tinker, "__version__", "?"),
    }
    META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[train] wrote {META_PATH}", flush=True)
    print(f"[train] DONE — base={base_model}  final_loss={final_loss:.4f}  "
          f"epochs_run={epochs_run}", flush=True)
    return 0


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #

def _accepts_model_path(fn) -> bool:
    import inspect
    try:
        return "model_path" in inspect.signature(fn).parameters
    except Exception:
        return False


def _first_sample_tokens(sample_resp) -> list[int]:
    seqs = getattr(sample_resp, "sequences", None) or []
    if not seqs:
        return []
    seq = seqs[0]
    toks = getattr(seq, "tokens", None)
    if toks is not None:
        try:
            return list(toks)
        except Exception:
            pass
    arr = getattr(seq, "tokens_np", None)
    if arr is not None:
        return [int(x) for x in np.asarray(arr).tolist()]
    tl = getattr(seq, "_tokens_list", None)
    if tl is not None:
        return list(tl)
    return []


def _extract_json(text: str) -> str:
    """Pull the first {...} JSON object out of a decoded sample."""
    start = text.find("{")
    if start < 0:
        return text
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return text[start:]


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser(description="One Curb Tinker LoRA SFT")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny end-to-end sanity run (~8 examples, few steps)")
    args = ap.parse_args()

    load_dotenv(_ROOT / ".env")
    api_key = os.environ.get("TINKER_API_KEY")
    if not api_key:
        print("ERROR: TINKER_API_KEY not found in environment/.env",
              file=sys.stderr)
        return 2

    # Confirm SDK identity (Thinking Machines Tinker training SDK).
    required = ["ServiceClient", "LoraConfig", "ForwardBackwardOutput"]
    missing = [n for n in required if not hasattr(tinker, n)]
    if missing or not hasattr(tinker.ServiceClient, "create_lora_training_client"):
        print(f"ERROR: installed 'tinker' is not the Thinking Machines training "
              f"SDK (missing: {missing or 'create_lora_training_client'})",
              file=sys.stderr)
        return 2

    service = tinker.ServiceClient(api_key=api_key)

    if args.smoke:
        return run_smoke(service)
    return run_full(service)


if __name__ == "__main__":
    raise SystemExit(main())
