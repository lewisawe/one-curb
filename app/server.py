"""One Curb, One Case — one-screen Flask app (plan item 7).

A single-file Flask server that turns a messy pedestrian-accessibility field
note into a structured civic report AND, prominently, a MISSING-EVIDENCE list
with the ONE concrete follow-up instruction ("here is what your report is still
missing — go measure it"). That follow-up is the product thesis.

It calls the fine-tuned Tinker sampler (resolved base id + the saved adapter
sampler weights, both recorded in ``models/train_meta.json`` by ``src/train.py``)
using the EXACT same preamble/schema/prompt as training — the module imports
``PREAMBLE``, ``render_prompt`` and the sampling helpers from ``src/train.py`` so
train and inference never drift. It parses the JSON output and renders it. On a
parse failure it shows the raw model output plus a clear error and does NOT
crash.

Run:
    .venv/bin/python app/server.py            # starts on 127.0.0.1:8000

``TINKER_API_KEY`` is loaded from ``.env`` via python-dotenv and is NEVER printed.
"""

from __future__ import annotations

import json
import os
import sys
import threading
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request

# --------------------------------------------------------------------------- #
# Paths & project imports (reuse the training preamble/schema verbatim)
# --------------------------------------------------------------------------- #

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import src.rubric as rubric  # noqa: E402
# Reuse the SAME preamble/prompt/parse helpers the model was trained with so
# inference matches training exactly.
from src.train import (  # noqa: E402
    PREAMBLE,
    render_prompt,
    _extract_json,
    _first_sample_tokens,
    _accepts_model_path,
)

MODELS_DIR = _ROOT / "models"
META_PATH = MODELS_DIR / "train_meta.json"

HOST = os.environ.get("ONECURB_HOST", "127.0.0.1")
PORT = int(os.environ.get("ONECURB_PORT", "8000"))

app = Flask(__name__, template_folder=str(_HERE / "templates"))


# --------------------------------------------------------------------------- #
# Lazy, thread-safe Tinker sampler (loaded once, reused across requests)
# --------------------------------------------------------------------------- #

class SamplerBundle:
    """Holds the loaded sampler, tokenizer, and resolved base-model id."""

    def __init__(self, sampling_client, tokenizer, base_model: str,
                 sampler_path: str) -> None:
        self.sampling_client = sampling_client
        self.tokenizer = tokenizer
        self.base_model = base_model
        self.sampler_path = sampler_path


_sampler_lock = threading.Lock()
_sampler: SamplerBundle | None = None
_sampler_error: str | None = None


def _read_meta() -> dict:
    if not META_PATH.exists():
        raise RuntimeError(
            f"missing {META_PATH} — run src/train.py before starting the app"
        )
    return json.loads(META_PATH.read_text(encoding="utf-8"))


def load_sampler() -> SamplerBundle:
    """Load (once) the fine-tuned Tinker sampler recorded in train_meta.json.

    Resolves the base model id + the saved adapter sampler weights path from
    ``models/train_meta.json`` (does NOT re-train). Raises with a clear message
    on any failure so the caller can surface it (and smoke_request can warn).
    """
    global _sampler, _sampler_error
    if _sampler is not None:
        return _sampler
    with _sampler_lock:
        if _sampler is not None:
            return _sampler

        load_dotenv(_ROOT / ".env")
        api_key = os.environ.get("TINKER_API_KEY")
        if not api_key:
            raise RuntimeError("TINKER_API_KEY not found in environment/.env")

        meta = _read_meta()
        base_model = meta.get("resolved_base_model")
        sampler_path = meta.get("sampler_weights_path")
        if not base_model:
            raise RuntimeError("train_meta.json has no 'resolved_base_model'")
        if not sampler_path:
            raise RuntimeError("train_meta.json has no 'sampler_weights_path'")

        import tinker  # local import so an import error is reported clearly

        service = tinker.ServiceClient(api_key=api_key)

        # Prefer loading the adapter sampler weights on top of the resolved base
        # model. Fall back to signatures the installed SDK accepts.
        sampling_client = None
        last_err: Exception | None = None
        for kwargs in (
            {"base_model": base_model, "model_path": sampler_path},
            {"model_path": sampler_path},
            {"base_model": base_model},
        ):
            try:
                sampling_client = service.create_sampling_client(**kwargs)
                break
            except TypeError as e:
                last_err = e
                continue
            except Exception as e:  # pragma: no cover - surfaced to caller
                last_err = e
                continue
        if sampling_client is None:
            raise RuntimeError(
                f"could not create Tinker sampling client: {last_err}"
            )

        tokenizer = _get_tokenizer(service, base_model, sampling_client)

        _sampler = SamplerBundle(sampling_client, tokenizer, base_model,
                                 sampler_path)
        _sampler_error = None
        return _sampler


def _get_tokenizer(service, base_model: str, sampling_client):
    """Obtain a tokenizer compatible with the base model.

    Tries the sampling client, then a tokenizer-only training client, then any
    service-level helper. Reuses the same tokenizer the training step used.
    """
    getter = getattr(sampling_client, "get_tokenizer", None)
    if callable(getter):
        try:
            return getter()
        except Exception:
            pass
    # training client exposes get_tokenizer() in this SDK
    try:
        tc = service.create_lora_training_client(base_model=base_model, rank=1)
        return tc.get_tokenizer()
    except Exception as e:
        raise RuntimeError(f"could not obtain tokenizer for {base_model}: {e}")


# --------------------------------------------------------------------------- #
# Core inference: messy note -> parsed gold report (+ raw text / error)
# --------------------------------------------------------------------------- #

def generate_report(note: str) -> dict:
    """Run one messy note through the fine-tuned sampler and parse the result.

    Returns a dict:
      { ok: bool,
        report: <parsed gold object> | None,
        raw: <raw decoded model text>,
        error: <str> | None,
        base_model: <str> }

    Never raises for a model-parse problem — a parse failure is reported with
    ok=False, the raw text, and an error string so the UI can show it without
    crashing. Infrastructure failures (no sampler) propagate to the caller.
    """
    import tinker
    from tinker import types

    bundle = load_sampler()
    tok = bundle.tokenizer

    prompt_text = render_prompt({"issue_description": note})
    prompt = types.ModelInput.from_ints(
        tok.encode(prompt_text, add_special_tokens=False)
    )
    sp = types.SamplingParams(max_tokens=256, temperature=0.0, top_p=1.0,
                              stop=["\n\n"])
    resp = bundle.sampling_client.sample(
        prompt=prompt, num_samples=1, sampling_params=sp
    ).result()
    raw = tok.decode(_first_sample_tokens(resp))

    result = {
        "ok": False,
        "report": None,
        "raw": raw,
        "error": None,
        "base_model": bundle.base_model,
    }

    try:
        obj = json.loads(_extract_json(raw))
    except Exception as e:
        result["error"] = f"model output was not valid JSON: {e}"
        return result

    if not isinstance(obj, dict):
        result["error"] = "model output JSON was not an object"
        return result

    # Normalize + validate against the gold schema (src/rubric.py). We keep the
    # object even if validation flags issues, but surface them to the UI.
    errs = rubric.validate_gold_object(
        {k: obj.get(k) for k in rubric.GOLD_FIELDS}
    )
    result["report"] = obj
    if errs:
        result["ok"] = False
        result["error"] = ("model output did not fully match the gold schema: "
                           + "; ".join(errs))
    else:
        result["ok"] = True
    return result


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #

@app.get("/")
def index():
    return render_template("index.html")


@app.get("/healthz")
def healthz():
    """Report whether the sampler can load (used by smoke_request if needed)."""
    try:
        load_sampler()
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 503


@app.post("/generate")
def generate():
    data = request.get_json(silent=True) or request.form
    note = (data.get("note") if hasattr(data, "get") else "") or ""
    note = note.strip()
    if not note:
        return jsonify({"ok": False, "error": "empty note", "report": None,
                        "raw": ""}), 400
    try:
        result = generate_report(note)
    except Exception as e:
        # Infrastructure failure (sampler won't load, Tinker error): clear
        # message, HTTP 503, no crash.
        return jsonify({
            "ok": False,
            "report": None,
            "raw": "",
            "error": f"sampler/Tinker error: {e}",
        }), 503
    status = 200
    return jsonify(result), status


def main() -> int:
    # Fail fast with a readable message if the key is absent, but do NOT print it.
    load_dotenv(_ROOT / ".env")
    if not os.environ.get("TINKER_API_KEY"):
        print("WARNING: TINKER_API_KEY not set — /generate will return 503 until "
              "it is available in .env", file=sys.stderr)
    print(f"[app] One Curb, One Case — serving on http://{HOST}:{PORT}",
          flush=True)
    app.run(host=HOST, port=PORT, debug=False, use_reloader=False, threaded=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
