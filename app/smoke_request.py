"""Smoke test for the One Curb, One Case app (plan item 7 verification).

Starts (or targets an already-running) Flask server, posts ONE messy note that
is deliberately missing evidence, and asserts:

  1. HTTP 200 from POST /generate
  2. the returned ``report`` validates against the gold schema
     (``validate_gold_object`` from ``src/rubric.py`` returns no errors)
  3. ``followup_instruction`` is NON-EMPTY (because evidence is missing)

Usage:
    .venv/bin/python app/smoke_request.py            # start a server, test, stop
    .venv/bin/python app/smoke_request.py --url http://127.0.0.1:8000   # target running

Exit code 0 on pass. Non-zero (with a printed reason) on failure. If the sampler
cannot load / Tinker errors, prints the exact server error and exits non-zero so
the caller can surface it. Never prints TINKER_API_KEY.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import src.rubric as rubric  # noqa: E402

# A deliberately messy note that is MISSING evidence (no dimension measured, no
# persistence, vague location) — so a correct model must flag missing_evidence
# and therefore emit a non-empty followup_instruction.
MESSY_NOTE = (
    "The curb by the pharmacy on Main St is all busted up and my mom's "
    "wheelchair can't get up it, she has to go in the road. been like that a while."
)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000


def _get(url: str, timeout: float = 5.0):
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, json.loads(r.read().decode("utf-8"))


def _post(url: str, payload: dict, timeout: float = 180.0):
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=body, method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        # read the structured error body the server returns on 4xx/5xx
        try:
            data = json.loads(e.read().decode("utf-8"))
        except Exception:
            data = {"error": f"HTTP {e.code}", "raw": ""}
        return e.code, data


def wait_until_up(base: str, timeout_s: float = 30.0) -> bool:
    """Poll the index route until the Flask process is accepting connections."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            req = urllib.request.Request(base + "/", method="GET")
            with urllib.request.urlopen(req, timeout=3.0) as r:
                if r.status == 200:
                    return True
        except Exception:
            time.sleep(0.5)
    return False


def run_assertions(base: str) -> int:
    print(f"[smoke] POST {base}/generate with a missing-evidence note", flush=True)
    status, data = _post(base + "/generate", {"note": MESSY_NOTE})

    if status != 200:
        print(f"[smoke] FAIL: expected HTTP 200, got {status}. "
              f"server error: {data.get('error')!r}", file=sys.stderr)
        if data.get("raw"):
            print("[smoke] raw model output:\n" + str(data.get("raw")),
                  file=sys.stderr)
        return 3

    report = data.get("report")
    if not isinstance(report, dict):
        print(f"[smoke] FAIL: response has no report object. "
              f"error={data.get('error')!r} raw={data.get('raw')!r}",
              file=sys.stderr)
        return 4

    # Assert schema validity using the SAME validator the gold labels use.
    obj = {k: report.get(k) for k in rubric.GOLD_FIELDS}
    errs = rubric.validate_gold_object(obj)
    if errs:
        print("[smoke] FAIL: report does not validate against the gold schema:",
              file=sys.stderr)
        for e in errs:
            print(f"           - {e}", file=sys.stderr)
        print("[smoke] raw model output:\n" + str(data.get("raw", "")),
              file=sys.stderr)
        return 5

    followup = report.get("followup_instruction") or ""
    if not followup.strip():
        print("[smoke] FAIL: followup_instruction is empty, but the note is "
              "missing evidence (missing_evidence="
              f"{report.get('missing_evidence')!r}).", file=sys.stderr)
        return 6

    print("[smoke] PASS", flush=True)
    print(f"[smoke]   HTTP 200: yes", flush=True)
    print(f"[smoke]   schema-valid: yes", flush=True)
    print(f"[smoke]   category: {report.get('category')}  "
          f"severity: {report.get('severity')}", flush=True)
    print(f"[smoke]   missing_evidence: {report.get('missing_evidence')}",
          flush=True)
    print(f"[smoke]   followup_instruction (non-empty): {followup}", flush=True)
    print(f"[smoke]   base_model: {data.get('base_model')}", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="One Curb app smoke test")
    ap.add_argument("--url", default=None,
                    help="target an already-running server (e.g. "
                         "http://127.0.0.1:8000); if omitted, start one here")
    ap.add_argument("--host", default=DEFAULT_HOST)
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = ap.parse_args()

    if args.url:
        base = args.url.rstrip("/")
        return run_assertions(base)

    # Start our own server subprocess, wait, test, tear down.
    base = f"http://{args.host}:{args.port}"
    env_port = str(args.port)
    print(f"[smoke] starting server subprocess on {base} ...", flush=True)
    proc = subprocess.Popen(
        [sys.executable, str(_HERE / "server.py")],
        cwd=str(_ROOT),
        env={**_server_env(env_port, args.host)},
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    try:
        if not wait_until_up(base, timeout_s=30.0):
            print("[smoke] FAIL: server did not come up within 30s.",
                  file=sys.stderr)
            _drain(proc)
            return 2
        return run_assertions(base)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()
        _drain(proc)


def _server_env(port: str, host: str) -> dict:
    import os
    env = dict(os.environ)
    env["ONECURB_PORT"] = port
    env["ONECURB_HOST"] = host
    return env


def _drain(proc: subprocess.Popen) -> None:
    """Print any captured server output (helps diagnose a sampler failure).

    The server never prints the API key; this is safe to echo.
    """
    try:
        if proc.stdout is not None:
            out = proc.stdout.read()
            if out:
                print("[smoke] --- server output ---", file=sys.stderr)
                print(out, file=sys.stderr)
    except Exception:
        pass


if __name__ == "__main__":
    raise SystemExit(main())
