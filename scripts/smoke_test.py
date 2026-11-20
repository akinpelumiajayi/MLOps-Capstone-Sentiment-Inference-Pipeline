"""Post-deploy smoke test (stdlib only, so it runs anywhere).

    python scripts/smoke_test.py http://localhost:8000

Waits for readiness, then checks that the deployed model gives sensible
answers on unambiguous inputs and that metrics are exposed. Exit code 1 on
any failure, so CI can gate a rollout on it.
"""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from typing import Any

CASES = [
    ("A brilliant, moving and beautifully acted film. One of the best of the year.", "positive"),
    ("A dull, tedious and badly written mess. Painfully boring from start to finish.", "negative"),
]


def call(base: str, path: str, payload: dict[str, Any] | None = None) -> tuple[int, Any]:
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(base + path, data=data, headers={"Content-Type": "application/json"})  # noqa: S310
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:  # noqa: S310 - operator-supplied URL
            body = resp.read().decode()
            return resp.status, json.loads(body) if "json" in resp.headers.get("Content-Type", "") else body
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode()


def wait_ready(base: str, timeout: float = 120) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if call(base, "/readyz")[0] == 200:
                return
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            pass
        time.sleep(2)
    raise SystemExit(f"FAIL: {base}/readyz not ready after {timeout}s")


def main() -> int:
    base = (sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000").rstrip("/")
    wait_ready(base)
    failures: list[str] = []

    for text, expected in CASES:
        status, body = call(base, "/predict", {"text": text})
        if status != 200 or body.get("label") != expected:
            failures.append(f"/predict {expected!r}: HTTP {status} {body}")
        else:
            print(f"ok   /predict -> {body['label']} ({body['confidence']:.2f})")

    status, body = call(base, "/predict/batch", {"texts": [t for t, _ in CASES]})
    if status != 200 or [p["label"] for p in body["predictions"]] != [e for _, e in CASES]:
        failures.append(f"/predict/batch: HTTP {status} {body}")
    else:
        print("ok   /predict/batch")

    if call(base, "/predict", {"text": "   "})[0] != 422:
        failures.append("/predict accepted blank input")
    else:
        print("ok   input validation")

    status, body = call(base, "/metrics")
    if status != 200 or "sentiment_predictions_total" not in str(body):
        failures.append("/metrics missing sentiment_predictions_total")
    else:
        print("ok   /metrics")

    for f in failures:
        print(f"FAIL {f}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
