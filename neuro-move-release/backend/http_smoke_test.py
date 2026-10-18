from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:8000"


def request(method: str, path: str, body=None):
    url = f"{BASE}{path}"
    data = None
    headers = {}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=180) as resp:
        raw = resp.read().decode("utf-8")
        return json.loads(raw)


def main() -> None:
    print("=== health ===")
    print(json.dumps(request("GET", "/health"), indent=2))

    print("\n=== datasets ===")
    print(request("GET", "/datasets"))

    print("\n=== train queue ===")
    job = request("POST", "/train", {"dataset": "BCI Competition IV 2a", "engine": "minirocket", "epochs": 2, "batch_size": 32})
    print(job)
    jid = job["job_id"]

    print("\n=== run preview ===")
    preview = request("POST", f"/train/{jid}/run-preview", {"dataset": "BCI Competition IV 2a", "engine": "minirocket", "epochs": 2, "batch_size": 32, "learning_rate": 1e-3})
    print(preview)

    print("\n=== status ===")
    time.sleep(1)
    print(request("GET", f"/train/{jid}"))

    print("\n=== infer ===")
    print(request("POST", "/infer", {"engine": "minirocket", "dataset": "BCI Competition IV 2a", "sample_count": 8, "channels": 22, "samples": 656}))

    print("\n=== evaluate ===")
    print(request("POST", "/evaluate", {"dataset": "BCI Competition IV 2a", "engine": "minirocket"}))

    print("\nHTTP smoke test complete")


if __name__ == "__main__":
    main()
