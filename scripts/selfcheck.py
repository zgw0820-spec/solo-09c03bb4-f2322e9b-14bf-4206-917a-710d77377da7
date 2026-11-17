#!/usr/bin/env python3
"""End-to-end self-check against a running service.

    python3 scripts/selfcheck.py [base-url]

Verifies, over real HTTP:
1. health + two distinct models
2. switching the active model
3. version pinning: requests accepted before a switch keep the old version,
   requests accepted after it use the new one, and no batch mixes versions
4. batching: 16 emails -> batches of at most 4
5. deadlines: a slow email times out, its batch-mates still succeed, and the
   late result never flips the timeout back to success
6. early dispatch when the earliest deadline is near
7. interleaved demo
8. shadow observation: candidate re-scores accepted emails asynchronously,
   disagreements are listed with both scores/versions, stopping excludes new
   emails while included ones keep settling, a slow shadow only affects the
   comparison status and never occupies online batches
"""

from __future__ import annotations

import sys
import time

import httpx

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
SPAM = "Congratulations! You are a winner. Claim your free prize now."
HAM = "Hi team, attached is the quarterly report for our meeting tomorrow."
CRYPTO = "Limited offer! Invest in crypto today and double your money fast."

failures: list[str] = []


def check(condition: bool, message: str) -> None:
    if condition:
        print(f"  ok   {message}")
    else:
        print(f"  FAIL {message}")
        failures.append(message)


def wait_job(client: httpx.Client, job_id: str, timeout: float = 10.0) -> dict:
    deadline = time.monotonic() + timeout
    state = {}
    while time.monotonic() < deadline:
        state = client.get(f"/api/jobs/{job_id}").json()
        if state["status"] in ("completed", "failed"):
            return state
        time.sleep(0.02)
    return state


def main() -> int:
    client = httpx.Client(base_url=BASE, timeout=30.0)

    print("1. health & models")
    health = client.get("/healthz").json()
    check(health["status"] == "ok", "service healthy")
    check({"v1", "v2"} <= set(health["models"]), "both v1 and v2 shipped")
    models = client.get("/api/models").json()["models"]
    sizes = {m["version"]: m["vocabulary_size"] for m in models}
    check(sizes["v1"] != sizes["v2"], f"models differ (v1={sizes['v1']}, v2={sizes['v2']} terms)")

    print("2. model switch")
    switched = client.post("/api/model", json={"version": "v2"}).json()
    check(switched["active"] == "v2", "active model switched to v2")
    check(client.get("/healthz").json()["active_model"] == "v2", "health reflects the switch")
    check(client.post("/api/model", json={"version": "nope"}).status_code == 404, "unknown version rejected")

    print("3. version pinning across a live switch")
    client.post("/api/model", json={"version": "v1"})
    job = client.post(
        "/api/jobs",
        json={
            "label": "pin",
            "emails": [
                {"text": SPAM, "deadline_ms": 900},
                {"text": SPAM, "deadline_ms": 900, "submit_after_ms": 260},
                {"text": SPAM, "deadline_ms": 900, "submit_after_ms": 420},
            ],
        },
    ).json()
    time.sleep(0.12)
    switch = client.post("/api/model", json={"version": "v2"}).json()
    check(switch["event"]["from"] == "v1" and switch["event"]["to"] == "v2", "switched v1 -> v2 mid-queue")
    pinned = wait_job(client, job["id"])
    versions = [e["model_version"] for e in pinned["entries"]]
    check(versions[0] == "v1", "email accepted before the switch kept v1")
    check(all(v == "v2" for v in versions[1:]), "emails accepted after the switch used v2")
    batches = client.get("/api/batches?limit=50").json()["batches"]
    for entry in pinned["entries"]:
        batch = next((b for b in batches if b["batch_id"] == entry["batch_id"]), None)
        if batch:
            expected = entry["model_version"]
            check(batch["model_version"] == expected, f"batch {batch['batch_id']} is pure {expected}")

    print("4. batching limits (16 emails)")
    client.post("/api/model", json={"version": "v1"})
    bulk = client.post(
        "/api/jobs",
        json={
            "label": "bulk16",
            "emails": [
                {"text": f"{SPAM} ref {i}", "deadline_ms": 1000}
                for i in range(16)
            ],
        },
    ).json()
    bulk_state = wait_job(client, bulk["id"])
    check(bulk_state["ok"] == 16, f"all 16 classified (ok={bulk_state['ok']})")
    bulk_ids = {e["batch_id"] for e in bulk_state["entries"]}
    check(len(bulk_ids) == 4, f"16 emails formed 4 batches (got {len(bulk_ids)})")
    sizes = [b["size"] for b in client.get("/api/batches?limit=50").json()["batches"] if b["batch_id"] in bulk_ids]
    check(max(sizes) <= 4 and sorted(sizes) == [4, 4, 4, 4], f"batch sizes {sorted(sizes)}")

    print("5. deadline timeout + late result cannot flip it")
    client.post("/api/model", json={"version": "v2"})
    timeout_job = client.post(
        "/api/classify",
        json={
            "label": "batch timeout",
            "emails": [
                {"text": SPAM, "deadline_ms": 100, "simulate_latency_ms": 600},
                {"text": CRYPTO, "deadline_ms": 1000},
            ],
        },
    ).json()
    late = next(e for e in timeout_job["entries"] if e["deadline_ms"] == 100)
    fast = next(e for e in timeout_job["entries"] if e["deadline_ms"] == 1000)
    check(late["status"] == "timeout", "slow email timed out")
    check(late["score"] is None, "timed-out email has no score")
    check(late["late_result_discarded"] is True, "late result explicitly discarded")
    check(fast["status"] == "ok", "batch-mate still succeeded")
    check(late["batch_id"] == fast["batch_id"], "they were in the same batch")
    time.sleep(0.7)
    reread = client.get(f"/api/jobs/{timeout_job['id']}").json()
    again = next(e for e in reread["entries"] if e["deadline_ms"] == 100)
    check(again["status"] == "timeout", "timeout still a timeout after the late thread finished")

    print("6. early dispatch when the earliest deadline is near")
    near = client.post(
        "/api/jobs",
        json={"label": "near", "emails": [{"text": SPAM, "deadline_ms": 150}]},
    ).json()
    near_state = wait_job(client, near["id"])
    entry = near_state["entries"][0]
    check(entry["status"] == "ok", "near-deadline request was dispatched early and scored")
    check(entry["latency_ms"] < 200, f"latency {entry['latency_ms']}ms < 200ms window")

    print("7. interleaved demo")
    demo = client.post("/api/demo/interleaved").json()
    demo_state = wait_job(client, demo["id"], timeout=10)
    old = [e for e in demo_state["entries"] if e["index"] <= 2]
    new = [e for e in demo_state["entries"] if e["index"] > 2]
    check({e["model_version"] for e in old} == {"v1"}, "demo old requests pinned to v1")
    check({e["model_version"] for e in new} == {"v2"}, "demo new requests used v2")
    slow = next(e for e in new if e["simulate_latency_ms"] > 0)
    check(slow["status"] == "timeout", "demo slow email timed out")
    check(client.get("/healthz").json()["active_model"] == "v2", "demo left v2 active")

    print("8. shadow observation (candidate re-scoring)")
    client.post("/api/model", json={"version": "v1"})
    check(client.post("/api/observation/stop").status_code == 409, "stop without observation rejected")
    started = client.post("/api/observation/start", json={"version": "v2"})
    check(started.status_code == 201, "observation started on candidate v2")
    obs_id = started.json()["observation"]["id"]
    submitted_before = client.get("/api/state").json()["worker"]["metrics"]["submitted"]

    t0 = time.monotonic()
    shadow_job = client.post(
        "/api/jobs",
        json={
            "label": "shadow",
            "emails": [
                {"text": CRYPTO, "deadline_ms": 900},
                {"text": SPAM, "deadline_ms": 900},
                {"text": CRYPTO, "deadline_ms": 900, "shadow_simulate_latency_ms": 5000},
            ],
        },
    ).json()
    shadow_state = wait_job(client, shadow_job["id"])
    online_elapsed = time.monotonic() - t0
    check(shadow_state["status"] == "completed", "shadowed job completed online")
    check(all(e["status"] == "ok" for e in shadow_state["entries"]), "all online conclusions ok")
    check(online_elapsed < 3.0, f"slow shadow did not delay the online job ({online_elapsed:.2f}s < 3s)")
    check(all(e["shadow"] for e in shadow_state["entries"]), "all 3 emails pinned to the observation")
    check(
        all(e["shadow"]["observation_id"] == obs_id for e in shadow_state["entries"]),
        "pins reference the active observation",
    )

    # Stop immediately: the slow shadow (5s > 3s shadow timeout) must still
    # settle, into the now-stopped observation.
    stopped = client.post("/api/observation/stop").json()["observation"]
    check(stopped["id"] == obs_id and stopped["status"] == "stopped", "observation stopped")
    after_job = client.post(
        "/api/jobs",
        json={"label": "post-stop", "emails": [{"text": CRYPTO, "deadline_ms": 900}]},
    ).json()
    after_state = wait_job(client, after_job["id"])
    check(after_state["entries"][0]["shadow"] is None, "email accepted after stop is not shadowed")

    # Wait for the slow shadow to time out, then verify the whole period.
    deadline = time.monotonic() + 8
    obs = {}
    while time.monotonic() < deadline:
        snapshot = client.get("/api/observation").json()
        obs = next((o for o in snapshot["observations"] if o["id"] == obs_id), {})
        if obs and obs.get("pending") == 0:
            break
        time.sleep(0.1)
    check(obs.get("included") == 3, f"observation included exactly 3 emails (got {obs.get('included')})")
    check(obs.get("pending") == 0, "all included comparisons settled after stop")
    check(obs.get("shadow_timeout") == 1, "slow shadow recorded as a shadow timeout (comparison status only)")
    check(obs.get("disagreements", 0) >= 1, "crypto disagreement counted on the observation")

    settled = client.get(f"/api/jobs/{shadow_job['id']}").json()["entries"]
    crypto = settled[0]
    check(crypto["label"] == "ham" and crypto["model_version"] == "v1", "online conclusion unchanged (v1 ham)")
    check(crypto["shadow"]["status"] == "ok", "crypto shadow scored")
    check(
        crypto["shadow"]["label"] == "spam" and crypto["shadow"]["candidate_version"] == "v2",
        "shadow says v2 spam -> opposite labels",
    )
    check(crypto["shadow"]["disagree"] is True, "disagreement flagged on the entry")
    check(
        crypto["score"] is not None and crypto["shadow"]["score"] is not None,
        "both online and shadow scores present",
    )
    slow_entry = settled[2]
    check(slow_entry["status"] == "ok", "online conclusion of slow-shadow email is ok")
    check(slow_entry["shadow"]["status"] == "timeout", "its shadow timed out without touching the online result")

    snapshot = client.get("/api/observation").json()
    items = snapshot["recent_disagreements"]
    check(
        any(i["observation_id"] == obs_id and i["online"]["model_version"] == "v1"
            and i["shadow"]["model_version"] == "v2" for i in items),
        "disagreement list carries both versions",
    )
    submitted_after = client.get("/api/state").json()["worker"]["metrics"]["submitted"]
    check(
        submitted_after - submitted_before == 4,
        f"shadow scoring never entered online batches (submitted +{submitted_after - submitted_before}, want 4)",
    )

    print()
    if failures:
        print(f"{len(failures)} check(s) failed")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
