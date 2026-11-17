"""Tests for shadow (candidate-model) observation.

Covers the isolation contract (shadow never touches the online worker or its
batches, never delays/rewrites online conclusions) and the observation-period
semantics (intake-time pairing, stop excludes new emails but keeps settling
included ones, late results belong to the original period).
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.batching import Worker  # noqa: E402
from app.model import LinearBoWModel, ModelRegistry, Prediction  # noqa: E402
from app.shadow import ShadowManager, ShadowRecord, ShadowRunner  # noqa: E402
from app import main as main_module  # noqa: E402


def make_registry() -> ModelRegistry:
    v1 = LinearBoWModel(
        version="v1",
        vocabulary=["spam", "free", "meeting"],
        idf=[1.0, 1.0, 1.0],
        weights=[2.0, 2.0, -2.0],
        bias=-0.5,
        threshold=0.5,
    )
    v2 = LinearBoWModel(
        version="v2",
        vocabulary=["spam", "free", "meeting", "winner"],
        idf=[1.0, 1.0, 1.0, 1.0],
        weights=[3.0, 1.0, -1.5, 4.0],
        bias=-0.5,
        threshold=0.5,
    )
    return ModelRegistry({"v1": v1, "v2": v2}, "v1")


def make_entry(**overrides):
    entry = SimpleNamespace(
        index=1,
        text="free spam",
        model_version="v1",
        result={"status": "ok", "score": 0.9, "label": "spam"},
        shadow=None,
        shadow_simulate_latency_ms=0,
    )
    for key, value in overrides.items():
        setattr(entry, key, value)
    return entry


async def wait_for(predicate, timeout: float = 3.0, tick: float = 0.005):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        await asyncio.sleep(tick)
    return predicate()


# ---------------------------------------------------------------------------
# ShadowRunner unit tests
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_shadow_runner_scores_with_candidate_version():
    runner = ShadowRunner(make_registry(), timeout_ms=1000)
    await runner.start()
    try:
        outcome = await runner.submit(text="free winner spam", candidate_version="v2")
        assert outcome["status"] == "ok"
        assert outcome["label"] == "spam"
        assert 0.0 <= outcome["score"] <= 1.0
        assert runner.metrics["ok"] == 1
    finally:
        await runner.stop()


@pytest.mark.asyncio
async def test_shadow_timeout_is_a_comparison_status():
    runner = ShadowRunner(make_registry(), timeout_ms=80)
    await runner.start()
    try:
        outcome = await runner.submit(
            text="free spam", candidate_version="v1", latency_sim_s=0.5
        )
        assert outcome["status"] == "timeout"
        assert outcome["reason"] == "shadow_deadline_exceeded"
        assert runner.metrics["timeout"] == 1
    finally:
        await runner.stop()


@pytest.mark.asyncio
async def test_shadow_scorer_failure_is_contained():
    def boom(model, text):
        raise RuntimeError("model exploded")

    runner = ShadowRunner(make_registry(), timeout_ms=1000, scorer=boom)
    await runner.start()
    try:
        outcome = await runner.submit(text="free spam", candidate_version="v1")
        assert outcome["status"] == "error"
        assert outcome["reason"] == "shadow_internal_error"
        assert runner.metrics["error"] == 1
    finally:
        await runner.stop()


@pytest.mark.asyncio
async def test_shadow_never_touches_online_worker_or_batches():
    """Shadow scoring must not submit to, batch with, or block the worker."""
    registry = make_registry()
    worker = Worker(registry, max_batch_size=4, batch_wait_ms=20)
    runner = ShadowRunner(registry, timeout_ms=1000)
    await worker.start()
    await runner.start()
    try:
        online = asyncio.create_task(
            worker.submit(
                text="free spam",
                model_version="v1",
                deadline_ms=800,
                request_id="online-1",
                request_batch_id="job",
            )
        )
        shadows = [
            runner.submit(text=f"free spam {i}", candidate_version="v2")
            for i in range(8)
        ]
        online_result, *shadow_outcomes = await asyncio.wait_for(
            asyncio.gather(online, *shadows), timeout=4
        )
        assert online_result["status"] == "ok"
        assert all(o["status"] == "ok" for o in shadow_outcomes)
        # The worker only ever saw the single online request.
        assert worker.metrics["submitted"] == 1
        assert worker.metrics["batches"] == 1
        assert all(b["size"] == 1 for b in worker.recent_batches(10))
        assert runner.metrics["ok"] == 8
    finally:
        await runner.stop()
        await worker.stop()


# ---------------------------------------------------------------------------
# ShadowManager unit tests
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_pin_only_while_observing_and_settlement_after_stop():
    manager = ShadowManager(make_registry(), timeout_ms=1000)
    await manager.start()
    try:
        assert manager.pin("v1") is None  # no active observation
        obs, _ = manager.start_observation("v2")
        entry = make_entry()
        entry.shadow = manager.pin("v1")
        assert entry.shadow is not None
        assert obs.included == 1

        manager.stop_observation()
        assert manager.pin("v1") is None  # stop excludes new emails

        # The already-pinned email still gets scored and settled, into the
        # stopped observation.
        manager.on_online_result(SimpleNamespace(id="job-1"), entry)
        assert await wait_for(lambda: entry.shadow.status == "ok")
        assert obs.shadow_ok == 1
        assert obs.settled == 1
        assert obs.pending == 0
        assert entry.shadow.candidate_version == "v2"
    finally:
        await manager.stop()


@pytest.mark.asyncio
async def test_online_failure_skips_comparison():
    manager = ShadowManager(make_registry(), timeout_ms=1000)
    await manager.start()
    try:
        obs, _ = manager.start_observation("v2")
        entry = make_entry(result={"status": "timeout", "reason": "deadline"})
        entry.shadow = manager.pin("v1")
        manager.on_online_result(SimpleNamespace(id="job-1"), entry)
        assert entry.shadow.status == "skipped"
        assert entry.shadow.reason == "online_timeout"
        assert obs.skipped == 1
        assert manager.runner.metrics["submitted"] == 0  # nothing scheduled
    finally:
        await manager.stop()


@pytest.mark.asyncio
async def test_restart_keeps_pairing_and_late_results_join_original_period():
    manager = ShadowManager(make_registry(), timeout_ms=2000)
    await manager.start()
    try:
        obs_a, _ = manager.start_observation("v2")
        old_entry = make_entry(shadow_simulate_latency_ms=300)
        old_entry.shadow = manager.pin("v1")
        old_entry.model_version = "v1"

        # Restart: new period, new candidate.  Old pin must not move.
        obs_b, replaced = manager.start_observation("v1")
        assert replaced is obs_a and obs_a.status == "stopped"
        new_entry = make_entry()
        new_entry.shadow = manager.pin("v2")
        new_entry.model_version = "v2"

        assert old_entry.shadow.observation_id == obs_a.id
        assert old_entry.shadow.candidate_version == "v2"
        assert new_entry.shadow.observation_id == obs_b.id
        assert new_entry.shadow.candidate_version == "v1"

        job = SimpleNamespace(id="job-x")
        manager.on_online_result(job, old_entry)
        manager.on_online_result(job, new_entry)
        assert await wait_for(lambda: obs_a.pending == 0 and obs_b.pending == 0)
        # The slow shadow of the old entry settled into the *stopped* obs A.
        assert obs_a.shadow_ok == 1 and obs_a.status == "stopped"
        assert obs_b.shadow_ok == 1
        assert old_entry.shadow.disagree is False  # v1 vs v2 both say spam here
    finally:
        await manager.stop()


@pytest.mark.asyncio
async def test_disagreement_recorded_with_both_scores_and_versions():
    registry = make_registry()
    manager = ShadowManager(registry, timeout_ms=1000)
    await manager.start()
    try:
        obs, _ = manager.start_observation("v2")
        # "winner" is unknown to v1 (ham-leaning) but strongly spam in v2.
        entry = make_entry(text="winner winner winner")
        entry.result = {"status": "ok", "score": 0.27, "label": "ham"}
        entry.shadow = manager.pin("v1")
        manager.on_online_result(SimpleNamespace(id="job-9"), entry)
        assert await wait_for(lambda: entry.shadow.status == "ok")
        assert entry.shadow.disagree is True
        assert obs.disagreements == 1
        item = obs.disagreement_items[0]
        assert item["online"]["model_version"] == "v1"
        assert item["shadow"]["model_version"] == "v2"
        assert item["online"]["label"] != item["shadow"]["label"]
        assert item["online"]["score"] is not None
        assert item["shadow"]["score"] is not None
    finally:
        await manager.stop()


# ---------------------------------------------------------------------------
# Gateway-level tests
# ---------------------------------------------------------------------------
@pytest.fixture()
def client():
    main_module.jobs.clear()
    main_module.switches.clear()
    main_module.active_version = main_module.registry.default_version
    main_module.shadow.reset()
    main_module.worker.metrics.update({key: 0 for key in main_module.worker.metrics})
    with TestClient(main_module.app) as test_client:
        yield test_client
    main_module.shadow.reset()


def wait_job(client: TestClient, job_id: str, timeout: float = 10.0) -> dict:
    deadline = time.monotonic() + timeout
    state = {}
    while time.monotonic() < deadline:
        state = client.get(f"/api/jobs/{job_id}").json()
        if state["status"] in ("completed", "failed"):
            return state
        time.sleep(0.02)
    return state


def wait_observation_settled(client: TestClient, obs_id: str, timeout: float = 10.0) -> dict:
    deadline = time.monotonic() + timeout
    snapshot = {}
    while time.monotonic() < deadline:
        snapshot = client.get("/api/observation").json()
        obs = next((o for o in snapshot["observations"] if o["id"] == obs_id), None)
        if obs and obs["pending"] == 0:
            return obs
        time.sleep(0.05)
    return obs or {}


CRYPTO = "Limited offer! Invest in crypto today and double your money fast."


def test_observation_lifecycle_endpoints(client):
    assert client.get("/api/observation").json()["active"] is None
    missing = client.post("/api/observation/start", json={"version": "v9"})
    assert missing.status_code == 404
    stopped_none = client.post("/api/observation/stop")
    assert stopped_none.status_code == 409

    started = client.post("/api/observation/start", json={"version": "v2"})
    assert started.status_code == 201
    first = started.json()["observation"]
    assert first["candidate_version"] == "v2"
    assert first["status"] == "active"
    assert client.get("/api/observation").json()["active"]["id"] == first["id"]

    # Restarting replaces the active period and keeps the old one listed.
    restarted = client.post("/api/observation/start", json={"version": "v1"})
    assert restarted.status_code == 201
    assert restarted.json()["replaced"]["id"] == first["id"]
    second = restarted.json()["observation"]
    snapshot = client.get("/api/observation").json()
    by_id = {o["id"]: o for o in snapshot["observations"]}
    assert by_id[first["id"]]["status"] == "stopped"
    assert by_id[second["id"]]["status"] == "active"

    stopped = client.post("/api/observation/stop")
    assert stopped.status_code == 200
    assert stopped.json()["observation"]["id"] == second["id"]
    assert client.get("/api/observation").json()["active"] is None


def test_no_observation_means_no_shadow(client):
    job = client.post(
        "/api/jobs",
        json={"label": "plain", "emails": [{"text": CRYPTO, "deadline_ms": 800}]},
    ).json()
    state = wait_job(client, job["id"])
    assert state["status"] == "completed"
    assert state["entries"][0]["shadow"] is None
    assert state["shadow"]["pinned"] == 0


def test_shadow_disagreement_listed_with_scores_and_versions(client):
    client.post("/api/model", json={"version": "v1"})
    client.post("/api/observation/start", json={"version": "v2"})
    job = client.post(
        "/api/jobs",
        json={"label": "shadowed", "emails": [{"text": CRYPTO, "deadline_ms": 900}]},
    ).json()
    state = wait_job(client, job["id"])
    entry = state["entries"][0]
    assert entry["status"] == "ok"
    assert entry["model_version"] == "v1"
    assert entry["shadow"] is not None

    obs_id = entry["shadow"]["observation_id"]
    obs = wait_observation_settled(client, obs_id)
    assert obs["included"] == 1
    assert obs["shadow_ok"] == 1

    state = client.get(f"/api/jobs/{job['id']}").json()
    entry = state["entries"][0]
    shadow = entry["shadow"]
    assert shadow["status"] == "ok"
    assert shadow["candidate_version"] == "v2"
    # v1 says ham, v2 says spam on the crypto text: labels must be opposite.
    assert entry["label"] == "ham"
    assert shadow["label"] == "spam"
    assert shadow["disagree"] is True
    assert entry["score"] is not None and shadow["score"] is not None
    assert state["shadow"]["disagreements"] == 1

    snapshot = client.get("/api/observation").json()
    assert snapshot["observations"][0]["disagreements"] == 1
    items = snapshot["recent_disagreements"]
    assert len(items) == 1
    assert items[0]["online"] == {
        "model_version": "v1",
        "score": entry["score"],
        "label": "ham",
    }
    assert items[0]["shadow"]["model_version"] == "v2"
    assert items[0]["shadow"]["label"] == "spam"


def test_stop_excludes_new_emails_but_included_keep_settling(client):
    client.post("/api/model", json={"version": "v1"})
    client.post("/api/observation/start", json={"version": "v2"})
    first = client.post(
        "/api/jobs",
        json={
            "label": "during",
            "emails": [
                {
                    "text": CRYPTO,
                    "deadline_ms": 900,
                    "shadow_simulate_latency_ms": 400,
                }
            ],
        },
    ).json()
    # Wait until the email is accepted (and thus pinned), then stop.
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        state = client.get(f"/api/jobs/{first['id']}").json()
        if state["entries"][0]["shadow"]:
            break
        time.sleep(0.02)
    obs_id = state["entries"][0]["shadow"]["observation_id"]
    client.post("/api/observation/stop")

    second = client.post(
        "/api/jobs",
        json={"label": "after", "emails": [{"text": CRYPTO, "deadline_ms": 900}]},
    ).json()
    second_state = wait_job(client, second["id"])
    assert second_state["entries"][0]["shadow"] is None

    # The first job's shadow still settles (into the stopped observation).
    obs = wait_observation_settled(client, obs_id)
    assert obs["status"] == "stopped"
    assert obs["included"] == 1
    assert obs["shadow_ok"] == 1
    first_state = client.get(f"/api/jobs/{first['id']}").json()
    assert first_state["entries"][0]["shadow"]["status"] == "ok"


def test_pairing_frozen_across_model_switch_and_observation_restart(client):
    """The key interleaving: switch + restart must not retarget accepted mail,
    and a late shadow result still belongs to the original observation."""
    client.post("/api/model", json={"version": "v1"})
    obs_a = client.post("/api/observation/start", json={"version": "v2"}).json()[
        "observation"
    ]
    job1 = client.post(
        "/api/jobs",
        json={
            "label": "old-pair",
            "emails": [
                {"text": CRYPTO, "deadline_ms": 900, "shadow_simulate_latency_ms": 700}
            ],
        },
    ).json()
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        state = client.get(f"/api/jobs/{job1['id']}").json()
        if state["entries"][0]["shadow"]:
            break
        time.sleep(0.02)
    assert state["entries"][0]["shadow"]["observation_id"] == obs_a["id"]

    # Switch the online model AND restart the observation with the roles
    # reversed, then accept a second email under the new pairing.
    client.post("/api/model", json={"version": "v2"})
    obs_b = client.post("/api/observation/start", json={"version": "v1"}).json()[
        "observation"
    ]
    job2 = client.post(
        "/api/jobs",
        json={"label": "new-pair", "emails": [{"text": CRYPTO, "deadline_ms": 900}]},
    ).json()

    wait_job(client, job1["id"])
    wait_job(client, job2["id"])
    obs_a_state = wait_observation_settled(client, obs_a["id"])
    obs_b_state = wait_observation_settled(client, obs_b["id"])

    entry1 = client.get(f"/api/jobs/{job1['id']}").json()["entries"][0]
    entry2 = client.get(f"/api/jobs/{job2['id']}").json()["entries"][0]
    # Pairings frozen at intake, mirror-imaged across the restart.
    assert (entry1["model_version"], entry1["shadow"]["candidate_version"]) == ("v1", "v2")
    assert entry1["shadow"]["observation_id"] == obs_a["id"]
    assert (entry2["model_version"], entry2["shadow"]["candidate_version"]) == ("v2", "v1")
    assert entry2["shadow"]["observation_id"] == obs_b["id"]
    # Both comparisons are disagreements (crypto flips between v1 and v2).
    assert entry1["shadow"]["disagree"] is True
    assert entry2["shadow"]["disagree"] is True
    # The slow shadow of entry1 finished long after obs A was stopped, yet it
    # was credited to obs A, not to the active obs B.
    assert obs_a_state["status"] == "stopped"
    assert obs_a_state["included"] == 1
    assert obs_a_state["shadow_ok"] == 1
    assert obs_a_state["disagreements"] == 1
    assert obs_b_state["included"] == 1
    assert obs_b_state["disagreements"] == 1


def test_shadow_is_async_and_never_uses_online_batches(client):
    """Online conclusions finish on time while a slow shadow is still running;
    the shadow never shows up in the online worker's queues or batches."""
    client.post("/api/model", json={"version": "v1"})
    client.post("/api/observation/start", json={"version": "v2"})
    before = client.get("/api/state").json()["worker"]["metrics"]
    started = time.monotonic()
    job = client.post(
        "/api/jobs",
        json={
            "label": "isolation",
            "emails": [
                {"text": CRYPTO, "deadline_ms": 900, "shadow_simulate_latency_ms": 1500},
                {"text": "Congratulations winner free prize", "deadline_ms": 900},
            ],
        },
    ).json()
    state = wait_job(client, job["id"])
    online_elapsed = time.monotonic() - started
    assert state["status"] == "completed"
    assert all(e["status"] == "ok" for e in state["entries"])
    # The 1.5s shadow did not delay the online job.
    assert online_elapsed < 1.4, online_elapsed
    # At job-completion time the slow shadow is still in flight.
    slow = state["entries"][0]
    assert slow["shadow"]["status"] in ("waiting", "pending")

    # Online worker only ever saw the two online emails.
    after = client.get("/api/state").json()["worker"]["metrics"]
    assert after["submitted"] - before["submitted"] == 2
    batches = client.get("/api/batches?limit=50").json()["batches"]
    own = [b for b in batches if b["request_ids"] and b["request_ids"][0].startswith(job["id"])]
    assert sum(b["size"] for b in own) == 2  # no shadow leaked into batches

    # The slow shadow eventually settles as a normal comparison.
    obs_id = slow["shadow"]["observation_id"]
    obs = wait_observation_settled(client, obs_id, timeout=6)
    assert obs["shadow_ok"] == 2
    settled = client.get(f"/api/jobs/{job['id']}").json()["entries"][0]
    assert settled["shadow"]["status"] == "ok"
    # The online conclusion was never rewritten by the shadow.
    assert settled["status"] == "ok"
    assert settled["label"] == "ham"


def test_shadow_timeout_only_marks_comparison(client):
    client.post("/api/model", json={"version": "v1"})
    client.post("/api/observation/start", json={"version": "v2"})
    runner = main_module.shadow.runner
    original_timeout = runner.timeout_s
    runner.timeout_s = 0.3
    try:
        job = client.post(
            "/api/jobs",
            json={
                "label": "slow-shadow",
                "emails": [
                    {
                        "text": CRYPTO,
                        "deadline_ms": 900,
                        "shadow_simulate_latency_ms": 1500,
                    }
                ],
            },
        ).json()
        state = wait_job(client, job["id"])
        entry = state["entries"][0]
        assert entry["status"] == "ok"  # online conclusion unaffected
        obs = wait_observation_settled(
            client, entry["shadow"]["observation_id"], timeout=6
        )
        assert obs["shadow_timeout"] == 1
        entry = client.get(f"/api/jobs/{job['id']}").json()["entries"][0]
        assert entry["shadow"]["status"] == "timeout"
        assert entry["shadow"]["reason"] == "shadow_deadline_exceeded"
        assert entry["label"] == "ham"  # online label untouched
    finally:
        runner.timeout_s = original_timeout


def test_demo_shadow_endpoint(client):
    result = client.post("/api/demo/shadow").json()
    job_id = result["job"]["id"]
    obs_id = result["observation"]["id"]
    state = wait_job(client, job_id)
    assert state["status"] == "completed"
    entries = state["entries"]
    # Emails 1-3 pinned during the observation, email 4 (after stop) not.
    assert all(e["shadow"] is not None for e in entries[:3])
    assert entries[3]["shadow"] is None
    assert all(e["shadow"]["observation_id"] == obs_id for e in entries[:3])
    assert all(e["model_version"] == "v1" for e in entries)
    # The crypto email disagrees between v1 and v2.
    obs = wait_observation_settled(client, obs_id, timeout=8)
    assert obs["status"] == "stopped"
    assert obs["included"] == 3
    assert obs["disagreements"] >= 1
    # The artificially slow shadow settled as a shadow timeout *after* the
    # observation stopped, still credited to the original period.
    entries = client.get(f"/api/jobs/{job_id}").json()["entries"]
    slow = entries[2]
    assert slow["status"] == "ok"
    assert slow["shadow"]["status"] == "timeout"
    assert obs["shadow_timeout"] == 1
    assert client.get("/api/observation").json()["active"] is None
