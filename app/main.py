"""FastAPI gateway for the spam moderation review console.

Intake pins the *current* active model version onto each accepted email, then
hands it to the single in-process worker.  A model switch therefore affects
only requests accepted from that moment on: emails already queued keep the
version they were accepted with.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .batching import Worker
from .model import ModelRegistry
from .shadow import ShadowManager, ShadowRecord

logger = logging.getLogger("moderation")
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = Path(os.environ.get("MODEL_DIR", BASE_DIR / "models"))
STATIC_DIR = Path(os.environ.get("STATIC_DIR", BASE_DIR / "static"))

MAX_EMAILS_PER_SUBMISSION = int(os.environ.get("MAX_EMAILS_PER_SUBMISSION", "16"))
DEADLINE_MIN_MS = 100
DEADLINE_MAX_MS = 1000
MAX_JOBS_KEPT = int(os.environ.get("MAX_JOBS_KEPT", "30"))

registry = ModelRegistry.load_dir(MODELS_DIR, os.environ.get("DEFAULT_MODEL_VERSION"))
worker = Worker(
    registry,
    max_batch_size=int(os.environ.get("MAX_BATCH_SIZE", "4")),
    batch_wait_ms=float(os.environ.get("BATCH_WAIT_MS", "200")),
    dispatch_guard_ms=float(os.environ.get("DISPATCH_GUARD_MS", "25")),
)
# Shadow (candidate-model) observation.  Runs fully off the online path:
# its own thread pool, concurrency limit and timeout; it never touches the
# online worker's queues or batches.
shadow = ShadowManager(
    registry,
    timeout_ms=float(os.environ.get("SHADOW_TIMEOUT_MS", "3000")),
    max_concurrency=int(os.environ.get("SHADOW_MAX_CONCURRENCY", "4")),
    thread_workers=int(os.environ.get("SHADOW_THREADS", "2")),
)

active_version: str = registry.default_version
switches: list[dict[str, Any]] = []
jobs: dict[str, dict[str, Any]] = {}
_background_tasks: set[asyncio.Task] = set()


def _spawn(coro, *, name: str) -> asyncio.Task:
    """Create a background task while keeping a strong reference to it."""
    task = asyncio.create_task(coro, name=name)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    return task


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _set_active(version: str, *, source: str, note: str | None = None) -> dict[str, Any]:
    global active_version
    if version not in registry:
        raise KeyError(version)
    previous = active_version
    event = {
        "at": _now_iso(),
        "from": previous,
        "to": version,
        "source": source,
        "changed": previous != version,
        "note": note,
    }
    active_version = version
    switches.append(event)
    if len(switches) > 100:
        del switches[: len(switches) - 100]
    logger.info("active model %s -> %s (%s)", previous, version, source)
    return event


# ---------------------------------------------------------------------------
# request models
# ---------------------------------------------------------------------------
class EmailIn(BaseModel):
    text: str = Field(min_length=1, max_length=8000)
    deadline_ms: int = Field(ge=DEADLINE_MIN_MS, le=DEADLINE_MAX_MS)
    # Demo / self-test affordances.  Both default to zero and only affect the
    # timing of this individual message, never the batching contract.
    submit_after_ms: int = Field(default=0, ge=0, le=5000)
    simulate_latency_ms: int = Field(default=0, ge=0, le=5000)
    # Demo-only: slow down the *shadow* re-score of this email (never the
    # online path) to show that a slow shadow only affects comparison status.
    shadow_simulate_latency_ms: int = Field(default=0, ge=0, le=10000)


class SubmitIn(BaseModel):
    label: str = Field(default="", max_length=80)
    emails: list[EmailIn] = Field(min_length=1, max_length=MAX_EMAILS_PER_SUBMISSION)


class SwitchIn(BaseModel):
    version: str
    note: str | None = Field(default=None, max_length=200)


class ObservationStartIn(BaseModel):
    version: str
    note: str | None = Field(default=None, max_length=200)


# ---------------------------------------------------------------------------
# jobs
# ---------------------------------------------------------------------------
@dataclass
class JobEntry:
    index: int
    text: str
    deadline_ms: int
    submit_after_ms: int
    simulate_latency_ms: int
    shadow_simulate_latency_ms: int = 0
    model_version: str | None = None
    accepted_at: str | None = None
    result: dict | None = None
    # Pinned at intake when an observation is active; never re-targeted.
    shadow: ShadowRecord | None = None

    def public(self) -> dict:
        result = self.result or {}
        return {
            "index": self.index,
            "text": self.text,
            "deadline_ms": self.deadline_ms,
            "submit_after_ms": self.submit_after_ms,
            "simulate_latency_ms": self.simulate_latency_ms,
            "shadow_simulate_latency_ms": self.shadow_simulate_latency_ms,
            "model_version": self.model_version,
            "accepted_at": self.accepted_at,
            "status": result.get("status", "accepted"),
            "score": result.get("score"),
            "label": result.get("label"),
            "threshold": result.get("threshold"),
            "batch_id": result.get("batch_id"),
            "request_id": result.get("request_id"),
            "latency_ms": result.get("latency_ms"),
            "reason": result.get("reason"),
            "late_result_discarded": result.get("late_result_discarded", False),
            "submitted_at": result.get("submitted_at"),
            "completed_at": result.get("completed_at"),
            "shadow": self.shadow.public() if self.shadow else None,
        }


@dataclass
class Job:
    id: str
    label: str
    created_at: str
    entries: list[JobEntry]
    status: str = "pending"
    started_at: str | None = None
    finished_at: str | None = None
    error: str | None = None
    events: list[dict[str, Any]] = field(default_factory=list)

    def public(self, *, include_entries: bool = True) -> dict:
        versions = sorted({e.model_version for e in self.entries if e.model_version})
        shadowed = [e for e in self.entries if e.shadow is not None]
        data = {
            "id": self.id,
            "label": self.label,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "error": self.error,
            "events": list(self.events),
            "size": len(self.entries),
            "settled": sum(1 for e in self.entries if e.result is not None),
            "ok": sum(1 for e in self.entries if (e.result or {}).get("status") == "ok"),
            "timeouts": sum(
                1 for e in self.entries if (e.result or {}).get("status") == "timeout"
            ),
            "errors": sum(
                1 for e in self.entries if (e.result or {}).get("status") == "error"
            ),
            "versions": versions,
            "shadow": {
                "pinned": len(shadowed),
                "settled": sum(
                    1 for e in shadowed if e.shadow.status in ("ok", "timeout", "error", "skipped")
                ),
                "disagreements": sum(1 for e in shadowed if e.shadow.disagree),
            },
        }
        if include_entries:
            data["entries"] = [e.public() for e in self.entries]
        return data


def _prune_jobs() -> None:
    if len(jobs) <= MAX_JOBS_KEPT:
        return
    completed = [jid for jid, job in jobs.items() if job.status == "completed"]
    for jid in completed[: max(0, len(jobs) - MAX_JOBS_KEPT)]:
        jobs.pop(jid, None)


def _accept_entry(job: Job, entry: JobEntry) -> asyncio.Task:
    """Intake one email: pin the active model *and* the active observation.

    Both pins happen here, synchronously, at acceptance time.  From then on
    the (online model, candidate model) pairing of this email is frozen —
    later model switches or observation restarts never retarget it.
    """
    version = active_version
    entry.model_version = version
    entry.accepted_at = _now_iso()
    record = shadow.pin(version)
    if record is not None:
        entry.shadow = record
    event: dict[str, Any] = {
        "at": entry.accepted_at,
        "type": "accepted",
        "index": entry.index,
        "model_version": version,
    }
    if record is not None:
        event["shadow_observation"] = record.observation_id
        event["shadow_candidate"] = record.candidate_version
    job.events.append(event)
    coro = worker.submit(
        text=entry.text,
        model_version=version,
        deadline_ms=entry.deadline_ms,
        request_id=f"{job.id}-{entry.index}",
        request_batch_id=job.id,
        latency_sim_ms=entry.simulate_latency_ms,
    )
    return asyncio.create_task(coro)


def _settle_entry(job: Job, entry: JobEntry, result: Any) -> None:
    """Store the final online conclusion, then maybe schedule the shadow.

    The shadow is fire-and-forget: it is only ever *scheduled* here, never
    awaited, so it cannot delay the online job, and it writes only to the
    entry's shadow record, never to the online result.
    """
    if isinstance(result, BaseException):
        entry.result = {
            "status": "error",
            "reason": "gateway_error",
            "detail": str(result),
        }
    else:
        entry.result = result
    shadow.on_online_result(job, entry)


async def _run_job(job: Job) -> None:
    job.status = "running"
    job.started_at = _now_iso()
    start = time.monotonic()
    tasks: list[asyncio.Task] = []
    try:
        for entry in job.entries:
            delay = start + entry.submit_after_ms / 1000.0 - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)
            tasks.append(_accept_entry(job, entry))
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for entry, result in zip(job.entries, results):
            _settle_entry(job, entry, result)
        job.status = "completed"
    except asyncio.CancelledError:
        job.status = "cancelled"
        job.error = "shutdown"
        raise
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("job %s failed", job.id)
        job.status = "failed"
        job.error = str(exc)
    finally:
        job.finished_at = _now_iso()
        _prune_jobs()


def _new_job(payload: SubmitIn, *, prefix: str = "job") -> Job:
    job = Job(
        id=f"{prefix}-{uuid.uuid4().hex[:10]}",
        label=payload.label or "未命名提交",
        created_at=_now_iso(),
        entries=[
            JobEntry(
                index=i + 1,
                text=email.text,
                deadline_ms=email.deadline_ms,
                submit_after_ms=email.submit_after_ms,
                simulate_latency_ms=email.simulate_latency_ms,
                shadow_simulate_latency_ms=email.shadow_simulate_latency_ms,
            )
            for i, email in enumerate(payload.emails)
        ],
    )
    jobs[job.id] = job
    return job


# ---------------------------------------------------------------------------
# app
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(_app: FastAPI):
    await worker.start()
    await shadow.start()
    logger.info(
        "models loaded: %s (active=%s, model_dir=%s)",
        ", ".join(registry.versions),
        active_version,
        MODELS_DIR,
    )
    try:
        yield
    finally:
        await shadow.stop()
        await worker.stop()


app = FastAPI(
    title="Spam Moderation Review Console",
    version="1.0.0",
    description="Version-pinned, deadline-aware batching spam classifier",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html", media_type="text/html")


@app.get("/healthz")
async def healthz() -> dict:
    return {"status": "ok", "active_model": active_version, "models": registry.versions}


@app.get("/api/models")
async def list_models() -> dict:
    models = []
    for version in registry.versions:
        model = registry.get(version)
        order = model.weights.argsort()
        models.append(
            {
                "version": version,
                "description": model.description,
                "threshold": model.threshold,
                "vocabulary_size": len(model.vocabulary),
                "top_spam_terms": [
                    {"term": model.vocabulary[i], "weight": round(float(model.weights[i]), 4)}
                    for i in order[::-1][:8]
                ],
                "top_ham_terms": [
                    {"term": model.vocabulary[i], "weight": round(float(model.weights[i]), 4)}
                    for i in order[:8]
                ],
            }
        )
    return {"active": active_version, "models": models}


@app.post("/api/model")
async def switch_model(payload: SwitchIn) -> dict:
    if payload.version not in registry:
        raise HTTPException(
            status_code=404,
            detail=f"未知模型版本 {payload.version!r}，可用: {registry.versions}",
        )
    event = _set_active(payload.version, source="console", note=payload.note)
    return {"active": active_version, "event": event, "models": registry.versions}


@app.get("/api/state")
async def state() -> dict:
    return {
        "active_model": active_version,
        "models": registry.versions,
        "switches": list(reversed(switches[-20:])),
        "worker": worker.snapshot(),
        "observation": shadow.active.public() if shadow.active else None,
        "config": {
            "max_batch_size": worker.max_batch_size,
            "batch_wait_ms": worker.batch_wait_s * 1000,
            "dispatch_guard_ms": worker.dispatch_guard_s * 1000,
            "max_emails_per_submission": MAX_EMAILS_PER_SUBMISSION,
            "deadline_min_ms": DEADLINE_MIN_MS,
            "deadline_max_ms": DEADLINE_MAX_MS,
        },
        "jobs": [job.public(include_entries=False) for job in
                 sorted(jobs.values(), key=lambda j: j.created_at, reverse=True)[:12]],
        "server_time": _now_iso(),
        "pending": worker.pending_count(),
    }


# ---------------------------------------------------------------------------
# shadow observation (candidate model re-scoring)
# ---------------------------------------------------------------------------
@app.get("/api/observation")
async def get_observation() -> dict:
    return shadow.snapshot()


@app.post("/api/observation/start", status_code=201)
async def start_observation(payload: ObservationStartIn) -> dict:
    """Open an observation period on a candidate version.

    If one is already active it is stopped first (a "restart"): emails
    accepted from this moment pair the *current* active model with the new
    candidate, while emails accepted earlier keep their original pairing.
    """
    if payload.version not in registry:
        raise HTTPException(
            status_code=404,
            detail=f"未知模型版本 {payload.version!r}，可用: {registry.versions}",
        )
    obs, previous = shadow.start_observation(payload.version)
    logger.info(
        "observation %s started by console (candidate=%s, replaced=%s, note=%s)",
        obs.id,
        payload.version,
        previous.id if previous else None,
        payload.note,
    )
    return {
        "observation": obs.public(),
        "replaced": previous.public() if previous else None,
        "active_model": active_version,
    }


@app.post("/api/observation/stop")
async def stop_observation() -> dict:
    """Close the active period.  Already-pinned emails keep settling."""
    try:
        obs = shadow.stop_observation()
    except LookupError:
        raise HTTPException(status_code=409, detail="当前没有进行中的观察") from None
    logger.info(
        "observation %s stopped by console (included=%d, pending=%d)",
        obs.id,
        obs.included,
        obs.pending,
    )
    return {"observation": obs.public()}


@app.post("/api/jobs", status_code=202)
async def create_job(payload: SubmitIn) -> dict:
    job = _new_job(payload)
    _spawn(_run_job(job), name=f"job-runner-{job.id}")
    return job.public()


@app.post("/api/classify")
async def classify(payload: SubmitIn) -> dict:
    """Synchronous convenience endpoint: accept all emails and await results."""
    job = _new_job(payload, prefix="sync")
    await _run_job(job)
    return job.public()


@app.get("/api/jobs")
async def list_jobs(limit: int = Query(default=12, ge=1, le=50)) -> dict:
    ordered = sorted(jobs.values(), key=lambda j: j.created_at, reverse=True)[:limit]
    return {"jobs": [job.public() for job in ordered]}


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str) -> dict:
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="任务不存在或已过期")
    return job.public()


@app.get("/api/batches")
async def batches(limit: int = Query(default=20, ge=1, le=100)) -> dict:
    return {"batches": worker.recent_batches(limit)}


@app.post("/api/demo/interleaved")
async def demo_interleaved() -> dict:
    """Schedule old-model work, switch models mid-queue, then add new work.

    Demonstrates: queued requests keep the version they were accepted with,
    later requests use the new version immediately, and batches never mix
    versions.  One new-version email is given artificial inference latency to
    show that a timeout is not overturned by a late result.
    """
    if "v1" not in registry or "v2" not in registry:
        raise HTTPException(status_code=409, detail="demo 需要 v1 与 v2 两个模型")
    _set_active("v1", source="demo", note="演示开始：先使用 v1")

    old_emails = [
        EmailIn(
            text="Congratulations! You are a winner. Claim your free prize now, click here.",
            deadline_ms=900,
        ),
        EmailIn(
            text="URGENT: your account will be suspended. Verify your account immediately.",
            deadline_ms=800,
        ),
    ]
    new_emails = [
        EmailIn(
            text="Limited offer! Invest in crypto today and double your money fast.",
            deadline_ms=500,
        ),
        EmailIn(
            text="Your parcel could not be delivered. Pay the small fee to reschedule.",
            deadline_ms=400,
        ),
        EmailIn(
            text="Final notice: send your password to unlock the refund.",
            deadline_ms=150,
            simulate_latency_ms=600,
        ),
    ]
    combined = SubmitIn(label="切换交错演示", emails=old_emails + new_emails)
    job = _new_job(combined, prefix="demo")

    async def orchestrate() -> None:
        job.status = "running"
        job.started_at = _now_iso()
        start = time.monotonic()
        tasks: list[asyncio.Task] = []
        try:
            for entry in job.entries[: len(old_emails)]:
                tasks.append(_accept_entry(job, entry))
            await asyncio.sleep(max(0.0, start + 0.06 - time.monotonic()))
            event = _set_active(
                "v2", source="demo", note="演示：切换到 v2（旧请求仍在排队）"
            )
            job.events.append(
                {
                    "at": event["at"],
                    "type": "model_switched",
                    "from": event["from"],
                    "to": event["to"],
                    "pending_old": worker.pending_count("v1"),
                }
            )
            await asyncio.sleep(max(0.0, start + 0.10 - time.monotonic()))
            for entry in job.entries[len(old_emails) :]:
                tasks.append(_accept_entry(job, entry))
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for entry, result in zip(job.entries, results):
                _settle_entry(job, entry, result)
            job.status = "completed"
        except asyncio.CancelledError:
            job.status = "cancelled"
            job.error = "shutdown"
            raise
        finally:
            job.finished_at = _now_iso()
            _prune_jobs()

    _spawn(orchestrate(), name=f"demo-runner-{job.id}")
    return job.public()


@app.post("/api/demo/shadow")
async def demo_shadow() -> dict:
    """Observation demo: start on v2, accept emails, stop, accept one more.

    Timeline (active model is switched to v1 first):
      1. observation opens on candidate v2
      2. three emails are accepted *during* the observation — one of them
         (crypto) flips label between v1 and v2, and one is given an
         artificially slow shadow so its comparison settles late
      3. the observation is stopped while the slow shadow is still running
      4. one more email is accepted *after* the stop — it gets no shadow

    Shows: intake-time pairing, disagreement listing, "no new emails after
    stop", and "late shadow results still settle into the original period".
    The online conclusions are never delayed by the slow shadow.
    """
    if "v1" not in registry or "v2" not in registry:
        raise HTTPException(status_code=409, detail="demo 需要 v1 与 v2 两个模型")
    _set_active("v1", source="demo", note="影子演示：先切回 v1")
    obs, _previous = shadow.start_observation("v2")

    emails = [
        # Disagreement expected: ham under v1, spam under v2.
        EmailIn(
            text="Limited offer! Invest in crypto today and double your money fast.",
            deadline_ms=900,
        ),
        EmailIn(
            text="Congratulations! You are a winner. Claim your free prize now, click here.",
            deadline_ms=900,
        ),
        # Slow shadow: settles (as a shadow timeout) only after the stop.
        EmailIn(
            text="Your parcel could not be delivered. Pay the small fee to reschedule.",
            deadline_ms=900,
            shadow_simulate_latency_ms=5000,
        ),
        # Accepted after the stop: must NOT be shadowed.
        EmailIn(
            text="Final notice: send your password to unlock the refund.",
            deadline_ms=900,
        ),
    ]
    job = _new_job(SubmitIn(label="影子观察演示", emails=emails), prefix="demo")

    async def orchestrate() -> None:
        job.status = "running"
        job.started_at = _now_iso()
        start = time.monotonic()
        tasks: list[asyncio.Task] = []
        try:
            job.events.append(
                {
                    "at": _now_iso(),
                    "type": "observation_started",
                    "observation_id": obs.id,
                    "candidate_version": obs.candidate_version,
                }
            )
            for entry in job.entries[:3]:
                tasks.append(_accept_entry(job, entry))
            # Stop the observation while email #3's shadow is still running.
            await asyncio.sleep(max(0.0, start + 0.15 - time.monotonic()))
            stopped = shadow.stop_observation()
            job.events.append(
                {
                    "at": stopped.stopped_at,
                    "type": "observation_stopped",
                    "observation_id": stopped.id,
                    "included": stopped.included,
                    "pending": stopped.pending,
                }
            )
            await asyncio.sleep(max(0.0, start + 0.20 - time.monotonic()))
            tasks.append(_accept_entry(job, job.entries[3]))
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for entry, result in zip(job.entries, results):
                _settle_entry(job, entry, result)
            job.status = "completed"
        except asyncio.CancelledError:
            job.status = "cancelled"
            job.error = "shutdown"
            raise
        finally:
            job.finished_at = _now_iso()
            _prune_jobs()

    _spawn(orchestrate(), name=f"demo-shadow-{job.id}")
    return {"job": job.public(), "observation": obs.public()}


@app.exception_handler(ValueError)
async def value_error_handler(_request, exc: ValueError) -> JSONResponse:
    return JSONResponse(status_code=400, content={"detail": str(exc)})
