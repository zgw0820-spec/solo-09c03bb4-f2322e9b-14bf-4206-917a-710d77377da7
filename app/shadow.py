"""Shadow (candidate-model) observation for emails already classified online.

Purpose
-------
The auditor on duty wants to know whether a *candidate* model would disagree
with the model currently serving traffic.  While an observation period is
open, every email accepted by the gateway is pinned — at intake time — with
the pair ``(online model version, candidate version)``.  Once the online
classification for such an email finishes successfully, the same text is
re-scored asynchronously by the pinned candidate version and the two labels
are compared.

Isolation contract
------------------
* Shadow scoring **never** goes through the online :class:`~app.batching.Worker`:
  it runs on its own thread pool with its own concurrency limit and its own
  timeout, so it cannot occupy an online batch, delay an online conclusion,
  or rewrite one.  A slow/failed shadow only changes the *comparison status*
  of that email.
* The ``(online, candidate)`` pairing of an already-accepted email is frozen
  at intake.  Switching the active model, stopping the observation, or
  starting a new observation never retargets it.
* A shadow result that arrives after its observation period was stopped (or
  after a new one started) is still settled into the *original* observation
  period's counters.
"""

from __future__ import annotations

import asyncio
import itertools
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from .model import LinearBoWModel, ModelRegistry, Prediction

logger = logging.getLogger(__name__)

Scorer = Callable[[LinearBoWModel, str], Prediction]

# Shadow record lifecycle:
#   waiting  -- pinned at intake, online conclusion not in yet
#   pending  -- online ok, shadow scoring queued/running
#   ok       -- shadow scored; ``disagree`` tells whether the label flipped
#   timeout  -- shadow exceeded its own deadline (comparison status only)
#   error    -- shadow scoring failed (comparison status only)
#   skipped  -- online conclusion was not ok, so there is nothing to compare
SHADOW_FINAL_STATES = ("ok", "timeout", "error", "skipped")


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


@dataclass
class ShadowRecord:
    """Per-email comparison state, pinned to one observation period."""

    observation_id: str
    candidate_version: str
    status: str = "waiting"
    score: float | None = None
    label: str | None = None
    threshold: float | None = None
    disagree: bool | None = None
    reason: str | None = None
    latency_ms: float | None = None
    started_at: str | None = None
    completed_at: str | None = None

    def public(self) -> dict[str, Any]:
        return {
            "observation_id": self.observation_id,
            "candidate_version": self.candidate_version,
            "status": self.status,
            "score": self.score,
            "label": self.label,
            "threshold": self.threshold,
            "disagree": self.disagree,
            "reason": self.reason,
            "latency_ms": self.latency_ms,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }


@dataclass
class Observation:
    """One observation period: a candidate version plus its settlement stats."""

    id: str
    candidate_version: str
    started_at: str
    status: str = "active"  # active | stopped
    stopped_at: str | None = None
    included: int = 0  # emails pinned to this period at intake
    shadow_ok: int = 0
    shadow_timeout: int = 0
    shadow_error: int = 0
    skipped: int = 0  # online conclusion was not ok -> nothing to compare
    disagreements: int = 0
    pairs: set[str] = field(default_factory=set)  # "online->candidate" seen
    disagreement_items: list[dict[str, Any]] = field(default_factory=list)

    @property
    def settled(self) -> int:
        return self.shadow_ok + self.shadow_timeout + self.shadow_error + self.skipped

    @property
    def pending(self) -> int:
        return max(0, self.included - self.settled)

    def public(self, *, include_items: bool = False) -> dict[str, Any]:
        data: dict[str, Any] = {
            "id": self.id,
            "candidate_version": self.candidate_version,
            "status": self.status,
            "started_at": self.started_at,
            "stopped_at": self.stopped_at,
            "included": self.included,
            "settled": self.settled,
            "pending": self.pending,
            "shadow_ok": self.shadow_ok,
            "shadow_timeout": self.shadow_timeout,
            "shadow_error": self.shadow_error,
            "skipped": self.skipped,
            "disagreements": self.disagreements,
            "pairs": sorted(self.pairs),
        }
        if include_items:
            data["disagreement_items"] = list(reversed(self.disagreement_items[-20:]))
        return data


class ShadowRunner:
    """Executes shadow scores off the online path.

    Uses a dedicated thread pool (never the online worker's executor) plus a
    semaphore, so a burst of shadow work cannot starve online inference
    threads, and a per-task timeout so a stuck shadow settles as ``timeout``
    instead of hanging the comparison forever.
    """

    def __init__(
        self,
        registry: ModelRegistry,
        *,
        timeout_ms: float = 3000.0,
        max_concurrency: int = 4,
        thread_workers: int = 2,
        scorer: Scorer | None = None,
    ) -> None:
        if max_concurrency < 1 or thread_workers < 1:
            raise ValueError("shadow concurrency/thread counts must be >= 1")
        if timeout_ms <= 0:
            raise ValueError("shadow timeout must be positive")
        self.registry = registry
        self.timeout_s = float(timeout_ms) / 1000.0
        self.max_concurrency = int(max_concurrency)
        self.thread_workers = int(thread_workers)
        self._scorer: Scorer = scorer or (lambda model, text: model.predict(text))
        self._executor: ThreadPoolExecutor | None = None
        self._sem: asyncio.Semaphore | None = None
        self._tasks: set[asyncio.Task] = set()
        self.metrics: dict[str, int] = {
            "submitted": 0,
            "ok": 0,
            "timeout": 0,
            "error": 0,
            "cancelled": 0,
        }

    # ------------------------------------------------------------------
    # lifecycle (mirrors Worker: rebind per-loop primitives on start)
    # ------------------------------------------------------------------
    async def start(self) -> None:
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=self.thread_workers, thread_name_prefix="shadow-score"
            )
        self._sem = asyncio.Semaphore(self.max_concurrency)
        logger.info(
            "shadow runner started (timeout=%.0fms, concurrency=%d, threads=%d)",
            self.timeout_s * 1000,
            self.max_concurrency,
            self.thread_workers,
        )

    async def stop(self) -> None:
        for task in list(self._tasks):
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None
        logger.info("shadow runner stopped")

    # ------------------------------------------------------------------
    # submission
    # ------------------------------------------------------------------
    def submit(self, *, text: str, candidate_version: str, latency_sim_s: float = 0.0) -> asyncio.Task:
        """Queue one shadow score; the task resolves to an outcome dict."""
        if self._sem is None or self._executor is None:
            raise RuntimeError("shadow runner is not started")
        # Validate eagerly so a pinned-but-unknown candidate fails fast.
        self.registry.get(candidate_version)
        self.metrics["submitted"] += 1
        task = asyncio.create_task(
            self._run(text, candidate_version, max(0.0, float(latency_sim_s))),
            name=f"shadow-{candidate_version}",
        )
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    async def _run(self, text: str, candidate_version: str, latency_sim_s: float) -> dict:
        assert self._sem is not None and self._executor is not None
        async with self._sem:
            started = time.monotonic()
            model = self.registry.get(candidate_version)
            try:
                loop = asyncio.get_running_loop()
                prediction = await asyncio.wait_for(
                    loop.run_in_executor(
                        self._executor, self._score, model, text, latency_sim_s
                    ),
                    timeout=self.timeout_s,
                )
            except asyncio.TimeoutError:
                self.metrics["timeout"] += 1
                return {
                    "status": "timeout",
                    "reason": "shadow_deadline_exceeded",
                    "latency_ms": round((time.monotonic() - started) * 1000, 3),
                }
            except asyncio.CancelledError:
                # Intentionally swallowed: the caller still wants a final
                # comparison status for the record instead of a stuck task.
                self.metrics["cancelled"] += 1
                return {
                    "status": "error",
                    "reason": "shadow_cancelled",
                    "latency_ms": round((time.monotonic() - started) * 1000, 3),
                }
            except Exception as exc:  # defensive: scorer bugs must not escape
                logger.exception("shadow scoring failed (%s)", candidate_version)
                self.metrics["error"] += 1
                return {
                    "status": "error",
                    "reason": "shadow_internal_error",
                    "detail": str(exc)[:200],
                    "latency_ms": round((time.monotonic() - started) * 1000, 3),
                }
            self.metrics["ok"] += 1
            return {
                "status": "ok",
                "score": round(float(prediction.score), 6),
                "label": prediction.label,
                "threshold": model.threshold,
                "latency_ms": round((time.monotonic() - started) * 1000, 3),
            }

    def _score(self, model: LinearBoWModel, text: str, latency_sim_s: float) -> Prediction:
        prediction = self._scorer(model, text)
        if latency_sim_s > 0:
            # Demo-only: emulate a slow shadow model.  The wait_for above
            # still fires on schedule; the late thread result is discarded.
            time.sleep(latency_sim_s)
        return prediction


class ShadowManager:
    """Owns observation periods and settles shadow outcomes onto job entries."""

    def __init__(
        self,
        registry: ModelRegistry,
        *,
        runner: ShadowRunner | None = None,
        timeout_ms: float = 3000.0,
        max_concurrency: int = 4,
        thread_workers: int = 2,
        scorer: Scorer | None = None,
        history_limit: int = 20,
        disagreement_limit: int = 100,
    ) -> None:
        self.registry = registry
        self.runner = runner or ShadowRunner(
            registry,
            timeout_ms=timeout_ms,
            max_concurrency=max_concurrency,
            thread_workers=thread_workers,
            scorer=scorer,
        )
        self.history_limit = int(history_limit)
        self.disagreement_limit = int(disagreement_limit)
        self.active: Observation | None = None
        self.observations: dict[str, Observation] = {}
        self._seq = itertools.count(1)

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------
    async def start(self) -> None:
        await self.runner.start()

    async def stop(self) -> None:
        await self.runner.stop()

    def reset(self) -> None:
        """Test hook: drop all observation state (runner metrics included)."""
        self.active = None
        self.observations.clear()
        for key in self.runner.metrics:
            self.runner.metrics[key] = 0

    # ------------------------------------------------------------------
    # observation periods
    # ------------------------------------------------------------------
    def start_observation(self, candidate_version: str) -> tuple[Observation, Observation | None]:
        """Open a new period; an already-active one is stopped first.

        Returns ``(new_observation, previous_observation_or_None)``.  Restarting
        never changes the pairing of emails accepted earlier: those keep the
        observation id and candidate version pinned at their intake time.
        """
        if candidate_version not in self.registry:
            raise KeyError(candidate_version)
        previous = self._stop_active()
        obs = Observation(
            id=f"obs-{next(self._seq):04d}",
            candidate_version=candidate_version,
            started_at=_iso_now(),
        )
        self.observations[obs.id] = obs
        self.active = obs
        self._prune()
        logger.info("observation %s started (candidate=%s)", obs.id, candidate_version)
        return obs, previous

    def stop_observation(self) -> Observation:
        """Close the active period.  Already-pinned emails keep settling."""
        if self.active is None:
            raise LookupError("no active observation")
        return self._stop_active()  # type: ignore[return-value]

    def _stop_active(self) -> Observation | None:
        obs = self.active
        if obs is None:
            return None
        obs.status = "stopped"
        obs.stopped_at = _iso_now()
        self.active = None
        logger.info(
            "observation %s stopped (included=%d, still to settle=%d)",
            obs.id,
            obs.included,
            obs.pending,
        )
        return obs

    def _prune(self) -> None:
        stopped = [o for o in self.observations.values() if o.status == "stopped"]
        stopped.sort(key=lambda o: o.started_at)
        while len(self.observations) > self.history_limit and stopped:
            victim = stopped.pop(0)
            # Never prune a period that still has comparisons to settle.
            if victim.pending > 0:
                continue
            self.observations.pop(victim.id, None)
            if len(self.observations) <= self.history_limit:
                break

    # ------------------------------------------------------------------
    # intake pairing
    # ------------------------------------------------------------------
    def pin(self, online_version: str) -> ShadowRecord | None:
        """Snapshot the active observation for one newly accepted email.

        Called synchronously at intake, next to where the gateway pins the
        online model version, so the (online, candidate) pair is atomic with
        acceptance.  Returns ``None`` when no observation is active.
        """
        obs = self.active
        if obs is None:
            return None
        obs.included += 1
        obs.pairs.add(f"{online_version}->{obs.candidate_version}")
        return ShadowRecord(
            observation_id=obs.id, candidate_version=obs.candidate_version
        )

    # ------------------------------------------------------------------
    # settlement
    # ------------------------------------------------------------------
    def on_online_result(self, job: Any, entry: Any) -> None:
        """React to one email's final online conclusion.

        Only successful online classifications are re-scored; anything else
        marks the comparison as skipped.  Scheduling is fire-and-forget: the
        online job never awaits the shadow.
        """
        record: ShadowRecord | None = getattr(entry, "shadow", None)
        if record is None:
            return
        result = getattr(entry, "result", None) or {}
        if result.get("status") != "ok":
            record.status = "skipped"
            record.reason = f"online_{result.get('status', 'unknown')}"
            record.completed_at = _iso_now()
            obs = self.observations.get(record.observation_id)
            if obs is not None:
                obs.skipped += 1
            return
        record.status = "pending"
        record.started_at = _iso_now()
        try:
            task = self.runner.submit(
                text=entry.text,
                candidate_version=record.candidate_version,
                latency_sim_s=getattr(entry, "shadow_simulate_latency_ms", 0) / 1000.0,
            )
        except Exception as exc:  # runner down / unknown version: comparison only
            logger.exception("could not schedule shadow for %s", record.observation_id)
            record.status = "error"
            record.reason = "shadow_schedule_failed"
            record.completed_at = _iso_now()
            obs = self.observations.get(record.observation_id)
            if obs is not None:
                obs.shadow_error += 1
            return

        def _settle(done: asyncio.Task, *, _job=job, _entry=entry) -> None:
            try:
                outcome = done.result()
            except asyncio.CancelledError:
                outcome = {"status": "error", "reason": "shadow_cancelled"}
            except Exception as exc:  # pragma: no cover - defensive
                outcome = {
                    "status": "error",
                    "reason": "shadow_internal_error",
                    "detail": str(exc)[:200],
                }
            self._apply_outcome(_job, _entry, outcome)

        task.add_done_callback(_settle)

    def _apply_outcome(self, job: Any, entry: Any, outcome: dict) -> None:
        """Settle a finished shadow score into the *original* observation.

        This runs whenever the shadow finishes — possibly long after the
        observation was stopped or replaced — and always credits the period
        the email was pinned to at intake.
        """
        record: ShadowRecord | None = getattr(entry, "shadow", None)
        if record is None:
            return
        obs = self.observations.get(record.observation_id)
        status = outcome.get("status", "error")
        record.status = status
        record.reason = outcome.get("reason")
        record.latency_ms = outcome.get("latency_ms")
        record.completed_at = _iso_now()
        if status == "ok":
            record.score = outcome.get("score")
            record.label = outcome.get("label")
            record.threshold = outcome.get("threshold")
            online = getattr(entry, "result", None) or {}
            record.disagree = bool(
                record.label is not None and record.label != online.get("label")
            )
            if obs is not None:
                obs.shadow_ok += 1
                if record.disagree:
                    obs.disagreements += 1
                    obs.disagreement_items.append(
                        {
                            "at": record.completed_at,
                            "observation_id": record.observation_id,
                            "job_id": getattr(job, "id", None),
                            "entry_index": getattr(entry, "index", None),
                            "text": (getattr(entry, "text", "") or "")[:160],
                            "online": {
                                "model_version": getattr(entry, "model_version", None),
                                "score": online.get("score"),
                                "label": online.get("label"),
                            },
                            "shadow": {
                                "model_version": record.candidate_version,
                                "score": record.score,
                                "label": record.label,
                            },
                        }
                    )
                    if len(obs.disagreement_items) > self.disagreement_limit:
                        del obs.disagreement_items[
                            : len(obs.disagreement_items) - self.disagreement_limit
                        ]
        elif status == "timeout":
            if obs is not None:
                obs.shadow_timeout += 1
        else:
            if obs is not None:
                obs.shadow_error += 1

    # ------------------------------------------------------------------
    # introspection
    # ------------------------------------------------------------------
    def snapshot(self, *, include_items: bool = True) -> dict[str, Any]:
        ordered = sorted(
            self.observations.values(), key=lambda o: o.started_at, reverse=True
        )
        disagreements: list[dict[str, Any]] = []
        if include_items:
            for obs in ordered:
                disagreements.extend(obs.disagreement_items)
            disagreements.sort(key=lambda item: item.get("at") or "", reverse=True)
            disagreements = disagreements[:20]
        return {
            "active": self.active.public() if self.active else None,
            "observations": [o.public() for o in ordered[:10]],
            "recent_disagreements": disagreements,
            "config": {
                "shadow_timeout_ms": round(self.runner.timeout_s * 1000, 3),
                "max_concurrency": self.runner.max_concurrency,
                "thread_workers": self.runner.thread_workers,
            },
            "metrics": dict(self.runner.metrics),
        }
