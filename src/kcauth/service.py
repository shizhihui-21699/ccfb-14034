"""Continuous keystroke-rhythm authentication service.

A long-running process that the terminal's keystroke agent talks to.  It owns
the models, the per-session detector state, and the persistence, so the agent
on the desk stays dumb (it only forwards key transitions).

Endpoints (JSON over HTTP; stdlib only, no framework):

    POST /v1/enroll          {user_id, sessions:[[event,...],...]}  -> {ok, windows}
    POST /v1/score           {user_id, session_id, events:[...]}    -> {decision, score, ...}
    POST /v1/reset           {user_id, session_id}                  -> {ok}
    GET  /v1/health                                                 -> {ok, users}
    GET  /v1/thresholds                                             -> operating point

`/v1/score` is the hot path.  The agent batches keystrokes (typically every
0.5-1.5 s, or every 20 keys) and posts them; the service appends them to the
session's rolling event buffer, recomputes window features on the new tail,
updates the detector, and returns the current decision.

Design notes that matter operationally:

  * Sessions are keyed by (user, session).  A new login gets a fresh detector
    state but reuses the user's model -- that is what "enrolled once" means.
  * The decision is the *agent's* action: OK / SUSPICIOUS / STEP_UP / LOCK.
    The service never locks a screen itself; the agent does, so this process
    can be restarted without unlocking anyone.
  * The persisted model is plain JSON so an offline re-fit (a new population
    profile) can be rolled out without losing per-user enrolment.
  * Nothing here logs keystrokes.  The event buffer is in memory and is
    dropped at session end.  This is a deliberate privacy property: a timing
    model does not need the text, and retaining it would make the service a
    keylogger.
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import numpy as np

from . import features as F
from . import models
from .detector import Decision, DetectorConfig, DetectorState
from .events import KeyEvent, parse_events

# how many events to keep per session for feature computation
SESSION_BUFFER = 900


@dataclass
class SessionRecord:
    user_id: str
    session_id: str
    events: list[KeyEvent] = field(default_factory=list)
    detector: DetectorState | None = None
    live: np.ndarray | None = None
    last_seen: float = 0.0
    scored_upto: int = 0

    def append(self, new: list[KeyEvent]) -> None:
        self.events.extend(new)
        # keep a bounded history; drop from the front in stride with the
        # window config so window boundaries stay stable
        if len(self.events) > SESSION_BUFFER * 2:
            drop = len(self.events) - SESSION_BUFFER
            self.events = self.events[drop:]
            self.scored_upto = max(0, self.scored_upto - drop)


class AuthService:
    def __init__(
        self,
        population: models.PopulationModel,
        config: DetectorConfig | None = None,
        store_dir: str | None = None,
    ) -> None:
        self.pop = population
        self.cfg = config or DetectorConfig()
        self.store_dir = store_dir
        self.users: dict[str, models.UserModel] = {}
        self.sessions: dict[tuple[str, str], SessionRecord] = {}
        self._lock = threading.RLock()
        if store_dir:
            os.makedirs(os.path.join(store_dir, "users"), exist_ok=True)
            self._load()

    # -- persistence -------------------------------------------------------
    def _user_path(self, uid: str) -> str:
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in uid)
        return os.path.join(self.store_dir, "users", f"{safe}.json")  # type: ignore[arg-type]

    def save_user(self, uid: str) -> None:
        if not self.store_dir:
            return
        with open(self._user_path(uid), "w") as fh:
            json.dump(self.users[uid].to_dict(), fh)

    def _load(self) -> None:
        d = os.path.join(self.store_dir, "users")  # type: ignore[arg-type]
        for name in sorted(os.listdir(d)):
            if not name.endswith(".json"):
                continue
            with open(os.path.join(d, name)) as fh:
                rec = json.load(fh)
            self.users[rec["uid"]] = models.UserModel.from_dict(rec, self.pop)

    def save_population(self) -> None:
        if not self.store_dir:
            return
        with open(os.path.join(self.store_dir, "population.json"), "w") as fh:
            json.dump(self.pop.to_dict(), fh)

    @classmethod
    def load_population(cls, store_dir: str) -> models.PopulationModel:
        with open(os.path.join(store_dir, "population.json")) as fh:
            return models.PopulationModel.from_dict(json.load(fh))

    # -- enrollment --------------------------------------------------------
    def enroll(self, user_id: str, sessions: list[list[KeyEvent]],
               prior_sessions: float = 1.0) -> dict[str, Any]:
        with self._lock:
            um = models.UserModel.enroll(user_id, self.pop, sessions, prior_sessions=prior_sessions)
            self.users[user_id] = um
            self.save_user(user_id)
            vb = np.ones(1) if self.pop.v_between is None else self.pop.v_between
            pv = np.ones(1) if um.post_var is None else um.post_var
            return {
                "ok": True,
                "user_id": user_id,
                "enrol_windows": um.n_windows,
                "enrol_sessions": um.n_sessions,
                # how much this user's posterior still leans on the population
                "prior_weight": float(np.mean(vb / (vb + pv))),
            }

    # -- scoring -----------------------------------------------------------
    def score(self, user_id: str, session_id: str, events: list[KeyEvent]) -> dict[str, Any]:
        with self._lock:
            if user_id not in self.users:
                raise KeyError(f"unknown user {user_id!r}")
            um = self.users[user_id]
            key = (user_id, session_id)
            rec = self.sessions.get(key)
            if rec is None:
                rec = SessionRecord(user_id=user_id, session_id=session_id,
                                    detector=DetectorState(user=um, cfg=self.cfg))
                self.sessions[key] = rec
            rec.append(events)
            rec.last_seen = time.time()

            decision, score, loop, nwin = self._advance(rec)
            return {
                "decision": decision.value,
                "score": score,
                "loop_score": loop,
                "windows_scored": nwin,
                "session_events": len(rec.events),
                "locked": decision is Decision.LOCK,
            }

    def _advance(self, rec: SessionRecord) -> tuple[Decision, float, float, int]:
        assert rec.detector is not None
        um = self.users[rec.user_id]
        evs = rec.events
        X = um.pop.feature_matrix(evs)
        if X is None or len(X) == 0:
            return rec.detector.decision, 0.0, 0.0, rec.detector.n_windows

        # liveness is computed over the raw stream tail
        live = F.streaming_liveness(evs, self.cfg.liveness)
        Z = um.pop.scaler.transform(X)  # type: ignore[union-attr]
        llr = um.score_rolling(Z, self.cfg.evidence_window)
        spans = um.pop.cfg.spans(len(evs))

        decision = rec.detector.decision
        score = rec.detector.history[-1].llr if rec.detector.history else 0.0
        loop = 0.0
        start_i = rec.scored_upto
        for i in range(start_i, len(spans)):
            a, b = spans[i]
            loop = float(live[b - 1, 0])
            decision = rec.detector.step(
                float(llr[i]), x=Z[i], loop_score=loop,
                stop_idx=b, t_ms=evs[b - 1].up_ms, window_start=a,
            )
            if decision is Decision.LOCK:
                break
        rec.scored_upto = len(spans)
        return decision, float(score), loop, rec.detector.n_windows

    def reset_session(self, user_id: str, session_id: str) -> None:
        with self._lock:
            self.sessions.pop((user_id, session_id), None)

    def gc_sessions(self, max_idle_s: float = 3600.0) -> int:
        with self._lock:
            now = time.time()
            stale = [k for k, r in self.sessions.items() if now - r.last_seen > max_idle_s]
            for k in stale:
                del self.sessions[k]
            return len(stale)


# ---------------------------------------------------------------------------
# HTTP front end
# ---------------------------------------------------------------------------


def make_handler(service: AuthService):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):  # quiet by default
            return

        def _send(self, code: int, payload: dict) -> None:
            body = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _body(self) -> dict:
            n = int(self.headers.get("Content-Length", "0"))
            if n <= 0:
                return {}
            return json.loads(self.rfile.read(n).decode())

        def do_GET(self):
            if self.path == "/v1/health":
                self._send(200, {"ok": True, "users": len(service.users),
                                 "sessions": len(service.sessions)})
            elif self.path == "/v1/thresholds":
                c = service.cfg
                self._send(200, {
                    "evidence_window": c.evidence_window,
                    "lock_llr": c.lock_llr, "stepup_llr": c.stepup_llr,
                    "loop_threshold": c.liveness.loop_threshold,
                })
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):
            try:
                body = self._body()
            except (json.JSONDecodeError, ValueError) as exc:
                self._send(400, {"error": f"bad json: {exc}"})
                return
            try:
                if self.path == "/v1/enroll":
                    sessions = [parse_events(s) for s in body.get("sessions", [])]
                    if not sessions:
                        self._send(400, {"error": "no sessions"})
                        return
                    self._send(200, service.enroll(body["user_id"], sessions,
                                                   float(body.get("prior_sessions", 1.0))))
                elif self.path == "/v1/score":
                    evs = parse_events(body.get("events", []))
                    self._send(200, service.score(body["user_id"], body["session_id"], evs))
                elif self.path == "/v1/reset":
                    service.reset_session(body["user_id"], body["session_id"])
                    self._send(200, {"ok": True})
                else:
                    self._send(404, {"error": "not found"})
            except (KeyError, ValueError) as exc:
                self._send(400, {"error": str(exc)})

    return Handler


def serve(service: AuthService, host: str = "127.0.0.1", port: int = 8099) -> ThreadingHTTPServer:
    httpd = ThreadingHTTPServer((host, port), make_handler(service))
    return httpd
