"""End-to-end demo: a trading terminal streaming keystrokes into the service.

Starts the service in-process, enrols a user, then replays a realistic desk
session batch-by-batch (as the native agent would), and reports what the agent
would have done.  Two scenarios:

    normal   the enrolled user works through the day (fatigue, a keyboard swap,
             Chinese IME, numeric order entry) -- the agent must stay quiet
    takeover the user walks away and an impostor continues -- the agent must
             lock, and we report how quickly

Run:  PYTHONPATH=src python scripts/demo_stream.py
"""

from __future__ import annotations

import sys

import numpy as np

from kcauth import corpus as C
from kcauth import evaluate as E
from kcauth import models, synth
from kcauth.detector import Decision
from kcauth.service import AuthService


def stream(svc: AuthService, user: str, sid: str, events, batch: int = 25) -> list[tuple[int, str, float]]:
    """Feed events in batches, exactly as the desk agent would POST them."""
    out = []
    for i in range(0, len(events), batch):
        chunk = events[i : i + batch]
        r = svc.score(user, sid, chunk)
        out.append((i + len(chunk), r["decision"], r["score"]))
    return out


def first_lock(trace, start_idx: int = 0):
    for n, dec, score in trace:
        if n >= start_idx and dec == Decision.LOCK.value:
            return n, score
    return None, None


def main() -> int:
    corp = C.build_corpus()
    pop = models.PopulationModel.fit(
        {u: [ev for (uu, ev) in corp.population_events if uu == u] for u in corp.train_uids}
    )
    cfg = E.calibrate(corp, pop, enrol_sessions=2, target_session_false_lock=0.02, evidence_window=16)
    svc = AuthService(pop, cfg)
    print(f"service ready. operating point: lock LLR {cfg.lock_llr:.2f}, "
          f"evidence window {cfg.evidence_window} score-windows "
          f"({cfg.evidence_window * pop.cfg.stride} keys)")

    takeovers = C.build_takeovers(corp, seed=7)

    # Results are reported for EVERY test user, not one hand-picked victim:
    # detection differs a lot between people, and a single trace would mislead.
    honest = {"n": 0, "lock": 0}
    tk = {k: {"n": 0, "lock": 0, "lat": []} for k in ("naive", "first_order", "adaptive")}
    rp = {}
    for user in corp.test_uids:
        svc.enroll(user, corp.enrolment[user])

        for s in corp.genuine:
            if s.actual != user:
                continue
            sid = f"day-{s.session_id}"
            trace = stream(svc, user, sid, s.events)
            honest["n"] += 1
            honest["lock"] += int(any(d == Decision.LOCK.value for _, d, _ in trace))
            svc.reset_session(user, sid)

        for t in takeovers:
            if t.owner != user:
                continue
            sid = f"tk-{t.strategy}"
            trace = stream(svc, user, sid, t.events)
            n, _ = first_lock(trace, t.switch_index)
            d = tk[t.strategy]
            d["n"] += 1
            if n is not None:
                d["lock"] += 1
                d["lat"].append((t.events[n - 1].up_ms - t.switch_t_ms) / 1000.0)
            svc.reset_session(user, sid)

        for s in corp.replay:
            if s.owner != user:
                continue
            sid = f"rp-{s.session_id}"
            trace = stream(svc, user, sid, s.events)
            n, _ = first_lock(trace)
            key = f"{s.state['kind']}/jitter {s.state['jitter']}"
            d = rp.setdefault(key, {"n": 0, "lock": 0})
            d["n"] += 1
            d["lock"] += int(n is not None)
            svc.reset_session(user, sid)

    print(f"\n--- honest working time ({len(corp.test_uids)} users x 6 states) ---")
    print(f"  sessions false-locked: {honest['lock']}/{honest['n']} "
          f"({honest['lock']/honest['n']:.1%})")

    print("\n--- takeover: user walks away, impostor sits down ---")
    for k, v in tk.items():
        if v["lock"]:
            med = sorted(v["lat"])[len(v["lat"]) // 2]
            print(f"  {k:12s} locked {v['lock']}/{v['n']}  median {med:.0f}s after the swap")
        else:
            print(f"  {k:12s} locked 0/{v['n']}")

    print("\n--- scripted replay ---")
    for k, v in sorted(rp.items()):
        print(f"  {k:18s} locked {v['lock']}/{v['n']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
