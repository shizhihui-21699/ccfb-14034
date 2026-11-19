"""Evaluation: the numbers the deployment decision actually turns on.

Reports, for a named operating point:

  false-lock rate      fraction of *honest* working time that ends in LOCK
                       (the thing users complain about), over all honest
                       states including fatigue and keyboard swaps
  step-up rate         fraction of honest sessions that merely get challenged
  detection latency    for a takeover at a known moment, how many keys and how
                       long until LOCK, per attacker capability
  detection rate       fraction of attacks ever locked
  replay detection     per replay style, separately, with the reason

All evaluation is on users and states held out from the population fit, and
enrolment scenarios are varied (1/2/3 sessions) so few-shot behaviour is visible
rather than asserted.
"""

from __future__ import annotations

import numpy as np

from typing import Sequence

from . import corpus as C
from . import features as F
from . import models
from .detector import Decision, DetectorConfig, run_detector


def _auc(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if not len(a) or not len(b):
        return float("nan")
    return float(np.mean((a[:, None] > b[None, :]) + 0.5 * (a[:, None] == b[None, :])))


def _live(events, cfg: DetectorConfig) -> np.ndarray:
    return F.streaming_liveness(events, cfg.liveness)


def calibrate(
    corp: C.Corpus,
    pop: models.PopulationModel,
    *,
    enrol_sessions: int = 2,
    evidence_window: int = 24,
    target_session_false_lock: float = 0.02,
    calibration_uids: Sequence[str] | None = None,
) -> DetectorConfig:
    """Choose the LOCK level from held-out honest data.

    The target is expressed in *sessions* false-locked, not windows, because a
    user experiences a lock, not a window.  We take the distribution of honest
    rolling scores on calibration users and set the threshold at the quantile
    that yields the requested session rate.  This is deliberately calibrated on
    a different set of users from the ones it is later measured on, so the
    reported false-lock rate is not self-fulfilling.
    """
    cfg = DetectorConfig(evidence_window=evidence_window)
    uids = list(calibration_uids) if calibration_uids else list(corp.train_uids)
    scores: list[np.ndarray] = []
    for u in uids:
        sess = [ev for (uu, ev) in corp.population_events if uu == u]
        if len(sess) < 2:
            continue
        # enrol on one calibration session, score the rest: honest held-out
        um = models.UserModel.enroll(u, pop, sess[:1])
        for ev in sess[1:]:
            Z = pop.feature_matrix(ev)
            if Z is None or len(Z) < evidence_window:
                continue
            scores.append(um.score_rolling(pop.scaler.transform(Z), evidence_window))
    if not scores:
        return cfg
    # per-session minimum is what decides a lock
    session_min = np.array([s.min() for s in scores])
    q = float(np.clip(target_session_false_lock, 1e-4, 0.5))
    cfg.lock_llr = float(np.quantile(session_min, q)) - 0.25
    cfg.stepup_llr = cfg.lock_llr * 0.55
    return cfg


def evaluate(
    corp: C.Corpus,
    pop: models.PopulationModel,
    *,
    enrol_sessions: int = 2,
    cfg: DetectorConfig | None = None,
    verbose: bool = False,
) -> dict:
    cfg = cfg or DetectorConfig()
    users = {}
    for u in corp.test_uids:
        users[u] = models.UserModel.enroll(u, pop, corp.enrolment[u][:enrol_sessions])

    # ---------------- honest time ----------------------------------------
    honest = {"sessions": 0, "locked": 0, "stepup": 0, "by_state": {}}
    honest_llr = []
    for u in corp.test_uids:
        for s in corp.genuine:
            if s.actual != u:
                continue
            st = run_detector(users[u], s.events, cfg, live=_live(s.events, cfg))
            honest["sessions"] += 1
            # detection of *honest* time is a false lock
            locked = any(r.decision is Decision.LOCK for r in st.history)
            stepup = any(r.decision is Decision.STEP_UP for r in st.history)
            honest["locked"] += int(locked)
            honest["stepup"] += int(stepup)
            key = f"{s.mode}/fat{s.state['fatigue']:.1f}/kb{s.state['keyboard']}"
            d = honest["by_state"].setdefault(key, {"n": 0, "locked": 0})
            d["n"] += 1
            d["locked"] += int(locked)
            if st.history:
                honest_llr.append(st.history[-1].llr)

    # ---------------- attacks from a cold session -------------------------
    attacks = {}
    for s in corp.impostor:
        u = s.owner
        st = run_detector(users[u], s.events, cfg, live=_live(s.events, cfg))
        d = attacks.setdefault(s.state["strategy"], {"n": 0, "locked": 0, "llr": []})
        d["n"] += 1
        d["locked"] += int(any(r.decision is Decision.LOCK for r in st.history))
        if st.history:
            d["llr"].append(st.history[-1].llr)

    replay_stats = {}
    for s in corp.replay:
        u = s.owner
        st = run_detector(users[u], s.events, cfg, live=_live(s.events, cfg))
        kind = s.state["kind"]
        d = replay_stats.setdefault(kind, {"n": 0, "locked": 0, "loop_detected": 0})
        d["n"] += 1
        d["locked"] += int(any(r.decision is Decision.LOCK for r in st.history))
        d["loop_detected"] += int(any(r.replay_flag for r in st.history))

    # ---------------- takeovers ------------------------------------------
    takeovers = C.build_takeovers(corp, seed=7)
    tk = {}
    for t in takeovers:
        u = t.owner
        st = run_detector(users[u], t.events, cfg, live=_live(t.events, cfg), stop_on_lock=True)
        d = tk.setdefault(t.strategy, {"n": 0, "locked": 0, "keys": [], "ms": []})
        d["n"] += 1
        if st.first_lock_idx is not None and st.first_lock_idx > t.switch_index:
            d["locked"] += 1
            d["keys"].append(st.first_lock_idx - t.switch_index)
            d["ms"].append(st.first_lock_t_ms - t.switch_t_ms)

    def summarise(d):
        out = {"n": d["n"], "detection_rate": d["locked"] / max(1, d["n"])}
        if d.get("keys"):
            out["keys_median"] = float(np.median(d["keys"]))
            out["keys_p90"] = float(np.percentile(d["keys"], 90))
            out["sec_median"] = float(np.median(d["ms"]) / 1000.0)
            out["sec_p90"] = float(np.percentile(d["ms"], 90) / 1000.0)
        return out

    return {
        "enrol_sessions": enrol_sessions,
        "honest": {
            "sessions": honest["sessions"],
            "false_lock_rate": honest["locked"] / max(1, honest["sessions"]),
            "stepup_rate": honest["stepup"] / max(1, honest["sessions"]),
            "by_state": honest["by_state"],
            "llr_median": float(np.median(honest_llr)) if honest_llr else float("nan"),
            "llr_p05": float(np.percentile(honest_llr, 5)) if honest_llr else float("nan"),
        },
        "attacks": {k: summarise({**v, "keys": [], "ms": []}) for k, v in attacks.items()},
        "replay": replay_stats,
        "takeovers": {k: summarise(v) for k, v in tk.items()},
        "attention_auc": {
            "naive": _auc(np.asarray([r for r in honest_llr]),
                          np.asarray(attacks.get("naive", {}).get("llr", []))),
        },
    }


def format_report(res: dict) -> str:
    L = []
    a = L.append
    e = res["enrol_sessions"]
    a(f"=== enrolment: {e} session(s) ===")
    h = res["honest"]
    a(f"honest sessions            {h['sessions']}   (all users x 6 states)")
    a(f"  false-lock rate           {h['false_lock_rate']:.1%}")
    a(f"  step-up (challenge) rate  {h['stepup_rate']:.1%}")
    a(f"  honest LLR median {h['llr_median']:+.2f}  p05 {h['llr_p05']:+.2f}")
    worst = sorted(h["by_state"].items(), key=lambda kv: -kv[1]["locked"] / max(1, kv[1]["n"]))[:3]
    for k, v in worst:
        a(f"    {k:34s} locked {v['locked']}/{v['n']}")
    a("")
    a("impostor (fresh session, 600+ keys)")
    for k, v in sorted(res["attacks"].items()):
        a(f"  {k:14s} detected {v['detection_rate']:.0%}  (n={v['n']})")
    a("")
    a("scripted replay")
    for k, v in sorted(res["replay"].items()):
        a(f"  {k:14s} locked {v['locked']}/{v['n']}  (loop-rule fired {v['loop_detected']}/{v['n']})")
    a("")
    a("takeover mid-session (attacker continues the logged-in session)")
    for k, v in sorted(res["takeovers"].items()):
        if "keys_median" in v:
            a(f"  {k:14s} detected {v['detection_rate']:.0%}  median {v['keys_median']:.0f} keys "
              f"/ {v['sec_median']:.1f}s   p90 {v['keys_p90']:.0f} keys / {v['sec_p90']:.1f}s")
        else:
            a(f"  {k:14s} detected {v['detection_rate']:.0%}")
    return "\n".join(L)
