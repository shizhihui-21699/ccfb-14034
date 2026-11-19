"""Unit and property tests.

Run with:  PYTHONPATH=src python -m pytest tests -q
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from kcauth import corpus as C
from kcauth import evaluate as E
from kcauth import features as F
from kcauth import layout, models, synth
from kcauth.detector import Decision, DetectorConfig, run_detector
from kcauth.events import KeyEvent, parse_events


# ---------------------------------------------------------------------------
# layout / events
# ---------------------------------------------------------------------------


def test_digraph_finger_model():
    assert layout.finger_of("a") == layout.L_PINKY
    assert layout.finger_of("j") == layout.R_INDEX
    assert layout.finger_of(" ") == layout.THUMB
    k = layout.digraph_kind("a", "a")
    assert k.same_key
    k = layout.digraph_kind("a", "s")   # both left pinky/ring, same hand
    assert k.same_hand and not k.hand_alt
    k = layout.digraph_kind("a", "j")   # alternate hands
    assert k.hand_alt


def test_parse_events_rejects_garbage():
    with pytest.raises(ValueError):
        parse_events([{"key": "a", "down": 10}])
    with pytest.raises(ValueError):
        parse_events([{"key": "ab", "down": 1, "up": 2}])
    with pytest.raises(ValueError):
        parse_events([{"key": "a", "down": 50, "up": 10}])
    with pytest.raises(ValueError):
        parse_events([])
    ev = parse_events([{"key": "a", "down": 100, "up": 190, "meta": {"mode": "code"}}])
    assert ev[0].dwell_ms == 90


# ---------------------------------------------------------------------------
# synthesiser
# ---------------------------------------------------------------------------


def test_synth_is_physically_plausible():
    rng = np.random.default_rng(3)
    u = synth.sample_traits(rng, "u")
    st = synth.make_state(rng, "s", "code", fatigue=0.3)
    ev = synth.synth_session(u, st, synth.gen_text(rng, "code", 400), rng)
    dwell = np.array([e.dwell_ms for e in ev])
    gaps = np.diff([e.down_ms for e in ev])
    assert dwell.min() >= 18
    assert np.median(dwell) < 200
    assert np.median(gaps) < 500
    # monotone downs, up > down
    assert np.all(gaps >= 0)
    assert all(e.up_ms > e.down_ms for e in ev)


def test_synth_pace_is_realistic():
    """A 700-key session must last minutes, not hours."""
    rng = np.random.default_rng(11)
    u = synth.sample_traits(rng, "u")
    st = synth.make_state(rng, "s", "code", fatigue=0.4)
    ev = synth.synth_session(u, st, synth.gen_text(rng, "code", 700), rng)
    minutes = (ev[-1].up_ms - ev[0].down_ms) / 60000.0
    assert 1.0 < minutes < 15.0, f"unrealistic session length {minutes:.1f} min"


def test_chinese_ime_has_commit_structure():
    rng = np.random.default_rng(5)
    toks = synth.gen_text(rng, "chinese_ime", 300)
    chars = "".join(t.ch for t in toks)
    assert any(c.isdigit() or c == " " for c in chars)


def test_raw_replay_is_a_loop():
    rng = np.random.default_rng(7)
    u = synth.sample_traits(rng, "u")
    st = synth.make_state(rng, "s", "code", fatigue=0.3)
    rec_toks = synth.gen_text(rng, "code", 500)
    raw = synth.record_raw_timing(u, st, rec_toks, rng)
    events = synth.synth_raw_replay(synth.gen_text(rng, "code", 600), raw, rng, 1_000_000)
    dwell = np.array([e.dwell_ms for e in events], float)
    gap = np.zeros_like(dwell)
    gap[1:] = np.diff([e.down_ms for e in events])
    score = F.loop_periodicity(dwell, gap)
    assert score > 0.8, f"raw replay not detected as periodic ({score:.2f})"


def test_live_stream_is_not_a_loop():
    rng = np.random.default_rng(9)
    u = synth.sample_traits(rng, "u")
    st = synth.make_state(rng, "s", "code", fatigue=0.3)
    ev = synth.synth_session(u, st, synth.gen_text(rng, "code", 900), rng)
    dwell = np.array([e.dwell_ms for e in ev], float)
    gap = np.zeros_like(dwell)
    gap[1:] = np.diff([e.down_ms for e in ev])
    assert F.loop_periodicity(dwell, gap) < 0.25


# ---------------------------------------------------------------------------
# model mathematics
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fitted():
    corp = C.build_corpus(seed=12345, n_train=14, n_test=4)
    pop = models.PopulationModel.fit(
        {u: [ev for (uu, ev) in corp.population_events if uu == u] for u in corp.train_uids}
    )
    return corp, pop


def test_variance_components_are_ordered(fitted):
    _, pop = fitted
    # every component must be positive; honest state drift must be real
    assert np.all(pop.v_between > 0)
    assert np.all(pop.v_state > 0)
    assert np.all(pop.v_noise > 0)
    # state drift should be at least as large as window noise for at least some
    # features -- otherwise the whole premise is wrong
    assert np.any(pop.v_state > pop.v_noise)
    assert np.all((pop.icc >= 0) & (pop.icc <= 1))


def test_state_drift_cancels_in_likelihood_ratio(fitted):
    """The defining property: making V_state larger must not systematically
    move the LLR, because it appears in both hypotheses.  If it did, honest
    drift could accumulate into false alarms."""
    _, pop = fitted
    um = models.UserModel.enroll("x", pop, [])
    um.mu = np.zeros(pop.v_between.shape)
    um.post_var = np.zeros_like(pop.v_between)
    m = np.zeros_like(pop.v_between)
    base = um.llr_terms(m, 10)
    pop.v_state = pop.v_state * 4.0
    after = um.llr_terms(m, 10)
    pop.v_state = pop.v_state / 4.0
    # both hypotheses inflate together, so the LLR must not blow up
    assert np.all(np.abs(after - base) < 1.0)


def test_enrolment_posterior_is_not_over_shrunk(fitted):
    """Regression: an earlier version shrank mu by 0.5 toward the population,
    which made an average impostor look closer than the real user."""
    corp, pop = fitted
    u = corp.test_uids[0]
    um = models.UserModel.enroll(u, pop, corp.enrolment[u][:2])
    Z = np.vstack([pop.scaler.transform(pop.feature_matrix(ev)) for ev in corp.enrolment[u][:2]])
    pooled = Z.mean(axis=0)
    # posterior mean should be close to the data mean, not halfway to zero
    corr = np.corrcoef(um.mu, pooled)[0, 1]
    assert corr > 0.9
    # the posterior shrinks toward the population (that is correct Bayesian
    # behaviour) but must keep most of the data's magnitude -- a factor of 0.5
    # or less is the bug this test guards against
    assert np.linalg.norm(um.mu) > 0.6 * np.linalg.norm(pooled)


def test_llr_is_positive_for_own_data(fitted):
    corp, pop = fitted
    u = corp.test_uids[0]
    um = models.UserModel.enroll(u, pop, corp.enrolment[u])
    own = [s for s in corp.genuine if s.actual == u][0]
    Z = pop.scaler.transform(pop.feature_matrix(own.events))
    llr = um.score_rolling(Z, 16)
    assert np.median(llr[-20:]) > 0


def test_few_shot_is_conservative(fitted):
    """With less enrolment, the model must not be MORE confident."""
    corp, pop = fitted
    u = corp.test_uids[1]
    one = models.UserModel.enroll(u, pop, corp.enrolment[u][:1])
    three = models.UserModel.enroll(u, pop, corp.enrolment[u])
    # posterior variance shrinks with data
    assert np.mean(one.post_var) > np.mean(three.post_var)


def test_adaptation_is_bounded(fitted):
    _, pop = fitted
    um = models.UserModel.enroll("x", pop, [])
    assert um.mu is not None and um.anchor_mu is not None
    start = um.mu.copy()
    far = np.full_like(um.mu, 50.0)
    for _ in range(500):
        um.adapt(far, lr=0.5)
    drift = np.abs(um.mu - start).sum()
    assert drift <= um.adapt_budget + 1e-6


def test_serialization_roundtrip(fitted):
    _, pop = fitted
    um = models.UserModel.enroll("x", pop, [])
    um.mu = np.linspace(-1, 1, len(F.FEATURE_NAMES))
    d = json.loads(json.dumps(um.to_dict()))
    back = models.UserModel.from_dict(d, pop)
    assert np.allclose(back.mu, um.mu)

    pd = json.loads(json.dumps(pop.to_dict()))
    pop2 = models.PopulationModel.from_dict(pd)
    assert np.allclose(pop2.v_between, pop.v_between)
    assert pop2.cfg.size == pop.cfg.size


# ---------------------------------------------------------------------------
# detector behaviour
# ---------------------------------------------------------------------------


def test_liveness_overrides_identity(fitted):
    corp, pop = fitted
    u = corp.test_uids[0]
    um = models.UserModel.enroll(u, pop, corp.enrolment[u])
    own = [s for s in corp.genuine if s.actual == u][0]
    cfg = DetectorConfig()
    # force every window to carry a replay loop score
    Z = pop.scaler.transform(pop.feature_matrix(own.events))
    llr = um.score_rolling(Z, cfg.evidence_window)
    from kcauth.detector import DetectorState

    st = DetectorState(user=um, cfg=cfg)
    dec = None
    for i, v in enumerate(llr):
        dec = st.step(float(v), loop_score=1.0, stop_idx=i, t_ms=i)
        break
    assert dec is Decision.LOCK


def test_honest_session_does_not_lock(fitted):
    corp, pop = fitted
    u = corp.test_uids[0]
    um = models.UserModel.enroll(u, pop, corp.enrolment[u])
    own = [s for s in corp.genuine if s.actual == u and s.mode == "code"
           and s.state["fatigue"] < 0.2][0]
    cfg = DetectorConfig()
    st = run_detector(um, own.events, cfg)
    assert st.decision is not Decision.LOCK


def test_replay_stream_locks(fitted):
    corp, pop = fitted
    u = corp.test_uids[0]
    um = models.UserModel.enroll(u, pop, corp.enrolment[u])
    raw = [s for s in corp.replay if s.owner == u and s.state["kind"] == "raw"
           and s.state["jitter"] == 0.0][0]
    cfg = DetectorConfig()
    live = F.streaming_liveness(raw.events, cfg.liveness)
    st = run_detector(um, raw.events, cfg, live=live)
    # A raw replay of 600 keys is caught long before the loop rule can see a
    # full period (that needs ~2x the recording length), so it is caught by
    # identity here.  The loop rule is exercised separately for long streams.
    assert st.decision is Decision.LOCK


def test_loop_rule_catches_long_raw_replay(fitted):
    """The loop rule needs at least one full period of overlap, so a short
    replay is invisible to it.  Give it a long stream and it must fire."""
    corp, pop = fitted
    u = corp.test_uids[0]
    um = models.UserModel.enroll(u, pop, corp.enrolment[u])
    rng = np.random.default_rng(4)
    rec = synth.record_raw_timing(corp.traits[u],
                                  synth.make_state(rng, "s", "code", fatigue=0.3),
                                  synth.gen_text(rng, "code", 240), rng)
    ev = synth.synth_raw_replay(synth.gen_text(rng, "code", 900), rec, rng, 5_000_000)
    cfg = DetectorConfig()
    live = F.streaming_liveness(ev, cfg.liveness)
    assert live[:, 0].max() > cfg.liveness.loop_threshold


# ---------------------------------------------------------------------------
# service
# ---------------------------------------------------------------------------


def test_service_enroll_score_reset(fitted, tmp_path):
    from kcauth.service import AuthService

    corp, pop = fitted
    svc = AuthService(pop, store_dir=str(tmp_path))
    u = corp.test_uids[0]
    r = svc.enroll(u, corp.enrolment[u])
    assert r["ok"] and r["enrol_windows"] > 0

    own = [s for s in corp.genuine if s.actual == u][0]
    out = svc.score(u, "s1", own.events[:200])
    assert out["decision"] in {d.value for d in Decision}
    assert out["windows_scored"] > 0

    # persisted and reloadable
    svc2 = AuthService(pop, store_dir=str(tmp_path))
    assert u in svc2.users
    svc.reset_session(u, "s1")
    assert (u, "s1") not in svc.sessions


def test_service_rejects_unknown_user(fitted):
    from kcauth.service import AuthService

    _, pop = fitted
    svc = AuthService(pop)
    with pytest.raises(KeyError):
        svc.score("nobody", "s", [KeyEvent("a", 0, 90)])


def test_service_http_roundtrip(fitted):
    import threading
    import urllib.request
    from kcauth.service import AuthService, serve

    corp, pop = fitted
    svc = AuthService(pop)
    u = corp.test_uids[0]
    svc.enroll(u, corp.enrolment[u])
    httpd = serve(svc, "127.0.0.1", 0)
    port = httpd.server_address[1]
    th = threading.Thread(target=httpd.serve_forever, daemon=True)
    th.start()
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/health") as resp:
            assert json.loads(resp.read())["ok"]
        ev = [e.as_dict() for e in corp.genuine[0].events[:120]]
        body = json.dumps({"user_id": u, "session_id": "s", "events": ev}).encode()
        req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/score", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req) as resp:
            out = json.loads(resp.read())
        assert "decision" in out
    finally:
        httpd.shutdown()
