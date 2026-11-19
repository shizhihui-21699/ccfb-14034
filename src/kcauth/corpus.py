"""Corpus construction: users, enrolment sessions, and adversarial scenarios.

Everything is seeded and reproducible.  Scenario *labels* live in metadata, so
the evaluator never has to infer ground truth from the data it is scoring.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from . import synth
from .events import KeyEvent

KINDS = ("naive", "first_order", "adaptive", "replay")


@dataclass
class SessionSpec:
    """A stretch of desk time with a known owner and state."""

    session_id: str
    owner: str                     # enrolled user id the desk *believes*
    actual: str                    # who is really typing
    mode: str
    events: list[KeyEvent]
    state: dict = field(default_factory=dict)
    switch_index: int | None = None   # event index where the owner changes
    switch_t_ms: int | None = None

    @property
    def n(self) -> int:
        return len(self.events)


@dataclass
class Corpus:
    traits: dict[str, synth.UserTraits]
    train_uids: list[str]
    test_uids: list[str]
    enrolment: dict[str, list[list[KeyEvent]]]
    genuine: list[SessionSpec]
    impostor: list[SessionSpec]
    replay: list[SessionSpec]
    population_events: list[tuple[str, list[KeyEvent]]]


def build_corpus(
    seed: int = 20261120,
    n_train: int = 24,
    n_test: int = 8,
    n_impostor_sessions: int = 2,
    enrol_chars: int = 900,
) -> Corpus:
    rng = np.random.default_rng(seed)

    uids = [f"trader{i:02d}" for i in range(n_train + n_test)]
    traits = {u: synth.sample_traits(rng, u) for u in uids}
    train_uids, test_uids = uids[:n_train], uids[n_train:]

    # ---------------- population calibration data -------------------------
    # This is a one-time, broadly-collected corpus (many people, many states).
    # Its whole job is to estimate the variance components -- especially
    # V_state, the honest drift envelope.  It must therefore span the same
    # state space the desk will actually see: end-of-day fatigue, a swapped
    # keyboard, and all three input modes.  Estimating V_state from calm
    # code-only sessions (an easy mistake) makes the deployed model treat
    # ordinary fatigue and board swaps as identity violations.
    pop_events: list[tuple[str, list[KeyEvent]]] = []
    _POP_STATES = [
        ("code", dict(fatigue=0.05)),
        ("code", dict(fatigue=0.85)),
        ("code", dict(fatigue=0.4, keyboard_swap=True, keyboard_id=1)),
        ("chinese_ime", dict(fatigue=0.35)),
        ("numeric", dict(fatigue=0.35)),
        ("chinese_ime", dict(fatigue=0.7)),
        ("code", dict(fatigue=0.6, keyboard_swap=True, keyboard_id=2)),
        ("numeric", dict(fatigue=0.75)),
    ]
    for u in train_uids:
        for j, (mode, kw) in enumerate(_POP_STATES):
            st = synth.make_state(rng, f"pop{j}", mode, **kw)
            toks = synth.gen_text(rng, mode, 420)
            ev = synth.synth_session(traits[u], st, toks, rng, t0_ms=1_000_000 * (j + 1))
            pop_events.append((u, ev))

    # ---------------- enrolment (what a new joiner provides) ---------------
    enrolment: dict[str, list[list[KeyEvent]]] = {}
    for u in uids:
        sess = []
        # 3 roughly-minute-long sessions: mostly code, one mixed
        plan = [("code", 1.0), ("code", 0.55), ("chinese_ime", 0.45)]
        for j, (mode, frac) in enumerate(plan):
            st = synth.make_state(rng, f"enrol{j}", mode, fatigue=0.15)
            toks = synth.gen_text(rng, mode, int(enrol_chars * frac))
            sess.append(synth.synth_session(traits[u], st, toks, rng, t0_ms=2_000_000 * (j + 1)))
        enrolment[u] = sess

    # ---------------- genuine sessions across states ----------------------
    genuine: list[SessionSpec] = []
    sid = 0
    for u in uids:
        scenarios = [
            ("code", dict(fatigue=0.0)),
            ("code", dict(fatigue=0.85)),                     # end of a long day
            ("code", dict(fatigue=0.35, keyboard_swap=True, keyboard_id=2)),
            ("chinese_ime", dict(fatigue=0.35)),
            ("numeric", dict(fatigue=0.35)),
            ("code", dict(fatigue=0.5, keyboard_swap=True, keyboard_id=3)),
        ]
        for mode, kw in scenarios:
            st = synth.make_state(rng, "genuine", mode, **kw)
            toks = synth.gen_text(rng, mode, 700)
            ev = synth.synth_session(traits[u], st, toks, rng, t0_ms=3_000_000 * (sid + 1))
            genuine.append(SessionSpec(
                session_id=f"g{sid:04d}", owner=u, actual=u, mode=mode, events=ev,
                state={"fatigue": st.fatigue, "keyboard": st.keyboard},
            ))
            sid += 1

    # ---------------- impostor sessions -----------------------------------
    impostor: list[SessionSpec] = []
    for u in test_uids:
        for j in range(n_impostor_sessions):
            strategy = KINDS[j % 3]  # naive / first_order / adaptive
            attacker = uids[(uids.index(u) + 5 + j) % len(uids)]
            t = synth.imitate_traits(traits[u], strategy, rng, list(traits.values()))
            st = synth.make_state(rng, "imp", "code", fatigue=0.4)
            toks = synth.gen_text(rng, "code", 700)
            ev = synth.synth_session(t, st, toks, rng, t0_ms=9_000_000 + sid * 1000)
            impostor.append(SessionSpec(
                session_id=f"i{sid:04d}", owner=u, actual=attacker, mode="code", events=ev,
                state={"strategy": strategy},
            ))
            sid += 1

    # ---------------- replay scenarios ------------------------------------
    # Two distinct attacks, reported separately because they are detectable by
    # different means and a system that handles one can miss the other:
    #   parametric  re-draws durations from a model of the recording -- it has
    #               the victim's marginal statistics and no liveness tells;
    #               only identity texture can catch it.
    #   raw         loops the recorded (dwell, gap) bytes -- invisible to
    #               identity statistics (it *is* the victim) but exposed by
    #               periodicity and millisecond-exact repeats.
    replay: list[SessionSpec] = []
    for u in test_uids:
        rec_toks = synth.gen_text(rng, "code", 500)
        rec_state = synth.make_state(rng, "rec", "code", fatigue=0.3)
        recording = synth.record_timing(traits[u], rec_state, rec_toks, rng)
        raw_rec = synth.record_raw_timing(traits[u], rec_state, rec_toks, rng)
        for j, jitter in enumerate((0.0, 0.02, 0.06)):
            toks = synth.gen_text(rng, "code", 600)
            stream = synth.replay_stream(recording, len(toks), rng, jitter=jitter)
            st = synth.make_state(rng, "replay", "code", fatigue=0.3)
            ev = synth.synth_session(traits[u], st, toks, rng, t0_ms=20_000_000 + sid * 1000,
                                     timing_stream=stream)
            replay.append(SessionSpec(
                session_id=f"r{sid:04d}", owner=u, actual=f"{u}-script", mode="code", events=ev,
                state={"kind": "parametric", "jitter": jitter},
            ))
            sid += 1
        for j, jitter_ms in enumerate((0.0, 1.0)):
            toks = synth.gen_text(rng, "code", 600)
            ev = synth.synth_raw_replay(toks, raw_rec, rng, t0_ms=30_000_000 + sid * 1000,
                                        jitter_ms=jitter_ms)
            replay.append(SessionSpec(
                session_id=f"r{sid:04d}", owner=u, actual=f"{u}-script", mode="code", events=ev,
                state={"kind": "raw", "jitter": jitter_ms},
            ))
            sid += 1

    return Corpus(
        traits=traits, train_uids=train_uids, test_uids=test_uids,
        enrolment=enrolment, genuine=genuine, impostor=impostor, replay=replay,
        population_events=pop_events,
    )


@dataclass
class TakeoverSpec:
    """Genuine typing, then someone else sits down at a known moment."""

    session_id: str
    owner: str
    actual: str
    events: list[KeyEvent]
    switch_index: int
    switch_t_ms: int
    strategy: str


def build_takeovers(corpus: Corpus, seed: int = 7, pre_chars: int = 320, post_chars: int = 520) -> list[TakeoverSpec]:
    """The realistic threat: the real user is logged in and working, walks away,
    and an attacker continues in the same session."""
    rng = np.random.default_rng(seed)
    uids = corpus.test_uids
    out: list[TakeoverSpec] = []
    for u in uids:
        for j, strategy in enumerate(("naive", "first_order", "adaptive")):
            attacker = uids[(uids.index(u) + 3 + j) % len(uids)]
            t = synth.imitate_traits(corpus.traits[u], strategy, rng, list(corpus.traits.values()))
            st_a = synth.make_state(rng, "pre", "code", fatigue=0.3)
            toks_a = synth.gen_text(rng, "code", pre_chars)
            ev_a = synth.synth_session(corpus.traits[u], st_a, toks_a, rng, t0_ms=40_000_000)
            st_b = synth.make_state(rng, "post", "code", fatigue=0.4)
            toks_b = synth.gen_text(rng, "code", post_chars)
            # the attacker sits down 20 s after the victim's last keystroke
            t0 = ev_a[-1].up_ms + 20_000
            ev_b = synth.synth_session(t, st_b, toks_b, rng, t0_ms=t0)
            # synth_session pins its own run of keys relative to t0, so rebase
            # the block to start exactly at the intended moment
            shift = t0 - ev_b[0].down_ms
            if shift:
                ev_b = [
                    type(e)(key=e.key, down_ms=e.down_ms + shift, up_ms=e.up_ms + shift, meta=e.meta)
                    for e in ev_b
                ]
            switch_idx = len(ev_a)
            out.append(TakeoverSpec(
                session_id=f"t{u}-{strategy}", owner=u, actual=attacker,
                events=ev_a + ev_b, switch_index=switch_idx,
                switch_t_ms=ev_b[0].down_ms, strategy=strategy,
            ))
    return out
