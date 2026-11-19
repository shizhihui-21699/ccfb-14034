"""Online detector: turns window scores into desk actions.

Two independent detectors run in parallel and are reported separately, because
they answer different questions and fail differently:

  identity   a running-mean likelihood ratio over content-conditioned window
             features (see models.UserModel.llr_terms).  Sustained negative
             evidence -> STEP_UP then LOCK.  The statistic is built so honest
             drift (fatigue, keyboard, input mode) cannot accumulate: it enters
             both hypotheses and cancels.

  liveness   a periodicity test on the raw (dwell, gap) stream.  A script that
             loops a recording is invisible to identity statistics -- it *is*
             the victim's timing -- but its loop is unmistakable.  Score near
             1.0 at some lag means the desk is being driven by a recording.

Deliberate design choices, each of which trades detection for usability:

  * the agent locks only on LOCK; STEP_UP asks for a second factor instead.
    False locks are the thing users complain about, so the system prefers to
    challenge rather than lock when it is unsure.
  * evidence must be SUSTAINED (`consecutive` bad windows and a CUSUM level),
    so one odd burst (a shouted question, a build finishing) cannot lock.
  * adaptation is bounded and only learns from clearly-genuine windows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence

import numpy as np

from . import features as F
from .events import KeyEvent
from .models import UserModel


class Decision(Enum):
    OK = "ok"
    SUSPICIOUS = "suspicious"
    STEP_UP = "step_up"      # require a second factor; never lock
    LOCK = "lock"


@dataclass
class DetectorConfig:
    # identity evidence: a ROLLING window of this many score-windows
    evidence_window: int = 24
    # decisions are a function of the rolling LLR and how long it has stayed
    # negative; the levels are in LLR units and were calibrated on held-out
    # honest states (see evaluate.calibrate)
    lock_llr: float = -3.0           # sustained negative evidence -> LOCK
    stepup_llr: float = -1.5        # weaker -> STEP_UP (challenge, never lock)
    consecutive: int = 4             # consecutive negative score-windows
    min_windows_before_lock: int = 8
    # liveness
    liveness: F.LivenessConfig = field(default_factory=F.LivenessConfig)
    replay_lock: bool = True
    # adaptation
    adapt_enabled: bool = True
    adapt_lr: float = 0.02
    adapt_min_llr: float = 1.5       # only clearly-genuine windows teach


@dataclass
class WindowRecord:
    stop_idx: int
    t_ms: int
    llr: float
    cusum: float
    decision: Decision
    loop_score: float = 0.0
    replay_flag: bool = False


@dataclass
class DetectorState:
    user: UserModel
    cfg: DetectorConfig
    cusum: float = 0.0
    streak: int = 0
    n_windows: int = 0
    decision: Decision = Decision.OK
    first_lock_idx: int | None = None
    first_lock_t_ms: int | None = None
    first_stepup_idx: int | None = None
    first_stepup_t_ms: int | None = None
    history: list[WindowRecord] = field(default_factory=list)

    def step(self, llr: float, *, x: np.ndarray | None = None, loop_score: float = 0.0,
             stop_idx: int = -1, t_ms: int = 0, window_start: int = 0) -> Decision:
        self.n_windows += 1
        lc = self.cfg.liveness
        replay = loop_score >= lc.loop_threshold

        if replay and self.cfg.replay_lock:
            dec = Decision.LOCK
        else:
            self.streak = self.streak + 1 if llr < 0 else 0
            if (self.n_windows >= self.cfg.min_windows_before_lock
                    and self.streak >= self.cfg.consecutive
                    and llr <= self.cfg.lock_llr):
                dec = Decision.LOCK
            elif llr <= self.cfg.stepup_llr:
                dec = Decision.STEP_UP
            elif llr < 0:
                dec = Decision.SUSPICIOUS
            else:
                dec = Decision.OK
                if (self.cfg.adapt_enabled and x is not None
                        and llr >= self.cfg.adapt_min_llr):
                    self.user.adapt(x, lr=self.cfg.adapt_lr)

        if dec is Decision.LOCK and self.first_lock_idx is None:
            self.first_lock_idx = stop_idx
            self.first_lock_t_ms = t_ms
        if dec in (Decision.STEP_UP, Decision.LOCK) and self.first_stepup_idx is None:
            self.first_stepup_idx = stop_idx
            self.first_stepup_t_ms = t_ms
        self.decision = dec
        self.history.append(WindowRecord(stop_idx, t_ms, llr, self.cusum, dec, loop_score, replay))
        return dec


def run_detector(
    user: UserModel,
    events: Sequence[KeyEvent],
    cfg: DetectorConfig | None = None,
    *,
    live: np.ndarray | None = None,
    stop_on_lock: bool = True,
) -> DetectorState:
    """Score a whole event stream (simulation / offline evaluation).

    `live` is the (n_events, 4) streaming-liveness matrix from
    features.streaming_liveness; if omitted, replay detection is disabled.
    """
    cfg = cfg or DetectorConfig()
    st = DetectorState(user=user, cfg=cfg)
    pop = user.pop
    X = pop.feature_matrix(events)
    if X is None or len(X) == 0:
        return st
    Z = pop.scaler.transform(X)  # type: ignore[union-attr]
    llr = user.score_rolling(Z, cfg.evidence_window)
    spans = pop.cfg.spans(len(events))
    for i, (a, b) in enumerate(spans):
        loop = float(live[b - 1, 0]) if live is not None else 0.0
        dec = st.step(float(llr[i]), x=Z[i], loop_score=loop,
                      stop_idx=b, t_ms=events[b - 1].up_ms, window_start=a)
        if dec is Decision.LOCK and stop_on_lock:
            break
    return st
