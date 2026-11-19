"""Scoring models: a random-effects formulation for continuous keyboard identity.

The naive approach -- enrol a user, then flag any window far from their mean --
fails on this problem for a structural reason, not a tuning reason.  A person's
honest state moves their timing as much as a different person does:

    variance of a session's mean across honest states (fatigue, keyboard,
    input mode)   ~ 0.7-0.8   in scaled units
    window-to-window noise within a state  ~ 0.3

So "distance from enrolment" ranks a tired user on a new keyboard *below* an
informed impostor.  Any threshold that stops the false locks also stops
detecting; any threshold that detects also locks honest users all day.

The fix is to model the state explicitly.  Write a window feature as

    x = mu_user + b_state + e_window
        b_state ~ N(0, V_state)     # honest drift: fatigue, keyboard, mode
        e_window ~ N(0, V_noise)    # motor noise within a state

and score a running mean m over n windows against two hypotheses:

    H1 same user, arbitrary state :  m ~ N(mu_A,  V_state + V_noise/n)
    H2 a different user           :  m ~ N(0,     V_between + V_state + V_noise/n)

The crucial property: V_state appears in BOTH variances.  Honest drift therefore
cancels out of the likelihood ratio -- it cannot accumulate into a false alarm no
matter how long the session runs -- while V_between, the only term that actually
distinguishes people, keeps contributing as n grows.  A feature earns its weight
only in proportion to V_between / (V_between + V_state), which is exactly its ICC
across honest states.  Features whose honest drift swamps their between-person
signal (interval spread, pause rate) are correctly ignored.

Two consequences that are reported honestly rather than papered over:
  * With little enrolment the posterior over mu_A is wide, so the model is
    conservative (few detections, few false locks).
  * A persistent state shift that exceeds V_state -- a genuinely different
    keyboard model, a hand injury -- will read as suspicious.  The step-up
    path (re-authentication) exists for exactly this, and adaptation moves the
    anchor slowly with a hard budget.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from . import features as F
from .events import KeyEvent

# ---------------------------------------------------------------------------
# windowing
# ---------------------------------------------------------------------------


@dataclass
class WindowConfig:
    size: int = 40
    stride: int = 10

    def spans(self, n: int) -> list[tuple[int, int]]:
        if n < self.size:
            return []
        out = []
        s = 0
        while s + self.size <= n:
            out.append((s, s + self.size))
            s += self.stride
        return out


def windows_for(events: Sequence[KeyEvent], pop: "PopulationModel", cfg: WindowConfig):
    """Yield (stop_index, stop_time_ms, feature_vector, (start, stop))."""
    res = pop.event_residuals(events)
    for a, b in cfg.spans(len(events)):
        yield b, events[b - 1].up_ms, F.window_features(res, a, b), (a, b)


# ---------------------------------------------------------------------------
# robust scaling
# ---------------------------------------------------------------------------


def _mad(x: np.ndarray) -> np.ndarray:
    m = np.median(x, axis=0)
    return 1.4826 * np.median(np.abs(x - m), axis=0)


@dataclass
class RobustScaler:
    center: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, X: np.ndarray) -> "RobustScaler":
        center = np.median(X, axis=0)
        scale = _mad(X)
        scale = np.where(scale < 1e-6, np.maximum(np.std(X, axis=0), 1e-3), scale)
        return cls(center=center, scale=scale)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (np.atleast_2d(X) - self.center) / self.scale

    def to_dict(self) -> dict:
        return {"center": self.center.tolist(), "scale": self.scale.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "RobustScaler":
        return cls(np.asarray(d["center"], float), np.asarray(d["scale"], float))


# ---------------------------------------------------------------------------
# population model
# ---------------------------------------------------------------------------


@dataclass
class PopulationModel:
    """Shared content model plus the random-effects variance components."""

    cfg: WindowConfig
    scaler: RobustScaler | None = None
    v_between: np.ndarray | None = None
    v_state: np.ndarray | None = None
    v_noise: np.ndarray | None = None
    v_between3: np.ndarray | None = None   # third-moment ratio (for t-approx)
    n_users: int = 0
    n_sessions: int = 0

    # -- construction ------------------------------------------------------
    @property
    def content(self) -> F.PopulationProfile:
        return self._content  # type: ignore[attr-defined]

    def event_residuals(self, events: Sequence[KeyEvent]):
        return self.content.event_residuals(events)

    def windows_of(self, events: Sequence[KeyEvent]) -> list[np.ndarray]:
        sel = self.cfg.spans(len(events))
        if not sel:
            return []
        res = self.event_residuals(events)
        return [F.window_features(res, a, b) for a, b in sel]

    def feature_matrix(self, events: Sequence[KeyEvent]) -> np.ndarray | None:
        xs = self.windows_of(events)
        if not xs:
            return None
        return np.asarray(xs)

    # -- fitting -----------------------------------------------------------
    @classmethod
    def fit(
        cls,
        per_user_sessions: dict[str, list[list[KeyEvent]]],
        cfg: WindowConfig | None = None,
        min_sessions_per_user: int = 1,
    ) -> "PopulationModel":
        cfg = cfg or WindowConfig()
        raw = [(u, ev) for u, sess in per_user_sessions.items() for ev in sess]
        pop = cls(cfg=cfg)
        pop._content = F.PopulationProfile.fit(raw)  # type: ignore[attr-defined]

        # raw window features for every session
        per_user: dict[str, list[np.ndarray]] = {}
        for uid, sessions in per_user_sessions.items():
            mats = []
            for ev in sessions:
                X = pop.feature_matrix(ev)
                if X is not None and len(X) >= 2:
                    mats.append(X)
            if mats:
                per_user[uid] = mats
        if not per_user:
            raise ValueError("no usable sessions to fit the population model")

        all_X = np.vstack([m for mats in per_user.values() for m in mats])
        scaler = RobustScaler.fit(all_X)
        per_user_z = {u: [scaler.transform(m) for m in mats] for u, mats in per_user.items()}

        # within-session noise (var of windows around the session mean)
        noise = []
        for mats in per_user_z.values():
            for Z in mats:
                if len(Z) >= 2:
                    noise.append(Z.var(axis=0, ddof=1))
        v_noise = np.nanmean(np.asarray(noise), axis=0) if noise else np.ones(all_X.shape[1])

        # state drift (var of session means within user) and between-user var
        state_parts, user_means, user_weights = [], [], []
        for u, mats in per_user_z.items():
            means = np.asarray([Z.mean(axis=0) for Z in mats])
            user_means.append(means.mean(axis=0))
            user_weights.append(sum(len(Z) for Z in mats))
            if len(means) >= 2:
                state_parts.append(means.var(axis=0, ddof=1))
        v_state = np.nanmean(np.asarray(state_parts), axis=0) if state_parts else np.zeros(all_X.shape[1])
        user_means = np.asarray(user_means)
        v_between = np.nanvar(user_means, axis=0, ddof=1) if len(user_means) > 1 else np.zeros(all_X.shape[1])

        # floors: never let noise go to 0 (would make any drift infinitely bad)
        v_noise = np.maximum(v_noise, 1e-3)
        v_state = np.maximum(v_state, 1e-2)
        v_between = np.maximum(v_between, 1e-3)

        pop.scaler = scaler
        pop.v_noise = v_noise
        pop.v_state = v_state
        pop.v_between = v_between
        pop.n_users = len(per_user)
        pop.n_sessions = len(all_X)
        return pop

    @property
    def icc(self) -> np.ndarray:
        assert self.v_between is not None and self.v_state is not None
        return self.v_between / (self.v_between + self.v_state + 1e-9)

    # -- serialization -----------------------------------------------------
    def to_dict(self) -> dict:
        c = self.content
        return {
            "cfg": {"size": self.cfg.size, "stride": self.cfg.stride},
            "scaler": self.scaler.to_dict() if self.scaler else None,
            "v_between": self.v_between.tolist() if self.v_between is not None else None,
            "v_state": self.v_state.tolist() if self.v_state is not None else None,
            "v_noise": self.v_noise.tolist() if self.v_noise is not None else None,
            "n_users": self.n_users,
            "n_sessions": self.n_sessions,
            "content": {
                "dwell_coef": c.dwell_coef.tolist() if c.dwell_coef is not None else None,
                "iki_coef": c.iki_coef.tolist() if c.iki_coef is not None else None,
                "dwell_scale": c.dwell_scale,
                "iki_scale": c.iki_scale,
                "n_sessions": c.n_sessions,
            },
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PopulationModel":
        cfg = WindowConfig(size=d["cfg"]["size"], stride=d["cfg"]["stride"])
        pop = cls(cfg=cfg, n_users=int(d["n_users"]), n_sessions=int(d.get("n_sessions", 0)))
        pop.scaler = RobustScaler.from_dict(d["scaler"]) if d["scaler"] else None
        for name in ("v_between", "v_state", "v_noise"):
            arr = d.get(name)
            setattr(pop, name, np.asarray(arr, float) if arr is not None else None)
        c = d["content"]
        pop._content = F.PopulationProfile(  # type: ignore[attr-defined]
            dwell_coef=np.asarray(c["dwell_coef"], float) if c["dwell_coef"] else None,
            iki_coef=np.asarray(c["iki_coef"], float) if c["iki_coef"] else None,
            dwell_scale=float(c["dwell_scale"]),
            iki_scale=float(c["iki_scale"]),
            n_sessions=int(c["n_sessions"]),
        )
        return pop


# ---------------------------------------------------------------------------
# per-user model
# ---------------------------------------------------------------------------

T_CAP = 16.0  # clip per-feature squared standardized deviation


@dataclass
class UserModel:
    uid: str
    pop: PopulationModel
    mu: np.ndarray | None = None
    post_var: np.ndarray | None = None   # Var(mu | enrolment data), per feature
    state_scale: np.ndarray | None = None  # per-user multiplier on V_state
    n_windows: int = 0
    n_sessions: int = 0
    anchor_mu: np.ndarray | None = None
    adapt_budget: float = 0.5     # L1 cap on drift from anchor, in population-sigma units

    # -- enrolment ---------------------------------------------------------
    @classmethod
    def enroll(cls, uid: str, pop: PopulationModel, sessions: Sequence[Sequence[KeyEvent]],
               prior_sessions: float = 1.0) -> "UserModel":
        """Posterior over mu_user given the enrolment sessions.

        Everything is conjugate Gaussian per feature:

            mu_user   ~ N(0, V_between)                    (population prior)
            session mean ~ N(mu_user, V_state + V_noise/n_s)
            window    ~ N(session mean, V_noise)           -> pooled mean

        so the enrolled mean is the precision-weighted combination of the
        population prior and the data, and we keep its posterior variance.  That
        variance is *essential* downstream: it is what makes a one-session
        enrollee conservative instead of overconfident.  Omitting it (or
        shrinking harder, as an earlier draft of this module did) biases the
        enrolled mean toward the population average, which makes an ordinary
        impostor scoring at the population average look *closer* than the real
        user -- an inversion that silently destroys detection.
        """
        m = cls(uid=uid, pop=pop)
        d = len(F.FEATURE_NAMES)
        v_between = pop.v_between
        v_state = pop.v_state
        v_noise = pop.v_noise
        assert v_between is not None and v_state is not None and v_noise is not None

        mats = [Z for Z in (pop.feature_matrix(ev) for ev in sessions) if Z is not None and len(Z)]
        if not mats:
            m.mu = np.zeros(d)
            m.post_var = v_between.copy()   # no data -> prior
            m.state_scale = np.ones(d)
            m.anchor_mu = m.mu.copy()
            return m

        Zs = [pop.scaler.transform(Z) for Z in mats]  # type: ignore[union-attr]
        n_eff = sum(len(Z) for Z in Zs)
        pooled = np.vstack(Zs).mean(axis=0)

        # Per-user state volatility.  V_state is a population average, but users
        # differ a lot in how much their pacing moves with fatigue and hardware:
        # a metronome-like typist barely changes, another swings wildly.  Using
        # the population average for everyone over-penalises the volatile ones,
        # which is precisely the "false-locked my own desk" complaint.  With two
        # or more enrolment sessions we can measure it directly and shrink the
        # estimate toward 1.0 in proportion to how little evidence there is.
        state_scale = np.ones(d)
        if len(Zs) >= 2:
            means = np.asarray([Z.mean(axis=0) for Z in Zs])
            df = len(Zs) - 1
            emp = means.var(axis=0, ddof=df)
            # Shrink toward the population value; weight = df / (df + k).
            # ONE-SIDED: only widen.  Two enrolment sessions that happen to be
            # similar are not evidence that the user is stable -- they are
            # mostly luck, and letting them narrow V_state makes the model
            # overconfident and *raises* false locks on the first unusual day
            # (measured: 4.2% -> 12.5%).  Widening is safe because V_state
            # cancels out of the likelihood ratio anyway.
            k = 2.0
            w = df / (df + k)
            ratio = emp / np.maximum(v_state, 1e-6)
            state_scale = np.clip((1.0 - w) + w * ratio, 1.0, 4.0)

        # Variance of the pooled mean around mu_user.
        #
        # The subtle part: honest state drift is drawn ONCE PER SESSION, not per
        # window.  So it averages down as V_state / n_sessions, while pure motor
        # noise averages down as V_noise / n_windows.  Getting this wrong --
        # treating state drift as if it averaged over windows -- makes a
        # multi-session enrollee no more certain than a one-session one, and
        # (because n_windows >> n_sessions) lets a single-session enrollee look
        # *more* confident than a three-session one.  Measured: a one-session
        # enrollee scored an impostor at -7.7 while a three-session enrollee
        # scored the same impostor at -5.6, i.e. backwards.
        n_sessions = len(Zs)
        obs_var = v_state * state_scale / n_sessions + v_noise / max(1.0, n_eff)
        # prior can be weakened to express low trust in the population (e.g. a
        # first enrollee); prior_sessions=1 is the neutral setting.
        prior_var = v_between / max(1e-6, prior_sessions)

        prec = 1.0 / prior_var + 1.0 / obs_var
        m.post_var = 1.0 / prec
        m.mu = pooled * (m.post_var / obs_var)     # posterior mean, prior at 0
        m.state_scale = state_scale
        m.n_windows = int(n_eff)
        m.n_sessions = len(Zs)
        m.anchor_mu = m.mu.copy()
        return m

    # -- scoring -----------------------------------------------------------
    def llr_terms(self, m: np.ndarray, n: int) -> np.ndarray:
        """Per-feature log-likelihood-ratio contribution for a running mean m
        over n windows.  Positive = genuine.

        V_state appears in both hypotheses, so honest drift cancels: no matter
        how long the session runs, a tired user on a new keyboard cannot
        accumulate a false alarm.  Only V_between separates the hypotheses, and
        it is the term that tightens as evidence accumulates.
        """
        assert self.mu is not None and self.post_var is not None
        v_noise = self.pop.v_noise
        v_state = self.pop.v_state
        v_between = self.pop.v_between
        assert v_noise is not None and v_state is not None and v_between is not None
        n = max(1, int(n))
        scale = self.state_scale if self.state_scale is not None else 1.0
        base = v_state * scale + v_noise / n
        s1 = base + self.post_var          # same user, state marginalised
        s2 = base + v_between              # a different user
        d1 = np.minimum(((m - self.mu) ** 2) / s1, T_CAP)
        d2 = np.minimum((m ** 2) / s2, T_CAP)
        return 0.5 * (np.log(s2 / s1) - d1 + d2)

    def score_rolling(self, mats: np.ndarray, window: int) -> np.ndarray:
        """LLR of the *trailing* mean over `window` windows, after each window.

        A running mean from login would let hundreds of genuine windows dilute a
        mid-session takeover to nothing, so the evidence statistic must be a
        rolling window.  `window` is a deployment knob traded against noise:
        short windows react faster but a single odd burst counts more.
        """
        out = np.empty(len(mats))
        csum = np.vstack([np.zeros((1, mats.shape[1])), np.cumsum(mats, axis=0)])
        for i in range(len(mats)):
            lo = max(0, i + 1 - window)
            n = i + 1 - lo
            out[i] = float(self.llr_terms((csum[i + 1] - csum[lo]) / n, n).sum())
        return out

    # -- bounded adaptation ------------------------------------------------
    def adapt(self, z_row: np.ndarray, lr: float = 0.02) -> None:
        """Slow anchored update; caller must pass only clearly-genuine windows.

        Adaptation also narrows the posterior a little: genuine evidence is
        evidence about this user's trait, though we cap how far the mean may
        drift from the enrolment anchor so an attacker who gets a handful of
        accepted windows cannot walk the model onto themselves.
        """
        assert self.mu is not None and self.anchor_mu is not None and self.post_var is not None
        self.post_var = self.post_var * (1.0 - 0.5 * lr)
        new = (1 - lr) * self.mu + lr * z_row
        drift = new - self.anchor_mu
        total = np.abs(drift).sum()
        if total > self.adapt_budget:
            new = self.anchor_mu + drift * (self.adapt_budget / total)
        self.mu = new

    def reset_anchor(self) -> None:
        assert self.mu is not None
        self.anchor_mu = self.mu.copy()

    # -- serialization -----------------------------------------------------
    def to_dict(self) -> dict:
        assert self.mu is not None
        return {
            "uid": self.uid, "mu": self.mu.tolist(),
            "post_var": self.post_var.tolist() if self.post_var is not None else None,
            "state_scale": self.state_scale.tolist() if self.state_scale is not None else None,
            "n_windows": self.n_windows,
            "n_sessions": self.n_sessions,
            "anchor_mu": self.anchor_mu.tolist() if self.anchor_mu is not None else None,
            "adapt_budget": self.adapt_budget,
        }

    @classmethod
    def from_dict(cls, d: dict, pop: PopulationModel) -> "UserModel":
        return cls(uid=d["uid"], pop=pop,
                   mu=np.asarray(d["mu"], float),
                   post_var=np.asarray(d["post_var"], float) if d.get("post_var") else None,
                   state_scale=np.asarray(d["state_scale"], float) if d.get("state_scale") else None,
                   n_windows=int(d["n_windows"]), n_sessions=int(d.get("n_sessions", 0)),
                   anchor_mu=np.asarray(d["anchor_mu"], float) if d["anchor_mu"] else None,
                   adapt_budget=float(d.get("adapt_budget", 0.5)))
