"""Feature extraction.

The central idea: never compare raw durations across content.  Every observed
log-duration is first predicted from *content* using a population regression
(character class, finger, digraph kind, boundary, input mode), and only the
residual -- how this person deviates from the population *on this content* -- is
used for identity.  This is what makes Chinese-IME pinyin, English prose and
numeric order entry comparable, and what makes a keyboard swap show up as a
dwell shift rather than as a content artefact.

A window of consecutive events is then summarised into a small robust vector:
location (median), spread (MAD), temporal texture (lag-1 autocorrelation,
dwell/interval coupling), and rates of discrete motor events (key overlap,
thinking pauses, crash stops).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from . import layout
from .events import KeyEvent

EPS = 1e-9


# ---------------------------------------------------------------------------
# population content model
# ---------------------------------------------------------------------------

_MODE_NAMES = ["code", "chinese_ime", "numeric", "prose"]
_KIND_NAMES = ["same_key", "same_finger", "same_hand", "hand_alt", "unknown"]
_CLASS_NAMES = ["letter", "symbol", "digit", "space", "enter", "backspace"]
_BOUNDARY = {"word": 1, "clause": 2}


def _mode_index(ev: KeyEvent) -> int:
    m = ev.meta.get("mode", "code")
    return _MODE_NAMES.index(m) if m in _MODE_NAMES else 0


def _kind_index(a: str, b: str) -> int:
    if a == b:
        return 0
    fa, fb = layout.finger_of(a), layout.finger_of(b)
    if fa is None or fb is None:
        return 4
    k = layout.digraph_kind(a, b)
    if k.same_finger:
        return 1
    if k.same_hand:
        return 2
    if k.hand_alt:
        return 3
    return 4


def _boundary_index(a: str, b: str) -> int:
    """Boundary class of the pair, inferred from content (0=motor,1=word,2=clause)."""
    if a in " \n\r" or b in " \n\r":
        return 2
    if a in ".,;:()[]{}\"'=+-*/<>!&|" or b in ".,;:()[]{}\"'=+-*/<>!&|":
        return 1
    return 0


def _dwell_row(ev: KeyEvent) -> np.ndarray:
    cls = layout.char_class(ev.key)
    f = layout.finger_of(ev.key)
    f = layout.THUMB if f is None else f
    row = np.zeros(len(_CLASS_NAMES) + 9 + len(_MODE_NAMES) + 1)
    row[_CLASS_NAMES.index(cls)] = 1.0
    row[len(_CLASS_NAMES) + f] = 1.0
    row[-len(_MODE_NAMES) - 1 + _mode_index(ev)] = 1.0
    row[-1] = 1.0  # intercept
    return row


def _iki_row(prev: KeyEvent, cur: KeyEvent) -> np.ndarray:
    n = len(_KIND_NAMES) + 3 + len(_CLASS_NAMES) * 2 + len(_MODE_NAMES) + 1
    row = np.zeros(n)
    row[_kind_index(prev.key, cur.key)] = 1.0
    row[len(_KIND_NAMES) + _boundary_index(prev.key, cur.key)] = 1.0
    o = len(_KIND_NAMES) + 3
    row[o + _CLASS_NAMES.index(layout.char_class(prev.key))] = 1.0
    o += len(_CLASS_NAMES)
    row[o + _CLASS_NAMES.index(layout.char_class(cur.key))] = 1.0
    o += len(_CLASS_NAMES)
    row[o + _mode_index(cur)] = 1.0
    row[-1] = 1.0
    return row


def _ridge(X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    d = X.shape[1]
    A = X.T @ X + lam * np.eye(d)
    A[0, 0] += lam * 0.1
    return np.linalg.solve(A, X.T @ y)


@dataclass
class PopulationProfile:
    """Content -> expected log-duration, learned from other users."""

    dwell_coef: np.ndarray | None = None
    iki_coef: np.ndarray | None = None
    dwell_scale: float = 0.3
    iki_scale: float = 0.6
    n_sessions: int = 0

    @classmethod
    def fit(cls, sessions: Sequence[tuple[str, list[KeyEvent]]], lam: float = 1.0) -> "PopulationProfile":
        Xd, yd, Xi, yi = [], [], [], []
        for _uid, evs in sessions:
            if len(evs) < 10:
                continue
            Xd.extend(_dwell_row(e) for e in evs)
            yd.extend(np.log(max(15.0, e.dwell_ms)) for e in evs)
            Xi.extend(_iki_row(evs[k - 1], evs[k]) for k in range(1, len(evs)))
            yi.extend(
                np.log(max(5.0, evs[k].down_ms - evs[k - 1].down_ms))
                for k in range(1, len(evs))
            )
        if len(yd) < 200:
            raise ValueError("not enough events to fit population profile")
        Xd = np.asarray(Xd)
        yd = np.asarray(yd)
        Xi = np.asarray(Xi)
        yi = np.asarray(yi)
        cd = _ridge(Xd, yd, lam)
        ci = _ridge(Xi, yi, lam)
        rd = yd - Xd @ cd
        ri = yi - Xi @ ci
        return cls(
            dwell_coef=cd,
            iki_coef=ci,
            dwell_scale=float(1.4826 * np.median(np.abs(rd - np.median(rd)))),
            iki_scale=float(1.4826 * np.median(np.abs(ri - np.median(ri)))),
            n_sessions=len(sessions),
        )

    def event_residuals(self, events: Sequence[KeyEvent]) -> "Residuals":
        if self.dwell_coef is None or self.iki_coef is None:
            raise RuntimeError("population profile is not fitted")
        n = len(events)
        Xd = np.array([_dwell_row(e) for e in events])
        zd = (np.log(np.maximum(15.0, np.array([e.dwell_ms for e in events], dtype=float)))
              - Xd @ self.dwell_coef) / max(EPS, self.dwell_scale)

        zi = np.zeros(n)
        valid = np.zeros(n, dtype=bool)
        if n > 1:
            Xi = np.array([_iki_row(events[k - 1], events[k]) for k in range(1, n)])
            raw = np.array([max(5.0, events[k].down_ms - events[k - 1].down_ms) for k in range(1, n)], dtype=float)
            zi[1:] = (np.log(raw) - Xi @ self.iki_coef) / max(EPS, self.iki_scale)
            valid[1:] = True

        downs = np.array([e.down_ms for e in events], dtype=float)
        ups = np.array([e.up_ms for e in events], dtype=float)
        overlap = np.zeros(n, dtype=bool)
        if n > 1:
            overlap[:-1] = downs[1:] < ups[:-1]
        pause = np.array([bool(e.meta.get("pause")) for e in events])
        return Residuals(
            zd=zd,
            zi=zi,
            zi_valid=valid,
            overlap=overlap,
            pause=pause,
            events=events,
            downs=downs,
            ups=ups,
        )


@dataclass
class Residuals:
    zd: np.ndarray
    zi: np.ndarray
    zi_valid: np.ndarray
    overlap: np.ndarray
    pause: np.ndarray
    events: Sequence[KeyEvent]
    downs: np.ndarray
    ups: np.ndarray

    def __len__(self) -> int:
        return len(self.zd)


# ---------------------------------------------------------------------------
# window features
# ---------------------------------------------------------------------------

FEATURE_NAMES = [
    "zd_med",
    "zi_med",
    "zd_spread",
    "zi_spread",
    "iki_ac1",
    "dwell_iki_corr",
    "overlap_rate",
    "pause_rate",
    "crash_rate",
    "zd_p90",
    "zi_p90",
    "zi_p10",
    "zd_iqr_ratio",
    # finger-contrast features: relative finger timing.  A different keyboard
    # scales every dwell roughly together, so *contrasts* between fingers
    # survive a board swap while the absolute level does not.  Which finger is
    # weak is a motor property of the person, so this is where identity lives.
    "fp_pinky",
    "fp_ring",
    "fp_mid",
    "fp_index",
    # same-finger digraph penalty relative to the overall interval
    "sf_penalty",
]

_FINGER_GROUPS = ("pinky", "ring", "mid", "index")


def _finger_group_index(ev: KeyEvent) -> int:
    f = layout.finger_of(ev.key)
    g = layout.finger_group(f)
    return _FINGER_GROUPS.index(g) if g in _FINGER_GROUPS else -1

REPLAY_FEATURE_NAMES = [
    "loop_score",
    "exact_rep_frac",
    "quantized_frac",
    "delta_entropy",
]

# how many recent durations to keep for the periodicity test
_LOOP_MIN_PERIOD = 20
_LOOP_MIN_OVERLAP = 40


def loop_periodicity(dwell: np.ndarray, gap: np.ndarray, tol: float = 1.5) -> float:
    """Detect a script looping over a fixed recording.

    For each candidate lag p, measure the fraction of positions i where
    (dwell[i], gap[i]) matches (dwell[i+p], gap[i+p]) within `tol` ms.  A live
    typist decorrelates within a few keys, so the fraction collapses to the
    chance level (about 1e-3 at millisecond resolution).  A replayer looping
    over an L-key recording shows a fraction near 1.0 at p = L, and as long as
    there is at least a few tens of keys of overlap one repeat is enough.

    Returns the best score over lags in [MIN_PERIOD, n - MIN_OVERLAP], or 0 if
    the window is too short.  The score is a match fraction, in [0, 1].
    """
    n = len(dwell)
    if n < _LOOP_MIN_PERIOD + _LOOP_MIN_OVERLAP:
        return 0.0
    best = 0.0
    max_p = n - _LOOP_MIN_OVERLAP
    for p in range(_LOOP_MIN_PERIOD, max_p + 1):
        m = n - p
        d = np.abs(dwell[:m] - dwell[p:]) <= tol
        g = np.abs(gap[:m] - gap[p:]) <= tol
        frac = float(np.mean(d & g))
        if frac > best:
            best = frac
    return best


def liveness_features(
    res: "Residuals",
    buf_dwell: np.ndarray,
    buf_gap: np.ndarray,
    start: int,
    stop: int,
) -> np.ndarray:
    """Features that separate a live human stream from a scripted replay.

    Measured on the WHOLE stream seen so far, not on the 40-key scoring window:
    a replay loops over a recording of hundreds of keys, so a short window
    cannot contain a full period and periodicity is mathematically invisible
    inside it.  `buf_dwell`/`buf_gap` carry the running raw (dwell, gap) history.

    loop_score      best lag-p match fraction of the running (dwell, gap) pair
                    sequence.  Live typing decorrelates within a few keys; a
                    looping script scores near 1.0 at p = recording length.
    exact_rep_frac  fraction of (dwell, gap) pairs that recur elsewhere in the
                    buffer (millisecond-exact repeats).
    quantized_frac  fraction of duplicate dwell values in the buffer.
    delta_entropy   normalized entropy of the sorted dwell-difference
                    histogram; scripts are more regular than people.
    """
    if len(buf_dwell) < _LOOP_MIN_PERIOD + _LOOP_MIN_OVERLAP:
        return np.zeros(len(REPLAY_FEATURE_NAMES))
    # cap the buffer so the lag search stays cheap on long sessions
    cap = 1200
    d = np.asarray(buf_dwell[-cap:], dtype=float)
    g = np.asarray(buf_gap[-cap:], dtype=float)

    score = loop_periodicity(d, g)

    pairs = np.stack([d, g], axis=1)
    uniq = np.unique(pairs, axis=0)
    exact_rep = float(1.0 - len(uniq) / len(pairs))
    quant = float(1.0 - len(np.unique(d)) / len(d))

    if len(d) > 2:
        edges = np.clip(np.round(np.diff(np.sort(d))).astype(int) + 64, 0, 128)
        hist = np.bincount(edges, minlength=129).astype(float)
        p = hist / max(EPS, hist.sum())
        ent = float(-np.sum(p[p > 0] * np.log2(p[p > 0]))) / 7.0
    else:
        ent = 0.0
    return np.array([score, exact_rep, quant, ent], dtype=float)


def streaming_liveness(events: Sequence[KeyEvent], config: "LivenessConfig") -> np.ndarray:
    """Per-event liveness features over a rolling raw-timing buffer.

    Returns an (n_events, 4) array; row k uses events[0:k+1].  The detector
    samples the row at each window boundary.
    """
    n = len(events)
    out = np.zeros((n, len(REPLAY_FEATURE_NAMES)))
    dwell = np.array([e.dwell_ms for e in events], dtype=float)
    gap = np.zeros(n)
    for k in range(1, n):
        gap[k] = events[k].down_ms - events[k - 1].down_ms
    stride = config.buffer_stride
    last = np.zeros(len(REPLAY_FEATURE_NAMES))
    for k in range(n):
        if k < 6 or (k % stride) not in (0,):
            out[k] = last
            continue
        lo = max(0, k + 1 - config.buffer_len)
        last = liveness_features(None, dwell[lo:k + 1], gap[lo:k + 1], 0, k + 1 - lo)  # type: ignore[arg-type]
        out[k] = last
    return out


@dataclass
class LivenessConfig:
    buffer_len: int = 900      # raw keys kept for the periodicity test
    buffer_stride: int = 25    # recompute every N keys
    loop_threshold: float = 0.9
    exact_rep_threshold: float = 0.02


def _logit(p: float) -> float:
    p = min(max(p, 0.01), 0.99)
    return float(np.log(p / (1.0 - p)))


def _fisher(r: float) -> float:
    r = min(max(r, -0.97), 0.97)
    return float(np.arctanh(r))


def _mad(x: np.ndarray) -> float:
    if len(x) == 0:
        return 0.0
    m = np.median(x)
    return float(1.4826 * np.median(np.abs(x - m)))


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 4:
        return 0.0
    sa, sb = np.std(a), np.std(b)
    if sa < EPS or sb < EPS:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _finger_contrasts(res: Residuals, start: int, stop: int) -> tuple[np.ndarray, float]:
    """Relative per-finger dwell, and the same-finger interval penalty.

    Both are contrasts against the window's own overall level, so they are
    invariant to anything that scales all dwells together -- a different
    keyboard, fatigue, a cold room.  What survives is the *ratio* between, say,
    the ring finger and the index finger, which is motoric and personal.
    """
    evs = res.events[start:stop]
    n = len(evs)
    out = np.zeros(len(_FINGER_GROUPS))
    if n < 8:
        return out, 0.0
    log_dwell = np.log(np.maximum(1.0, np.array([e.dwell_ms for e in evs], dtype=float)))
    centre = np.median(log_dwell)
    group_of = np.array([_finger_group_index(e) for e in evs])
    counts = np.zeros(len(_FINGER_GROUPS))
    for g in range(len(_FINGER_GROUPS)):
        sel = group_of == g
        counts[g] = sel.sum()
        if counts[g] >= 3:
            # shrink small groups toward 0 so a rare finger cannot dominate
            shrink = counts[g] / (counts[g] + 6.0)
            out[g] = shrink * (float(np.median(log_dwell[sel])) - centre)
        else:
            out[g] = 0.0

    # same-finger digraph penalty minus the window's ordinary interval level
    zi = res.zi[start:stop][res.zi_valid[start:stop]]
    sf = []
    for k in range(1, n):
        a_ch, b_ch = evs[k - 1].key, evs[k].key
        f1, f2 = layout.finger_of(a_ch), layout.finger_of(b_ch)
        if f1 is not None and f1 == f2 and a_ch != b_ch and res.zi_valid[start + k]:
            sf.append(res.zi[start + k])
    if len(sf) >= 3 and len(zi) >= 5:
        shrink = len(sf) / (len(sf) + 6.0)
        sf_pen = shrink * (float(np.median(sf)) - float(np.median(zi)))
    else:
        sf_pen = 0.0
    return out, float(sf_pen)


def window_features(res: Residuals, start: int, stop: int) -> np.ndarray:
    """Robust feature vector for events[start:stop]."""
    zd = res.zd[start:stop]
    v = res.zi_valid[start:stop]
    zi = res.zi[start:stop][v]
    ov = res.overlap[start:stop]
    pa = res.pause[start:stop]
    downs = res.downs[start:stop]
    ups = res.ups[start:stop]

    gaps = np.diff(downs) if len(downs) > 1 else np.array([0.0])
    crash = (gaps < 45.0) & (gaps > 0)

    zd_med = float(np.median(zd)) if len(zd) else 0.0
    zi_med = float(np.median(zi)) if len(zi) else 0.0
    zd_spread = float(np.log(max(1e-3, _mad(zd))))
    zi_spread = float(np.log(max(1e-3, _mad(zi))))
    iki_ac1 = _safe_corr(zi[:-1], zi[1:]) if len(zi) > 3 else 0.0

    # pair dwell and interval on the same index where both are valid
    pair = np.zeros(len(zd), dtype=bool)
    pair[1:] = res.zi_valid[start + 1 : stop]
    if pair.sum() >= 4:
        corr = _safe_corr(zd[pair], res.zi[start:stop][pair])
    else:
        corr = 0.0

    o_rate = _logit(float(np.mean(ov))) if len(ov) else 0.0
    p_rate = _logit(float(np.mean(pa))) if len(pa) else 0.0
    c_rate = _logit(float(np.mean(crash))) if len(crash) else 0.0
    zd_p90 = float(np.percentile(zd, 90)) if len(zd) else 0.0
    zi_p90 = float(np.percentile(zi, 90)) if len(zi) else 0.0
    zi_p10 = float(np.percentile(zi, 10)) if len(zi) else 0.0
    iqr = np.percentile(zd, 75) - np.percentile(zd, 25) if len(zd) else 0.0
    iqr_ratio = float(np.log(max(1e-3, iqr) / max(1e-3, 2.0 * _mad(zd))))

    fp, sf_pen = _finger_contrasts(res, start, stop)

    return np.array([
        zd_med, zi_med, zd_spread, zi_spread,
        _fisher(iki_ac1), _fisher(corr),
        o_rate, p_rate, c_rate,
        zd_p90, zi_p90, zi_p10, iqr_ratio,
        fp[0], fp[1], fp[2], fp[3], sf_pen,
    ], dtype=float)


# ---------------------------------------------------------------------------
# replay / liveness features (computed on normalized timing streams)
# ---------------------------------------------------------------------------


def _norm_stream(res: Residuals) -> np.ndarray:
    """2 x n matrix of standardized log-dwell and log-gap, robustly centred."""
    n = len(res)
    s = np.zeros((2, n))
    s[0] = res.zd
    if n > 1:
        s[1, 1:] = res.zi[1:]
    for r in range(2):
        row = s[r]
        if len(row):
            s[r] = (row - np.median(row)) / max(EPS, _mad(row))
    return s


