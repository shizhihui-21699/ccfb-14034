"""Synthetic keystroke corpus.

Why synthesise?  Real keystroke-timing corpora are (a) small, (b) not from a
trading desk, and (c) cannot contain the adversarial cases we must report on
(an informed impersonator, a scripted replay).  So we build a generative model
that encodes what the literature agrees on, and we make its *hard cases* hard on
purpose:

  honest variation
    - per-user latent motor traits (dwell, inter-key interval, finger bias,
      burstiness, key-overlap, correction habits)
    - state drift: fatigue over a session, time of day, a *different keyboard*,
      and the three input modes of the job (English code, Chinese IME, numeric)

  adversaries
    - naive impostor: a different person
    - informed impostor: has watched recordings, matches every first-order
      statistic (means, variances, per-class dwell, per-bigram-kind interval)
    - adaptive impostor: also partially reproduces second-order texture
      (burstiness, dwell/interval coupling, key overlap)
    - script replay: the recorded timing stream re-emitted, optionally jittered

Ground truth (user id, state, attacker strategy, switch point) is carried in the
metadata so the evaluation harness can compute detection latency and false-lock
rates without any guessing.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Iterable, Sequence

import numpy as np

from . import layout
from .events import KeyEvent

# ---------------------------------------------------------------------------
# latent user traits
# ---------------------------------------------------------------------------


@dataclass
class UserTraits:
    uid: str
    # global motor scales (log space offsets, 1.0 == population median)
    dwell_scale: float = 0.0
    iki_scale: float = 0.0
    # within-user variability
    dwell_sigma: float = 0.24
    iki_sigma: float = 0.30
    # temporal texture
    ar1: float = 0.25           # autocorrelation of motor interval residuals
    rho_di: float = 0.0         # dwell <-> interval coupling
    overlap: float = 0.18       # P(next key goes down before this one comes up)
    # content-conditioned behaviours
    alt_bias: float = 0.0       # extra speed on hand alternation (log)
    samekey_bias: float = 0.0   # extra slowness on repeated key (log)
    finger_bias: np.ndarray = field(default_factory=lambda: np.zeros(9))
    pause_mult: float = 1.0     # how often this person stops to think
    error_rate: float = 0.03    # typos per character
    ime_speed: float = 1.0      # how fast they drive the IME candidate list
    commit_style: float = 0.5   # 0=always digits, 1=always space to commit


def sample_traits(rng: np.random.Generator, uid: str) -> UserTraits:
    """Population prior.  Dispersions are chosen so that a keyboard swap or a
    fatigued afternoon is *within* the honest range: that is exactly what makes
    the false-lock problem real.

    `finger_bias` is drawn with meaningful spread and is a *relative* pattern
    (zero-mean across fingers).  That is deliberate: it is the one trait that a
    different keyboard cannot mimic (a board scales all fingers together) and
    that a first-order impersonator who only matched marginal dwell/interval
    statistics will not reproduce.  Real motor variability between fingers is
    large, and stable within a person (cf. neural and biomechanical models of
    typing), so this is a defensible place to put identity.
    """
    finger = rng.normal(0.0, 0.085, size=9)
    finger -= finger.mean()
    return UserTraits(
        uid=uid,
        dwell_scale=rng.normal(0.0, 0.13),
        iki_scale=rng.normal(0.0, 0.10),
        dwell_sigma=float(np.clip(rng.normal(0.24, 0.035), 0.13, 0.40)),
        iki_sigma=float(np.clip(rng.normal(0.30, 0.05), 0.16, 0.48)),
        ar1=float(np.clip(rng.normal(0.25, 0.09), -0.05, 0.60)),
        rho_di=float(np.clip(rng.normal(0.0, 0.16), -0.45, 0.45)),
        overlap=float(np.clip(rng.normal(0.18, 0.09), 0.01, 0.55)),
        alt_bias=float(rng.normal(0.0, 0.05)),
        samekey_bias=float(rng.normal(0.0, 0.05)),
        finger_bias=finger,
        pause_mult=float(np.clip(rng.normal(1.0, 0.25), 0.4, 2.0)),
        error_rate=float(np.clip(rng.normal(0.03, 0.015), 0.005, 0.09)),
        ime_speed=float(np.clip(rng.normal(1.0, 0.18), 0.6, 1.6)),
        commit_style=float(np.clip(rng.normal(0.5, 0.22), 0.0, 1.0)),
    )


@dataclass
class State:
    """Everything outside the person that moves their timing."""

    name: str
    fatigue: float = 0.0        # 0..1
    keyboard: str = "kb-0"
    log_dwell_mult: float = 0.0
    log_iki_mult: float = 0.0
    dwell_sigma_mult: float = 1.0
    iki_sigma_mult: float = 1.0
    ar1_mult: float = 1.0
    overlap_mult: float = 1.0
    pause_mult: float = 1.0
    error_mult: float = 1.0
    mode: str = "code"


def make_state(
    rng: np.random.Generator,
    name: str,
    mode: str,
    fatigue: float | None = None,
    keyboard_swap: bool = False,
    keyboard_id: int = 0,
) -> State:
    if fatigue is None:
        fatigue = float(np.clip(rng.beta(2.0, 3.0), 0.0, 1.0))
    # fatigue: slower, more variable, more typos, fewer long bursts
    log_d = 0.20 * fatigue
    log_i = 0.16 * fatigue
    dsm = 1.0 + 0.35 * fatigue
    ism = 1.0 + 0.40 * fatigue
    ar1m = 1.0 - 0.25 * fatigue
    ovm = 1.0 - 0.35 * fatigue
    pm = 1.0 + 0.45 * fatigue
    em = 1.0 + 2.5 * fatigue
    # time of day / thermal drift, independent of fatigue
    td = rng.normal(0.0, 0.03)
    log_d += td
    log_i += 0.8 * td
    if keyboard_swap:
        # a new board changes travel and actuation force: dwell moves a lot,
        # interval much less, and consistency changes.
        log_d += rng.normal(0.0, 0.16)
        dsm *= float(np.clip(rng.normal(1.0, 0.22), 0.55, 1.8))
        log_i += rng.normal(0.0, 0.04)
        ism *= float(np.clip(rng.normal(1.0, 0.12), 0.7, 1.5))
        ovm *= float(np.clip(rng.normal(1.0, 0.18), 0.6, 1.5))
    if mode == "chinese_ime":
        pm *= 1.25          # candidate-list scanning and pinyin thinking
        log_i += 0.04
    elif mode == "numeric":
        pm *= 0.8
        log_d -= 0.02
    return State(
        name=name,
        fatigue=fatigue,
        keyboard=f"kb-{keyboard_id}",
        log_dwell_mult=log_d,
        log_iki_mult=log_i,
        dwell_sigma_mult=dsm,
        iki_sigma_mult=ism,
        ar1_mult=ar1m,
        overlap_mult=ovm,
        pause_mult=pm,
        error_mult=em,
        mode=mode,
    )


# ---------------------------------------------------------------------------
# text sources
# ---------------------------------------------------------------------------

_CODE_LINES = [
    "def calc_pnl(price, qty, fee=0.0003):",
    "    gross = price * qty",
    "    return gross - fee * gross",
    "order = book.best_bid(qty=100)",
    "if order.filled and order.px > limit:",
    "    log.info('fill %s @ %s', order.id, order.px)",
    "for sym in universe:",
    "    pos[sym] = np.zeros(len(tenors))",
    "risk = sum(abs(v) for v in pos.values())",
    "while not done:",
    "    tick = feed.next()",
    "    assert tick.px > 0, tick",
    "class Walker(Strategy):",
    "    def on_quote(self, q): pass",
    "px = (bid + ask) / 2.0",
    "vol = np.sqrt(np.sum(w * w * cov))",
    "cache[key] = (ts, val)",
    "return {'ok': True, 'n': n}",
]

_PROSE_WORDS = (
    "the quick brown fox jumps over lazy dog market maker spread hedge "
    "position delta gamma vega risk limit order fill cancel amend quote "
    "venue broker client ticket notional exposure margin settlement"
).split()

_PINYIN = (
    "wo ni ta men zhe ge mai ru chu ting pan jia ge shu liang wei tuo "
    "cheng jiao che dan feng xian tou cun bao zheng jin zhang hu jie suan "
    "qing suan dui chong shou yi ce lue xin hao can shu mo xing hui ce "
    "shi chang liu dong xing bo dong lv qi quan qi huo xian huo wai hui"
).split()


@dataclass
class Token:
    ch: str
    # 0 = ordinary motor boundary, 1 = word boundary, 2 = clause boundary
    boundary: int = 0
    # IME bookkeeping (used only to shape timing)
    ime_pause_before: float = 1.0


def gen_code_text(rng: np.random.Generator, n_chars: int) -> list[Token]:
    toks: list[Token] = []
    while len(toks) < n_chars:
        line = _CODE_LINES[rng.integers(len(_CODE_LINES))]
        if rng.random() < 0.25:  # sometimes write prose comments
            words = [ _PROSE_WORDS[i] for i in rng.integers(0, len(_PROSE_WORDS), size=6) ]
            line = "# " + " ".join(words)
        for i, ch in enumerate(line):
            b = 0
            if ch == " ":
                b = 1
            elif ch in "()[]{}:,.":
                b = 2
            toks.append(Token(ch, b))
        toks.append(Token("\n", 2))
    return toks[:n_chars]


def gen_ime_text(rng: np.random.Generator, n_chars: int) -> list[Token]:
    """A plausible Chinese-IME stream: pinyin letters, then a commit (space or
    digit), occasional candidate scrolling, occasional typo+backspace."""
    toks: list[Token] = []
    while len(toks) < n_chars:
        syl = _PINYIN[rng.integers(len(_PINYIN))]
        if rng.random() < 0.45:
            syl = syl + _PINYIN[rng.integers(len(_PINYIN))]
        pause = float(np.clip(rng.normal(1.0, 0.5), 0.3, 2.5))
        for i, ch in enumerate(syl):
            if i == 0:
                toks.append(Token(ch, 1, ime_pause_before=pause))
            else:
                toks.append(Token(ch, 0))
        if rng.random() < 0.10:  # wrong pinyin, backspace twice
            for _ in range(2):
                toks.append(Token("\b", 0))
            for i in range(min(3, len(syl))):
                toks.append(Token(syl[i], 0))
        r = rng.random()
        if r < 0.55:
            toks.append(Token(" ", 2))           # commit with space
        elif r < 0.9:
            toks.append(Token(str(int(rng.integers(1, 5))), 2))  # pick candidate
        else:
            toks.append(Token("\n", 2))
    return toks[:n_chars]


def gen_numeric_text(rng: np.random.Generator, n_chars: int) -> list[Token]:
    toks: list[Token] = []
    while len(toks) < n_chars:
        r = rng.random()
        if r < 0.55:
            n = int(rng.integers(1, 6))
            s = "".join(str(int(d)) for d in rng.integers(0, 10, size=n))
            if rng.random() < 0.4:
                s = s[:-1] + "." + s[-1:]
            for i, ch in enumerate(s):
                toks.append(Token(ch, 0))
            toks.append(Token("\n", 2))
        else:
            for _ in range(int(rng.integers(3, 8))):
                toks.append(Token(str(int(rng.integers(0, 10))), 0))
            toks.append(Token(",", 1))
    return toks[:n_chars]


_TEXT_SOURCES = {
    "code": gen_code_text,
    "prose": gen_code_text,
    "chinese_ime": gen_ime_text,
    "numeric": gen_numeric_text,
}


def gen_text(rng: np.random.Generator, mode: str, n_chars: int) -> list[Token]:
    return _TEXT_SOURCES[mode](rng, n_chars)


# ---------------------------------------------------------------------------
# timing synthesis
# ---------------------------------------------------------------------------

_PAUSE_PROB = {1: 0.07, 2: 0.11, 0: 0.004}  # per boundary class
# Thinking pauses: roughly lognormal around 1.3 s, median ~1.3 s, 99th
# percentile ~8 s.  An earlier draft used mu=6.3 (median ~550 s) which made a
# 700-key session last 1.5 hours and turned latency reporting into nonsense.
_PAUSE_MU = 7.15
_PAUSE_SIGMA = 0.75


def _pause_prob(boundary: int, mode: str, prev: str) -> float:
    p = _PAUSE_PROB.get(boundary, 0.01)
    if mode == "chinese_ime":
        if prev in " \n" or prev.isdigit():
            p *= 1.6
    elif mode == "numeric":
        p *= 1.2
    return p


def synth_session(
    traits: UserTraits,
    state: State,
    tokens: Sequence[Token],
    rng: np.random.Generator,
    t0_ms: int = 1_000_000,
    *,
    timing_stream: np.ndarray | None = None,
    meta_extra: dict | None = None,
) -> list[KeyEvent]:
    """Render a token stream into key events.

    If `timing_stream` is given (shape [n,2]: log-dwell, log-iki residuals) the
    motor noise is taken from it instead of being sampled.  This is how a script
    replay is modelled: the *timing* comes from a recording while the *content*
    is whatever the script injects.
    """
    n = len(tokens)
    if n == 0:
        return []
    finger = np.array([layout.finger_of(t.ch) for t in tokens], dtype=object)
    cls = [layout.char_class(t.ch) for t in tokens]

    log_dwell = np.empty(n)
    log_iki = np.empty(n)
    is_pause = np.zeros(n, dtype=bool)

    eps_prev = 0.0
    for k, tok in enumerate(tokens):
        f = finger[k]
        if f is None:
            f = layout.THUMB
        base_d = layout.BASE_DWELL_MS[cls[k]] * layout.FINGER_DWELL_MULT[f]
        log_dwell[k] = (
            np.log(base_d)
            + traits.dwell_scale
            + traits.finger_bias[f]
            + state.log_dwell_mult
        )

        if k == 0:
            log_iki[k] = np.log(400.0)
        else:
            prev = tokens[k - 1].ch
            mult = layout.digraph_mult(prev, tok.ch)
            b = tok.boundary
            if b >= 1:
                mult *= 1.0 + 0.05 * b
            log_iki[k] = (
                np.log(layout.BASE_PPI_MS)
                + np.log(mult)
                + traits.iki_scale
                + state.log_iki_mult
            )
            kind = layout.digraph_kind(prev, tok.ch)
            if kind.hand_alt:
                log_iki[k] += traits.alt_bias
            if kind.same_key:
                log_iki[k] += traits.samekey_bias

    # --- motor residual process -------------------------------------------
    if timing_stream is not None:
        log_dwell += timing_stream[:, 0]
        log_iki += timing_stream[:, 1]
        pause_mask = timing_stream[:, 2] > 0.5 if timing_stream.shape[1] > 2 else np.zeros(n, bool)
        is_pause |= pause_mask
    else:
        ar1 = float(np.clip(traits.ar1 * state.ar1_mult, -0.2, 0.8))
        sd = 0.6 * traits.dwell_sigma * state.dwell_sigma_mult
        si = traits.iki_sigma * state.iki_sigma_mult
        rho = float(np.clip(traits.rho_di, -0.6, 0.6))
        eps = np.empty(n)
        for k in range(n):
            eps[k] = ar1 * eps_prev + rng.normal(0.0, si)
            eps_prev = eps[k]
        noise_d = rho * eps + np.sqrt(max(1e-6, 1.0 - rho * rho)) * rng.normal(0.0, sd, n)
        log_dwell += noise_d
        log_iki += eps

        # cognitive pauses: content-driven, heavy tailed
        for k, tok in enumerate(tokens):
            prev = tokens[k - 1].ch if k > 0 else " "
            p = _pause_prob(tok.boundary, state.mode, prev) * traits.pause_mult * state.pause_mult
            if tok.ime_pause_before != 1.0:
                p = p * tok.ime_pause_before
            if k > 0 and rng.random() < p:
                is_pause[k] = True

    # clip to physically plausible ranges before exponentiating
    np.clip(log_dwell, np.log(22.0), np.log(520.0), out=log_dwell)
    np.clip(log_iki, np.log(26.0), np.log(2600.0), out=log_iki)

    ikis = np.exp(log_iki)
    if timing_stream is None:
        for k in range(1, n):
            if is_pause[k]:
                # an absolute thinking pause, NOT a multiplier on the previous
                # interval: multiplying by exp(7.15) turned 200 ms into ~4 min
                # and made a 700-key session last hours.
                ikis[k] = float(np.exp(rng.normal(_PAUSE_MU, _PAUSE_SIGMA)))
    dwells = np.exp(log_dwell)

    # key overlap: next key goes down before this one comes up
    overlap_p = float(np.clip(traits.overlap * state.overlap_mult, 0.0, 0.75))
    t = float(t0_ms)
    downs = np.empty(n)
    ups = np.empty(n)
    rng_overlap = rng if timing_stream is None else np.random.default_rng(int(t0_ms) & 0xFFFF)
    for k in range(n):
        if k > 0:
            t += float(ikis[k])
        downs[k] = t
        d = float(dwells[k])
        if k + 1 < n and rng_overlap.random() < overlap_p:
            d = max(d, float(ikis[k + 1]) * float(rng_overlap.uniform(1.02, 1.7)))
        if k > 0 and is_pause[k]:
            d = max(d, float(dwells[k]))
        ups[k] = downs[k] + min(d, 900.0)

    # enforce monotone downs and sane dwells
    downs = np.maximum.accumulate(downs)
    ups = np.maximum(ups, downs + 18.0)

    base_meta = {
        "mode": state.mode,
        "kb": state.keyboard,
        "fatigue": round(float(state.fatigue), 3),
    }
    if meta_extra:
        base_meta.update(meta_extra)

    events: list[KeyEvent] = []
    for k, tok in enumerate(tokens):
        m = dict(base_meta)
        if is_pause[k]:
            m["pause"] = 1
        f = finger[k]
        m["finger"] = int(f) if f is not None else -1
        events.append(
            KeyEvent(
                key=tok.ch,
                down_ms=int(round(downs[k])),
                up_ms=int(round(ups[k])),
                meta=m,
            )
        )
    return events


# ---------------------------------------------------------------------------
# adversary strategies
# ---------------------------------------------------------------------------


def imitate_traits(
    victim: UserTraits,
    strategy: str,
    rng: np.random.Generator,
    population: Sequence[UserTraits],
) -> UserTraits:
    """Build the traits an attacker of a given capability would need to pass.

    first_order: copies every marginal statistic (scales, per-class dwell,
        per-bigram interval, variances) but keeps population-average temporal
        texture.  This is what you get from watching recordings without ever
        practising on the victim's actual machine.
    adaptive: also reproduces a fraction of the second-order texture.
    """
    pop_ar1 = float(np.mean([u.ar1 for u in population]))
    pop_rho = float(np.mean([u.rho_di for u in population]))
    pop_ov = float(np.mean([u.overlap for u in population]))
    pop_alt = float(np.mean([u.alt_bias for u in population]))
    pop_sk = float(np.mean([u.samekey_bias for u in population]))

    if strategy == "first_order":
        return replace(
            victim,
            ar1=pop_ar1,
            rho_di=pop_rho,
            overlap=pop_ov,
            alt_bias=pop_alt,
            samekey_bias=pop_sk,
            finger_bias=np.zeros(9),
            # residual noise of the impersonator is human, so about population
            # average; matching the victim's sigma is easy from recordings
        )
    if strategy == "adaptive":
        return replace(
            victim,
            ar1=0.6 * victim.ar1 + 0.4 * pop_ar1,
            rho_di=0.5 * victim.rho_di + 0.5 * pop_rho,
            overlap=0.8 * victim.overlap + 0.2 * pop_ov,
            alt_bias=0.6 * victim.alt_bias + 0.4 * pop_alt,
            samekey_bias=0.6 * victim.samekey_bias + 0.4 * pop_sk,
            finger_bias=0.55 * victim.finger_bias,
        )
    if strategy == "naive":
        return sample_traits(rng, victim.uid + "-impostor")
    raise ValueError(f"unknown strategy {strategy!r}")


def record_timing(
    traits: UserTraits,
    state: State,
    tokens: Sequence[Token],
    rng: np.random.Generator,
) -> np.ndarray:
    """Capture the motor residual stream of a genuine session (for replay)."""
    # Re-derive residuals by synthesising and regressing out the deterministic
    # part, so a replay carries exactly the recorded timing structure.
    n = len(tokens)
    events = synth_session(traits, state, tokens, rng, t0_ms=500_000)
    log_dwell = np.log(np.array([e.dwell_ms for e in events], dtype=float))
    iki = np.array([np.nan] + [events[k].down_ms - events[k - 1].down_ms for k in range(1, n)], dtype=float)
    stream = np.zeros((n, 3))
    for k, tok in enumerate(tokens):
        f = layout.finger_of(tok.ch) or layout.THUMB
        base = np.log(layout.BASE_DWELL_MS[layout.char_class(tok.ch)] * layout.FINGER_DWELL_MULT[f])
        stream[k, 0] = log_dwell[k] - base
        if k > 0:
            mult = layout.digraph_mult(tokens[k - 1].ch, tok.ch)
            stream[k, 1] = np.log(max(1.0, iki[k])) - np.log(layout.BASE_PPI_MS * mult)
        stream[k, 2] = 1.0 if events[k].meta.get("pause") else 0.0
    return stream


def replay_stream(recording: np.ndarray, n: int, rng: np.random.Generator, jitter: float = 0.0) -> np.ndarray:
    """Parametric replay: sample timing *statistics* from the recording.

    This is the sophisticated replayer -- it re-draws durations from a model
    fitted to the recording rather than re-emitting the bytes.  Statistically
    it is a first-order impersonator: it reproduces the marginal distributions
    but not the fine texture.  Liveness features cannot see it; only identity
    statistics can.  Reported separately for exactly that reason.
    """
    reps = int(np.ceil(n / len(recording)))
    stream = np.tile(recording, (reps, 1))[:n].copy()
    if jitter > 0:
        stream[:, 0] += rng.normal(0.0, jitter, n)
        stream[:, 1] += rng.normal(0.0, jitter, n)
    return stream


def record_raw_timing(traits: UserTraits, state: State, tokens: Sequence[Token],
                      rng: np.random.Generator) -> np.ndarray:
    """Record the raw (dwell_ms, gap_ms) pairs of a genuine session."""
    ev = synth_session(traits, state, tokens, rng, t0_ms=500_000)
    out = np.zeros((len(ev), 2))
    out[0, 1] = 400.0
    for k, e in enumerate(ev):
        out[k, 0] = e.dwell_ms
        if k > 0:
            out[k, 1] = e.down_ms - ev[k - 1].down_ms
    return out


def synth_raw_replay(tokens: Sequence[Token], recorded: np.ndarray,
                     rng: np.random.Generator, t0_ms: int,
                     *, jitter_ms: float = 0.0, meta_extra: dict | None = None) -> list[KeyEvent]:
    """A script that replays recorded raw timing, looping over the recording.

    Every duration is a value that came out of a recording; a looping script
    therefore repeats the same (dwell, gap) pair every recording-length.  That
    periodicity -- not any identity statistic -- is what gives it away.
    """
    n = len(tokens)
    L = len(recorded)
    events: list[KeyEvent] = []
    t = float(t0_ms)
    base_meta = {"mode": "code", "kb": "kb-0", "fatigue": 0.3, "replay": "raw"}
    if meta_extra:
        base_meta.update(meta_extra)
    for k, tok in enumerate(tokens):
        d, g = recorded[k % L]
        if jitter_ms > 0:
            d = d + rng.normal(0.0, jitter_ms)
            g = g + rng.normal(0.0, jitter_ms)
        if k > 0:
            t += max(20.0, float(g))
        down = int(round(t))
        up = int(round(t + max(18.0, float(d))))
        f = layout.finger_of(tok.ch)
        m = dict(base_meta)
        m["finger"] = int(f) if f is not None else -1
        events.append(KeyEvent(key=tok.ch, down_ms=down, up_ms=up, meta=m))
    return events
