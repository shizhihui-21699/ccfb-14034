"""Produce the figures used in the report.

    PYTHONPATH=src python scripts/make_figures.py
"""

from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from kcauth import corpus as C
from kcauth import evaluate as E
from kcauth import features as F
from kcauth import models
from kcauth.detector import Decision, run_detector

OUT = os.path.join(os.path.dirname(__file__), "..", "out", "figures")
os.makedirs(OUT, exist_ok=True)

INK = "#1b1b1f"
MUTED = "#6b6b76"
GRID = "#dcdce3"
ACCENT = "#b3261e"      # impostor / lock
CALM = "#1f5f8b"        # honest
GREEN = "#2e6b4f"       # detected / good


def _style(ax):
    ax.set_facecolor("white")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)


def fig_variance_components(pop):
    names = F.FEATURE_NAMES
    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(9, 6.2))
    _style(ax)
    vb = pop.v_between
    vs = pop.v_state
    vn = pop.v_noise
    ax.barh(y - 0.26, vb, height=0.25, color=CALM, label="V_between (identity)")
    ax.barh(y, vs, height=0.25, color="#c9a227", label="V_state (honest drift)")
    ax.barh(y + 0.26, vn, height=0.25, color=MUTED, alpha=0.55, label="V_noise (motor)")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{n}   ICC {i:.2f}" for n, i in zip(names, pop.icc)], fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel("variance component (log scale, scaled units)")
    ax.set_title("Honest drift (V_state) is comparable to identity (V_between) for every feature",
                 color=INK, fontsize=12.5, loc="left", pad=12)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "variance_components.png"), dpi=170)
    plt.close(fig)


def fig_score_distributions(corp, pop, cfg, enrol=2):
    users = {u: models.UserModel.enroll(u, pop, corp.enrolment[u][:enrol]) for u in corp.test_uids}
    W = cfg.evidence_window

    def scores(evs_filter):
        out = {}
        for u in corp.test_uids:
            um = users[u]
            for s in evs_filter(u):
                Z = pop.scaler.transform(pop.feature_matrix(s.events))
                v = um.score_rolling(Z, W)
                key = s.state.get("strategy") or s.state.get("kind") or (
                    "honest:" + s.mode)
                out.setdefault(key, []).extend(v.tolist())
        return out

    honest = scores(lambda u: [s for s in corp.genuine if s.actual == u])
    attacks = scores(lambda u: [s for s in corp.impostor if s.owner == u])
    replays = scores(lambda u: [s for s in corp.replay if s.owner == u])

    fig, ax = plt.subplots(figsize=(9.4, 5.0))
    _style(ax)
    all_honest = np.concatenate([np.asarray(v) for v in honest.values()])
    bins = np.linspace(-18, 18, 90)
    ax.hist(all_honest, bins=bins, density=True, color=CALM, alpha=0.55,
            label="honest (all states)", zorder=2)
    for key, col, lab in (("naive", ACCENT, "impostor: unfamiliar"),
                          ("first_order", "#d4763a", "impostor: informed")):
        if key in attacks:
            ax.hist(np.asarray(attacks[key]), bins=bins, density=True, histtype="step",
                    linewidth=2.0, color=col, label=lab, zorder=3)
    ax.axvline(cfg.lock_llr, color=INK, linestyle="--", linewidth=1.4)
    ax.annotate(f"lock level\n{cfg.lock_llr:.1f}", xy=(cfg.lock_llr, ax.get_ylim()[1] * 0.78),
                xytext=(6, 0), textcoords="offset points", fontsize=8.5, color=MUTED)
    ax.set_xlabel("identity score (LLR, rolling mean)")
    ax.set_ylabel("density")
    ax.set_title("Honest variation overlaps the informed impostor — by design, not by accident",
                 color=INK, fontsize=12.5, loc="left", pad=10)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "score_distributions.png"), dpi=170)
    plt.close(fig)


def fig_takeover_trace(corp, pop, cfg, strategy="naive", enrol=2):
    takeovers = C.build_takeovers(corp, seed=7)
    # show a REPRESENTATIVE case: the trace whose detection latency is closest
    # to the median, so the figure matches the table rather than cherry-picking
    # the easiest or hardest victim.
    cands = []
    for t in takeovers:
        if t.strategy != strategy:
            continue
        um = models.UserModel.enroll(t.owner, pop, corp.enrolment[t.owner][:enrol])
        live = F.streaming_liveness(t.events, cfg.liveness)
        st = run_detector(um, t.events, cfg, live=live, stop_on_lock=True)
        if st.first_lock_idx is not None:
            cands.append((st.first_lock_idx - t.switch_index, t, st))
    assert cands, f"no detectable {strategy} takeover to plot"
    med = float(np.median([c[0] for c in cands]))
    _, t, st = min(cands, key=lambda c: abs(c[0] - med))
    fig, ax = plt.subplots(figsize=(9.4, 4.6))
    _style(ax)
    idx = np.array([r.stop_idx for r in st.history])
    llr = np.array([r.llr for r in st.history])
    ax.plot(idx, llr, color=CALM, linewidth=1.7, zorder=3)
    ax.axhline(0, color=GRID, linewidth=1.0)
    ax.axhline(cfg.lock_llr, color=INK, linestyle="--", linewidth=1.3)
    ax.axvline(t.switch_index, color=ACCENT, linewidth=1.6, linestyle=":")
    ax.annotate("user leaves,\nimpostor sits", xy=(t.switch_index, ax.get_ylim()[1] * 0.7),
                xytext=(10, 0), textcoords="offset points", fontsize=9, color=ACCENT)
    if st.first_lock_idx is not None:
        ax.axvline(st.first_lock_idx, color=ACCENT, linewidth=1.2)
        ax.annotate(f"LOCK  +{st.first_lock_idx - t.switch_index} keys\n"
                    f"({(st.first_lock_t_ms - t.switch_t_ms)/1000:.0f}s)",
                    xy=(st.first_lock_idx, cfg.lock_llr), xytext=(-96, -48),
                    textcoords="offset points", fontsize=9, color=ACCENT,
                    arrowprops=dict(arrowstyle="-", color=ACCENT, linewidth=1.0))
    ax.set_xlabel("keystrokes into the session")
    ax.set_ylabel("identity score (LLR)")
    ax.set_title(f"Mid-session takeover ({strategy} impostor): evidence collapses after the swap",
                 color=INK, fontsize=12.5, loc="left", pad=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "takeover_trace.png"), dpi=170)
    plt.close(fig)
    return st


def fig_operating_curve(corp, pop, enrol=2):
    cfg0 = E.calibrate(corp, pop, enrol_sessions=enrol, target_session_false_lock=0.02,
                       evidence_window=16)
    rows = []
    for fp in (0.005, 0.01, 0.02, 0.04, 0.08, 0.15):
        cfg = E.calibrate(corp, pop, enrol_sessions=enrol, target_session_false_lock=fp,
                          evidence_window=16)
        r = E.evaluate(corp, pop, enrol_sessions=enrol, cfg=cfg)
        rows.append((r["honest"]["false_lock_rate"], r["honest"]["stepup_rate"],
                     r["attacks"].get("naive", {}).get("detection_rate", 0),
                     r["attacks"].get("first_order", {}).get("detection_rate", 0),
                     r["takeovers"].get("naive", {}).get("detection_rate", 0)))
    rows = np.array(rows)
    fig, ax = plt.subplots(figsize=(8.6, 4.9))
    _style(ax)
    x = rows[:, 0] * 100
    ax.plot(x, rows[:, 2] * 100, "o-", color=ACCENT, linewidth=2, markersize=7,
            label="unfamiliar impostor (fresh session)")
    ax.plot(x, rows[:, 4] * 100, "s--", color="#d4763a", linewidth=1.8, markersize=6,
            label="unfamiliar impostor (mid-session takeover)")
    ax.plot(x, rows[:, 3] * 100, "^-", color=MUTED, linewidth=1.8, markersize=6,
            label="informed impostor")
    ax.set_xlabel("false-lock rate on honest sessions (%)")
    ax.set_ylabel("detection rate (%)")
    ax.set_title("Operating curve: what each false-lock budget buys",
                 color=INK, fontsize=12.5, loc="left", pad=10)
    ax.legend(frameon=False, fontsize=9)
    ax.set_ylim(-4, 104)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "operating_curve.png"), dpi=170)
    plt.close(fig)
    return rows


def fig_liveness():
    """Replay detection: loop score for live vs scripted streams."""
    from kcauth import synth

    rng = np.random.default_rng(2)
    u = synth.sample_traits(rng, "demo")
    st = synth.make_state(rng, "s", "code", fatigue=0.3)
    cfg = F.LivenessConfig()
    live_scores, raw_scores = [], []
    for _ in range(12):
        ev = synth.synth_session(u, st, synth.gen_text(rng, "code", 950), rng)
        live_scores.append(F.streaming_liveness(ev, cfg)[:, 0].max())
        rec = synth.record_raw_timing(u, st, synth.gen_text(rng, "code", 240), rng)
        ev2 = synth.synth_raw_replay(synth.gen_text(rng, "code", 900), rec, rng, 5_000_000)
        raw_scores.append(F.streaming_liveness(ev2, cfg)[:, 0].max())

    fig, ax = plt.subplots(figsize=(8.2, 3.9))
    _style(ax)
    ax.scatter(np.zeros(len(live_scores)) + np.random.default_rng(1).normal(0, 0.035, len(live_scores)),
               live_scores, s=64, color=CALM, alpha=0.75, label="live typing", zorder=3)
    ax.scatter(np.ones(len(raw_scores)) + np.random.default_rng(2).normal(0, 0.035, len(raw_scores)),
               raw_scores, s=64, color=ACCENT, alpha=0.8, label="scripted replay", zorder=3)
    ax.axhline(cfg.loop_threshold, color=INK, linestyle="--", linewidth=1.3)
    ax.annotate("loop threshold", xy=(1.42, cfg.loop_threshold), xytext=(0, 5),
                textcoords="offset points", fontsize=8.5, color=MUTED, ha="right")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["live typing", "scripted replay"])
    ax.set_ylabel("best loop-periodicity score")
    ax.set_ylim(-0.03, 1.05)
    ax.set_xlim(-0.5, 1.5)
    ax.set_title("Replay is caught by periodicity, not by identity",
                 color=INK, fontsize=12.5, loc="left", pad=10)
    ax.legend(frameon=False, fontsize=9, loc="center right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "liveness.png"), dpi=170)
    plt.close(fig)


def main() -> int:
    corp = C.build_corpus()
    pop = models.PopulationModel.fit(
        {u: [ev for (uu, ev) in corp.population_events if uu == u] for u in corp.train_uids}
    )
    cfg = E.calibrate(corp, pop, enrol_sessions=2, target_session_false_lock=0.02,
                      evidence_window=16)
    fig_variance_components(pop)
    fig_score_distributions(corp, pop, cfg)
    fig_takeover_trace(corp, pop, cfg)
    rows = fig_operating_curve(corp, pop)
    fig_liveness()
    print(f"figures written to {os.path.abspath(OUT)}")
    for f in sorted(os.listdir(OUT)):
        print("  ", f)
    return 0


if __name__ == "__main__":
    sys.exit(main())
