"""Command-line entry points.

    python -m kcauth.cli corpus   [--seed N] [--out DIR]     build & fit, save
    python -m kcauth.cli evaluate [--enrol 2] [--fp 0.02]    full report
    python -m kcauth.cli sweep                                 operating-point table
    python -m kcauth.cli demo                                  stream a takeover live
    python -m kcauth.cli serve   [--port 8099]                 run the service
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

from . import corpus as C
from . import evaluate as E
from . import features as F
from . import models
from .detector import Decision, DetectorConfig, run_detector


def _fit(seed: int = 20261120, save: str | None = None):
    corp = C.build_corpus(seed=seed)
    pop = models.PopulationModel.fit(
        {u: [ev for (uu, ev) in corp.population_events if uu == u] for u in corp.train_uids}
    )
    if save:
        os.makedirs(save, exist_ok=True)
        with open(os.path.join(save, "population.json"), "w") as fh:
            json.dump(pop.to_dict(), fh)
    return corp, pop


def cmd_corpus(args) -> None:
    corp, pop = _fit(args.seed, args.out)
    print(f"population fit on {pop.n_users} users / {pop.n_sessions} sessions")
    print(f"  users: {len(corp.train_uids)} train, {len(corp.test_uids)} test")
    print(f"  genuine sessions {len(corp.genuine)}  impostor {len(corp.impostor)}  replay {len(corp.replay)}")
    if args.out:
        print(f"  saved to {args.out}")


def cmd_evaluate(args) -> None:
    corp, pop = _fit(args.seed)
    cfg = E.calibrate(corp, pop, enrol_sessions=args.enrol,
                      target_session_false_lock=args.fp,
                      evidence_window=args.window)
    print(f"operating point: target {args.fp:.0%} false-lock -> lock LLR {cfg.lock_llr:.2f}")
    res = E.evaluate(corp, pop, enrol_sessions=args.enrol, cfg=cfg)
    print(E.format_report(res))


def cmd_sweep(args) -> None:
    corp, pop = _fit(args.seed)
    print(f"{'target':>7s} {'lockLLR':>8s} {'FA':>6s} {'stepup':>7s} "
          f"{'naive':>6s} {'1stord':>7s} {'tk-naive':>9s} {'tk-1st':>7s}")
    for fp in (0.01, 0.02, 0.05, 0.10):
        cfg = E.calibrate(corp, pop, enrol_sessions=args.enrol, target_session_false_lock=fp)
        r = E.evaluate(corp, pop, enrol_sessions=args.enrol, cfg=cfg)
        a = r["attacks"]
        t = r["takeovers"]
        print(f"{fp:7.0%} {cfg.lock_llr:8.2f} {r['honest']['false_lock_rate']:6.1%} "
              f"{r['honest']['stepup_rate']:7.1%} "
              f"{a.get('naive',{}).get('detection_rate',0):6.0%} "
              f"{a.get('first_order',{}).get('detection_rate',0):7.0%} "
              f"{t.get('naive',{}).get('detection_rate',0):9.0%} "
              f"{t.get('first_order',{}).get('detection_rate',0):7.0%}")


def cmd_demo(args) -> None:
    corp, pop = _fit(args.seed)
    cfg = E.calibrate(corp, pop, enrol_sessions=2, target_session_false_lock=0.02)
    takeovers = C.build_takeovers(corp, seed=7)
    t = next(x for x in takeovers if x.strategy == args.strategy)
    um = models.UserModel.enroll(t.owner, pop, corp.enrolment[t.owner][:2])
    live = F.streaming_liveness(t.events, cfg.liveness)
    st = run_detector(um, t.events, cfg, live=live, stop_on_lock=True)
    print(f"victim {t.owner}, attacker continues the session after "
          f"{t.switch_index} keys (strategy: {t.strategy})")
    print(f"lock threshold LLR {cfg.lock_llr:.2f}, evidence window {cfg.evidence_window} windows")
    print(f"{'key#':>6s} {'score':>8s} {'decision':>12s}")
    for r in st.history:
        if r.stop_idx >= t.switch_index - 60:
            mark = "  <-- takeover" if r.stop_idx - pop.cfg.stride <= t.switch_index <= r.stop_idx else ""
            print(f"{r.stop_idx:6d} {r.llr:8.2f} {r.decision.value:>12s}{mark}")
    if st.first_lock_idx is not None:
        print(f"\nLOCKED after {st.first_lock_idx - t.switch_index} keys "
              f"({(st.first_lock_t_ms - t.switch_t_ms)/1000:.0f}s) of the impostor typing")
    else:
        print("\nnot detected within this session")


def cmd_serve(args) -> None:
    from .service import AuthService, serve
    pop = AuthService.load_population(args.store) if os.path.exists(
        os.path.join(args.store, "population.json")) else _fit()[1]
    svc = AuthService(pop, store_dir=args.store)
    svc.save_population()
    print(f"keystroke-auth service on http://{args.host}:{args.port}  "
          f"(users loaded: {len(svc.users)})", flush=True)
    serve(svc, args.host, args.port).serve_forever()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="kcauth")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("corpus"); p.add_argument("--seed", type=int, default=20261120)
    p.add_argument("--out"); p.set_defaults(fn=cmd_corpus)

    p = sub.add_parser("evaluate"); p.add_argument("--seed", type=int, default=20261120)
    p.add_argument("--enrol", type=int, default=2); p.add_argument("--fp", type=float, default=0.02)
    p.add_argument("--window", type=int, default=24); p.set_defaults(fn=cmd_evaluate)

    p = sub.add_parser("sweep"); p.add_argument("--seed", type=int, default=20261120)
    p.add_argument("--enrol", type=int, default=2); p.set_defaults(fn=cmd_sweep)

    p = sub.add_parser("demo"); p.add_argument("--seed", type=int, default=20261120)
    p.add_argument("--strategy", default="naive"); p.set_defaults(fn=cmd_demo)

    p = sub.add_parser("serve"); p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8099); p.add_argument("--store", default="out/store")
    p.set_defaults(fn=cmd_serve)

    args = ap.parse_args(argv)
    args.fn(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
