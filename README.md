# kcauth — continuous keystroke-rhythm authentication

Scores, continuously while the trader works, whether the person typing is the
enrolled trader — from key timing only, never key content — and tells the
desktop agent to challenge or lock when it is not.

**Read [REPORT.md](REPORT.md) for the measured results** (detection latency,
false-lock rate, what beats it and what doesn't) and the deployment
recommendations. Short version:

- strangers at an already-logged-in desk are caught in **~40 s**, 6 of 8 times,
  at a **6–8%** session false-lock rate that is **0%** on a normal day;
- an attacker who has studied recordings is caught only ~25% of the time, and
  the system challenges rather than locks — that is a real limit, stated up
  front;
- scripted replay is caught by a periodicity test when unjittered, and needs
  session binding to be caught otherwise.

```bash
PYTHONPATH=src python -m kcauth.cli evaluate
PYTHONPATH=src python -m pytest tests -q
```

Runtime needs numpy only. Python 3.11+.
