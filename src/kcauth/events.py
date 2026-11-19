"""Wire format for keystroke events.

The terminal agent (native hook) emits one record per key transition.  Times are
integer milliseconds on a monotonic clock; only *differences* are ever used, so
the epoch is irrelevant and no wall-clock privacy is leaked.

    {"key": "a", "down": 1712345678123, "up": 1712345678218, "meta": {...}}

`meta` is optional and reserved for things a native hook can see and an
out-of-process replayer usually cannot: hardware scan code, IME composition
state, key-repeat flags.  The scoring core never requires it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable


@dataclass(frozen=True)
class KeyEvent:
    key: str
    down_ms: int
    up_ms: int
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def dwell_ms(self) -> int:
        return self.up_ms - self.down_ms

    def as_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"key": self.key, "down": self.down_ms, "up": self.up_ms}
        if self.meta:
            d["meta"] = self.meta
        return d


def parse_events(raw: Iterable[dict[str, Any]]) -> list[KeyEvent]:
    """Parse and validate a batch of wire records.

    Raises ValueError on anything a timing model cannot use.  Deliberately
    strict: silently dropping malformed events is how replay layers hide.
    """
    out: list[KeyEvent] = []
    for i, rec in enumerate(raw):
        try:
            key = rec["key"]
            down = int(rec["down"])
            up = int(rec["up"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"event[{i}] malformed: {exc!r}") from exc
        if not isinstance(key, str) or len(key) != 1:
            raise ValueError(f"event[{i}]: key must be a single character, got {key!r}")
        if up < down:
            raise ValueError(f"event[{i}]: up < down ({up} < {down})")
        if up - down > 2000:
            raise ValueError(f"event[{i}]: implausible dwell {up - down} ms")
        out.append(KeyEvent(key=key, down_ms=down, up_ms=up, meta=dict(rec.get("meta") or {})))
    if not out:
        raise ValueError("empty event batch")
    return out


@dataclass
class Session:
    session_id: str
    user_id: str
    events: list[KeyEvent] = field(default_factory=list)

    def extend(self, events: Iterable[KeyEvent]) -> None:
        self.events.extend(events)

    def __len__(self) -> int:
        return len(self.events)
