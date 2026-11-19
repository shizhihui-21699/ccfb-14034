"""QWERTY layout model: character -> (hand, finger, key class).

Only the information a timing model actually needs is encoded here: which finger
produced the keystroke (motor cost) and which character class it belongs to
(content cost).  Everything is keyed by a canonical lowercase character.
"""

from __future__ import annotations

from dataclasses import dataclass

# finger ids: 0=L pinky, 1=L ring, 2=L middle, 3=L index, 4=thumb,
#             5=R index, 6=R middle, 7=R ring, 8=R pinky
L_PINKY, L_RING, L_MID, L_INDEX, THUMB, R_INDEX, R_MID, R_RING, R_PINKY = range(9)

HAND_OF_FINGER = {
    L_PINKY: "L",
    L_RING: "L",
    L_MID: "L",
    L_INDEX: "L",
    THUMB: "L",
    R_INDEX: "R",
    R_MID: "R",
    R_RING: "R",
    R_PINKY: "R",
}

_FINGER_GROUPS: dict[int, str] = {
    L_PINKY: "pinky",
    R_PINKY: "pinky",
    L_RING: "ring",
    R_RING: "ring",
    L_MID: "mid",
    R_MID: "mid",
    L_INDEX: "index",
    R_INDEX: "index",
    THUMB: "thumb",
}

_CLASSES = ("letter", "symbol", "digit", "space", "enter", "backspace")

_CHAR_TO_FINGER: dict[str, int] = {}

def _assign(chars: str, finger: int) -> None:
    for ch in chars:
        _CHAR_TO_FINGER[ch] = finger

# left hand rows
_assign("1qaz", L_PINKY)
_assign("2wsx", L_RING)
_assign("3edc", L_MID)
_assign("45rtfgvb", L_INDEX)
# right hand rows
_assign("67yuhjnm", R_INDEX)
_assign("8ik,", R_MID)
_assign("9ol.", R_RING)
_assign("0-=p[]\\;'/", R_PINKY)
# shifted symbols map through their base key
for ch in "!@#$%^&*()_+{}|:\"<>?":
    base = "1234567890-=[]\\;',./"[ "!@#$%^&*()_+{}|:\"<>?".index(ch) ]
    _CHAR_TO_FINGER[ch] = _CHAR_TO_FINGER[base]


def char_class(ch: str) -> str:
    """Coarse content class used for content-conditioned residuals."""
    if ch == " ":
        return "space"
    if ch in "\n\r":
        return "enter"
    if ch in "\b\x08":
        return "backspace"
    if ch.isdigit():
        return "digit"
    if ch.isalpha():
        return "letter"
    return "symbol"


def finger_of(ch: str) -> int | None:
    if ch == " ":
        return THUMB
    return _CHAR_TO_FINGER.get(ch)


def finger_group(finger: int | None) -> str:
    if finger is None:
        return "other"
    return _FINGER_GROUPS.get(finger, "other")


@dataclass(frozen=True)
class DigraphKind:
    same_key: bool
    same_finger: bool
    same_hand: bool
    hand_alt: bool


def digraph_kind(a: str, b: str) -> DigraphKind:
    fa, fb = finger_of(a), finger_of(b)
    if fa is None or fb is None:
        return DigraphKind(False, False, False, False)
    same_key = a == b
    same_finger = (fa == fb) and not same_key
    same_hand = HAND_OF_FINGER[fa] == HAND_OF_FINGER[fb]
    return DigraphKind(
        same_key=same_key,
        same_finger=same_finger,
        same_hand=same_hand,
        hand_alt=not same_hand,
    )


# ---------------------------------------------------------------------------
# reference population timing tables (ms).  These are *medians under the
# population prior*; the deployed population model is re-estimated from the
# enrollment corpus (see features.PopulationProfile).
# ---------------------------------------------------------------------------

BASE_DWELL_MS = {
    "letter": 95.0,
    "symbol": 108.0,
    "digit": 102.0,
    "space": 76.0,
    "enter": 92.0,
    "backspace": 86.0,
}

FINGER_DWELL_MULT = {
    L_PINKY: 1.13,
    L_RING: 1.07,
    L_MID: 1.01,
    L_INDEX: 0.96,
    THUMB: 0.90,
    R_INDEX: 0.96,
    R_MID: 1.01,
    R_RING: 1.07,
    R_PINKY: 1.13,
}

BASE_PPI_MS = 185.0
DIGRAPH_PPI_MULT = {
    "same_key": 1.32,
    "same_finger": 1.24,
    "same_hand": 1.06,
    "hand_alt": 0.93,
    "after_space": 0.96,
    "to_space": 1.04,
    "after_enter": 1.12,
    "unknown": 1.10,
}


def digraph_mult(a: str, b: str) -> float:
    """Shared (population) multiplier for a bigram.  Used by the synth only;
    the feature layer learns its own table from the corpus."""
    fa, fb = finger_of(a), finger_of(b)
    if fa is None or fb is None:
        return DIGRAPH_PPI_MULT["unknown"]
    m = 1.0
    if a == b:
        return DIGRAPH_PPI_MULT["same_key"]
    if fa == fb:
        return DIGRAPH_PPI_MULT["same_finger"]
    if HAND_OF_FINGER[fa] != HAND_OF_FINGER[fb]:
        m *= DIGRAPH_PPI_MULT["hand_alt"]
    else:
        m *= DIGRAPH_PPI_MULT["same_hand"]
    if a == " ":
        m *= DIGRAPH_PPI_MULT["after_space"]
    if b == " ":
        m *= DIGRAPH_PPI_MULT["to_space"]
    if a in "\n\r":
        m *= DIGRAPH_PPI_MULT["after_enter"]
    return m
