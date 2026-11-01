"""Word-level diff between two FOMC statements: a readable redline and a list of changed blocks.

The first statement is diffed against "" (everything is an insert).
"""

from difflib import SequenceMatcher


def _opcodes(prev: str, curr: str):
    a, b = prev.split(), curr.split()
    # autojunk=False: the heuristic treats common words ("the", "rate") as junk on long texts and garbles the diff.
    for op, i1, i2, j1, j2 in SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        yield op, " ".join(a[i1:i2]), " ".join(b[j1:j2])


def redline(prev: str, curr: str) -> str:
    parts = []
    for op, old, new in _opcodes(prev, curr):
        if op == "equal":
            parts.append(old)
        elif op == "delete":
            parts.append(f"[-{old}-]")
        elif op == "insert":
            parts.append(f"{{+{new}+}}")
        else:  # replace
            parts.append(f"[-{old}-] {{+{new}+}}")
    return " ".join(parts)


def changes(prev: str, curr: str) -> list[dict]:
    return [{"op": op, "old": old, "new": new} for op, old, new in _opcodes(prev, curr) if op != "equal"]
