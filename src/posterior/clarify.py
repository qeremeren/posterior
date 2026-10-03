"""Ask the user a question only when the readings of their question would change the answer.

Two readings are compared by the labels they produce at the same past anchors. If they pick the
same entities with the same labels, the difference does not matter and nobody is asked.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from .spec import Spec, run_labels

SLOT_ORDER = ("entity", "event", "op", "value", "filter", "population", "horizon_days")


@dataclass
class Clarification:
    question: str
    options: list[str]  # one plain-language description per reading
    readings: list[int]  # indexes into the formulation's readings
    slot: str
    agreement: float

    def to_dict(self) -> dict[str, Any]:
        return {"question": self.question, "options": self.options, "readings": self.readings,
                "slot": self.slot, "agreement": round(self.agreement, 4)}


def agreement(con: Any, a: Spec, b: Spec, anchors: list[datetime]) -> float:
    """1.0 when both readings label the same entities the same way, 0.0 when they share nothing."""
    if a.entity != b.entity or a.task_type != b.task_type:
        return 0.0
    la = run_labels(con, a.label_sql(), anchors, a.horizon_days)
    lb = run_labels(con, b.label_sql(), anchors, b.horizon_days)
    key = ["timestamp", a.entity_key]
    m = la.merge(lb, on=key, how="outer", suffixes=("_a", "_b"), indicator=True)
    if m.empty:
        return 1.0
    both = m[m["_merge"] == "both"]
    coverage = len(both) / len(m)
    if both.empty:
        return 0.0
    ya, yb = both["target_a"].astype(float), both["target_b"].astype(float)
    if a.task_type == "binary_classification":
        same = float((ya == yb).mean())
    else:
        same = float(pd.Series(ya).corr(pd.Series(yb), method="spearman")) if ya.nunique() > 1 and yb.nunique() > 1 else float(np.allclose(ya, yb))
        same = max(0.0, same)
    return coverage * same


def first_difference(a: Spec, b: Spec) -> str:
    for slot in SLOT_ORDER:
        if getattr(a, slot) != getattr(b, slot):
            return slot
    return "op"


def clarify(con: Any, readings: list[Spec], anchors: list[datetime], threshold: float = 0.85,
            min_share: float = 0.4, max_options: int = 3) -> Clarification | None:
    """Return one multiple-choice question, or None when the plausible readings agree."""
    if len(readings) < 2:
        return None
    top = readings[0]
    rivals = []
    for i, r in enumerate(readings[1:], start=1):
        if r.prob < min_share * top.prob:
            continue
        agr = agreement(con, top, r, anchors)
        if agr < threshold:
            rivals.append((i, r, agr))
    if not rivals:
        return None
    rivals = rivals[: max_options - 1]
    slot = first_difference(top, rivals[0][1])
    names = {"entity": "what to predict for", "event": "which activity to look at",
             "op": "what outcome you mean", "value": "which amount to use", "filter": "which kind of event counts",
             "population": "who should get a prediction", "horizon_days": "how far ahead to look"}
    return Clarification(
        question=f"Your question can be read in more than one way, and the readings give different answers. "
                 f"Which one did you mean ({names[slot]})?",
        options=[top.describe()] + [r.describe() for _, r, _ in rivals],
        readings=[0] + [i for i, _, _ in rivals],
        slot=slot,
        agreement=min(a for _, _, a in rivals),
    )
