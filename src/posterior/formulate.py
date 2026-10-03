"""Turn a plain-language question into ranked predictive task readings.

The question is split into slots (entity, event, operation, value, filter, window, population).
Each slot is one typed question to a decision model. A small beam keeps the likely options per
slot, invalid combinations are dropped by the grammar, and joint probabilities rank the readings.
"""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass, field
from typing import Any

from .decider import Choice, Noul
from .schema import Schema
from .spec import Filter, Spec

HORIZONS = {7: "one week", 14: "two weeks", 30: "one month", 60: "two months", 90: "three months, a quarter",
            180: "half a year", 365: "one year"}
_UNITS = {"day": 1, "days": 1, "week": 7, "weeks": 7, "month": 30, "months": 30, "quarter": 91,
          "quarters": 91, "year": 365, "years": 365, "gün": 1, "hafta": 7, "ay": 30, "yıl": 365}
_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "six": 6, "a": 1, "an": 1, "next": 1, "bir": 1,
          "iki": 2, "üç": 3}


def parse_horizon(question: str) -> int | None:
    """Read an explicit window ("next 30 days", "next quarter", "3 months", "30 gün") if there is one."""
    q = question.lower()
    # English units stand alone; Turkish units take suffixes ("30 günde", "3 ayda").
    m = re.search(r"(\d+)\s*-?\s*(?:(days?|weeks?|months?|quarters?|years?)(?![a-z])|(gün|hafta|ay|yıl))", q)
    if m:
        return int(m.group(1)) * _UNITS[m.group(2) or m.group(3)]
    m = re.search(r"\b(one|two|three|four|six|a|an|next|bir|iki|üç)\s+(day|week|month|quarter|year)s?\b", q)
    if m:
        return _WORDS[m.group(1)] * _UNITS[m.group(2)]
    return None


_NUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "ten": 10}


def parse_threshold(question: str) -> float | None:
    """"more than 2 invitations" gives 2, "at least 3" gives 2 (the label is count > threshold)."""
    m = re.search(r"\b(more than|over|above|at least)\s+(\d+|one|two|three|four|five|ten)\b", question.lower())
    if not m:
        return None
    n = int(m.group(2)) if m.group(2).isdigit() else _NUM[m.group(2)]
    return float(n - 1 if m.group(1) == "at least" else n)


def parse_lookback(question: str) -> int | None:
    """"if they attended in the last 14 days" sets how far back recent activity is looked for."""
    m = re.search(r"\b(?:in|within|during|over) the (?:last|past|previous) (\d+) (days?|weeks?|months?)", question.lower())
    return int(m.group(1)) * _UNITS[m.group(2)] if m else None


def shortlist(options: dict[str, str], question: str, k: int = 40) -> dict[str, str]:
    """Keep the k options that share the most words with the question (decision models cap the options)."""
    if len(options) <= k:
        return options
    words = set(re.findall(r"[a-z]+", question.lower()))

    def overlap(item: tuple[str, str]) -> int:
        return len(words & set(re.findall(r"[a-z]+", (item[0] + " " + item[1]).replace("_", " ").lower())))

    ranked = sorted(options.items(), key=overlap, reverse=True)
    return dict(ranked[:k])


@dataclass
class Formulation:
    question: str
    readings: list[Spec]
    trace: list[dict[str, Any]] = field(default_factory=list)  # every decision asked, for the report

    @property
    def best(self) -> Spec:
        return self.readings[0]


class Formulator:
    def __init__(self, schema: Schema, decider: Any, keep: float = 0.08, beam: int = 3, top_k: int = 5):
        self.schema = schema
        self.decider = decider
        self.keep = keep
        self.beam = beam
        self.top_k = top_k
        self.trace: list[dict[str, Any]] = []

    # ------------------------------------------------------------------ helpers
    def _ask(self, slot: str, state: dict[str, Any], q: Choice | Noul) -> Any:
        ans = self.decider.ask_one(state, q)
        self.trace.append({"slot": slot, "question": q.instructions, "answer": ans,
                           "options": getattr(q, "options", None)})
        return ans

    def _kept(self, probs: dict[str, float]) -> list[tuple[str, float]]:
        ranked = sorted(probs.items(), key=lambda kv: -kv[1])
        out = [kv for kv in ranked if kv[1] >= self.keep][: self.beam]
        return out or ranked[:1]

    def _state(self, question: str, *tables: str) -> dict[str, Any]:
        names = tables or tuple(self.schema.tables)
        return {"question": question,
                "database": [self.schema.tables[n].describe(max_cols=10) for n in names]}

    # ------------------------------------------------------------------ slots
    def entity(self, question: str) -> list[tuple[str, float]]:
        cands = self.schema.entity_candidates()
        if len(cands) == 1:
            return [(cands[0], 1.0)]
        q = Choice(
            instructions=("The question asks for a prediction about each row of one table. Which table's rows "
                          "is the question about?"),
            options=shortlist({c: f"one prediction per row of {c}" for c in cands}, question),
        )
        return self._kept(self._ask("entity", self._state(question), q))

    def event(self, question: str, entity: str) -> list[tuple[tuple[str, str], float]]:
        links = self.schema.links_to(entity)
        if len(links) == 1:
            return [(links[0], 1.0)]
        labels = {f"{ev}.{col}" if sum(e == ev for e, _ in links) > 1 else ev: (ev, col) for ev, col in links}
        q = Choice(
            instructions=(f"Which kind of event does the question count or look at for each {entity} row? "
                          "Pick the table whose rows are those events."),
            options={lab: f"{ev} rows linked to {entity} through {col}" for lab, (ev, col) in labels.items()},
        )
        ans = self._ask("event", self._state(question, entity, *[e for e, _ in links]), q)
        return [(labels[lab], p) for lab, p in self._kept(ans)]

    def op(self, question: str, entity: str, event: str) -> list[tuple[str, float]]:
        # Atomic questions about the question's wording alone. The schema is left out on purpose:
        # measured on labelled examples, adding it to the state made these answers worse.
        state = {"question": question}
        kind = self._ask("answer type", state, Choice(
            instructions=f"What kind of answer does the question want for each {entity}?",
            options={"yes_or_no": f"a yes or no: will something happen to the {entity} or not",
                     "number": f"a number: how many, how much, or what average for the {entity}"}))
        probs: dict[str, float] = {}
        p_yes = kind["yes_or_no"]
        if p_yes >= self.keep:
            outcome = self._ask("stops or acts", state, Choice(
                instructions=f"What outcome does the question ask about for each {entity}?",
                options={"stops": f"the {entity} stops, churns, leaves, goes inactive or has no activity",
                         "acts": f"the {entity} does something: buys, sells, receives, visits, finishes, gets"}))
            probs["not_exists"] = p_yes * outcome["stops"]
            probs["exists"] = p_yes * outcome["acts"]
        p_num = kind["number"]
        if p_num >= self.keep:
            which = self._ask("number type", state, Choice(
                instructions="Which number does the question ask for?",
                options={"count": "a count: how many times something happens (orders, visits, purchases)",
                         "total": "a total amount added up (revenue, sales, spend)",
                         "average": "an average value (average score, average position, average rating)"}))
            probs["count"] = p_num * which["count"]
            probs["sum"] = p_num * which["total"]
            probs["mean"] = p_num * which["average"]
        return self._kept(probs)

    def value(self, question: str, event: str) -> list[tuple[str, float]]:
        t = self.schema.tables[event]
        nums = [c.name for c in t.columns.values() if c.kind == "numeric" and c.name not in t.fkeys]
        if not nums:
            return []
        if len(nums) == 1:
            return [(nums[0], 1.0)]
        q = Choice(
            instructions=f"Which column of {event} holds the amount or value the question totals or averages?",
            options=shortlist({c: f"{c}, e.g. {', '.join(t.columns[c].samples[:3])}" for c in nums}, question),
        )
        return self._kept(self._ask("value", self._state(question, event), q))

    def filters(self, question: str, event: str) -> list[tuple[Filter | None, float]]:
        t = self.schema.tables[event]
        cands: dict[str, Filter] = {}
        for c in t.columns.values():
            if c.name in t.fkeys or c.name in (t.pkey, t.time_col):
                continue
            if c.kind in ("categorical", "bool") and c.n_distinct <= 30:
                for v in c.samples[:6]:
                    cands[f"{c.name} = {v}"] = Filter(c.name, "=", v)
            if c.kind == "numeric" and c.n_distinct <= 60:
                for n in {int(x) for x in re.findall(r"\b(\d{1,3})\b", question)}:
                    if n not in (7, 14, 30, 60, 90, 180, 365):
                        cands[f"{c.name} <= {n}"] = Filter(c.name, "<=", n)
                        cands[f"{c.name} >= {n}"] = Filter(c.name, ">=", n)
        if not cands:
            return [(None, 1.0)]
        kind = self._ask("filter?", {"question": question}, Choice(
            instructions="Which activity does the question count?",
            options={"all": "all activity counts",
                     "some": "only activity of a particular kind counts, such as a top-3 finish, a cancelled "
                             "order, an ignored invitation or a 5-star review"}))
        p_any = kind["some"]
        out: list[tuple[Filter | None, float]] = [(None, 1 - p_any)]
        if p_any >= self.keep:
            q = Choice(instructions=f"Which condition picks the {event} rows the question is about?",
                       options=shortlist({k: k for k in cands}, question))
            for label, p in self._kept(self._ask("filter", self._state(question, event), q)):
                out.append((cands[label], p_any * p))
        return sorted(out, key=lambda kv: -kv[1])

    def horizon(self, question: str) -> list[tuple[int, float]]:
        h = parse_horizon(question)
        if h is not None:
            self.trace.append({"slot": "horizon", "question": "read from the text", "answer": {str(h): 1.0}})
            return [(h, 1.0)]
        q = Choice(instructions="How far ahead does the question look?",
                   options={str(k): v for k, v in HORIZONS.items()})
        return [(int(k), p) for k, p in self._kept(self._ask("horizon", {"question": question}, q))]

    def population(self, question: str, entity: str, event: str) -> dict[str, float]:
        state = {"question": question}
        p_recent = self._ask("refers to earlier activity?", state, Noul(
            instructions=(f"Does the question contain a word like stop, churn, again, return, keep or continue "
                          f"that refers to earlier activity of the {entity}?")))
        p_window = self._ask("only those with events?", state, Noul(
            instructions=(f"Is the question only about {entity}s that do have activity in the coming period, "
                          f"asking what that activity will be like (its average, its result, its outcome)?")))
        rest = 1 - p_recent
        return {"recent": p_recent, "in_window": rest * p_window, "all": rest * (1 - p_window)}

    # ------------------------------------------------------------------ search
    def run(self, question: str) -> Formulation:
        self.trace = []
        horizons = self.horizon(question)
        readings: dict[tuple, Spec] = {}
        for entity, p_ent in self.entity(question):
            key = self.schema.tables[entity].pkey or ""
            for (event, link), p_ev in self.event(question, entity):
                time_col = self.schema.tables[event].time_col or ""
                ops = self.op(question, entity, event)
                values = self.value(question, event)
                filters = self.filters(question, event)
                pops = self.population(question, entity, event)
                thr, lookback = parse_threshold(question), parse_lookback(question)
                ops_ext: list[tuple[tuple[str, float | None], float]] = []
                for op, p in ops:
                    if op == "exists" and thr is not None:
                        # "more than 2 invitations": a yes/no answer about a count crossing a threshold.
                        ops_ext += [(("count", thr), p * 0.9), (("exists", None), p * 0.1)]
                    else:
                        ops_ext.append(((op, None), p))
                for ((op, t_hold), p_op), (flt, p_f), (h, p_h) in itertools.product(ops_ext, filters, horizons):
                    vals = values if op in ("sum", "mean") else [(None, 1.0)]
                    for v, p_v in vals:
                        valid = {}
                        pop_probs = dict(pops)
                        if op == "not_exists" or lookback:
                            # Only something that was active can stop, and "if they attended in the last
                            # 14 days" names the active population outright.
                            pop_probs["recent"] = max(pop_probs["recent"], 0.8)
                        for pop, p_pop in pop_probs.items():
                            s = Spec(entity, key, event, link, time_col, op, h, pop, value=v, filter=flt,
                                     threshold=t_hold, lookback_days=lookback if pop == "recent" else None)
                            if s.valid():
                                valid[pop] = (s, p_pop)
                        z = sum(p for _, p in valid.values()) or 1.0
                        for pop, (s, p_pop) in valid.items():
                            s.slot_probs = {"entity": p_ent, "event": p_ev, "op": p_op, "value": p_v,
                                            "filter": p_f, "horizon": p_h, "population": p_pop / z}
                            s.prob = p_ent * p_ev * p_op * p_v * p_f * p_h * (p_pop / z)
                            if s.prob >= 1e-4 and (s.key() not in readings or readings[s.key()].prob < s.prob):
                                readings[s.key()] = s
        ranked = sorted(readings.values(), key=lambda s: -s.prob)[: self.top_k]
        z = sum(s.prob for s in ranked) or 1.0
        for s in ranked:
            s.prob = s.prob / z
        if not ranked:
            raise ValueError("No valid reading of the question fits this database.")
        return Formulation(question, ranked, list(self.trace))
