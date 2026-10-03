"""The predictive task grammar and its compiler to RelArena task files.

A ``Spec`` is a small typed record: which entity, which events, what operation over them, which
filter, which window and which population. Every field is filled by a typed decision, and the
compiler turns a spec into label SQL deterministically, so no model ever writes SQL.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from typing import Any

import pandas as pd

OPS = ("not_exists", "exists", "count", "sum", "mean")
POPULATIONS = ("all", "recent", "in_window")


@dataclass(frozen=True)
class Filter:
    column: str
    op: str  # = != <= >= < >
    value: Any

    def sql(self, alias: str) -> str:
        v = self.value
        lit = str(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else "'" + str(v).replace("'", "''") + "'"
        return f'{alias}."{self.column}" {self.op} {lit}'

    def text(self) -> str:
        return f"{self.column} {self.op} {self.value}"


@dataclass
class Spec:
    entity: str
    entity_key: str
    event: str
    link: str
    time_col: str
    op: str
    horizon_days: int
    population: str
    value: str | None = None
    filter: Filter | None = None
    threshold: float | None = None
    lookback_days: int | None = None
    prob: float = 1.0
    slot_probs: dict[str, float] = field(default_factory=dict)

    # ------------------------------------------------------------------ meaning
    @property
    def task_type(self) -> str:
        if self.op in ("exists", "not_exists") or self.threshold is not None:
            return "binary_classification"
        return "regression"

    def valid(self) -> bool:
        if self.op not in OPS or self.population not in POPULATIONS:
            return False
        if self.op in ("sum", "mean") and not self.value:
            return False
        if self.op == "not_exists" and self.population == "in_window":
            return False
        if self.op == "exists" and self.filter is None and self.population == "in_window":
            return False  # every entity in the window has an event: the label would be constant
        if self.op == "mean" and self.population != "in_window":
            return False
        return True

    def key(self) -> tuple:
        return (self.entity, self.event, self.link, self.op, self.value, self.filter, self.threshold,
                self.horizon_days, self.population)

    def describe(self) -> str:
        who = {"all": f"every row of {self.entity}",
               "recent": f"each {self.entity} row with a {self.event} event in the last {self.lookback_days or self.horizon_days} days",
               "in_window": f"each {self.entity} row with a {self.event} event in the next {self.horizon_days} days"}[self.population]
        kind = f" where {self.filter.text()}" if self.filter else ""
        what = {"not_exists": f"no {self.event} event{kind}",
                "exists": f"at least one {self.event} event{kind}",
                "count": f"the number of {self.event} events{kind}" + (f" is above {self.threshold:g}" if self.threshold is not None else ""),
                "sum": f"the total {self.value} over {self.event} events{kind}",
                "mean": f"the average {self.value} over {self.event} events{kind}"}[self.op]
        return f"For {who}: {what} in the next {self.horizon_days} days"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["filter"] = None if self.filter is None else asdict(self.filter)
        d["task_type"] = self.task_type
        d["description"] = self.describe()
        return d

    # ------------------------------------------------------------------ compiler
    def label_sql(self) -> str:
        """RelArena label query. ``{timedelta}`` is substituted by the runner."""
        tc, pk = f'"{self.time_col}"', f'"{self.entity_key}"'
        win = f"e.{tc} > t.timestamp AND e.{tc} <= t.timestamp + INTERVAL '{{timedelta}}'"
        filt = f" AND {self.filter.sql('e')}" if self.filter else ""
        if self.population == "in_window":
            cond = self.filter.sql("e") if self.filter else "TRUE"
            if self.op == "exists":
                agg = f"MAX(CASE WHEN {cond} THEN 1 ELSE 0 END)"
            elif self.op == "count":
                agg = "COUNT(*)" if not self.filter else f"SUM(CASE WHEN {cond} THEN 1 ELSE 0 END)"
                if self.threshold is not None:
                    agg = f"CASE WHEN {agg} > {self.threshold:g} THEN 1 ELSE 0 END"
            elif self.op == "sum":
                agg = f'SUM(CASE WHEN {cond} THEN e."{self.value}" ELSE 0 END)'
            else:
                agg = f'AVG(CASE WHEN {cond} THEN e."{self.value}" END)'
            sql = (f'SELECT t.timestamp AS timestamp, e."{self.link}" AS {pk}, {agg} AS target\n'
                   f'FROM timestamp_df t\nJOIN "{self.event}" e ON {win}\n'
                   f'WHERE e."{self.link}" IS NOT NULL\nGROUP BY t.timestamp, e."{self.link}"')
            if self.op == "mean":
                sql = f"SELECT * FROM (\n{sql}\n) WHERE target IS NOT NULL"
            return sql

        sub = f'FROM "{self.event}" e WHERE e."{self.link}" = x.{pk} AND {win}{filt}'
        if self.op == "not_exists":
            expr = f"CAST(NOT EXISTS (SELECT 1 {sub}) AS INTEGER)"
        elif self.op == "exists":
            expr = f"CAST(EXISTS (SELECT 1 {sub}) AS INTEGER)"
        elif self.op == "count":
            expr = f"(SELECT COUNT(*) {sub})"
            if self.threshold is not None:
                expr = f"CAST({expr} > {self.threshold:g} AS INTEGER)"
        elif self.op == "sum":
            expr = f'(SELECT COALESCE(SUM(e."{self.value}"), 0) {sub})'
        else:
            expr = f'(SELECT AVG(e."{self.value}") {sub})'
        where = "TRUE"
        if self.population == "recent":
            lb = self.lookback_days or self.horizon_days
            back = "INTERVAL '{timedelta}'" if lb == self.horizon_days else f"INTERVAL '{lb} days'"
            where = (f'EXISTS (SELECT 1 FROM "{self.event}" e WHERE e."{self.link}" = x.{pk} '
                     f"AND e.{tc} > t.timestamp - {back} AND e.{tc} <= t.timestamp)")
        return (f"SELECT t.timestamp AS timestamp, x.{pk} AS {pk}, {expr} AS target\n"
                f'FROM timestamp_df t, "{self.entity}" x\nWHERE {where}')

    def task_yaml(self, val_ts: datetime, test_ts: datetime, database: str = "db.yaml") -> dict[str, Any]:
        return {
            "database": database,
            "entity_table": self.entity,
            "entity_col": self.entity_key,
            "time_col": "timestamp",
            "target_col": "target",
            "task_type": self.task_type,
            "timedelta": f"{self.horizon_days} days",
            "num_eval_timestamps": 1,
            "val_timestamp": val_ts.strftime("%Y-%m-%d %H:%M:%S"),
            "test_timestamp": test_ts.strftime("%Y-%m-%d %H:%M:%S"),
            "query": self.label_sql() + "\n",
        }


def choose_splits(con: Any, event_sql: str, time_col: str, horizon_days: int) -> tuple[datetime, datetime]:
    """Put the test window at the end of the period where the data is still dense.

    Many exports thin out at the end (Olist's last weeks hold a handful of orders). The effective end
    is the last day whose trailing 7-day volume is at least a quarter of the typical 7-day volume.
    """
    daily = con.execute(
        f'SELECT CAST("{time_col}" AS DATE) AS d, count(*) AS n FROM {event_sql} '
        f'WHERE "{time_col}" IS NOT NULL GROUP BY 1 ORDER BY 1'
    ).df()
    daily["d"] = pd.to_datetime(daily["d"])
    s = daily.set_index("d")["n"].asfreq("D", fill_value=0)
    week = s.rolling(7, min_periods=1).sum()
    typical = week[week > 0].median()
    dense = week[week >= 0.25 * typical]
    end = dense.index.max() if len(dense) else s.index.max()
    start = s.index.min()
    h = timedelta(days=horizon_days)
    test = (end - h).to_pydatetime()
    val = test - h
    if val - 3 * h < start:
        raise ValueError(
            f"Not enough history for a {horizon_days}-day window: data runs {start.date()} to {end.date()}."
        )
    return val, test


def run_labels(con: Any, sql: str, anchors: list[datetime], horizon_days: int) -> pd.DataFrame:
    """Execute a label query on chosen anchors, the way the RelArena runner does."""
    con.register("timestamp_df", pd.DataFrame({"timestamp": pd.to_datetime(anchors)}))
    try:
        return con.execute(sql.replace("{timedelta}", f"{horizon_days} days")).df()
    finally:
        con.unregister("timestamp_df")


def anchors_before(test: datetime, horizon_days: int, n: int) -> list[datetime]:
    h = timedelta(days=horizon_days)
    return [test - i * h for i in range(n)][::-1]


def safe_float(x: Any) -> float | None:
    try:
        f = float(x)
        return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return None
