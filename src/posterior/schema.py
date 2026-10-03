"""Load a folder of tables into DuckDB and work out its relational structure.

Keys, links and timestamps are found in code from the data itself. The decision model is asked
only where the data cannot settle it, for example which of several timestamps marks the event.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import duckdb

from .decider import Choice

TIME_TYPES = ("TIMESTAMP", "DATE", "TIMESTAMP WITH TIME ZONE", "TIMESTAMP_NS", "TIMESTAMP_MS", "TIMESTAMP_S")
NUM_TYPES = ("TINYINT", "SMALLINT", "INTEGER", "BIGINT", "HUGEINT", "UTINYINT", "USMALLINT", "UINTEGER",
             "UBIGINT", "FLOAT", "DOUBLE", "REAL")
ID_NAME = re.compile(r"(^id$|_id$|Id$|ID$|_key$)")


@dataclass
class Column:
    name: str
    dtype: str
    kind: str  # id | time | numeric | categorical | text | bool | other
    n_distinct: int
    null_frac: float
    samples: list[str]


@dataclass
class Table:
    name: str
    source: str
    n_rows: int
    columns: dict[str, Column]
    pkey: str | None = None
    time_col: str | None = None
    fkeys: dict[str, str] = field(default_factory=dict)  # column -> parent table
    # Event time borrowed from a parent row when the table has none: (fk column, parent, parent time col)
    derived_time: tuple[str, str, str] | None = None

    @property
    def is_event(self) -> bool:
        return self.time_col is not None

    def describe(self, max_cols: int = 14) -> str:
        parts = []
        for c in list(self.columns.values())[:max_cols]:
            tag = c.kind
            if c.name == self.pkey:
                tag = "key"
            elif c.name in self.fkeys:
                tag = f"link to {self.fkeys[c.name]}"
            elif c.name == self.time_col:
                tag = "event time"
            ex = ", ".join(c.samples[:3])
            parts.append(f"{c.name} [{tag}{'; e.g. ' + ex if ex and tag not in ('key',) and not tag.startswith('link') else ''}]")
        more = f" and {len(self.columns) - max_cols} more" if len(self.columns) > max_cols else ""
        role = "event table, one row per event" if self.is_event else "table, one row per item"
        return f"{self.name} ({role}, {self.n_rows} rows): " + "; ".join(parts) + more


def _clean_names(stems: list[str]) -> dict[str, str]:
    """Strip a prefix most files share (``olist_``) and suffixes like ``_dataset``."""
    prefix = ""
    if len(stems) > 2:
        first = stems[0].split("_")[0] + "_"
        if sum(s.startswith(first) for s in stems) >= 0.5 * len(stems):
            prefix = first
    out = {}
    for s in stems:
        n = s[len(prefix):] if prefix and s.startswith(prefix) else s
        n = re.sub(r"_(dataset|data|table)$", "", n)
        out[s] = n or s
    return out


class Schema:
    def __init__(self, con: duckdb.DuckDBPyConnection, tables: dict[str, Table]):
        self.con = con
        self.tables = tables

    # ------------------------------------------------------------------ loading
    @classmethod
    def load(cls, path: str | Path, decider: Any = None, threads: int | None = None) -> "Schema":
        path = Path(path)
        con = duckdb.connect()
        con.execute(f"SET threads={threads or max(1, (os.cpu_count() or 4) - 1)}")
        sources: dict[str, str] = {}
        if path.is_dir():
            files = sorted([p for p in path.iterdir() if p.suffix.lower() in (".csv", ".parquet")])
            names = _clean_names([p.stem for p in files])
            for p in files:
                name = names[p.stem]
                if name in sources:  # two files clean to the same name: keep the raw stem
                    name = p.stem
                reader = "read_parquet" if p.suffix.lower() == ".parquet" else "read_csv_auto"
                extra = "" if reader == "read_parquet" else ", sample_size=-1"
                con.execute(f'CREATE TABLE "{name}" AS SELECT * FROM {reader}(\'{p}\'{extra})')
                sources[name] = str(p)
        else:
            src = duckdb.connect(str(path), read_only=True)
            for (name,) in src.execute("SELECT table_name FROM information_schema.tables").fetchall():
                df = src.execute(f'SELECT * FROM "{name}"').arrow()
                con.register("_tmp", df)
                con.execute(f'CREATE TABLE "{name}" AS SELECT * FROM _tmp')
                con.unregister("_tmp")
                sources[name] = f"{path}:{name}"
        tables = {n: cls._profile(con, n, s) for n, s in sources.items()}
        schema = cls(con, tables)
        schema._find_keys()
        schema._find_links()
        schema._find_times(decider)
        return schema

    @staticmethod
    def _profile(con: duckdb.DuckDBPyConnection, name: str, source: str) -> Table:
        n_rows = con.execute(f'SELECT count(*) FROM "{name}"').fetchone()[0]
        cols: dict[str, Column] = {}
        for cname, ctype, *_ in con.execute(f'DESCRIBE "{name}"').fetchall():
            q = f'"{cname}"'
            nd, nn = con.execute(f'SELECT count(DISTINCT {q}), count({q}) FROM "{name}"').fetchone()
            samples = [str(r[0])[:40] for r in con.execute(
                f'SELECT {q} FROM "{name}" WHERE {q} IS NOT NULL GROUP BY {q} ORDER BY count(*) DESC LIMIT 5'
            ).fetchall()]
            ctype_u = ctype.upper()
            if any(ctype_u.startswith(t) for t in TIME_TYPES):
                kind = "time"
            elif ctype_u == "BOOLEAN":
                kind = "bool"
            elif ID_NAME.search(cname) and (ctype_u.startswith("VARCHAR") or ctype_u in NUM_TYPES):
                kind = "id"
            elif ctype_u in NUM_TYPES or ctype_u.startswith("DECIMAL"):
                kind = "numeric"
            elif ctype_u.startswith("VARCHAR"):
                avg_len = con.execute(f'SELECT avg(length({q})) FROM "{name}"').fetchone()[0] or 0
                if nd <= 50 or nd <= 0.02 * max(nn, 1):
                    kind = "categorical"
                elif avg_len > 30:
                    kind = "text"
                elif nd > 0.9 * max(nn, 1) and avg_len >= 16:
                    kind = "id"
                else:
                    kind = "categorical"
            else:
                kind = "other"
            cols[cname] = Column(cname, ctype, kind, nd, 1 - nn / max(n_rows, 1), samples)
        return Table(name, source, n_rows, cols)

    def _find_keys(self) -> None:
        def stem(name: str) -> str:
            return (name[:-1] if name.endswith("s") else name).lower()

        for t in self.tables.values():
            singular = t.name[:-1] if t.name.endswith("s") else t.name
            # A column named after another table (order_id in order_items) points at that table's key,
            # even when it happens to be unique here (one item per order).
            foreign = {f"{stem(o)}_id" for o in self.tables if o != t.name} | {f"{o.lower()}_id" for o in self.tables if o != t.name}
            foreign -= {f"{singular.lower()}_id", f"{t.name.lower()}_id"}
            unique = [c for c in t.columns.values()
                      if c.n_distinct == t.n_rows and c.null_frac == 0 and c.kind in ("id", "numeric", "categorical")
                      and c.name.lower() not in foreign]
            if not unique:
                continue

            def rank(c: Column) -> tuple[int, int]:
                n = c.name.lower()
                if n in ("id", f"{t.name.lower()}_id", f"{singular.lower()}_id", f"{singular.lower()}id"):
                    return (0, 0)
                if c.kind == "id":
                    return (1, list(t.columns).index(c.name))
                return (2, list(t.columns).index(c.name))

            t.pkey = sorted(unique, key=rank)[0].name

    def _find_links(self, min_inclusion: float = 0.9) -> None:
        parents = [t for t in self.tables.values() if t.pkey]
        for t in self.tables.values():
            for c in t.columns.values():
                if c.name == t.pkey or c.kind not in ("id", "numeric", "categorical"):
                    continue
                for p in parents:
                    if p.name == t.name:
                        continue
                    pk = p.pkey or ""
                    cn, pn = c.name.lower(), pk.lower()
                    singular = p.name[:-1].lower() if p.name.endswith("s") else p.name.lower()
                    if not (cn == pn or (len(pn) > 3 and cn.endswith(pn)) or cn in (f"{singular}_id", f"{singular}id")):
                        continue
                    inc = self.con.execute(
                        f'SELECT avg(CASE WHEN p."{pk}" IS NULL THEN 0 ELSE 1 END) FROM '
                        f'(SELECT DISTINCT "{c.name}" AS v FROM "{t.name}" WHERE "{c.name}" IS NOT NULL) x '
                        f'LEFT JOIN "{p.name}" p ON x.v = p."{pk}"'
                    ).fetchone()[0]
                    if inc is not None and inc >= min_inclusion:
                        t.fkeys[c.name] = p.name
                        break

    def _find_times(self, decider: Any = None) -> None:
        own = {n: [c.name for c in t.columns.values() if c.kind == "time" and c.null_frac <= 0.05]
               for n, t in self.tables.items()}
        # A child row (an order item, a payment) has no key of its own and points at a parent that
        # has timestamps. Its event time may be its own column or the parent's event time.
        children = {n for n, t in self.tables.items()
                    if not t.pkey and any(own[p] for p in t.fkeys.values())}
        for n, t in self.tables.items():
            if n in children or not own[n]:
                continue
            t.time_col = own[n][0] if len(own[n]) == 1 else self._pick_time(t, {c: c for c in own[n]}, decider)
        for n in children:
            t = self.tables[n]
            options: dict[str, tuple[str, str, str] | None] = {c: None for c in own[n]}
            for col, parent in t.fkeys.items():
                ptime = self.tables[parent].time_col
                if ptime:
                    options[f"{parent}.{ptime}"] = (col, parent, ptime)
            if not options:
                continue
            label = next(iter(options)) if len(options) == 1 else self._pick_time(
                t, {k: (k if v is None else f"{k} (time of the linked {v[1]} row)") for k, v in options.items()},
                decider)
            via = options[label]
            if via is None:
                t.time_col = label
            else:
                t.derived_time = via
                t.time_col = via[2] if via[2] not in t.columns else f"{via[1]}_{via[2]}"
                self._materialize_derived(t)

    def _materialize_derived(self, t: Table) -> None:
        """Store the borrowed event time as a real column, as RelArena asks for event tables."""
        col, parent, ptime = t.derived_time  # type: ignore[misc]
        pk = self.tables[parent].pkey
        self.con.execute(
            f'CREATE OR REPLACE TABLE "{t.name}" AS SELECT c.*, p."{ptime}" AS "{t.time_col}" '
            f'FROM "{t.name}" c LEFT JOIN "{parent}" p ON c."{col}" = p."{pk}"'
        )
        dtype = self.tables[parent].columns[ptime].dtype
        t.columns[t.time_col] = Column(t.time_col, dtype, "time", 0, 0.0, [])

    @staticmethod
    def _pick_time(t: Table, options: dict[str, str], decider: Any) -> str:
        """``options`` maps a label to its description. Returns the chosen label."""
        if decider is not None:
            q = Choice(
                instructions=(
                    f"Rows of the table '{t.name}' are events. Which timestamp marks the moment a row's "
                    "event happened, when the row came into existence? Deadlines, estimates and later "
                    "updates are not the event time."
                ),
                options=options,
            )
            ans = decider.ask_one({"table": t.describe()}, q)
            return max(ans, key=ans.__getitem__)
        hint = re.compile(r"(purchase|creat|start|order|date$|timestamp$|^date|^time)", re.I)
        for label in options:
            if hint.search(label) and not re.search(r"(limit|deadline|estimat|due)", label, re.I):
                return label
        return next(iter(options))

    # ------------------------------------------------------------------ queries
    def links_to(self, entity: str) -> list[tuple[str, str]]:
        """Event tables that point at ``entity``: [(event table, fk column)]."""
        out = []
        for t in self.tables.values():
            if not t.is_event:
                continue
            for col, parent in t.fkeys.items():
                if parent == entity:
                    out.append((t.name, col))
        return out

    def entity_candidates(self) -> list[str]:
        return [t.name for t in self.tables.values() if t.pkey and self.links_to(t.name)]

    def time_range(self, table: str) -> tuple[Any, Any]:
        t = self.tables[table]
        return self.con.execute(f'SELECT min("{t.time_col}"), max("{t.time_col}") FROM {self.view(table)}').fetchone()

    def view(self, table: str) -> str:
        return f'"{table}"'

    def summary(self) -> str:
        return "\n".join(t.describe() for t in self.tables.values())

    # ------------------------------------------------------------------ export
    def materialize(self, out_dir: str | Path, columns: dict[str, list[str]] | None = None) -> dict[str, Any]:
        """Write every table to Parquet and return a RelArena database YAML dict."""
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        db: dict[str, Any] = {}
        for name, t in self.tables.items():
            fname = f"{name}.parquet"
            self.con.execute(f"COPY (SELECT * FROM {self.view(name)}) TO '{out / fname}' (FORMAT parquet)")
            entry: dict[str, Any] = {"path": fname}
            if t.pkey:
                entry["pkey"] = t.pkey
            if t.time_col:
                entry["time_col"] = t.time_col
            if t.fkeys:
                entry["fkeys"] = dict(t.fkeys)
            if columns and name in columns:
                keep = list(columns[name])
                for must in [t.pkey, t.time_col, *t.fkeys]:
                    if must and must not in keep:
                        keep.insert(0, must)
                entry["columns"] = keep
            db[name] = entry
        return db
