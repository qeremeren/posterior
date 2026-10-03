"""Leakage audit: decide, column by column, whether a value was known at prediction time.

Two independent kinds of evidence are combined:

* **Meaning** (decision model). One batched call per table asks, for every column, whether it is
  a value that is filled in or updated after the row's moment (a status, a delivery date, a
  running total) instead of a fixed attribute.
* **Data**. For event tables: does the column look different on the newest rows, with more empty
  values or a shifted distribution? Columns filled in later are still empty on recent rows.
  For the entity table: does the column alone predict the label suspiciously well? This check
  uses TabPFN when it is available.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from .decider import Noul
from .schema import Schema
from .spec import Spec


@dataclass
class ColumnAudit:
    table: str
    column: str
    role: str  # entity | event | dimension
    meaning: float  # p(value is filled in or updated later), decision model
    evidence: float | None  # 0..1 from the data, None when no check applies
    evidence_kind: str
    evidence_value: float | None
    score: float
    verdict: str  # keep | drop
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class _Scorer:
    """Univariate AUC of one column against the label. TabPFN when possible, else boosting."""

    def __init__(self) -> None:
        self.name = "sklearn"
        self._tabpfn = None
        backend = os.environ.get("POSTERIOR_TABPFN", "client")
        try:
            if backend == "local":
                from tabpfn import TabPFNClassifier  # type: ignore

                self._tabpfn = TabPFNClassifier
                self.name = "tabpfn-local"
            elif os.environ.get("TABPFN_TOKEN"):
                import tabpfn_client  # type: ignore

                tabpfn_client.set_access_token(os.environ["TABPFN_TOKEN"])
                self._tabpfn = tabpfn_client.TabPFNClassifier
                self.name = "tabpfn-client"
        except Exception:  # noqa: BLE001
            self._tabpfn = None

    def auc(self, x: pd.Series, y: pd.Series, t: pd.Series | None = None, max_rows: int = 1200,
            seed: int = 0) -> tuple[float, float] | None:
        """Out-of-fold AUC of the column (with the anchor time) and of the anchor time alone.

        A date column is measured relative to the anchor ("days until the last order"), which is how a
        snapshot date leaks.
        """
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import StratifiedKFold

        df = pd.DataFrame({"x": x.values, "y": y.values, "t": (t.values if t is not None else 0)}).dropna(subset=["y"])
        if df["y"].nunique() != 2 or len(df) < 40:
            return None
        if len(df) > max_rows:
            df = df.sample(max_rows, random_state=seed)
        ts = pd.to_datetime(df["t"]) if t is not None else None
        if np.issubdtype(df["x"].dtype, np.datetime64) or (df["x"].dtype == object and _looks_like_dates(df["x"])):
            xv = (pd.to_datetime(df["x"], errors="coerce") - (ts if ts is not None else 0)).dt.total_seconds() / 86400
        elif df["x"].dtype == object or str(df["x"].dtype).startswith(("string", "category")):
            xv = df["x"].astype("category").cat.codes.astype(float)
        else:
            xv = df["x"].astype(float)
        tv = (ts.astype("int64") // 10**9).astype(float) if ts is not None else pd.Series(0.0, index=df.index)
        yv = df["y"].astype(int).to_numpy()
        both = np.column_stack([xv.fillna(-1e9).to_numpy(), tv.to_numpy()])
        return self._oof_auc(both, yv, seed), self._oof_auc(tv.to_numpy()[:, None], yv, seed)

    def _oof_auc(self, X: np.ndarray, y: np.ndarray, seed: int) -> float:
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import StratifiedKFold

        oof = np.zeros(len(y))
        for tr, te in StratifiedKFold(2, shuffle=True, random_state=seed).split(X, y):
            model = self._model()
            model.fit(X[tr], y[tr])
            oof[te] = model.predict_proba(X[te])[:, 1]
        a = roc_auc_score(y, oof)
        return float(max(a, 1 - a))

    def _model(self) -> Any:
        if self._tabpfn is not None:
            try:
                return self._tabpfn()
            except Exception:  # noqa: BLE001
                pass
        from sklearn.ensemble import HistGradientBoostingClassifier

        self.name = "sklearn"
        return HistGradientBoostingClassifier(max_iter=100)


def _looks_like_dates(s: pd.Series) -> bool:
    import warnings

    sample = s.dropna().astype(str).head(20)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return len(sample) > 0 and pd.to_datetime(sample, errors="coerce").notna().mean() > 0.9


class Auditor:
    def __init__(self, schema: Schema, decider: Any, drop_at: float = 0.6, prior_only_drop_at: float = 0.75,
                 use_meaning: bool = True, use_data: bool = True):
        self.schema = schema
        self.decider = decider
        self.use_meaning = use_meaning
        self.use_data = use_data
        self.drop_at = drop_at
        self.prior_only_drop_at = prior_only_drop_at
        self.scorer = _Scorer()

    # ------------------------------------------------------------------ evidence
    def _meaning(self, table: str) -> dict[str, float]:
        t = self.schema.tables[table]
        cols = self._audited_columns(table)
        if not cols:
            return {}
        if t.is_event:
            make = lambda c: Noul(  # noqa: E731
                f"Is '{c}' filled in or changed some time after the moment in '{t.time_col}', such as a delivery "
                f"date, a final status, an approval time or a later answer, rather than recorded at that moment?")
        else:
            make = lambda c: Noul(  # noqa: E731
                f"Is '{c}' a value that keeps being updated as things happen, such as a status, a last-activity "
                f"date, a running total or a lifetime count, rather than a fixed attribute of the {table} row?")
        return self.decider.ask({"table": t.describe(max_cols=30)}, {c: make(c) for c in cols})

    def _audited_columns(self, table: str) -> list[str]:
        t = self.schema.tables[table]
        skip = {t.pkey, t.time_col, *t.fkeys}
        return [c for c in t.columns if c not in skip and t.columns[c].kind != "id"]

    def _age_shift(self, table: str, col: str) -> float | None:
        """How differently a column looks on the newest rows than on rows from the middle of history."""
        t = self.schema.tables[table]
        tc = t.time_col
        q = f'"{col}"'
        stats = self.schema.con.execute(
            f'WITH r AS (SELECT {q} AS v, percent_rank() OVER (ORDER BY "{tc}") AS pr FROM "{table}" '
            f'WHERE "{tc}" IS NOT NULL) '
            "SELECT avg(CASE WHEN pr >= 0.97 THEN (v IS NULL)::INT END), "
            "avg(CASE WHEN pr BETWEEN 0.4 AND 0.8 THEN (v IS NULL)::INT END) FROM r"
        ).fetchone()
        null_recent, null_mid = (stats[0] or 0.0), (stats[1] or 0.0)
        shift = max(0.0, null_recent - null_mid) * 2
        if t.columns[col].kind in ("categorical", "bool"):
            dist = self.schema.con.execute(
                f'WITH r AS (SELECT CAST({q} AS VARCHAR) AS v, percent_rank() OVER (ORDER BY "{tc}") AS pr '
                f'FROM "{table}" WHERE "{tc}" IS NOT NULL), '
                "a AS (SELECT v, count(*) FILTER (WHERE pr >= 0.97) AS n_new, "
                "count(*) FILTER (WHERE pr BETWEEN 0.4 AND 0.8) AS n_mid FROM r GROUP BY v) "
                "SELECT 0.5 * sum(abs(n_new / (SELECT sum(n_new) FROM a) - n_mid / (SELECT sum(n_mid) FROM a))) FROM a"
            ).fetchone()[0]
            shift = max(shift, float(dist or 0.0))
        return float(min(1.0, shift))

    def _label_power(self, spec: Spec, train: pd.DataFrame, col: str) -> tuple[float, float] | None:
        ent = self.schema.tables[spec.entity]
        if spec.task_type != "binary_classification" or train.empty:
            return None
        values = self.schema.con.execute(f'SELECT "{ent.pkey}" AS k, "{col}" AS x FROM "{spec.entity}"').df()
        df = train.merge(values, left_on=spec.entity_key, right_on="k", how="left")
        return self.scorer.auc(df["x"], df["target"], df["timestamp"])

    # ------------------------------------------------------------------ run
    def run(self, spec: Spec, train: pd.DataFrame) -> list[ColumnAudit]:
        out: list[ColumnAudit] = []
        for name, t in self.schema.tables.items():
            meaning = self._meaning(name) if self.use_meaning else {}
            role = "entity" if name == spec.entity else ("event" if t.is_event else "dimension")
            for col in self._audited_columns(name):
                prior = float(meaning.get(col, 0.0))
                ev: float | None = None
                kind, raw = "none", None
                if not self.use_data:
                    pass
                elif t.is_event:
                    raw = self._age_shift(name, col)
                    kind, ev = "newest rows differ", raw
                elif role == "entity":
                    aucs = self._label_power(spec, train, col)
                    if aucs is not None:
                        auc, base = aucs
                        kind, raw = f"AUC with the anchor time vs {base:.2f} without the column ({self.scorer.name})", auc
                        # Suspicious when the column adds a lot over the anchor time alone. On Olist the
                        # injected snapshot columns add 0.13 to 0.26 AUC; ordinary seller attributes add ~0.
                        ev = float(np.clip((auc - base - 0.05) / 0.15, 0, 1))
                out.append(self._decide(name, col, role, prior, ev, kind, raw))
        return out

    def _decide(self, table: str, col: str, role: str, prior: float, ev: float | None, kind: str,
                raw: float | None) -> ColumnAudit:
        # The data checks can confirm a leak but cannot clear a column: an export taken after every
        # order was delivered shows no empty delivery dates on its newest rows. So strong evidence of
        # either kind is enough on its own, and weaker evidence of both kinds adds up (noisy-OR).
        if ev is None:
            score, drop = prior, prior >= self.prior_only_drop_at
        else:
            score = 1 - (1 - prior) * (1 - ev)
            # Either kind of evidence decides alone only when it is strong; otherwise both must point the
            # same way, so a genuinely predictive attribute with an innocent meaning is kept.
            drop = (prior >= self.prior_only_drop_at or ev >= 0.9
                    or (score >= self.drop_at and min(prior, ev) >= 0.25))
        parts = [f"meaning: updated later with p={prior:.2f}"]
        if raw is not None:
            parts.append(f"{kind}: {raw:.2f}")
        return ColumnAudit(table, col, role, round(prior, 4), None if ev is None else round(ev, 4), kind,
                           None if raw is None else round(raw, 4), round(score, 4),
                           "drop" if drop else "keep", "; ".join(parts))


def allow_list(schema: Schema, audits: list[ColumnAudit]) -> dict[str, list[str]]:
    dropped = {(a.table, a.column) for a in audits if a.verdict == "drop"}
    return {n: [c for c in t.columns if (n, c) not in dropped and t.columns[c].kind != "text"]
            for n, t in schema.tables.items()}
