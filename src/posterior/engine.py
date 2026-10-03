"""Posterior: one call from a plain-language question to calibrated, audited predictions."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .audit import Auditor, ColumnAudit, allow_list
from .clarify import Clarification, clarify
from .decider import make_decider
from .formulate import Formulation, Formulator
from .schema import Schema
from .spec import Spec, anchors_before, choose_splits, run_labels


@dataclass
class Result:
    question: str
    status: str  # done | needs_clarification | formulated
    readings: list[Spec]
    chosen: int
    clarification: Clarification | None = None
    splits: dict[str, str] = field(default_factory=dict)
    audit: list[ColumnAudit] = field(default_factory=list)
    learn: dict[str, Any] | None = None
    predictions: pd.DataFrame | None = None  # live predictions, highest first
    files: dict[str, str] = field(default_factory=dict)
    trace: list[dict[str, Any]] = field(default_factory=list)
    seconds: float = 0.0

    @property
    def spec(self) -> Spec:
        return self.readings[self.chosen]

    def to_dict(self, max_predictions: int = 50) -> dict[str, Any]:
        preds = None
        if self.predictions is not None:
            preds = json.loads(self.predictions.head(max_predictions).to_json(orient="records", date_format="iso"))
        return {
            "question": self.question,
            "status": self.status,
            "readings": [{"index": i, "probability": round(s.prob, 4), **s.to_dict()} for i, s in enumerate(self.readings)],
            "chosen": self.chosen,
            "clarification": self.clarification.to_dict() if self.clarification else None,
            "splits": self.splits,
            "audit": {"dropped": [a.to_dict() for a in self.audit if a.verdict == "drop"],
                      "kept": len([a for a in self.audit if a.verdict == "keep"])},
            "backtest": self.learn,
            "predictions": preds,
            "n_predictions": 0 if self.predictions is None else len(self.predictions),
            "files": self.files,
            "seconds": round(self.seconds, 1),
        }


class Posterior:
    def __init__(self, schema: Schema, decider: Any, work_root: str | Path = "runs", backend: str | None = None):
        self.schema = schema
        self.decider = decider
        self.work_root = Path(work_root)
        self.backend = backend
        self.formulator = Formulator(schema, decider)
        self.auditor = Auditor(schema, decider)

    @classmethod
    def connect(cls, path: str | Path, model: str | None = None, work_root: str | Path = "runs",
                backend: str | None = None, cache_path: str | Path = "cache/decisions.jsonl") -> "Posterior":
        decider = make_decider(model, cache_path=cache_path)
        schema = Schema.load(path, decider=decider)
        return cls(schema, decider, work_root=work_root, backend=backend)

    # ------------------------------------------------------------------ steps
    def formulate(self, question: str) -> tuple[Formulation, Clarification | None, tuple[datetime, datetime]]:
        f = self.formulator.run(question)
        best = f.best
        val, test = choose_splits(self.schema.con, f'"{best.event}"', best.time_col, best.horizon_days)
        anchors = anchors_before(val, best.horizon_days, 4)
        clar = clarify(self.schema.con, f.readings, anchors)
        return f, clar, (val, test)

    def _slug(self, question: str, spec: Spec, dropped: list[str]) -> str:
        h = hashlib.sha1(json.dumps([question, str(spec.key()), sorted(dropped)]).encode()).hexdigest()[:10]
        return f"{spec.entity}-{spec.op}-{spec.horizon_days}d-{h}"

    def ask(self, question: str, reading: int | None = None, live: bool = True, auto: bool = True,
            run_model: bool = True) -> Result:
        """Formulate, clarify, audit and learn.

        ``reading`` picks one of the readings (for example after a clarification). With ``auto`` the most
        likely reading runs even when a clarification is open; without it the call stops and asks.
        """
        t0 = time.time()
        f, clar, (val, test) = self.formulate(question)
        chosen = reading if reading is not None else 0
        if reading is not None:
            clar = None
        spec = f.readings[chosen]
        if spec is not f.best:
            val, test = choose_splits(self.schema.con, f'"{spec.event}"', spec.time_col, spec.horizon_days)
        result = Result(question, "formulated", f.readings, chosen, clar, trace=f.trace,
                        splits={"val": val.isoformat(), "test": test.isoformat()})
        if clar is not None and not auto:
            result.status = "needs_clarification"
            result.seconds = time.time() - t0
            return result

        h = timedelta(days=spec.horizon_days)
        train_anchors = [a for a in anchors_before(val - h, spec.horizon_days, 24)]
        train = run_labels(self.schema.con, spec.label_sql(), train_anchors, spec.horizon_days)
        result.audit = self.auditor.run(spec, train)
        if not run_model:
            result.seconds = time.time() - t0
            return result

        dropped = [f"{a.table}.{a.column}" for a in result.audit if a.verdict == "drop"]
        work = self.work_root / self._slug(question, spec, dropped)
        cached = work / "result.json"
        if cached.exists() and (work / "predictions.parquet").exists():
            saved = json.loads(cached.read_text())
            result.learn = saved["backtest"]
            result.predictions = pd.read_parquet(work / "predictions.parquet")
            result.files = saved["files"]
            result.status = "done"
            result.seconds = time.time() - t0
            return result

        from .learn import Learner, live_task

        data_dir = work / "data"
        db = self.schema.materialize(data_dir, columns=allow_list(self.schema, result.audit))
        (work / "db.yaml").write_text(yaml.safe_dump(db, sort_keys=False))
        task = spec.task_yaml(val, test, database="db.yaml")
        (work / "task.yaml").write_text(yaml.safe_dump(task, sort_keys=False, width=1000))
        live_path, live_entities = None, None
        if live:
            end = test + h
            lt = live_task(task, end, spec.horizon_days)
            live_path = work / "live_task.yaml"
            live_path.write_text(yaml.safe_dump(lt, sort_keys=False, width=1000))
            live_entities = self._eligible(spec, end)
        learner = Learner(self.backend)
        lr = learner.run(spec, work / "task.yaml", data_dir, self.schema.con, cache_dir=work / "dfs-cache",
                         live_task_path=live_path, live_entities=live_entities)
        result.learn = lr.summary()
        preds = lr.live_predictions if lr.live_predictions is not None else lr.predictions
        col = [c for c in preds.columns if c.endswith("_pred")][0]
        preds = preds.rename(columns={col: "prediction"}).sort_values("prediction", ascending=False)
        result.predictions = preds.reset_index(drop=True)
        result.status = "done"
        result.files = {"task": str(work / "task.yaml"), "database": str(work / "db.yaml"),
                        "predictions": str(work / "predictions.parquet")}
        result.predictions.to_parquet(work / "predictions.parquet")
        result.seconds = time.time() - t0
        cached.write_text(json.dumps(result.to_dict(), indent=2, default=str))
        return result

    def _eligible(self, spec: Spec, at: datetime) -> list[Any] | None:
        """Entities that should get a live prediction at ``at`` (the population, judged on the past)."""
        if spec.population == "all":
            return None
        lb = spec.lookback_days or spec.horizon_days
        rows = self.schema.con.execute(
            f'SELECT DISTINCT "{spec.link}" FROM "{spec.event}" WHERE "{spec.time_col}" > ? AND "{spec.time_col}" <= ? '
            f'AND "{spec.link}" IS NOT NULL', [at - timedelta(days=lb), at]).fetchall()
        return [r[0] for r in rows] or None
