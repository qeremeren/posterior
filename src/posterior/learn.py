"""Fit TabPFN-3.5 through TabPFN-Rel on a compiled task, backtest it and predict.

TabPFN-Rel replays the database at many past anchor times, builds cross-table features with deep
feature synthesis, and lets TabPFN-3.5 learn from those rows in context. No training loop.
The backtest compares against two constant baselines, as RelArena recommends.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .spec import Spec, run_labels

MODEL_IDS = {"local": "tabpfn-rel-local-latest", "client": "tabpfn-rel-client-latest"}


def load_env(path: str | Path = "~/.config/posterior/env") -> None:
    """Read KEY=VALUE lines (for example TABPFN_TOKEN) without overriding the environment."""
    p = Path(path).expanduser()
    if p.exists():
        for line in p.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


@dataclass
class LearnResult:
    backend: str
    task_type: str
    metrics: dict[str, float]
    baselines: dict[str, dict[str, float]]
    test_rows: int
    test_timestamp: str
    predictions: pd.DataFrame
    live_predictions: pd.DataFrame | None = None
    seconds: float = 0.0
    notes: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        return {
            "backend": self.backend, "task_type": self.task_type, "metrics": self.metrics,
            "baselines": self.baselines, "test_rows": self.test_rows,
            "test_timestamp": self.test_timestamp, "seconds": round(self.seconds, 1), "notes": self.notes,
        }


def _metrics(task_type: str, y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import brier_score_loss, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score

    if task_type == "binary_classification":
        out = {"positive_rate": float(np.mean(y)), "brier": float(brier_score_loss(y, np.clip(p, 0, 1)))}
        if len(np.unique(y)) == 2:
            out["roc_auc"] = float(roc_auc_score(y, p))
        return out
    return {"mae": float(mean_absolute_error(y, p)), "rmse": float(np.sqrt(mean_squared_error(y, p))),
            "r2": float(r2_score(y, p)) if len(y) > 1 else float("nan")}


class Learner:
    def __init__(self, backend: str | None = None, seed: int = 0):
        load_env()
        self.backend = backend or os.environ.get("POSTERIOR_TABPFN", "client")
        if self.backend not in MODEL_IDS:
            raise ValueError(f"backend must be one of {list(MODEL_IDS)}")
        self.seed = seed

    def _auth(self) -> None:
        if self.backend == "client":
            token = os.environ.get("TABPFN_TOKEN")
            if not token:
                raise RuntimeError("TABPFN_TOKEN is not set (put it in ~/.config/posterior/env).")
            import tabpfn_client

            tabpfn_client.set_access_token(token)

    def run(
        self,
        spec: Spec,
        task_path: Path,
        data_dir: Path,
        con: Any,
        cache_dir: Path | None = None,
        live_task_path: Path | None = None,
        live_entities: list[Any] | None = None,
    ) -> LearnResult:
        from relarena_core.userdb import PredictiveContext, PredictiveQuery, PredictiveQuerySpec

        self._auth()
        t0 = time.time()
        task = yaml.safe_load(task_path.read_text())
        pq_spec = PredictiveQuerySpec.from_yaml(str(task_path), data_dir=str(data_dir))
        ctx = PredictiveContext(pq_spec)
        fitted = ctx.fit(MODEL_IDS[self.backend], n_trials=0, seed=self.seed, cache_dir=cache_dir)
        labels = ctx.compute_test_labels()
        ent, tcol, target = task["entity_col"], task["time_col"], task["target_col"]
        preds = fitted.predict(PredictiveQuery(entities=labels[ent].tolist(), at_timestamp="test_timestamp"))
        scored = labels.merge(preds, on=[tcol, ent], how="inner")
        y = scored[target].astype(float).to_numpy()
        p = scored[f"{target}_pred"].astype(float).to_numpy()
        metrics = _metrics(spec.task_type, y, p)

        baselines = self._baselines(spec, task, con, scored, ent, target)
        notes = []
        if len(scored) < len(labels):
            notes.append(f"{len(labels) - len(scored)} test rows had no prediction")

        live = None
        if live_task_path is not None:
            live_spec = PredictiveQuerySpec.from_yaml(str(live_task_path), data_dir=str(data_dir))
            live_ctx = PredictiveContext(live_spec)
            live_fit = live_ctx.fit(MODEL_IDS[self.backend], n_trials=0, seed=self.seed, cache_dir=cache_dir)
            sel = live_entities if live_entities else "all"
            live = live_fit.predict(PredictiveQuery(entities=sel, at_timestamp="test_timestamp"))
        return LearnResult(self.backend, spec.task_type, metrics, baselines, len(scored), str(task["test_timestamp"]),
                           scored, live, time.time() - t0, notes)

    def _baselines(self, spec: Spec, task: dict, con: Any, scored: pd.DataFrame, ent: str, target: str) -> dict:
        """Constant (global) and constant (per entity) baselines fitted on the training anchors."""
        val = pd.Timestamp(task["val_timestamp"]).to_pydatetime()
        h = timedelta(days=spec.horizon_days)
        start = con.execute(f'SELECT min("{spec.time_col}") FROM "{spec.event}"').fetchone()[0]
        anchors = []
        a = val - h
        while a - h >= pd.Timestamp(start).to_pydatetime() and len(anchors) < 60:
            anchors.append(a)
            a -= h
        if not anchors:
            return {}
        train = run_labels(con, spec.label_sql(), sorted(anchors), spec.horizon_days)
        if train.empty:
            return {}
        y = scored[target].astype(float).to_numpy()
        g = float(train["target"].astype(float).mean())
        per = train.groupby(spec.entity_key)["target"].mean()
        p_ent = scored[ent].map(per).fillna(g).astype(float).to_numpy()
        out = {"constant_global": _metrics(spec.task_type, y, np.full_like(y, g)),
               "constant_per_entity": _metrics(spec.task_type, y, p_ent)}
        return out


def live_task(task: dict[str, Any], end: datetime, horizon_days: int) -> dict[str, Any]:
    """The same task moved forward so its cutoff is the end of the data: predictions for now."""
    t = dict(task)
    t["test_timestamp"] = end.strftime("%Y-%m-%d %H:%M:%S")
    t["val_timestamp"] = (end - timedelta(days=horizon_days)).strftime("%Y-%m-%d %H:%M:%S")
    return t
