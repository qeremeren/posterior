"""HTTP API.

* ``POST /v1/formulate``: readings and clarification, no TabPFN fit (fast)
* ``POST /v1/ask``: the full answer (formulate, audit, learn, predict), synchronous
* ``POST /v1/tasks`` and ``GET /v1/tasks/{id}``: the same, in the background
* ``POST /v1/systemone`` (alias ``/v1/decisions``): TypeSafe/Jev wire format. A yes/no question about
  the future of an entity named in ``state`` is answered from the database's own history by TabPFN;
  every other question is passed through to the decision model unchanged.
"""

from __future__ import annotations

import contextlib
import sys
import threading
import traceback
import uuid
from typing import Any

import httpx
from pydantic import BaseModel

from .decider import Noul
from .engine import Posterior, Result


class AskBody(BaseModel):
    question: str
    reading: int | None = None
    live: bool = True
    auto: bool = True
    run_model: bool = True


class FormulateBody(BaseModel):
    question: str


def _entity_ref(post: Posterior, state: Any) -> tuple[str, Any] | None:
    """Find ``{<entity primary key>: value}`` at the top level or one level down in ``state``."""
    keys = {t.pkey: name for name, t in post.schema.tables.items() if t.pkey}
    if not isinstance(state, dict):
        return None
    layers = [state] + [v for v in state.values() if isinstance(v, dict)]
    for layer in layers:
        for k, v in layer.items():
            if k in keys and isinstance(v, (str, int)):
                return keys[k], v
    return None


class Service:
    """Thread-safe wrapper: DuckDB and the TabPFN fit run one at a time."""

    def __init__(self, post: Posterior):
        self.post = post
        self.lock = threading.Lock()
        self.results: dict[str, Result] = {}
        self.tasks: dict[str, dict[str, Any]] = {}

    def ask(self, question: str, reading: int | None = None, live: bool = True, auto: bool = True,
            run_model: bool = True) -> Result:
        key = f"{question}|{reading}|{live}|{auto}|{run_model}"
        if key in self.results:
            return self.results[key]
        with self.lock, contextlib.redirect_stdout(sys.stderr):
            res = self.post.ask(question, reading=reading, live=live, auto=auto, run_model=run_model)
        if res.status == "done":
            self.results[key] = res
        return res

    def formulate(self, question: str) -> dict[str, Any]:
        with self.lock:
            f, clar, (val, test) = self.post.formulate(question)
        d = Result(question, "formulated", f.readings, 0, clar,
                   splits={"val": val.isoformat(), "test": test.isoformat()}).to_dict()
        d.pop("audit")
        d["decisions"] = f.trace
        return d

    def start(self, question: str, **kw: Any) -> str:
        tid = uuid.uuid4().hex[:12]
        self.tasks[tid] = {"id": tid, "status": "running", "question": question}

        def work() -> None:
            try:
                res = self.ask(question, **kw)
                self.tasks[tid].update(status=res.status, result=res.to_dict())
            except Exception as e:  # noqa: BLE001
                self.tasks[tid].update(status="error", error=f"{type(e).__name__}: {e}",
                                       detail=traceback.format_exc()[-2000:])

        threading.Thread(target=work, daemon=True).start()
        return tid

    def systemone(self, body: dict[str, Any]) -> dict[str, Any]:
        state, questions = body.get("state"), body.get("questions") or {}
        ref = _entity_ref(self.post, state)
        answers: dict[str, Any] = {}
        forward: dict[str, Any] = {}
        for qid, q in questions.items():
            text = q.get("instructions") if isinstance(q.get("instructions"), str) else qid
            if ref and q.get("type") == "noul" and self._is_future(text):
                p = self._future_probability(text, *ref)
                if p is not None:
                    answers[qid] = {"type": "noul", "noul": round(p, 4)}
                    continue
            forward[qid] = q
        model = self.post.decider.name.split("+")[0]
        if forward:
            dec = getattr(self.post.decider, "members", [self.post.decider])[0]
            r = httpx.post(f"{dec.base_url}/v1/systemone", timeout=180,
                           headers={"Authorization": f"Bearer {dec.api_key}"},
                           json={"model": dec.model, "state": state, "questions": forward})
            r.raise_for_status()
            answers.update(r.json()["answers"])
            model = r.json().get("model", model)
        ordered = {k: answers[k] for k in questions if k in answers}
        return {"model": f"posterior/{model}", "answers": ordered, "usage": {"input_tokens": 0, "output_tokens": 0}}

    def _is_future(self, question: str) -> bool:
        """Route with the decision model: a prediction about what comes next, or a question about the text?"""
        p = self.post.decider.ask_one({"question": question}, Noul(
            "Does this question ask for a prediction about what will happen to someone or something in the "
            "coming period, rather than about the meaning or content of the text it comes with?"))
        return float(p) >= 0.5

    def _future_probability(self, question: str, table: str, entity_id: Any) -> float | None:
        try:
            res = self.ask(question)
        except Exception:  # noqa: BLE001
            return None
        spec = res.spec
        if res.status != "done" or spec.entity != table or spec.task_type != "binary_classification":
            return None
        preds = res.predictions
        if preds is None or spec.entity_key not in preds.columns:
            return None
        hit = preds[preds[spec.entity_key].astype(str) == str(entity_id)]
        if hit.empty:
            return None
        return float(hit["prediction"].iloc[0])


def create_app(post: Posterior):  # noqa: ANN201
    from fastapi import FastAPI, HTTPException

    svc = Service(post)
    app = FastAPI(title="Posterior", version="0.1.0",
                  description="Ask a database a question about the future.")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"ok": True, "decider": post.decider.name, "tables": len(post.schema.tables)}

    @app.get("/v1/schema")
    def schema() -> dict[str, Any]:
        return {"tables": {n: {"rows": t.n_rows, "key": t.pkey, "time": t.time_col, "links": t.fkeys,
                               "time_borrowed_from": t.derived_time[1] if t.derived_time else None,
                               "columns": {c.name: c.kind for c in t.columns.values()}}
                           for n, t in post.schema.tables.items()},
                "entities": post.schema.entity_candidates()}

    @app.post("/v1/formulate")
    def formulate(body: FormulateBody) -> dict[str, Any]:
        try:
            return svc.formulate(body.question)
        except ValueError as e:
            raise HTTPException(422, str(e)) from e

    @app.post("/v1/ask")
    def ask(body: AskBody) -> dict[str, Any]:
        try:
            return svc.ask(body.question, body.reading, body.live, body.auto, body.run_model).to_dict()
        except ValueError as e:
            raise HTTPException(422, str(e)) from e

    @app.post("/v1/tasks")
    def start(body: AskBody) -> dict[str, Any]:
        return {"id": svc.start(body.question, reading=body.reading, live=body.live, auto=body.auto,
                                run_model=body.run_model)}

    @app.get("/v1/tasks/{tid}")
    def task(tid: str) -> dict[str, Any]:
        if tid not in svc.tasks:
            raise HTTPException(404, f"no task {tid}")
        return svc.tasks[tid]

    @app.post("/v1/systemone")
    @app.post("/v1/decisions")
    def systemone(body: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(body.get("questions"), dict) or "state" not in body:
            raise HTTPException(422, "body needs 'state' and 'questions'")
        return svc.systemone(body)

    @app.get("/v1/models")
    def models() -> dict[str, Any]:
        return {"data": [{"id": "posterior", "object": "model", "owned_by": "posterior",
                          "decider": post.decider.name}]}

    app.state.service = svc
    return app


def serve(db: str, model: str | None = None, backend: str | None = None, runs: str = "runs",
          host: str = "127.0.0.1", port: int = 8787) -> None:
    import uvicorn

    post = Posterior.connect(db, model=model, work_root=runs, backend=backend)
    uvicorn.run(create_app(post), host=host, port=port)
