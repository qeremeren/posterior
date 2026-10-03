"""MCP server: agents ask a question instead of preparing a DataFrame.

    claude mcp add posterior -- posterior mcp --db path/to/tables

Everything the engine prints is sent to stderr, so the stdio transport stays clean.
"""

from __future__ import annotations

from typing import Any

from .api import Service
from .engine import Posterior


def build(post: Posterior):  # noqa: ANN201
    from mcp.server.fastmcp import FastMCP

    svc = Service(post)
    mcp = FastMCP("posterior", instructions=(
        "Posterior answers questions about the future of a database, such as 'which sellers will stop "
        "selling in the next 30 days'. Call describe_database first, then formulate to see how the question "
        "is read. If a clarification comes back, show its options to the user and call ask with the chosen "
        "reading. ask runs TabPFN-3.5 and can take a few minutes; later calls for the same question are instant."))

    @mcp.tool()
    def describe_database() -> dict[str, Any]:
        """Tables, keys, links and event times found in the database, and which tables can be predicted for."""
        return {"tables": {n: {"rows": t.n_rows, "key": t.pkey, "event_time": t.time_col, "links": t.fkeys}
                           for n, t in post.schema.tables.items()},
                "entities": post.schema.entity_candidates()}

    @mcp.tool()
    def formulate(question: str) -> dict[str, Any]:
        """Read a question into ranked predictive tasks, and say whether the user must choose between them.

        No model is fitted, so this is fast.
        """
        d = svc.formulate(question)
        d.pop("decisions", None)
        return d

    @mcp.tool()
    def ask(question: str, reading: int | None = None, live: bool = True, top: int = 20) -> dict[str, Any]:
        """Answer a question about the future: readings, leakage audit, backtest and the top predictions.

        ``reading`` selects one of the readings returned by formulate (default: the most likely).
        """
        return svc.ask(question, reading=reading, live=live).to_dict(max_predictions=top)

    @mcp.tool()
    def predict(question: str, entity_ids: list[str], reading: int | None = None) -> dict[str, Any]:
        """Predictions for specific entities (for example seller ids) under a question already asked."""
        res = svc.ask(question, reading=reading)
        if res.predictions is None:
            return {"error": "no predictions", "status": res.status}
        key = res.spec.entity_key
        df = res.predictions[res.predictions[key].astype(str).isin([str(e) for e in entity_ids])]
        return {"entity_key": key, "predictions": df.to_dict(orient="records"),
                "missing": sorted(set(map(str, entity_ids)) - set(df[key].astype(str)))}

    @mcp.tool()
    def audit(question: str, reading: int | None = None) -> dict[str, Any]:
        """Which columns would leak the future for this question, with the evidence for each. No model fit."""
        res = svc.ask(question, reading=reading, run_model=False)
        return {"reading": res.spec.describe(),
                "columns": [a.to_dict() for a in res.audit]}

    return mcp


def run(db: str, model: str | None = None, backend: str | None = None, runs: str = "runs",
        http: bool = False, port: int = 8788) -> None:
    import contextlib
    import sys

    with contextlib.redirect_stdout(sys.stderr):
        post = Posterior.connect(db, model=model, work_root=runs, backend=backend)
    server = build(post)
    if http:
        server.settings.port = port
        server.run(transport="streamable-http")
    else:
        server.run(transport="stdio")
