"""posterior ask | formulate | schema | serve | mcp"""

from __future__ import annotations

import argparse
import json
import sys


def _print_result(d: dict, top: int) -> None:
    print(f"\nQuestion: {d['question']}\n")
    print("Readings")
    for r in d["readings"]:
        mark = "→" if r["index"] == d["chosen"] else " "
        print(f"  {mark} {r['probability']:.2f}  {r['description']}")
    c = d.get("clarification")
    if c:
        print(f"\nClarification (agreement {c['agreement']:.0%}): {c['question']}")
        for i, o in zip(c["readings"], c["options"]):
            print(f"    [{i}] {o}")
    a = d.get("audit") or {}
    if a.get("dropped") is not None:
        print(f"\nLeakage audit: kept {a['kept']} columns, dropped {len(a['dropped'])}")
        for x in a["dropped"]:
            print(f"    - {x['table']}.{x['column']}: {x['reason']}")
    b = d.get("backtest")
    if b:
        print(f"\nBacktest ({b['backend']}, {b['test_rows']} rows at {b['test_timestamp']})")
        print("    model     " + "  ".join(f"{k}={v:.3f}" for k, v in b["metrics"].items()))
        for name, m in (b.get("baselines") or {}).items():
            print(f"    {name:9s} " + "  ".join(f"{k}={v:.3f}" for k, v in m.items()))
    if d.get("predictions"):
        print(f"\nPredictions (top {min(top, len(d['predictions']))} of {d['n_predictions']})")
        for row in d["predictions"][:top]:
            print("    " + "  ".join(f"{k}={v}" for k, v in row.items()))
    print(f"\n{d['status']} in {d['seconds']}s")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="posterior", description="Ask a database a question about the future.")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--db", required=True, help="folder of CSV/Parquet tables, or a DuckDB file")
        sp.add_argument("--model", default=None, help="decision model, e.g. winnow:e4b or winnow:e4b+laya")
        sp.add_argument("--backend", default=None, choices=["client", "local"], help="TabPFN backend")
        sp.add_argument("--runs", default="runs", help="working directory for compiled tasks")

    a = sub.add_parser("ask", help="formulate, audit, learn and predict")
    common(a)
    a.add_argument("question")
    a.add_argument("--reading", type=int, default=None, help="use this reading instead of the most likely")
    a.add_argument("--no-live", action="store_true", help="skip the live forecast (one fewer TabPFN fit)")
    a.add_argument("--no-model", action="store_true", help="stop after the audit")
    a.add_argument("--strict", action="store_true", help="stop and ask when readings disagree")
    a.add_argument("--json", action="store_true")
    a.add_argument("--top", type=int, default=10)

    f = sub.add_parser("formulate", help="show the readings and any clarification, no TabPFN")
    common(f)
    f.add_argument("question")
    f.add_argument("--json", action="store_true")

    s = sub.add_parser("schema", help="show the inferred keys, links and event times")
    common(s)

    sv = sub.add_parser("serve", help="HTTP API (/v1/ask, /v1/systemone, ...)")
    common(sv)
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8787)

    m = sub.add_parser("mcp", help="MCP server over stdio (or --http)")
    common(m)
    m.add_argument("--http", action="store_true", help="serve streamable HTTP instead of stdio")
    m.add_argument("--port", type=int, default=8788)

    args = p.parse_args(argv)
    import contextlib

    from .engine import Posterior

    out = sys.stdout
    # Libraries print progress to stdout; keep it for the answer (and valid JSON).
    quiet = contextlib.redirect_stdout(sys.stderr)

    if args.cmd == "serve":
        from .api import serve

        serve(args.db, model=args.model, backend=args.backend, runs=args.runs, host=args.host, port=args.port)
        return 0
    if args.cmd == "mcp":
        from .mcp_server import run

        run(args.db, model=args.model, backend=args.backend, runs=args.runs, http=args.http, port=args.port)
        return 0

    with quiet:
        post = Posterior.connect(args.db, model=args.model, work_root=args.runs, backend=args.backend)
    if args.cmd == "schema":
        for t in post.schema.tables.values():
            print(f"{t.name}: key={t.pkey} time={t.time_col} links={t.fkeys}"
                  + (f" (time borrowed from {t.derived_time[1]})" if t.derived_time else ""))
        print("\nentities:", ", ".join(post.schema.entity_candidates()))
        return 0
    if args.cmd == "formulate":
        with contextlib.redirect_stdout(sys.stderr):
            fm, clar, (val, test) = post.formulate(args.question)
        from .engine import Result

        d = Result(args.question, "formulated", fm.readings, 0, clar,
                   splits={"val": val.isoformat(), "test": test.isoformat()}).to_dict()
        d.pop("audit")
        print(json.dumps(d, indent=2, default=str)) if args.json else _print_result(d, 0)
        return 0
    with contextlib.redirect_stdout(sys.stderr):
        res = post.ask(args.question, reading=args.reading, live=not args.no_live, auto=not args.strict,
                       run_model=not args.no_model)
    d = res.to_dict(max_predictions=max(args.top, 50))
    with contextlib.redirect_stdout(out):
        print(json.dumps(d, indent=2, default=str)) if args.json else _print_result(d, args.top)
    return 0


if __name__ == "__main__":
    sys.exit(main())
