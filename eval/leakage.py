"""Leakage audit experiment on Olist with injected leaks.

Four seller columns are computed from the whole history, future included:
last order date, active/inactive status, lifetime item count and lifetime revenue. They are added to
the sellers table, and the audit runs three ways: decision-model meaning only, data evidence only, and
both. Then TabPFN-3.5 is fitted twice on the leaky database: once with every column, once after the
audit. The original columns are compared with Prior Labs' hand-curated allow-list for Olist.

    python eval/leakage.py --olist data/olist --model winnow:e4b [--fit]
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import timedelta
from pathlib import Path

import pandas as pd

from posterior.audit import Auditor
from posterior.decider import make_decider
from posterior.engine import Posterior
from posterior.schema import Schema
from posterior.spec import anchors_before, choose_splits, run_labels

QUESTION = "Which sellers will stop selling in the next 30 days?"
INJECTED = {"seller_last_order_date", "seller_status", "seller_lifetime_items", "seller_lifetime_revenue"}
# Columns Prior Labs left out of its Olist example (examples/olist_database.yaml in tabpfn-rel) among
# the tables it used: post-purchase order fields and review text. Everything else there was kept.
PRIOR_LABS_DROPPED = {("orders", "order_status"), ("orders", "order_approved_at"),
                      ("orders", "order_delivered_carrier_date"), ("orders", "order_delivered_customer_date"),
                      ("orders", "order_estimated_delivery_date"), ("order_reviews", "review_comment_title"),
                      ("order_reviews", "review_comment_message"), ("order_reviews", "review_creation_date"),
                      ("order_items", "shipping_limit_date")}
PRIOR_LABS_TABLES = {"sellers", "customers", "products", "orders", "order_items", "order_reviews"}


def make_leaky(src: Path, dst: Path) -> Path:
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.glob("olist_*.csv"):
        if "sellers" not in f.name:
            shutil.copy(f, dst / f.name)
    shutil.copy(src / "product_category_name_translation.csv", dst)
    orders = pd.read_csv(src / "olist_orders_dataset.csv", parse_dates=["order_purchase_timestamp"])
    items = pd.read_csv(src / "olist_order_items_dataset.csv").merge(
        orders[["order_id", "order_purchase_timestamp"]], on="order_id")
    g = items.groupby("seller_id")
    sellers = pd.read_csv(src / "olist_sellers_dataset.csv")
    end = items["order_purchase_timestamp"].max()
    last = g["order_purchase_timestamp"].max()
    sellers["seller_last_order_date"] = sellers.seller_id.map(last)
    sellers["seller_status"] = (sellers["seller_last_order_date"] >= end - timedelta(days=60)).map(
        {True: "active", False: "inactive"})
    sellers["seller_lifetime_items"] = sellers.seller_id.map(g.size()).fillna(0).astype(int)
    sellers["seller_lifetime_revenue"] = sellers.seller_id.map(g["price"].sum()).fillna(0).round(2)
    sellers.to_csv(dst / "olist_sellers_dataset.csv", index=False)
    return dst


def score(audits, mode: str) -> dict:
    drop = {(a.table, a.column) for a in audits if a.verdict == "drop"}
    inj = {("sellers", c) for c in INJECTED}
    tp = len(drop & inj)
    fp_orig = sorted(f"{t}.{c}" for t, c in drop - inj)
    orig = [(a.table, a.column) for a in audits if (a.table, a.column) not in inj and a.table in PRIOR_LABS_TABLES]
    agree = sum(((t, c) in drop) == ((t, c) in PRIOR_LABS_DROPPED) for t, c in orig)
    return {"mode": mode, "injected_caught": f"{tp}/{len(inj)}", "recall": tp / len(inj),
            "agreement_with_prior_labs_allow_list": round(agree / max(1, len(orig)), 3),
            "dropped_original_columns": fp_orig,
            "columns": [a.to_dict() for a in audits]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--olist", default="data/olist")
    ap.add_argument("--model", default="winnow:e4b")
    ap.add_argument("--fit", action="store_true", help="also fit TabPFN-3.5 with and without the audit")
    ap.add_argument("--out", default="eval/results")
    args = ap.parse_args()
    leaky = make_leaky(Path(args.olist), Path(args.olist).parent / "olist-leaky")
    decider = make_decider(args.model, cache_path="cache/decisions.jsonl")
    schema = Schema.load(leaky, decider=decider)
    post = Posterior(schema, decider, work_root="runs/leakage")
    f = post.formulator.run(QUESTION)
    spec = f.best
    val, test = choose_splits(schema.con, f'"{spec.event}"', spec.time_col, spec.horizon_days)
    h = timedelta(days=spec.horizon_days)
    train = run_labels(schema.con, spec.label_sql(), anchors_before(val - h, spec.horizon_days, 24), spec.horizon_days)
    results = {"question": QUESTION, "reading": spec.describe(), "modes": []}
    for mode, auditor in [("meaning only", Auditor(schema, decider, use_data=False)),
                          ("data only", Auditor(schema, decider, use_meaning=False)),
                          ("both", Auditor(schema, decider))]:
        r = score(auditor.run(spec, train), mode)
        results["modes"].append(r)
        print(f"{mode:13s} injected caught {r['injected_caught']}  agreement with Prior Labs allow-list "
              f"{r['agreement_with_prior_labs_allow_list']:.0%}  dropped originals: {r['dropped_original_columns']}")
    if args.fit:
        for label, use_audit in [("no audit", False), ("with audit", True)]:
            res = post.ask(QUESTION, live=False, audit=use_audit)
            results[f"tabpfn_{label.replace(' ', '_')}"] = res.learn
            print(f"TabPFN-3.5 {label}: {res.learn['metrics']}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "leakage.json").write_text(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
