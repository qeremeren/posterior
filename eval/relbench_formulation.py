"""Formulation accuracy on RelBench tasks, against Prior Labs' hand-written RelArena task files.

For each task, the input is RelBench's own one-line task description (the task class docstring).
Posterior reads it into ranked readings using the gold database schema, compiles the best reading
to label SQL, and both label queries run on the same four anchors before the task's validation
cutoff. Agreement = share of (anchor, entity) pairs present in both label sets x share of those
pairs with the same label (exact for yes/no, within 0.1% for numbers).

    python eval/relbench_formulation.py --data ~/data/relbench --model winnow:e4b

Gold files: eval/relbench_v1 (copied from github.com/PriorLabs/relarena, Apache-2.0).
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from posterior.clarify import clarify
from posterior.decider import make_decider
from posterior.formulate import Formulator
from posterior.schema import Schema
from posterior.spec import Spec, anchors_before, run_labels

HERE = Path(__file__).parent

# RelBench task docstrings, verbatim (typos included). driver-position is absent from relbench 2.x and
# is taken from relbench 1.x. The four Amazon docstrings name no window, so "over the next quarter"
# (the gold 91 days) is appended to them.
QUESTIONS = {
    "rel-f1/driver-dnf": "Predict the if each driver will DNF (not finish) a race in the next 1 month.",
    "rel-f1/driver-top3": "Predict if each driver will qualify in the top-3 for a race within the next 1 month.",
    "rel-f1/driver-position": "Predict the average finishing position of each driver all races in the next 2 months.",
    "rel-event/user-attendance": "Predict the number of events a user will go to in the next seven days 7 days.",
    "rel-event/user-repeat": "Predict whether a user will attend an event in the next 7 days if they have already attended an event in the last 14 days.",
    "rel-event/user-ignore": "Predict whether a user will ignore more than 2 event invitations in the next 7 days.",
    "rel-hm/user-churn": "Predict the churn for a customer (no transactions) in the next week.",
    "rel-hm/item-sales": "Predict the total sales for an article (the sum of prices of the associated transactions) in the next week.",
    "rel-amazon/user-churn": "Churn for a customer is 1 if the customer does not review any product in the time window, else 0. Time window: over the next quarter.",
    "rel-amazon/user-ltv": "LTV (life-time value) for a customer is the sum of prices of products that the customer reviews in the time window. Time window: over the next quarter.",
    "rel-amazon/item-churn": "Churn for a product is 1 if the product recieves at least one review in the time window, else 0. Time window: over the next quarter.",
    "rel-amazon/item-ltv": "LTV (life-time value) for a product is the numer of times the product is purchased in the time window multiplied by price. Time window: over the next quarter.",
    "rel-trial/study-outcome": "Predict if the trials in the next 1 year will achieve its primary outcome.",
    "rel-trial/study-adverse": "Predict the number of affected patients with severe advsere events/death for the trial in the next 1 year.",
    "rel-trial/site-success": "Predict the success rate of a trial site in the next 1 year.",
    "rel-stack/user-engagement": "Predict if a user will make any votes/posts/comments in the next 2 years.",
    "rel-stack/post-votes": "Predict the number of upvotes that an existing question will receive in the next 2 years.",
    "rel-stack/user-badge": "Predict if each user will receive in a new badge the next 2 years.",
    "rel-avito/ad-ctr": "Assuming the ad will be clicked in the next 4 days, predict the Click-Through-Rate (CTR) for each ad.",
    "rel-avito/user-visits": "Predict whether each customer will visit more than one ad in the next 4 days.",
    "rel-avito/user-clicks": "Predict whether the each customer will click on more than one ads in the next 4 days.",
}

# Gold tasks whose label the grammar can state exactly. The rest need a union of tables, a window
# function, a two-hop link, a set filter (status IN (...)) or a join to another table's value.
EXPRESSIBLE = {"rel-f1/driver-top3", "rel-f1/driver-position", "rel-event/user-ignore", "rel-hm/user-churn",
               "rel-hm/item-sales", "rel-amazon/user-churn", "rel-amazon/item-churn"}


def gold_labels(con, task: dict, anchors) -> pd.DataFrame:
    days = int(str(task["timedelta"]).split()[0])
    df = run_labels(con, task["query"], anchors, days)
    return df.rename(columns={task["time_col"]: "timestamp", task["target_col"]: "gold"})


def agreement(ours: pd.DataFrame, gold: pd.DataFrame, key: str, gold_key: str, task_type: str) -> dict:
    g = gold.rename(columns={gold_key: key})[["timestamp", key, "gold"]]
    o = ours[["timestamp", key, "target"]]
    g[key] = g[key].astype(str)
    o = o.assign(**{key: o[key].astype(str)})
    m = o.merge(g, on=["timestamp", key], how="outer", indicator=True)
    if m.empty:
        return {"coverage": 0.0, "label_match": 0.0, "agreement": 0.0, "n_gold": len(g), "n_ours": len(o)}
    both = m[m["_merge"] == "both"]
    cov = len(both) / len(m)
    if both.empty:
        same = 0.0
    elif task_type == "binary_classification":
        same = float((both["target"].astype(float) == both["gold"].astype(float)).mean())
    else:
        a, b = both["target"].astype(float), both["gold"].astype(float)
        same = float((np.abs(a - b) <= 1e-3 * np.maximum(1, np.abs(b))).mean())
    return {"coverage": round(cov, 4), "label_match": round(same, 4), "agreement": round(cov * same, 4),
            "n_gold": len(g), "n_ours": len(o)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="folder with one sub-folder of parquet tables per dataset")
    ap.add_argument("--model", default="winnow:e4b")
    ap.add_argument("--out", default=str(HERE / "results"))
    ap.add_argument("--only", nargs="*", default=None, help="dataset names to run")
    args = ap.parse_args()
    decider = make_decider(args.model, cache_path=Path("cache") / "decisions.jsonl")
    rows = []
    for ds_dir in sorted((HERE / "relbench_v1").iterdir()):
        ds = ds_dir.name
        data = Path(args.data).expanduser() / ds
        if (args.only and ds not in args.only) or not (data / ".done").exists():
            continue
        t0 = time.time()
        schema = Schema.from_relarena(ds_dir / "db.yaml", data)
        print(f"{ds}: loaded in {time.time() - t0:.0f}s", flush=True)
        for task_file in sorted(ds_dir.glob("*.yaml")):
            if task_file.name == "db.yaml":
                continue
            name = f"{ds}/{task_file.stem}"
            task = yaml.safe_load(task_file.read_text())
            question = QUESTIONS[name]
            days = int(str(task["timedelta"]).split()[0])
            val = pd.Timestamp(task["val_timestamp"]).to_pydatetime()
            anchors = anchors_before(val - timedelta(days=days), days, 4)
            row = {"task": name, "question": question, "expressible": name in EXPRESSIBLE,
                   "gold_entity": task["entity_table"], "gold_type": task["task_type"], "gold_days": days}
            t1 = time.time()
            calls0 = decider.calls
            try:
                f = Formulator(schema, decider).run(question)
                gold = gold_labels(schema.con, task, anchors)
                scored = []
                for i, r in enumerate(f.readings):
                    if r.entity != task["entity_table"]:
                        scored.append({"coverage": 0.0, "label_match": 0.0, "agreement": 0.0})
                        continue
                    ours = run_labels(schema.con, r.label_sql(), anchors, r.horizon_days)
                    scored.append(agreement(ours, gold, r.entity_key, task["entity_col"], r.task_type))
                best = f.readings[0]
                # What the user ends up with: the top reading, or the best option of the one clarification.
                clar = clarify(schema.con, f.readings, anchors)
                offered = clar.readings if clar else [0]
                row["clarified"] = bool(clar)
                row["with_clarification"] = max(scored[i]["agreement"] for i in offered)
                row.update({
                    "reading": best.describe(), "probability": round(best.prob, 3),
                    "entity_ok": best.entity == task["entity_table"],
                    "type_ok": best.task_type == task["task_type"],
                    "horizon_ok": best.horizon_days == days,
                    "top1": scored[0], "top1_agreement": scored[0]["agreement"],
                    "best_of_k": max(s["agreement"] for s in scored), "k": len(scored),
                    "readings": [r.to_dict() | {"prob": round(r.prob, 3)} for r in f.readings],
                })
            except Exception as e:  # noqa: BLE001
                row.update({"error": f"{type(e).__name__}: {e}"[:300], "top1_agreement": 0.0, "best_of_k": 0.0,
                            "with_clarification": 0.0, "clarified": False,
                            "entity_ok": False, "type_ok": False, "horizon_ok": False})
            row["seconds"] = round(time.time() - t1, 1)
            row["decisions"] = decider.calls - calls0
            rows.append(row)
            print(f"  {name:28s} top1={row['top1_agreement']:.2f} clarified={row['with_clarification']:.2f} "
                  f"best={row['best_of_k']:.2f} "
                  f"entity={row['entity_ok']} type={row['type_ok']} {row.get('reading', row.get('error'))}", flush=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"relbench_formulation_{args.model.replace(':', '_').replace('+', '_')}.json").write_text(
        json.dumps(rows, indent=2, default=str))
    df = pd.DataFrame(rows)
    if not df.empty:
        summary = {
            "model": args.model, "tasks": len(df),
            "entity_accuracy": round(df["entity_ok"].mean(), 3), "type_accuracy": round(df["type_ok"].mean(), 3),
            "horizon_accuracy": round(df["horizon_ok"].mean(), 3),
            "mean_top1_agreement": round(df["top1_agreement"].mean(), 3),
            "mean_with_clarification": round(df["with_clarification"].mean(), 3),
            "mean_best_of_k": round(df["best_of_k"].mean(), 3),
            "clarification_rate": round(df["clarified"].mean(), 3),
            "expressible_tasks": int(df["expressible"].sum()),
            "expressible_top1_agreement": round(df.loc[df["expressible"], "top1_agreement"].mean(), 3),
            "expressible_with_clarification": round(df.loc[df["expressible"], "with_clarification"].mean(), 3),
            "expressible_best_of_k": round(df.loc[df["expressible"], "best_of_k"].mean(), 3),
        }
        print(json.dumps(summary, indent=2))
        (out / f"relbench_formulation_{args.model.replace(':', '_').replace('+', '_')}_summary.json").write_text(
            json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
