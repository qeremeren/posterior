"""Offline fixtures: a small synthetic marketplace and a scripted decision model."""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from posterior.decider import Choice, Noul


def make_marketplace(root: Path, n_sellers: int = 80, days: int = 540, seed: int = 0, leak: bool = False) -> Path:
    """Sellers who sell for a while and then stop; orders with a status and a delivery date filled later."""
    rng = np.random.default_rng(seed)
    start = datetime(2022, 1, 1)
    sellers = pd.DataFrame({
        "seller_id": [f"s{i:03d}" for i in range(n_sellers)],
        "seller_state": rng.choice(["SP", "RJ", "MG", "PR"], n_sellers),
    })
    customers = pd.DataFrame({"customer_id": [f"c{i:04d}" for i in range(400)],
                              "customer_state": rng.choice(["SP", "RJ", "MG"], 400)})
    life = {s: (rng.integers(0, days // 2), rng.integers(days // 3, days + 200)) for s in sellers.seller_id}
    orders, items = [], []
    oid = 0
    for d in range(days):
        day = start + timedelta(days=d)
        for s, (a, b) in life.items():
            if a <= d <= b and rng.random() < 0.25:
                oid += 1
                ts = day + timedelta(hours=int(rng.integers(0, 23)))
                delivered = ts + timedelta(days=int(rng.integers(2, 12)))
                status = "delivered" if delivered < start + timedelta(days=days) else "shipped"
                orders.append({"order_id": f"o{oid:06d}", "customer_id": f"c{rng.integers(0, 400):04d}",
                               "order_status": status, "order_purchase_timestamp": ts,
                               "order_delivered_customer_date": delivered if status == "delivered" else pd.NaT})
                items.append({"order_id": f"o{oid:06d}", "order_item_id": 1, "seller_id": s,
                              "price": float(np.round(rng.gamma(2, 30), 2))})
    orders = pd.DataFrame(orders)
    items = pd.DataFrame(items)
    if leak:  # snapshot columns computed from the whole history, future included
        last = items.merge(orders[["order_id", "order_purchase_timestamp"]]).groupby("seller_id")[
            "order_purchase_timestamp"].max()
        sellers["seller_last_order_date"] = sellers.seller_id.map(last)
        end = start + timedelta(days=days)
        sellers["seller_status"] = np.where(sellers.seller_last_order_date > end - timedelta(days=45), "active", "inactive")
    root.mkdir(parents=True, exist_ok=True)
    sellers.to_csv(root / "sellers.csv", index=False)
    customers.to_csv(root / "customers.csv", index=False)
    orders.to_csv(root / "orders.csv", index=False)
    items.to_csv(root / "order_items.csv", index=False)
    return root


class FakeDecider:
    """Answers by keyword rules, so tests are deterministic and need no model server."""

    name = "fake"
    model = "fake"
    base_url = "http://decider.invalid"
    api_key = "local"

    def __init__(self, prefer: dict[str, str] | None = None, noul: dict[str, float] | None = None):
        self.prefer = prefer or {}
        self.noul = noul or {}
        self.calls = 0

    def _choice(self, q: Choice) -> dict[str, float]:
        labels = list(q.options)
        pick = None
        for pattern, label in self.prefer.items():
            if re.search(pattern, q.instructions, re.I) and label in labels:
                pick = label
        if pick is None:
            pick = labels[0]
        rest = (1 - 0.9) / max(len(labels) - 1, 1)
        return {lab: (0.9 if lab == pick else rest) for lab in labels}

    def _noul(self, q: Noul) -> float:
        for pattern, p in self.noul.items():
            if re.search(pattern, q.instructions, re.I):
                return p
        return 0.1

    def ask(self, state, questions):
        self.calls += 1
        return {k: (self._choice(q) if isinstance(q, Choice) else self._noul(q)) for k, q in questions.items()}

    def ask_one(self, state, question):
        return self.ask(state, {"q": question})["q"]


@pytest.fixture()
def market(tmp_path: Path) -> Path:
    return make_marketplace(tmp_path / "market")


@pytest.fixture()
def leaky_market(tmp_path: Path) -> Path:
    return make_marketplace(tmp_path / "leaky", leak=True)


@pytest.fixture()
def churn_decider() -> FakeDecider:
    return FakeDecider(
        prefer={"subject of the question": "sellers", "kind of answer": "yes_or_no", "outcome does": "stops",
                "timestamp marks": "orders.order_purchase_timestamp"},
        noul={"came into existence": 0.9, "stop, churn, again": 0.9, "keeps being updated": 0.1,
              "filled in or changed": 0.2, "status": 0.85},
    )
