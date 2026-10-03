from datetime import datetime

import pandas as pd
import pytest

from posterior.audit import Auditor, allow_list
from posterior.clarify import agreement, clarify
from posterior.formulate import Formulator, parse_horizon
from posterior.schema import Schema
from posterior.spec import Filter, Spec, anchors_before, choose_splits, run_labels

from .conftest import FakeDecider


@pytest.mark.parametrize("text,days", [
    ("Which sellers will stop selling in the next 30 days?", 30),
    ("How much revenue next month?", 30),
    ("Who will churn next quarter?", 91),
    ("in 2 weeks", 14),
    ("over the next 3 months", 90),
    ("önümüzdeki 30 günde", 30),
    ("Which drivers will finish in the top 3?", None),
])
def test_parse_horizon(text, days):
    assert parse_horizon(text) == days


def test_schema_finds_keys_links_and_borrowed_time(market, churn_decider):
    s = Schema.load(market, decider=churn_decider)
    assert s.tables["sellers"].pkey == "seller_id"
    assert s.tables["orders"].pkey == "order_id"
    assert s.tables["order_items"].pkey is None
    assert s.tables["order_items"].fkeys == {"order_id": "orders", "seller_id": "sellers"}
    assert s.tables["orders"].fkeys == {"customer_id": "customers"}
    # Two timestamps on orders: the delivery date is filled later, the purchase time is the event.
    assert s.tables["orders"].time_col == "order_purchase_timestamp"
    assert s.tables["order_items"].derived_time == ("order_id", "orders", "order_purchase_timestamp")
    assert "order_purchase_timestamp" in s.tables["order_items"].columns
    assert s.links_to("sellers") == [("order_items", "seller_id")]
    assert "sellers" in s.entity_candidates()


def _churn_spec() -> Spec:
    return Spec("sellers", "seller_id", "order_items", "seller_id", "order_purchase_timestamp",
                "not_exists", 30, "recent")


GOLD_CHURN = """SELECT timestamp, seller_id,
    CAST(NOT EXISTS (SELECT 1 FROM order_items WHERE order_items.seller_id = sellers.seller_id
        AND order_purchase_timestamp > timestamp AND order_purchase_timestamp <= timestamp + INTERVAL '{timedelta}'
    ) AS INTEGER) AS churn
  FROM timestamp_df, sellers
  WHERE EXISTS (SELECT 1 FROM order_items WHERE order_items.seller_id = sellers.seller_id
        AND order_purchase_timestamp > timestamp - INTERVAL '{timedelta}' AND order_purchase_timestamp <= timestamp)"""


def test_compiled_churn_matches_hand_written_sql(market, churn_decider):
    s = Schema.load(market, decider=churn_decider)
    anchors = anchors_before(datetime(2023, 3, 1), 30, 6)
    ours = run_labels(s.con, _churn_spec().label_sql(), anchors, 30)
    gold = run_labels(s.con, GOLD_CHURN, anchors, 30)
    m = ours.merge(gold, on=["timestamp", "seller_id"], how="outer", indicator=True)
    assert len(ours) > 50
    assert (m["_merge"] == "both").all()
    assert (m["target"] == m["churn"]).all()


def test_every_operation_compiles_and_runs(market, churn_decider):
    s = Schema.load(market, decider=churn_decider)
    anchors = anchors_before(datetime(2023, 3, 1), 30, 3)
    base = dict(entity="sellers", entity_key="seller_id", event="order_items", link="seller_id",
                time_col="order_purchase_timestamp", horizon_days=30)
    cases = [("not_exists", "all", None, None), ("exists", "recent", None, None), ("count", "all", None, None),
             ("sum", "all", "price", None), ("mean", "in_window", "price", None),
             ("exists", "in_window", None, Filter("price", ">=", 100)), ("count", "in_window", None, Filter("price", "<", 20))]
    for op, pop, value, flt in cases:
        sp = Spec(op=op, population=pop, value=value, filter=flt, **base)
        assert sp.valid(), sp
        df = run_labels(s.con, sp.label_sql(), anchors, 30)
        assert list(df.columns) == ["timestamp", "seller_id", "target"], sp.describe()
        assert len(df) > 0, sp.describe()
        assert df["target"].notna().all()


def test_grammar_rejects_meaningless_combinations():
    base = dict(entity="sellers", entity_key="seller_id", event="order_items", link="seller_id",
                time_col="t", horizon_days=30)
    assert not Spec(op="not_exists", population="in_window", **base).valid()
    assert not Spec(op="exists", population="in_window", **base).valid()
    assert not Spec(op="sum", population="all", **base).valid()
    assert not Spec(op="mean", population="all", value="price", **base).valid()
    assert Spec(op="exists", population="in_window", filter=Filter("price", ">", 1), **base).valid()


def test_task_yaml_shape():
    t = _churn_spec().task_yaml(datetime(2023, 1, 1), datetime(2023, 1, 31))
    assert t["task_type"] == "binary_classification"
    assert t["timedelta"] == "30 days"
    assert t["entity_table"] == "sellers" and t["entity_col"] == "seller_id"
    assert "{timedelta}" in t["query"]


def test_splits_cut_off_the_thin_tail(market, churn_decider):
    s = Schema.load(market, decider=churn_decider)
    val, test = choose_splits(s.con, '"order_items"', "order_purchase_timestamp", 30)
    assert (test - val).days == 30
    end = s.con.execute('SELECT max(order_purchase_timestamp) FROM order_items').fetchone()[0]
    assert test <= pd.Timestamp(end).to_pydatetime()


def test_formulation_reads_a_churn_question(market, churn_decider):
    s = Schema.load(market, decider=churn_decider)
    f = Formulator(s, churn_decider).run("Which sellers will stop selling in the next 30 days?")
    best = f.best
    assert (best.entity, best.op, best.population, best.horizon_days) == ("sellers", "not_exists", "recent", 30)
    assert abs(sum(r.prob for r in f.readings) - 1) < 1e-6
    assert all(r.valid() for r in f.readings)
    assert any(t["slot"] == "entity" for t in f.trace)


def test_agreement_and_clarification(market, churn_decider):
    s = Schema.load(market, decider=churn_decider)
    a = _churn_spec()
    anchors = anchors_before(datetime(2023, 3, 1), 30, 4)
    assert agreement(s.con, a, a, anchors) == pytest.approx(1.0)
    b = Spec("sellers", "seller_id", "order_items", "seller_id", "order_purchase_timestamp", "exists", 30, "recent")
    assert agreement(s.con, a, b, anchors) == pytest.approx(0.0)  # the exact opposite labels
    a.prob, b.prob = 0.55, 0.45
    c = clarify(s.con, [a, b], anchors)
    assert c is not None and c.slot == "op" and c.readings == [0, 1]
    same = Spec("sellers", "seller_id", "order_items", "seller_id", "order_purchase_timestamp", "not_exists", 30, "recent")
    same.prob = 0.45
    assert clarify(s.con, [a, same], anchors) is None


def test_audit_drops_snapshot_columns_from_the_data_alone(leaky_market, monkeypatch):
    monkeypatch.delenv("TABPFN_TOKEN", raising=False)
    monkeypatch.setenv("POSTERIOR_TABPFN", "client")
    # A decision model that suspects nothing: the label-power check must catch the leaks by itself.
    d = FakeDecider(prefer={"timestamp marks": "orders.order_purchase_timestamp"},
                    noul={"came into existence": 0.9, ".": 0.05})
    s = Schema.load(leaky_market, decider=d)
    spec = _churn_spec()
    val, test = choose_splits(s.con, '"order_items"', "order_purchase_timestamp", 30)
    train = run_labels(s.con, spec.label_sql(), anchors_before(val, 30, 10), 30)
    audits = Auditor(s, d).run(spec, train)
    verdict = {(a.table, a.column): a.verdict for a in audits}
    assert verdict[("sellers", "seller_last_order_date")] == "drop"
    assert verdict[("sellers", "seller_status")] == "drop"
    assert verdict[("sellers", "seller_state")] == "keep"
    allowed = allow_list(s, audits)
    assert "seller_status" not in allowed["sellers"] and "seller_state" in allowed["sellers"]


def test_audit_follows_a_confident_meaning_prior(market):
    d = FakeDecider(prefer={"timestamp marks": "orders.order_purchase_timestamp"},
                    noul={"came into existence": 0.9, "order_status": 0.9, ".": 0.05})
    s = Schema.load(market, decider=d)
    audits = Auditor(s, d).run(_churn_spec(), pd.DataFrame(columns=["timestamp", "seller_id", "target"]))
    verdict = {(a.table, a.column): a.verdict for a in audits}
    assert verdict[("orders", "order_status")] == "drop"
    assert verdict[("order_items", "price")] == "keep"


def test_materialize_writes_relarena_database(market, churn_decider, tmp_path):
    s = Schema.load(market, decider=churn_decider)
    db = s.materialize(tmp_path / "data", columns={"sellers": ["seller_id"]})
    assert db["order_items"]["time_col"] == "order_purchase_timestamp"
    assert db["order_items"]["fkeys"] == {"order_id": "orders", "seller_id": "sellers"}
    assert db["sellers"]["columns"] == ["seller_id"]
    assert (tmp_path / "data" / "order_items.parquet").exists()
    back = pd.read_parquet(tmp_path / "data" / "order_items.parquet")
    assert "order_purchase_timestamp" in back.columns
