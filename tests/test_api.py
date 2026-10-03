import pandas as pd
import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from posterior.api import Service, create_app  # noqa: E402
from posterior.engine import Posterior, Result  # noqa: E402
from posterior.schema import Schema  # noqa: E402
from posterior.spec import Spec  # noqa: E402


@pytest.fixture()
def client(market, churn_decider, tmp_path):
    post = Posterior(Schema.load(market, decider=churn_decider), churn_decider, work_root=tmp_path / "runs")
    return TestClient(create_app(post))


def test_health_and_schema(client):
    assert client.get("/health").json()["ok"] is True
    s = client.get("/v1/schema").json()
    assert s["tables"]["order_items"]["time_borrowed_from"] == "orders"
    assert "sellers" in s["entities"]


def test_formulate_returns_ranked_readings(client):
    r = client.post("/v1/formulate", json={"question": "Which sellers will stop selling in the next 30 days?"})
    assert r.status_code == 200
    d = r.json()
    assert d["readings"][0]["op"] == "not_exists"
    assert d["readings"][0]["population"] == "recent"
    assert d["decisions"]


def test_ask_without_model_runs_the_audit(client):
    r = client.post("/v1/ask", json={"question": "Which sellers will stop selling in the next 30 days?",
                                     "run_model": False})
    d = r.json()
    assert d["status"] == "formulated"
    assert d["audit"]["kept"] > 0


def test_systemone_answers_future_questions_from_predictions(client, monkeypatch):
    svc: Service = client.app.state.service
    spec = Spec("sellers", "seller_id", "order_items", "seller_id", "order_purchase_timestamp", "not_exists", 30, "recent")
    fake = Result("q", "done", [spec], 0, predictions=pd.DataFrame({"seller_id": ["s001", "s002"], "prediction": [0.81, 0.12]}))
    monkeypatch.setattr(svc, "ask", lambda *a, **k: fake)
    monkeypatch.setattr(svc, "_is_future", lambda text: " will " in f" {text.lower()} ")
    forwarded = {}

    class Resp:
        def __init__(self, body):
            self.body = body

        def raise_for_status(self):
            pass

        def json(self):
            return self.body

    def fake_post(url, json, headers, timeout):
        forwarded.update(json["questions"])
        return Resp({"model": "fake", "answers": {k: {"type": "noul", "noul": 0.5} for k in json["questions"]}})

    monkeypatch.setattr("posterior.api.httpx.post", fake_post)
    body = {"model": "posterior", "state": {"seller_id": "s001", "note": "late shipments lately"},
            "questions": {"churn": {"type": "noul", "instructions": "Will this seller stop selling in the next 30 days?"},
                          "angry": {"type": "noul", "instructions": "Does the note sound like a complaint?"}}}
    d = client.post("/v1/systemone", json=body).json()
    assert d["answers"]["churn"] == {"type": "noul", "noul": 0.81}
    assert list(d["answers"]) == ["churn", "angry"]
    assert list(forwarded) == ["angry"]


def test_systemone_rejects_bad_body(client):
    assert client.post("/v1/systemone", json={"questions": {}}).status_code == 422
