"""Client for decision models that speak TypeSafe's ``/v1/systemone`` wire format.

Ollaya serves open decision models (winnow, clef, kev, laya, ...) on this format, and TypeSafe's
Jev speaks it natively, so one client covers both. Answers come back as probabilities:
a ``Choice`` gives ``{label: p}``, a ``Noul`` gives ``p(statement is true)``.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Union

import httpx

DEFAULT_URL = "http://127.0.0.1:11435"
DEFAULT_MODEL = "winnow:e4b"


@dataclass(frozen=True)
class Choice:
    """Pick one option. ``options`` maps a label to its description."""

    instructions: str
    options: dict[str, str]

    def wire(self) -> dict[str, Any]:
        return {"type": "choice", "instructions": self.instructions, "criteria": dict(self.options)}


@dataclass(frozen=True)
class Noul:
    """Probability that a statement is true."""

    instructions: str
    true: str | None = None
    false: str | None = None

    def wire(self) -> dict[str, Any]:
        q: dict[str, Any] = {"type": "noul", "instructions": self.instructions}
        if self.true or self.false:
            q["criteria"] = {"true": self.true, "false": self.false}
        return q


Question = Union[Choice, Noul]
Answer = Union[dict[str, float], float]


class DeciderError(RuntimeError):
    pass


class _DiskCache:
    """Append-only JSONL cache so repeated runs and evaluations are reproducible and free."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._data: dict[str, Any] = {}
        if path.exists():
            for line in path.read_text().splitlines():
                if line.strip():
                    row = json.loads(line)
                    self._data[row["k"]] = row["v"]

    def get(self, key: str) -> Any:
        return self._data.get(key)

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._data[key] = value
            with self.path.open("a") as f:
                f.write(json.dumps({"k": key, "v": value}) + "\n")


class Decider:
    """One decision model behind a ``/v1/systemone`` endpoint."""

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        cache_path: str | Path | None = None,
        timeout: float = 180.0,
    ):
        self.model = model or os.environ.get("POSTERIOR_DECIDER_MODEL", DEFAULT_MODEL)
        self.base_url = (base_url or os.environ.get("POSTERIOR_DECIDER_URL", DEFAULT_URL)).rstrip("/")
        self.api_key = api_key or os.environ.get("POSTERIOR_DECIDER_KEY", "local")
        self._client = httpx.Client(timeout=timeout)
        self._cache = _DiskCache(Path(cache_path)) if cache_path else None
        self.calls = 0
        self.cache_hits = 0

    @property
    def name(self) -> str:
        return self.model

    def ask(self, state: Any, questions: dict[str, Question]) -> dict[str, Answer]:
        body = {
            "model": self.model,
            "state": state,
            "questions": {k: q.wire() for k, q in questions.items()},
        }
        key = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
        if self._cache is not None:
            hit = self._cache.get(key)
            if hit is not None:
                self.cache_hits += 1
                return hit
        r = self._client.post(
            f"{self.base_url}/v1/systemone",
            json=body,
            headers={"Authorization": f"Bearer {self.api_key}"},
        )
        self.calls += 1
        if r.status_code >= 400:
            raise DeciderError(f"{self.model} returned {r.status_code}: {r.text[:400]}")
        out: dict[str, Answer] = {}
        for k, a in r.json()["answers"].items():
            if a["type"] == "noul":
                out[k] = float(a["noul"])
            else:
                out[k] = {label: float(p) for label, p in a["probabilities"].items()}
        if self._cache is not None:
            self._cache.set(key, out)
        return out

    def ask_one(self, state: Any, question: Question) -> Answer:
        return self.ask(state, {"q": question})["q"]


class Ensemble:
    """Average the answers of several decision models. Disagreement lowers confidence."""

    def __init__(self, members: list[Decider]):
        if not members:
            raise ValueError("an ensemble needs at least one decision model")
        self.members = members

    @property
    def name(self) -> str:
        return "+".join(m.name for m in self.members)

    @property
    def calls(self) -> int:
        return sum(m.calls for m in self.members)

    def ask(self, state: Any, questions: dict[str, Question]) -> dict[str, Answer]:
        answers = [m.ask(state, questions) for m in self.members]
        out: dict[str, Answer] = {}
        for k in questions:
            first = answers[0][k]
            if isinstance(first, float):
                out[k] = sum(a[k] for a in answers) / len(answers)  # type: ignore[misc]
            else:
                out[k] = {
                    label: sum(a[k][label] for a in answers) / len(answers)  # type: ignore[index]
                    for label in first
                }
        return out

    def ask_one(self, state: Any, question: Question) -> Answer:
        return self.ask(state, {"q": question})["q"]


def make_decider(spec: str | None = None, cache_path: str | Path | None = None) -> Decider | Ensemble:
    """``"winnow:e4b"`` gives one model, ``"winnow:e4b+clef"`` an ensemble of both."""
    spec = spec or os.environ.get("POSTERIOR_DECIDER_MODEL", DEFAULT_MODEL)
    names = [s for s in spec.split("+") if s]
    if len(names) == 1:
        return Decider(names[0], cache_path=cache_path)
    return Ensemble([Decider(n, cache_path=cache_path) for n in names])


def top(answer: dict[str, float]) -> tuple[str, float]:
    label = max(answer, key=answer.__getitem__)
    return label, answer[label]
