"""Tests: signature building, request validation, and endpoint behavior with a mocked LLM."""

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.main import DecisionRequest, app, build_signature, decisions

client = TestClient(app)


def full_request() -> dict:
    return {
        "state": "payouts failed",
        "questions": {
            "is_urgent": {"type": "boolean", "instructions": "urgent?"},
            "count": {"type": "integer", "instructions": "how many?"},
            "score": {"type": "number", "instructions": "score 0-10"},
            "summary": {"type": "string", "instructions": "one line"},
        },
    }


# --- offline: signature + validation ---


def test_signature_all_types_and_confidence_fields():
    sig = build_signature(DecisionRequest.model_validate(full_request()).questions)
    out = sig.output_fields
    assert set(out) == {
        "is_urgent", "count", "score", "summary",
        "is_urgent_confidence", "count_confidence", "score_confidence", "summary_confidence",
    }
    assert out["is_urgent"].annotation is bool
    assert out["count"].annotation is int
    assert out["score"].annotation is float
    assert out["summary"].annotation is str
    assert out["is_urgent_confidence"].annotation is float


def test_rejects_unknown_type():
    with pytest.raises(ValidationError):
        DecisionRequest.model_validate(
            {"state": "s", "questions": {"x": {"type": "noul", "instructions": "?"}}}
        )


def test_rejects_empty_questions():
    with pytest.raises(ValidationError):
        DecisionRequest.model_validate({"state": "s", "questions": {}})


def test_rejects_non_identifier_name():
    req = DecisionRequest.model_validate(
        {"state": "s", "questions": {"not a name": {"type": "boolean", "instructions": "?"}}}
    )
    with pytest.raises(HTTPException) as exc:
        decisions(req)
    assert exc.value.status_code == 422


# --- endpoint with a fake dspy.Predict (no LLM calls) ---


@pytest.fixture
def fake_predict(monkeypatch):
    holder = {}

    class FakePredict:
        def __init__(self, sig):
            holder["sig"] = sig
            holder["state"] = None

        def __call__(self, **kwargs):
            holder["state"] = kwargs.get("state")
            result = holder.get("result")
            if isinstance(result, Exception):
                raise result
            return result or {}

    monkeypatch.setattr("app.main.dspy.Predict", FakePredict)
    return holder


def test_root(fake_predict):
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["service"] == "jev-dspy"


def test_happy_path_types_and_confidence(fake_predict):
    fake_predict["result"] = {
        "is_urgent": True, "is_urgent_confidence": 0.92,
        "count": 3, "count_confidence": 0.8,
        "score": 7.5, "score_confidence": 0.7,
        "summary": "Escalate now", "summary_confidence": 0.9,
    }
    r = client.post("/api/v1/decisions", json=full_request())
    assert r.status_code == 200
    body = r.json()
    assert body["is_urgent"] == {"value": True, "confidence": 0.92}
    assert body["count"] == {"value": 3, "confidence": 0.8}
    assert body["score"] == {"value": 7.5, "confidence": 0.7}
    assert body["summary"] == {"value": "Escalate now", "confidence": 0.9}
    # state reaches the predictor; signature was built from these questions
    assert fake_predict["state"] == "payouts failed"
    assert "is_urgent" in fake_predict["sig"].output_fields


def test_confidence_clamped_to_unit_interval(fake_predict):
    fake_predict["result"] = {"is_urgent": True, "is_urgent_confidence": 1.5}
    r = client.post(
        "/api/v1/decisions",
        json={"state": "s", "questions": {"is_urgent": {"type": "boolean", "instructions": "?"}}},
    )
    assert r.json()["is_urgent"]["confidence"] == 1.0


def test_llm_failure_returns_502(fake_predict):
    fake_predict["result"] = RuntimeError("gateway down")
    r = client.post(
        "/api/v1/decisions",
        json={"state": "s", "questions": {"is_urgent": {"type": "boolean", "instructions": "?"}}},
    )
    assert r.status_code == 502
    assert "LLM call failed" in r.json()["detail"]


def test_incomplete_answer_returns_502(fake_predict):
    fake_predict["result"] = {"is_urgent": True}  # missing confidence field
    r = client.post(
        "/api/v1/decisions",
        json={"state": "s", "questions": {"is_urgent": {"type": "boolean", "instructions": "?"}}},
    )
    assert r.status_code == 502


def test_bad_request_returns_422(fake_predict):
    for payload in (
        {"state": "s", "questions": {"x": {"type": "noul", "instructions": "?"}}},
        {"state": "s", "questions": {}},
        {"state": "s", "questions": {"not a name": {"type": "boolean", "instructions": "?"}}},
        {"questions": {"x": {"type": "boolean", "instructions": "?"}}},  # no state
    ):
        r = client.post("/api/v1/decisions", json=payload)
        assert r.status_code == 422, payload
