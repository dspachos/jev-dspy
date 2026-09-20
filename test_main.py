# Self-check: signature building and request validation (no LLM calls).
from fastapi import HTTPException
from pydantic import ValidationError

from app.main import DecisionRequest, build_signature, decisions


def check_signature_covers_all_types():
    req = DecisionRequest.model_validate(
        {
            "state": "s",
            "questions": {
                "is_urgent": {"type": "boolean", "instructions": "urgent?"},
                "count": {"type": "integer", "instructions": "how many?"},
                "score": {"type": "number", "instructions": "score 0-10"},
                "summary": {"type": "string", "instructions": "one line"},
            },
        }
    )
    sig = build_signature(req.questions)
    out = sig.output_fields
    assert set(out) == {"is_urgent", "count", "score", "summary",
                        "is_urgent_confidence", "count_confidence", "score_confidence", "summary_confidence"}
    assert out["is_urgent"].annotation is bool
    assert out["count"].annotation is int
    assert out["score"].annotation is float
    assert out["summary"].annotation is str


def check_rejects_bad_type():
    try:
        DecisionRequest.model_validate({"state": "s", "questions": {"x": {"type": "noul", "instructions": "?"}}})
    except ValidationError:
        return
    raise AssertionError("'noul' should be rejected")


def check_rejects_bad_name():
    req = DecisionRequest.model_validate({"state": "s", "questions": {"not a name": {"type": "boolean", "instructions": "?"}}})
    try:
        decisions(req)
    except HTTPException as e:
        assert e.status_code == 422
        return
    raise AssertionError("invalid identifier should be rejected")


if __name__ == "__main__":
    check_signature_covers_all_types()
    check_rejects_bad_type()
    check_rejects_bad_name()
    print("ok")
