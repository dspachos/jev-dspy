"""jev-dspy: DSPy-powered, type-safe decision API backed by an OpenAI-compatible LLM gateway."""

import keyword
import os

import dspy
from fastapi import FastAPI, HTTPException
from typing import Literal
from pydantic import BaseModel

MODEL = os.environ.get("LLM_MODEL", "claude-4-5-haiku")
PY_TYPES: dict[str, type] = {"boolean": bool, "integer": int, "number": float, "string": str}


class Question(BaseModel):
    type: Literal["boolean", "integer", "number", "string"]
    instructions: str


class DecisionRequest(BaseModel):
    state: str
    questions: dict[str, Question]


class Answer(BaseModel):
    value: bool | int | float | str
    confidence: float


lm = dspy.LM(
    f"openai/{MODEL}",
    api_base=os.environ.get("LLM_BASE_URL"),
    api_key=os.environ.get("LLM_API_KEY"),
    temperature=0.0,
)
dspy.configure(lm=lm)

app = FastAPI(title="jev-dspy decisions API")


@app.get("/")
def root() -> dict:
    return {"service": "jev-dspy", "model": MODEL}


def build_signature(questions: dict[str, Question]) -> type[dspy.Signature]:
    """One dynamic signature per request: every question becomes a typed output field."""
    fields = {"state": (str, dspy.InputField(desc="The situation to analyze"))}
    for name, q in questions.items():
        fields[name] = (PY_TYPES[q.type], dspy.OutputField(desc=q.instructions))
        fields[f"{name}_confidence"] = (
            float,
            dspy.OutputField(desc=f"Confidence from 0.0 to 1.0 that your '{name}' answer is correct."),
        )
    return dspy.make_signature(
        fields, instructions="Given the situation, answer every question. Be decisive and accurate."
    )


@app.post("/api/v1/decisions")
def decisions(req: DecisionRequest) -> dict[str, Answer]:
    for name in req.questions:
        # ponytail: DSPy field names must be identifiers; add alias mapping if arbitrary keys are ever needed
        if not name.isidentifier() or keyword.iskeyword(name):
            raise HTTPException(422, f"question name {name!r} must be a valid identifier")

    try:
        pred = dspy.Predict(build_signature(req.questions))(state=req.state)
    except Exception as exc:
        raise HTTPException(502, f"LLM call failed: {exc}") from exc

    answers = {}
    for name in req.questions:
        try:
            answers[name] = Answer(
                value=pred[name],
                confidence=min(1.0, max(0.0, float(pred[f"{name}_confidence"]))),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise HTTPException(502, f"model returned an incomplete answer for {name!r}") from exc
    return answers
