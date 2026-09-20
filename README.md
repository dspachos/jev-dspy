# jev-dspy

> [!WARNING]
> Experimental project — under active development, API and behavior may change without notice.

A type-safe, question-answering API built on [DSPy](https://dspy.ai) and backed by any
OpenAI-compatible LLM gateway.

Send a situation plus a set of typed questions, and get back a JSON decision for each —
parsed to the type you asked for, with a confidence score.

```json
{
  "is_urgent":   {"value": true, "confidence": 0.92},
  "severity":    {"value": 4,    "confidence": 0.88},
  "next_action": {"value": "Escalate to payment support immediately", "confidence": 0.9}
}
```

## Quickstart

```bash
# Requires: LLM_BASE_URL and LLM_API_KEY in your environment
uv run uvicorn app.main:app --reload --port 8000
```

The server is now at `http://localhost:8000`, with interactive docs at
[`http://localhost:8000/docs`](http://localhost:8000/docs).

## Configuration

| Variable       | Purpose                                  | Default              |
|----------------|------------------------------------------|----------------------|
| `LLM_BASE_URL` | Gateway base URL (OpenAI-compatible)     | — *(required)*       |
| `LLM_API_KEY`  | Gateway API token                        | — *(required)*       |
| `LLM_MODEL`    | Model name on the gateway                | `claude-4-5-haiku`   |

The app fails fast at request time with a `502` if the gateway is unreachable or the
credentials are rejected.

## API

### `POST /api/v1/decisions`

**Request**

| Field       | Type                          | Description                                    |
|-------------|-------------------------------|------------------------------------------------|
| `state`     | `string`                      | The situation to analyze                       |
| `questions` | `map[string, Question]`       | Each question gets one typed answer            |

Each `Question` has:

| Field          | Type     | Description                          |
|----------------|----------|--------------------------------------|
| `type`         | `string` | `boolean` \| `integer` \| `number` \| `string` |
| `instructions` | `string` | Natural-language question for the model |

**Example**

```bash
curl -s http://localhost:8000/api/v1/decisions \
  -H "Content-Type: application/json" \
  -d '{
    "state": "My payouts have failed for three days and nobody has replied.",
    "questions": {
      "is_urgent":   {"type": "boolean", "instructions": "Does this need urgent attention?"},
      "severity":    {"type": "integer", "instructions": "Severity from 1 (minor) to 5 (critical)"},
      "next_action": {"type": "string",  "instructions": "The single best next action, max 8 words"}
    }
  }' | python3 -m json.tool
```

**Response** — JSON only, one entry per question, preserving your question names:

```json
{
  "is_urgent":   {"value": true, "confidence": 0.92},
  "severity":    {"value": 4,    "confidence": 0.88},
  "next_action": {"value": "Escalate to payment support immediately with documentation", "confidence": 0.9}
}
```

- `value` is coerced to the requested type by DSPy's typed-output parsing — a `boolean`
  question comes back as a real JSON boolean, `integer` as a real number, and so on.
- `confidence` is the model's self-reported certainty, clamped to `[0.0, 1.0]`.

### Rules & errors

| Status | Meaning                                                                 |
|--------|-------------------------------------------------------------------------|
| `422`  | Invalid request: unknown `type`, missing fields, or a question name that is not a valid identifier (names become DSPy signature fields, so `is_urgent` is fine but `my question` is not) |
| `502`  | The LLM call failed (gateway down, bad credentials) or returned an incomplete/unparseable answer |

### `GET /`

Service metadata: `{"service": "jev-dspy", "model": "claude-4-5-haiku"}`.

## How it works

Each request compiles into a dynamic [DSPy signature](https://dspy.ai/learn/programming/signatures/):
every question becomes a typed output field plus a paired `_confidence` field. DSPy handles
prompt construction, output parsing and type coercion; the app just maps the parsed
prediction back onto your original question names.

```
POST /api/v1/decisions
        │
        ▼
build_signature(questions)   →  state -> is_urgent: bool, is_urgent_confidence: float, ...
        │
        ▼
dspy.Predict(sig)(state)     →  typed, parsed prediction (claude-4-5-haiku via LiteLLM)
        │
        ▼
{name: {value, confidence}}  →  JSON response
```

## Roadmap

The long-term objective is to fine-tune a small language model on a specific domain and
build a complete, end-to-end model-to-decision pipeline. The work is organized into the
following phases.

| Phase | Objective | Status |
|-------|-----------|--------|
| **1. Baseline** | Typed decision API served by a general-purpose hosted model via DSPy, establishing the request/response contract and the confidence-reporting behavior. | ✅ Delivered |
| **2. Dataset construction** | Curate a domain-specific corpus of `(state, questions, answers)` examples, including distillation from larger models and human review, forming the training and evaluation splits. | ⏳ Planned |
| **3. Evaluation harness** | Define quantitative metrics (type-coercion accuracy, answer agreement, calibration of confidence scores) and automate them as regression gates. | ⏳ Planned |
| **4. Fine-tuning** | Fine-tune a small open model on the domain dataset, with acceptance criteria derived from the Phase 3 harness. | ⏳ Planned |
| **5. Self-hosted serving** | Expose the fine-tuned model behind the same OpenAI-compatible gateway interface, so the API remains unchanged for consumers. | ⏳ Planned |
| **6. End-to-end pipeline** | Productionize the full model-to-decision flow: automated retraining, prompt/program optimization via DSPy teleprompters, monitoring, and rollout. | ⏳ Planned |

Each phase is expected to land incrementally on top of the existing API surface; no phase
introduces breaking changes to consumers without prior notice.

## Development

```bash
uv run python test_main.py   # offline self-check: signature building + validation
```

Project layout:

```
app/main.py      # FastAPI app, DSPy wiring, the endpoint
test_main.py     # self-checks (no LLM calls)
```
