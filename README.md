# jev-dspy

Type-safe question-answering API built on [DSPy](https://dspy.ai), backed by the Amazee AI LiteLLM gateway.

## Env vars

| Variable | Purpose | Default |
|---|---|---|
| `AMAZEEAI_BASE_URL` | LiteLLM base URL (OpenAI-compatible) | — (required) |
| `AMAZEEAI_API_KEY` | LiteLLM token | — (required) |
| `AMAZEEAI_MODEL` | Model name | `claude-4-5-haiku` |

## Run

```bash
uv run uvicorn app.main:app --reload --port 8000
```

Interactive docs at `http://localhost:8000/docs`.

## API

`POST /api/v1/decisions`

```bash
curl http://localhost:8000/api/v1/decisions \
  -H "Content-Type: application/json" \
  -d '{
    "state": "My payouts have failed for three days and nobody has replied.",
    "questions": {
      "is_urgent": {"type": "boolean", "instructions": "Does this need urgent attention?"}
    }
  }'
```

Response (JSON only):

```json
{"is_urgent": {"value": true, "confidence": 0.92}}
```

- `type`: `boolean` | `integer` | `number` | `string` — answers are parsed to that type; anything else is a 422.
- Question names must be valid identifiers (they become DSPy signature fields).
- Every answer carries a `confidence` between 0.0 and 1.0.
- LLM failures surface as `502` with a JSON detail.

## Tests

```bash
uv run python test_main.py
```
