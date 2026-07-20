"""Shared llama.cpp OpenAI-compatible client for clean + eval."""
from __future__ import annotations

import json

import convention_spec as spec

try:
    import requests
except ImportError:
    requests = None


def require_server(port: int, hint: str) -> None:
    """Fail early with `hint` if the model server is not up."""
    if requests is None:
        raise SystemExit("The `requests` package is missing. Run: make setup")
    models_url = f"http://localhost:{port}/v1/models"
    try:
        requests.get(models_url, timeout=3)
    except requests.exceptions.RequestException:
        raise SystemExit(hint.format(port=port))


def call_model(
    record: dict,
    *,
    model_name: str = "qwen3-0.6b-cleaner",
    port: int = 8080,
    timeout: int = 120,
) -> dict | None:
    """POST one record to llama-server; return parsed JSON or None."""
    if requests is None:
        raise RuntimeError("`requests` not installed; --live needs it")
    url = f"http://localhost:{port}/v1/chat/completions"
    payload = {
        "model": model_name,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": spec.system_prompt("mdm_record")},
            {"role": "user", "content": json.dumps(record, ensure_ascii=False)},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "mdm_record",
                "schema": spec.BLOCK_SCHEMAS["mdm_record"],
            },
        },
    }
    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    try:
        return json.loads(r.json()["choices"][0]["message"]["content"])
    except (json.JSONDecodeError, KeyError):
        return None
