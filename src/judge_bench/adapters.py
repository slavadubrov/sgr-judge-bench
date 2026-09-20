"""Provider request/response adapters; all use the same semantic validator."""

import asyncio
import json
import os
import time
from datetime import UTC, datetime
from typing import Protocol

import httpx

from .core import POLICY, InvalidOutput, dumps, rubric, schema_for, spec_for, validate_output


class Judge(Protocol):
    async def evaluate(self, case: dict, profile: str) -> dict: ...


def request_body(config, case, profile):
    spec = spec_for(case)
    # Gold labels, source metadata and annotator reasons NEVER enter a provider payload.
    state = case["input"]
    body = {"model": config["model"], **config["options"]}
    if config["adapter"] == "jev":
        questions = {}
        for field, d in spec.items():
            question = {"type": d["kind"], "instructions": POLICY + " " + d["question"]}
            if d["kind"] == "noul":
                question["instructions"] += " " + dumps(d["labels"])
            elif d["kind"] == "score":
                question["criteria"] = list(d["labels"].values())
            else:
                question["criteria"] = d["labels"]
            questions[field] = question
        return {**body, "state": state, "questions": questions}
    messages = [
        {"role": "system", "content": rubric(spec, profile)},
        {"role": "user", "content": dumps(state)},
    ]
    if config["adapter"] == "responses":
        return {
            **body,
            "input": messages,
            "max_output_tokens": config["max_output"],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "evaluation",
                    "strict": True,
                    "schema": schema_for(spec, profile),
                }
            },
        }
    body.update(messages=messages, max_tokens=config["max_output"], stream=False)
    if config["output_mode"] == "strict_tool":
        body["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": "submit_evaluation",
                    "description": "Submit the bounded judgment. No action is executed.",
                    "strict": True,
                    "parameters": schema_for(spec, profile),
                },
            }
        ]
        body["tool_choice"] = {"type": "function", "function": {"name": "submit_evaluation"}}
    else:
        body["response_format"] = {"type": "json_object"}
    return body


def parse_response(config, raw, case, profile):
    spec = spec_for(case)
    if config["adapter"] == "jev":
        answers = raw["answers"]
        if set(answers) != set(spec):
            raise InvalidOutput("schema:answer_fields")
        value = {}
        for field, d in spec.items():
            answer = answers[field]
            if answer.get("type") != d["kind"]:
                raise InvalidOutput("schema:primitive_type")
            value[field] = answer["noul"] if d["kind"] == "noul" else answer["probabilities"]
        parsed = validate_output(value, spec, "P")
        parsed["native_diagnostics"] = answers  # Confidence/Score are not p(correct).
        return parsed
    value = response_json(config, raw)
    return validate_output(value, spec, profile)


def response_json(config, raw):
    """Extract JSON once for final judgments and SGR stage schemas."""
    if config["adapter"] == "responses":
        if raw.get("status") != "completed":
            raise InvalidOutput("truncation_or_incomplete")
        text = "".join(
            c["text"]
            for item in raw.get("output", [])
            if item.get("type") == "message"
            for c in item.get("content", [])
            if c.get("type") == "output_text"
        )
    else:
        choice = raw["choices"][0]
        if choice.get("finish_reason") == "length":
            raise InvalidOutput("truncation")
        if config["output_mode"] == "strict_tool":
            calls = choice["message"].get("tool_calls", [])
            if len(calls) != 1 or calls[0]["function"]["name"] != "submit_evaluation":
                raise InvalidOutput("schema:tool_call")
            text = calls[0]["function"]["arguments"]
        else:
            text = choice["message"].get("content")
    if not text:
        raise InvalidOutput("empty_response")
    try:
        value = json.loads(text)
    except (ValueError, TypeError) as exc:
        raise InvalidOutput("malformed_json") from exc
    return value


def usage_of(config, raw):
    u = raw.get("usage", {})
    is_responses = config["adapter"] in {"responses", "jev"}
    inp = u.get("input_tokens" if is_responses else "prompt_tokens")
    out = u.get("output_tokens" if is_responses else "completion_tokens")
    details = u.get("input_tokens_details" if is_responses else "prompt_tokens_details") or {}
    out_details = (
        u.get("output_tokens_details" if is_responses else "completion_tokens_details") or {}
    )
    cached = u.get("prompt_cache_hit_tokens", details.get("cached_tokens"))
    return {
        "input_tokens": inp,
        "output_tokens": out,
        "cached_input_tokens": cached,
        "cache_write_tokens": details.get(
            "cache_write_tokens", details.get("cache_creation_tokens")
        ),
        "reasoning_tokens": out_details.get("reasoning_tokens"),
        "provider_cost_usd": u.get("cost"),
        "raw": u,
    }


def price(config, usage, started_at, holiday_status=None):
    rates = dict(config["rates"])
    tariff = "standard"
    dt = datetime.fromisoformat(started_at)
    multiplier = 1
    if config["provider"] == "DeepSeek":
        scheduled_peak = dt.weekday() < 5 and (1 <= dt.hour < 4 or 6 <= dt.hour < 10)
        if scheduled_peak and holiday_status is None:
            return {"usd": None, "reason": "Chinese_public_holiday_status_unknown", "rates": rates}
        multiplier = 2 if scheduled_peak and not holiday_status else 1
        tariff = "peak" if multiplier == 2 else "off_peak"
    inp, out, cache, write = (
        usage.get(k)
        for k in ("input_tokens", "output_tokens", "cached_input_tokens", "cache_write_tokens")
    )
    for key in rates:
        if rates[key] is not None:
            rates[key] *= multiplier
    if inp is not None and inp > config.get("long_context_threshold", float("inf")):
        for key in ("input", "cached_input", "cache_write"):
            if rates[key] is not None:
                rates[key] *= config["long_input_multiplier"]
        rates["output"] *= config["long_output_multiplier"]
        tariff += "_long_context"
    result = {
        "usd": None,
        "rates": rates,
        "tariff": tariff,
        "pricing_source": config["pricing_source"],
        "price_date": config["price_date"],
        "status": "published_rate_estimate_not_invoice",
    }
    if all(v == 0 for v in rates.values()):
        return {**result, "usd": 0.0, "all_uncached_usd": 0.0, "all_cached_usd": 0.0}
    if inp is None or (out is None and rates["output"] != 0):
        return {**result, "reason": "missing_billable_usage"}
    if any(type(v) not in (int, float) or v < 0 for v in (inp, out or 0, cache or 0, write or 0)):
        return {**result, "reason": "invalid_usage"}
    # Missing cache breakdown is never silently billed as zero cached tokens.
    if cache is None and rates["cached_input"] not in (None, rates["input"]):
        return {
            **result,
            "reason": "missing_cache_breakdown",
            "all_uncached_usd": (inp * rates["input"] + (out or 0) * rates["output"]) / 1e6,
        }
    if (cache or 0) + (write or 0) > inp:
        return {**result, "reason": "inconsistent_input_categories"}
    if write and rates["cache_write"] is None:
        return {**result, "reason": "unknown_cache_write_rate"}
    # Unknown write category cannot prove absence of surcharged writes on OpenAI.
    if write is None and config["provider"] == "OpenAI":
        return {
            **result,
            "reason": "cache_write_usage_not_exposed",
            "all_uncached_usd": (inp * rates["input"] + (out or 0) * rates["output"]) / 1e6,
            "usd_bounds": [
                (
                    (inp - (cache or 0)) * rates[r]
                    + (cache or 0) * rates["cached_input"]
                    + (out or 0) * rates["output"]
                )
                / 1e6
                for r in ("input", "cache_write")
            ],
        }
    total = (
        (inp - (cache or 0) - (write or 0)) * rates["input"]
        + (cache or 0) * (rates["cached_input"] or rates["input"])
        + (write or 0) * (rates["cache_write"] or 0)
        + (out or 0) * rates["output"]
    )
    return {
        **result,
        "usd": total / 1e6,
        "all_uncached_usd": (inp * rates["input"] + (out or 0) * rates["output"]) / 1e6,
        "all_cached_usd": None
        if rates["cached_input"] is None
        else (inp * rates["cached_input"] + (out or 0) * rates["output"]) / 1e6,
    }


class HTTPJudge:
    def __init__(self, config, client, *, timeout=30, holiday_status=None):
        self.config, self.client, self.timeout = config, client, timeout
        self.holiday_status = holiday_status
        self.calls = 0

    async def evaluate(self, case, profile):
        return await self.request(
            request_body(self.config, case, profile),
            lambda raw: parse_response(self.config, raw, case, profile),
        )

    async def request(self, body, parse):
        started = time.perf_counter()
        stamp = datetime.now(UTC).isoformat()
        record = {
            "started_at": stamp,
            "request": body,
            "raw_output": None,
            "parsed": None,
            "status": "pending",
            "error": None,
            "connection": "first" if self.calls == 0 else "reused_client",
            "usage": usage_of(self.config, {}),
            "model_returned": None,
            "request_count": 0,
        }
        key = os.environ.get(self.config["key_env"])
        if not key:
            record.update(status="unavailable", error="missing_key:" + self.config["key_env"])
        else:
            self.calls += 1
            record["request_count"] = 1
            try:
                async with asyncio.timeout(self.timeout):
                    response = await self.client.post(
                        self.config["endpoint"],
                        json=body,
                        headers={"Authorization": "Bearer " + key},
                        timeout=self.timeout,
                    )
                    record["http_status"] = response.status_code
                    record["request_id"] = response.headers.get(
                        "x-request-id", response.headers.get("request-id")
                    )
                    record["raw_output"] = response.text
                    try:
                        raw = response.json()
                    except ValueError as exc:
                        raise InvalidOutput("malformed_transport_json") from exc
                    record["usage"] = usage_of(self.config, raw)
                    record["model_returned"] = raw.get("model")
                    record["system_fingerprint"] = raw.get("system_fingerprint")
                    record["provider_returned"] = raw.get("provider")
                    response.raise_for_status()
                    record["parsed"] = parse(raw)
                    record["reasoning_content_present"] = any(
                        bool(c.get("message", {}).get("reasoning_content"))
                        for c in raw.get("choices", [])
                    )
                    record["status"] = "ok"
            except (TimeoutError, httpx.TimeoutException):
                record.update(status="timeout", error="timeout", right_censored=True)
            except httpx.HTTPStatusError as exc:
                record.update(status="http_error", error="http_" + str(exc.response.status_code))
            except httpx.RequestError as exc:
                record.update(status="transport_error", error=type(exc).__name__)
            except (InvalidOutput, KeyError, TypeError, IndexError, ValueError) as exc:
                record.update(
                    status="parse_error",
                    error=str(exc) if isinstance(exc, InvalidOutput) else type(exc).__name__,
                )
        record["latency_s"] = time.perf_counter() - started
        record["cost"] = price(self.config, record["usage"], stamp, self.holiday_status)
        return record
