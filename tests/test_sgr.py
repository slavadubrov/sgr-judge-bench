"""One focused regression suite: real source binding, failure handling and hybrid accounting."""

import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx

from judge_bench.adapters import HTTPJudge
from judge_bench.runner import load_models
from judge_bench.sgr import (
    aggregate,
    aggregate_native,
    materialize_checks,
    native_case,
    parse_stage,
    stage_request,
)
from judge_bench.tabfact import canaries, report, run

MODELS = load_models(Path(__file__).resolve().parents[1] / "config/tabfact.json")


class SGR(unittest.TestCase):
    def test_projection_and_semantic_guards(self):
        raw = canaries()[0]["input"]
        checks = [{"question": raw["claim"], "columns": ["c0", "c1"]}]
        views = materialize_checks(raw, checks)
        self.assertEqual(views["check_1"]["cells"]["r0:c1"], "assistant manager")
        result = {
            "coverage_complete": True,
            "missing_conditions": [],
            "results": {
                "check_1": {
                    "finding": "Ada has role assistant manager",
                    "evidence": ["r0:c0", "r0:c1"],
                    "status": "SUPPORTED",
                }
            },
        }
        self.assertEqual(aggregate(raw, checks, result), "ENTAILED")
        result["results"]["check_1"]["evidence"] = ["r0:c99"]
        with self.assertRaises(ValueError):
            aggregate(raw, checks, result)
        with self.assertRaises(ValueError):
            materialize_checks(raw, [{"question": "x", "columns": ["c1", "c1"]}])
        with self.assertRaises(ValueError):
            aggregate_native({"coverage": "no", "check_1": "SUPPORTED"})
        with self.assertRaises(ValueError):
            aggregate_native({"coverage": "yes", "check_1": "UNRESOLVED"})
        self.assertEqual(aggregate_native({"coverage": "yes", "check_1": "REFUTED"}), "REFUTED")
        for name in ("terra", "deepseek-json", "glm-current"):
            body, schema = stage_request(MODELS[name], raw, "assess", views)
            self.assertNotIn("gold", json.dumps(body))
            self.assertEqual(set(schema["properties"]["results"]["properties"]), {"check_1"})
        self.assertEqual(set(native_case(raw, views)["dimensions"]), {"coverage", "check_1"})

    def test_provider_error_retains_usage(self):
        async def check():
            body, schema = stage_request(MODELS["terra"], canaries()[0]["input"], "direct")
            async with httpx.AsyncClient(
                transport=httpx.MockTransport(
                    lambda _: httpx.Response(
                        200,
                        json={
                            "status": "completed",
                            "usage": {"input_tokens": 10, "output_tokens": 2},
                            "output": [
                                {
                                    "type": "message",
                                    "content": [
                                        {"type": "output_text", "text": '{"label":"WRONG"}'}
                                    ],
                                }
                            ],
                        },
                    )
                )
            ) as client:
                judge = HTTPJudge(MODELS["terra"], client)
                with patch.dict(os.environ, {MODELS["terra"]["key_env"]: "test"}):
                    record = await judge.request(
                        body, lambda raw: parse_stage(MODELS["terra"], raw, schema)
                    )
            self.assertEqual(record["status"], "parse_error")
            self.assertEqual(record["usage"]["input_tokens"], 10)

        asyncio.run(check())

    def test_end_to_end_and_shared_plan_accounting(self):
        async def fake(judge, body, parser):
            if judge.config["adapter"] == "jev":
                answers = {
                    k: (
                        {"type": "noul", "noul": 0.99}
                        if q["type"] == "noul"
                        else {
                            "type": "choice",
                            "probabilities": {
                                v: float(v in ("ENTAILED", "SUPPORTED")) for v in q["criteria"]
                            },
                        }
                    )
                    for k, q in body["questions"].items()
                }
                raw = {"answers": answers}
            else:
                properties = body["text"]["format"]["schema"]["properties"]
                if "label" in properties:
                    value = {"label": "ENTAILED"}
                elif "checks" in properties:
                    value = {
                        "checks": [
                            {"question": "Ada is an assistant manager.", "columns": ["c0", "c1"]}
                        ]
                    }
                else:
                    value = {
                        "coverage_complete": True,
                        "missing_conditions": [],
                        "results": {
                            "check_1": {
                                "finding": "Ada is assistant manager",
                                "evidence": ["r0:c0", "r0:c1"],
                                "status": "SUPPORTED",
                            }
                        },
                    }
                raw = {
                    "status": "completed",
                    "output": [
                        {
                            "type": "message",
                            "content": [{"type": "output_text", "text": json.dumps(value)}],
                        }
                    ],
                }
            return {
                "parsed": parser(raw),
                "status": "ok",
                "error": None,
                "request_count": 1,
                "cost": {"usd": 0.001},
                "latency_s": 0.1,
                "request": body,
                "raw_output": json.dumps(raw),
            }

        with tempfile.TemporaryDirectory() as directory, patch.object(HTTPJudge, "request", fake):
            root = Path(directory) / "run"
            result = asyncio.run(
                run(canaries()[:1], {k: MODELS[k] for k in ("terra", "jev")}, root, 1)
            )
            self.assertEqual(result["unique_actual_requests"], 5)
            self.assertEqual(result["arms"]["jev/hybrid"]["attributed_requests"], 2)
            self.assertEqual(result["arms"]["jev/hybrid"]["correct"], 1)
            self.assertEqual(report(root), result)
            self.assertEqual(result["missing_records"], 0)
            primary = asyncio.run(
                run(
                    canaries()[:1],
                    {k: MODELS[k] for k in ("terra", "jev")},
                    Path(directory) / "primary",
                    1,
                    include_hybrid=False,
                )
            )
            self.assertEqual(primary["unique_actual_requests"], 4)
            self.assertEqual(set(primary["arms"]), {"terra/direct", "terra/sgr", "jev/direct"})
            blocked = asyncio.run(
                run(
                    canaries()[:1],
                    {k: MODELS[k] for k in ("terra", "jev")},
                    Path(directory) / "blocked",
                    0.0000001,
                )
            )
            self.assertEqual(blocked["unique_actual_requests"], 0)
            self.assertEqual(blocked["missing_records"], 0)
            self.assertTrue(all(a["valid"] == 0 for a in blocked["arms"].values()))


if __name__ == "__main__":
    unittest.main()
