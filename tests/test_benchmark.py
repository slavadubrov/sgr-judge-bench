import asyncio
import json
import math
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx

from judge_bench.adapters import HTTPJudge, parse_response, price, request_body, usage_of
from judge_bench.core import DIMENSIONS, InvalidOutput, validate_output, write_json, write_jsonl
from judge_bench.datasets import deduplicate_train, partition, rag_cases
from judge_bench.metrics import (
    calibration,
    choose_threshold,
    classification,
    fit_temperature,
    paired_bootstrap,
    source_error_bound,
    temperature_scale,
)
from judge_bench.runner import canaries, load_models, load_records, make_plan

ROOT = Path(__file__).resolve().parents[1]
MODELS = load_models(ROOT / "config/models.json")


class Adapters(unittest.TestCase):
    def test_no_reference_leakage_and_same_semantics(self):
        for model in MODELS.values():
            c = canaries()[0]
            altered = {
                **c,
                "gold": {"claim_relation": "SECRET_GOLD"},
                "reason": "SECRET_REASON",
                "metadata": {"gold": "SECRET_META"},
            }
            for profile in model["profiles"]:
                self.assertEqual(
                    request_body(model, c, profile), request_body(model, altered, profile)
                )
                self.assertNotIn("SECRET", json.dumps(request_body(model, altered, profile)))

    def test_reasoning_and_structured_modes(self):
        c = canaries()[0]
        self.assertEqual(request_body(MODELS["luna"], c, "D")["reasoning"], {"effort": "none"})
        strict = request_body(MODELS["deepseek-strict"], c, "P")
        self.assertTrue(strict["tools"][0]["function"]["strict"])
        self.assertEqual(strict["thinking"]["type"], "disabled")
        self.assertEqual(request_body(MODELS["glm-current"], c, "D")["thinking"]["type"], "enabled")

    def test_jev_noul_and_choice(self):
        c = canaries()[3]
        result = parse_response(
            MODELS["jev"],
            {"answers": {"unsupported_claim_present": {"type": "noul", "noul": 0.2}}},
            c,
            "P",
        )
        self.assertEqual(result["labels"]["unsupported_claim_present"], "no")
        result = parse_response(
            MODELS["jev"],
            {
                "answers": {
                    "claim_relation": {
                        "type": "choice",
                        "choice": "contradicted",
                        "confidence": 0.1,
                        "probabilities": {
                            "supported": 0.8,
                            "contradicted": 0.1,
                            "insufficient_evidence": 0.1,
                        },
                    }
                }
            },
            canaries()[0],
            "P",
        )
        self.assertEqual(result["labels"]["claim_relation"], "supported")
        self.assertNotEqual(result["probabilities"]["claim_relation"]["supported"], 0.1)

    def test_provider_parsers(self):
        c = canaries()[0]
        content = '{"claim_relation":"supported"}'
        self.assertEqual(
            parse_response(
                MODELS["luna"],
                {
                    "status": "completed",
                    "output": [
                        {"type": "message", "content": [{"type": "output_text", "text": content}]}
                    ],
                },
                c,
                "D",
            )["labels"]["claim_relation"],
            "supported",
        )
        for name in ("deepseek-json", "glm-current", "glm-control"):
            parse_response(
                MODELS[name],
                {"choices": [{"finish_reason": "stop", "message": {"content": content}}]},
                c,
                "D",
            )
        parse_response(
            MODELS["deepseek-strict"],
            {
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "message": {
                            "tool_calls": [
                                {"function": {"name": "submit_evaluation", "arguments": content}}
                            ]
                        },
                    }
                ]
            },
            c,
            "D",
        )
        for content in ("", "{}", "```json\n{}\n```", '{"claim_relation":"invented"}'):
            with self.assertRaises(InvalidOutput):
                parse_response(
                    MODELS["deepseek-json"],
                    {"choices": [{"message": {"content": content}}]},
                    c,
                    "D",
                )

    def test_probability_validation_and_rounding(self):
        spec = {"claim_relation": DIMENSIONS["claim_relation"]}
        for p in (
            {"supported": 0.8, "contradicted": 0.8, "insufficient_evidence": 0},
            {"supported": math.nan, "contradicted": 0.1, "insufficient_evidence": 0.1},
            {"supported": True, "contradicted": 0, "insufficient_evidence": 0},
        ):
            with self.assertRaises(InvalidOutput):
                validate_output({"claim_relation": p}, spec, "P")
        parsed = validate_output(
            {
                "claim_relation": {
                    "supported": 0.8,
                    "contradicted": 0.1,
                    "insufficient_evidence": 0.10001,
                }
            },
            spec,
            "P",
        )
        self.assertTrue(parsed["normalization"]["claim_relation"]["normalized"])

    def test_usage_and_pricing(self):
        u = usage_of(
            MODELS["deepseek-json"],
            {
                "usage": {
                    "prompt_tokens": 1000,
                    "prompt_cache_hit_tokens": 500,
                    "completion_tokens": 50,
                    "completion_tokens_details": {"reasoning_tokens": 20},
                }
            },
        )
        value = price(MODELS["deepseek-json"], u, "2026-09-19T12:00:00+00:00")
        self.assertAlmostEqual(value["usd"], (500 * 0.15 + 500 * 0.003 + 50 * 0.6) / 1e6)
        self.assertIsNone(price(MODELS["deepseek-json"], u, "2026-09-21T02:00:00+00:00")["usd"])
        self.assertAlmostEqual(
            price(MODELS["deepseek-json"], u, "2026-09-21T02:00:00+00:00", False)["usd"],
            value["usd"] * 2,
        )
        self.assertIsNone(usage_of(MODELS["jev"], {})["reasoning_tokens"])
        self.assertEqual(price(MODELS["glm-control"], {}, "2026-09-19T12:00:00+00:00")["usd"], 0)
        self.assertIsNone(
            price(
                MODELS["luna"],
                {"input_tokens": 100, "output_tokens": 10, "cached_input_tokens": 0},
                "2026-09-19T12:00:00+00:00",
            )["usd"]
        )

    def test_transport_raw_preserved_failure(self):
        async def run():
            async def handler(request):
                return httpx.Response(
                    200,
                    json={
                        "model": "deepseek-flash",
                        "choices": [{"message": {"content": "not json"}}],
                        "usage": {
                            "prompt_tokens": 10,
                            "prompt_cache_hit_tokens": 0,
                            "completion_tokens": 3,
                        },
                    },
                )

            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "local-test-secret"}):
                    result = await HTTPJudge(MODELS["deepseek-json"], client).evaluate(
                        canaries()[0], "D"
                    )
            self.assertEqual(result["status"], "parse_error")
            self.assertIn("not json", result["raw_output"])
            self.assertIsNotNone(result["cost"]["usd"])
            self.assertNotIn("local-test-secret", json.dumps(result))

        asyncio.run(run())


class DataAndMetrics(unittest.TestCase):
    def test_human_spans_include_implicit_true_and_source_join(self):
        source = {
            "source_id": "1",
            "task_type": "QA",
            "source_info": {"question": "Q", "passages": "C"},
        }
        row = {
            "id": "a",
            "source_id": "1",
            "split": "test",
            "response": "A",
            "labels": [{"implicit_true": True}],
        }
        cases, excluded = rag_cases([source], [row])
        self.assertFalse(excluded)
        self.assertEqual(cases[0]["gold"]["unsupported_claim_present"], "yes")
        self.assertEqual(cases[0]["input"]["context"], "C")

    def test_split_no_source_leakage(self):
        rows = []
        for i in range(30):
            for response in range(2):
                rows.append(
                    {
                        "id": f"{i}:{response}",
                        "group": str(i),
                        "input": {"context": str(i)},
                        "task": "ragtruth",
                        "dataset": "ragtruth",
                        "official_split": "train",
                        "gold": {"unsupported_claim_present": "no"},
                    }
                )
        a, _ = partition(rows)
        b, _ = partition(rows)
        self.assertEqual(a, b)
        for group in {r["group"] for r in a}:
            self.assertEqual(len({r["split"] for r in a if r["group"] == group}), 1)
        self.assertEqual({r["split"] for r in a}, {"development", "calibration", "routing"})

    def test_overlap_removes_train_only(self):
        rows = [
            {
                "group": "a",
                "official_split": "train",
                "input": {"context": "This is the same evidence."},
            },
            {
                "group": "b",
                "official_split": "test",
                "input": {"context": "THIS IS THE SAME EVIDENCE!"},
            },
        ]
        clean, overlaps = deduplicate_train(rows)
        self.assertEqual([r["group"] for r in clean], ["b"])
        self.assertEqual(len(overlaps), 1)

    def test_failures_are_false_negatives(self):
        score = classification(
            ["yes", "yes", "no", "no"], ["yes", None, "no", "yes"], ["no", "yes"]
        )
        self.assertEqual(score["accuracy"], 0.5)
        self.assertEqual(score["confusion_matrix"]["yes"]["invalid"], 1)
        self.assertAlmostEqual(score["macro_f1"], (0.5 + 2 / 3) / 2)
        self.assertEqual(score["false_pass_rate"], 0)

    def test_calibration_binary_and_multiclass(self):
        result = calibration(
            ["yes", "no"], [{"yes": 0.8, "no": 0.2}, {"yes": 0.3, "no": 0.7}], ["no", "yes"]
        )
        self.assertAlmostEqual(result["brier"], 0.065)
        self.assertEqual(len(result["bins"]), 15)
        multi = calibration(["a"], [{"a": 0.5, "b": 0.25, "c": 0.25}], ["a", "b", "c"])
        self.assertAlmostEqual(multi["brier"], 0.375)

    def test_temperature_and_source_bound(self):
        p = {"no": 0.8, "yes": 0.2}
        self.assertAlmostEqual(temperature_scale(p, 1)["no"], 0.8)
        result = fit_temperature(["no", "yes"], [p, p], ["no", "yes"])
        self.assertGreater(result["temperature"], 1)
        self.assertGreater(
            source_error_bound([False] * 6, ["one"] * 6)[
                "upper_95_probability_source_has_any_error"
            ],
            0.9,
        )

    def test_cascade_escalation_and_free_fallback(self):
        a = [
            {
                "status": "ok",
                "confidence": 0.99,
                "gold": "no",
                "prediction": "no",
                "cost": {"usd": 0.001},
            }
        ] * 10
        b = [{"cost": {"usd": 0}}] * 10
        self.assertIsNone(choose_threshold(a, b)["threshold"])
        b = [{"cost": {"usd": 0.01}}] * 10
        self.assertEqual(choose_threshold(a, b)["threshold"], 0.5)

    def test_paired_bootstrap_identical(self):
        result = paired_bootstrap(
            ["no", "yes"] * 3,
            ["no", "yes"] * 3,
            ["no", "yes"] * 3,
            ["a", "a", "b", "b", "c", "c"],
            ["no", "yes"],
            100,
        )
        self.assertEqual(result["macro_f1"]["ci95"], [0, 0])
        self.assertEqual(result["groups"], 3)

    def test_plan_and_interruption_keep_all_slots(self):
        models = {"jev": MODELS["jev"]}
        plan = make_plan(canaries(), models, "test")
        self.assertEqual(plan, make_plan(canaries(), models, "test"))
        with tempfile.TemporaryDirectory() as root:
            write_json(Path(root) / "manifest.json", {"phase": "test"})
            write_jsonl(Path(root) / "plan.jsonl", plan)
            _, rows = load_records(root)
            self.assertEqual(len(rows), 5)
            self.assertTrue(all(r["status"] == "not_attempted" for r in rows))


if __name__ == "__main__":
    unittest.main()


class RunnerIntegration(unittest.TestCase):
    def test_provider_concurrency_and_raw_plan_coverage(self):
        from judge_bench.runner import execute

        async def run():
            active = {}
            maximum = {}

            class StubJudge:
                def __init__(self, config, client, **kwargs):
                    self.config = config

                async def evaluate(self, case, profile):
                    provider = self.config["provider"]
                    active[provider] = active.get(provider, 0) + 1
                    maximum[provider] = max(maximum.get(provider, 0), active[provider])
                    await asyncio.sleep(0.001)
                    active[provider] -= 1
                    field = next(iter(case["gold"]))
                    return {
                        "status": "ok",
                        "error": None,
                        "model_returned": self.config["model"],
                        "request_count": 1,
                        "cost": {"usd": 0.001},
                        "usage": {},
                        "latency_s": 0.001,
                        "parsed": {"labels": {field: case["gold"][field]}, "probabilities": {}},
                        "raw_output": "unit-test fixture",
                    }

            models = {name: MODELS[name] for name in ["deepseek-json", "deepseek-strict", "jev"]}
            with tempfile.TemporaryDirectory() as root:
                with patch("judge_bench.runner.HTTPJudge", StubJudge), patch("builtins.print"):
                    await execute(
                        canaries(),
                        models,
                        Path(root) / "run",
                        phase="preflight",
                        budget=1,
                        location="offline-test",
                    )
                _, rows = load_records(Path(root) / "run")
                self.assertEqual(len(rows), 25)
                self.assertTrue(all(r["status"] == "ok" for r in rows))
                self.assertTrue(all(value == 1 for value in maximum.values()))
                completion = json.loads((Path(root) / "run/completion.json").read_text())
                self.assertAlmostEqual(completion["budget_accounted_usd"], 0.025)

        asyncio.run(run())

    def test_report_includes_failures_and_generates_artifacts(self):
        from judge_bench.report import report

        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            plans = make_plan(canaries(), {"jev": MODELS["jev"]}, "test")
            for p in plans:
                p["case"]["dataset"] = "ragtruth" if p["case"]["task"] == "ragtruth" else "anli-R1"
            write_json(root / "run/manifest.json", {"phase": "test"})
            write_jsonl(root / "run/plan.jsonl", plans)
            actual = []
            for p in plans:
                actual.append(
                    {
                        **p,
                        "status": "parse_error",
                        "error": "malformed_json",
                        "parsed": None,
                        "request_count": 1,
                        "usage": {},
                        "latency_s": 0.1,
                        "cost": {"usd": 0.001},
                    }
                )
            write_jsonl(root / "run/attempts.jsonl", actual)
            report([root / "run"], root / "results", resamples=100)
            result = json.loads((root / "results/metrics.json").read_text())
            self.assertEqual(result["results"]["test/ragtruth/jev/P"]["quality"]["accuracy"], 0)
            self.assertTrue((root / "results/report.md").exists())
            self.assertTrue((root / "results/summary.csv").exists())

    def test_candidate_order_survives_persistence(self):
        from judge_bench.core import dumps

        c = canaries()[0]
        d = json.loads(json.dumps(DIMENSIONS["claim_relation"]))
        order = list(reversed(d["labels"]))
        d["candidate_order"] = order
        c = {**c, "dimensions": {"claim_relation": d}}
        c = json.loads(dumps(c))
        body = request_body(MODELS["jev"], c, "P")
        self.assertEqual(list(body["questions"]["claim_relation"]["criteria"]), order)

    def test_repeat_plan_has_four_new_calls_per_case(self):
        cases = canaries()[:1]
        count = 0
        for block in (1, 2, 3):
            count += len(
                make_plan(cases, {"jev": MODELS["jev"]}, "repeat", [cases[0]["id"]], block)
            )
        self.assertEqual(count, 4)


class FullWorkflow(unittest.TestCase):
    def test_budgeted_workflow_offline(self):
        from judge_bench.core import digest
        from judge_bench.experiment import experiment
        from judge_bench.runner import code_hash

        async def run():
            class Stub:
                def __init__(self, config, client, **kwargs):
                    self.config = config

                async def evaluate(self, case, profile):
                    field = next(iter(case["gold"]))
                    gold = case["gold"][field]
                    labels = list(DIMENSIONS[field]["labels"])
                    probabilities = {
                        label: 0.9 if label == gold else 0.1 / (len(labels) - 1) for label in labels
                    }
                    return {
                        "status": "ok",
                        "error": None,
                        "model_returned": self.config["model"],
                        "request_count": 1,
                        "cost": {"usd": 0.00001 if self.config["adapter"] == "jev" else 0.0001},
                        "usage": {},
                        "latency_s": 0.001 if self.config["adapter"] == "jev" else 0.01,
                        "parsed": {
                            "labels": {field: gold},
                            "probabilities": {field: probabilities} if profile == "P" else {},
                        },
                        "raw_output": "offline workflow fixture",
                    }

            models = {name: MODELS[name] for name in ["jev", "luna"]}
            with tempfile.TemporaryDirectory() as root:
                root = Path(root)
                cases = []
                for split in ["development", "calibration", "routing", "test"]:
                    for c in canaries():
                        row = {
                            **c,
                            "id": split + c["id"],
                            "group": split + c["id"],
                            "split": split,
                            "dataset": "ragtruth" if c["task"] == "ragtruth" else "anli-R1",
                        }
                        cases.append(row)
                write_jsonl(root / "data/cases.jsonl", cases)
                write_jsonl(root / "data/probes.jsonl", [])
                write_json(root / "data/manifest.json", {"fixture": True})
                write_json(
                    root / "data/subsets.json",
                    {
                        "repeat_ids": [
                            c["id"]
                            for c in cases
                            if c["split"] == "test" and c["task"] == "ragtruth"
                        ]
                    },
                )
                write_json(
                    root / "admission.json",
                    {
                        "admission": {
                            name + ":" + profile: {"admitted": True, "config_hash": digest(config)}
                            for name, config in models.items()
                            for profile in config["profiles"]
                        },
                        "code_hash": code_hash(),
                    },
                )
                with patch("judge_bench.runner.HTTPJudge", Stub), patch("builtins.print"):
                    result = await experiment(
                        root / "data",
                        models,
                        root / "admission.json",
                        root / "experiment",
                        1,
                        0.001,
                        "offline-test",
                    )
                ledger = json.loads((root / "experiment/ledger.json").read_text())
                self.assertLessEqual(ledger["accounted_usd"], 1)
                self.assertEqual(len(ledger["completed_runs"]), 9)
                self.assertTrue(Path(result["report"]).exists())

        asyncio.run(run())
