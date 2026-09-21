"""Offline replay must preserve scores, pairing direction and every failed case."""

import csv
import gzip
import json
import tempfile
import unittest
from html.parser import HTMLParser
from pathlib import Path
from unittest.mock import patch

from judge_bench.adapters import HTTPJudge
from judge_bench.demo import build
from judge_bench.sgr import stage_request


class Demo(unittest.TestCase):
    def test_frozen_replay_without_provider(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(
                HTTPJudge, "request", side_effect=AssertionError("Offline demo called a provider")
            ),
        ):
            root = Path(directory) / "demo"
            page = Path(build(root)).read_text()
            provenance = json.loads((root / "provenance.json").read_text())
            self.assertTrue(
                all(Path(p).name == p for p in provenance["dataset_manifest"]["excluded_files"])
            )
            bundle = json.loads(
                gzip.decompress(
                    Path(__file__)
                    .resolve()
                    .parents[1]
                    .joinpath("src/judge_bench/article.json.gz")
                    .read_bytes()
                )
            )
            self.assertTrue(
                all("version_limit" not in model for model in bundle["manifest"]["models"].values())
            )
            summary = json.loads((root / "summary.json").read_text())
            pairs = json.loads((root / "jev-comparison.json").read_text())
            self.assertEqual(summary["unique_actual_requests"], 1200)
            self.assertEqual(summary["missing_records"], 0)
            self.assertEqual(len(summary["arms"]), 7)
            self.assertEqual(summary["arms"]["jev/direct"]["correct"], 110)
            self.assertEqual(summary["arms"]["luna/sgr"]["correct"], 111)
            self.assertEqual(summary["arms"]["luna/direct_guided"]["correct"], 102)
            self.assertNotIn("luna/direct", summary["arms"])
            self.assertAlmostEqual(pairs["luna/sgr"]["jev_minus_comparator"], -1 / 120)
            self.assertAlmostEqual(
                pairs["deepseek-json/direct_guided"]["jev_minus_comparator"], 10 / 120
            )
            self.assertNotIn("deepseek-json/direct", pairs)
            self.assertEqual(pairs["luna/sgr"]["jev_only_correct"], 6)
            self.assertEqual(pairs["luna/sgr"]["comparator_only_correct"], 7)
            self.assertGreater(pairs["terra/sgr"]["ci95"][1], 0)
            prompts = json.loads((root / "prompts.json").read_text())
            self.assertIn("CONJUNCTION", prompts["direct_guided"])
            self.assertEqual(set(prompts), {"direct_guided", "plan", "assess", "jev"})
            for stage in ("direct_guided", "plan"):
                request, _ = stage_request(
                    bundle["manifest"]["models"]["luna"], bundle["cases"][0]["input"], stage
                )
                self.assertEqual(prompts[stage], request["input"][0]["content"])
            self.assertTrue(
                all(
                    "request_id" not in call and "raw_output" not in call
                    for call in bundle["calls"]
                )
            )
            self.assertNotIn("short prompt", page)
            self.assertNotIn("historical", page)
            self.assertIn("Luna / Direct", page)
            with (root / "predictions.csv").open() as f:
                self.assertEqual(len(list(csv.DictReader(f))), 840)
            self.assertEqual(page.count('class="case"'), 120)
            self.assertEqual(page.count(" trace</summary>"), 480)
            self.assertNotIn("SGR", page)
            self.assertIn("Jev versus one-call structured-output LLM judges", page)
            self.assertEqual(page.count("Jev / Native decision trace</summary>"), 120)
            self.assertNotIn("<script src=", page)
            self.assertIn('label for="search"', page)
            HTMLParser().feed(page)
            with self.assertRaises(ValueError):
                build(root)


if __name__ == "__main__":
    unittest.main()
