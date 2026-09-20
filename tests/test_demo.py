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
            self.assertEqual(summary["unique_actual_requests"], 1596)
            self.assertEqual(summary["missing_records"], 0)
            self.assertEqual(len(summary["arms"]), 10)
            self.assertEqual(summary["arms"]["jev/direct"]["correct"], 108)
            self.assertEqual(summary["arms"]["luna/sgr"]["correct"], 114)
            self.assertEqual(summary["arms"]["jev/hybrid"]["correct"], 55)
            self.assertAlmostEqual(pairs["luna/sgr"]["jev_minus_comparator"], -0.05)
            self.assertAlmostEqual(pairs["deepseek-json/direct"]["jev_minus_comparator"], 0.05)
            self.assertEqual(pairs["luna/sgr"]["jev_only_correct"], 4)
            self.assertEqual(pairs["luna/sgr"]["comparator_only_correct"], 10)
            self.assertAlmostEqual(pairs["luna/sgr"]["ci95"][0], -13 / 120)
            self.assertTrue(all(p["ci95"][0] < 0 < p["ci95"][1] for p in pairs.values()))
            with (root / "predictions.csv").open() as f:
                self.assertEqual(len(list(csv.DictReader(f))), 1200)
            self.assertEqual(page.count('class="case"'), 120)
            self.assertEqual(page.count(" trace</summary>"), 1200)
            self.assertNotIn("<script src=", page)
            self.assertIn('label for="search"', page)
            HTMLParser().feed(page)
            with self.assertRaises(ValueError):
                build(root)


if __name__ == "__main__":
    unittest.main()
