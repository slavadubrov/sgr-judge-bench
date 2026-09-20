"""Pilot gates must stop before scaling a broken adapter."""

import importlib.util
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "pilot", Path(__file__).parents[1] / "scripts/pilot.py"
)
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


class PilotGateTests(unittest.TestCase):
    def rows(self):
        return [
            {
                "judge": "judge",
                "case": {
                    "id": str(i),
                    "task": "ragtruth",
                    "gold": {"unsupported_claim_present": "yes" if i % 2 else "no"},
                },
                "status": "ok",
                "parsed": {
                    "labels": {
                        "unsupported_claim_present": ("no" if i % 2 else "yes")
                        if i < 12
                        else ("yes" if i % 2 else "no")
                    }
                },
            }
            for i in range(60)
        ]

    def test_varied_imperfect_outputs_pass_but_pathologies_stop(self):
        rows = self.rows()
        self.assertTrue(pilot.sanity(rows)["continue"])
        for row in rows[:13]:
            row.update(status="parse_error", parsed=None)
        self.assertFalse(pilot.sanity(rows)["continue"])
        rows = self.rows()
        for row in rows:
            row["parsed"]["labels"]["unsupported_claim_present"] = "no"
        self.assertFalse(pilot.sanity(rows)["continue"])
        for row in rows:
            row["parsed"]["labels"] = row["case"]["gold"].copy()
        self.assertFalse(pilot.sanity(rows)["continue"])
        for row in rows:
            row["parsed"]["labels"]["unsupported_claim_present"] = (
                "yes" if row["case"]["gold"]["unsupported_claim_present"] == "no" else "no"
            )
        self.assertFalse(pilot.sanity(rows)["continue"])

    def test_one_bad_judge_stops_everyone(self):
        good = self.rows()
        bad = [
            {**r, "judge": "broken", "status": "parse_error", "parsed": None} for r in self.rows()
        ]
        self.assertFalse(pilot.sanity(good + bad)["continue"])


if __name__ == "__main__":
    unittest.main()
