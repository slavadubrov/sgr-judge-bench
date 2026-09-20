"""Render all saved cases and traces; no web dependencies."""

import argparse

from judge_bench.review import render

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run")
    p.add_argument("out")
    args = p.parse_args()
    render(args.run, args.out)
