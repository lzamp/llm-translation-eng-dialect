#!/usr/bin/env python3
"""Average human_score values across scored prediction files in data/results."""

import json
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "data" / "results"

def average_scores(path: Path) -> tuple[float, int]:
    scores = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        val = entry.get("human_score")
        if isinstance(val, (int, float)):
            scores.append(val)
    if not scores:
        return float("nan"), 0
    return sum(scores) / len(scores), len(scores)

def main():
    files = sorted(RESULTS_DIR.glob("*.jsonl"))
    if not files:
        print(f"No .jsonl files found in {RESULTS_DIR}")
        return

    print(f"{'File':<45} {'Avg score':>10} {'N':>6}")
    print("-" * 63)
    for path in files:
        avg, n = average_scores(path)
        if n == 0:
            print(f"{path.name:<45} {'no scores':>10}")
        else:
            print(f"{path.name:<45} {avg:>10.3f} {n:>6}")

if __name__ == "__main__":
    main()
