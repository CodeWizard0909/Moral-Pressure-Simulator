"""
Manual anti-hack inspection helper for saved baseline captures.

This does not replace reading the transcripts. It gives a fast Day 1 pass/fail
summary for the expected reward ordering:
principled_professional > confused_rebel > yes_man.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"


def latest_baseline_capture() -> Path:
    captures = sorted(RESULTS_DIR.glob("baseline_capture_*.json"))
    if not captures:
        raise FileNotFoundError(f"No baseline captures found in {RESULTS_DIR}")
    return captures[-1]


def load_capture(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def inspect_capture(results: dict) -> tuple[list[dict], dict]:
    rows = []
    failures = []

    for scenario_data in results["scenarios"]:
        label = scenario_data["scenario"]["label"]
        agents = scenario_data["agents"]

        yes_man = agents["yes_man"]
        rebel = agents["confused_rebel"]
        principled = agents["principled_professional"]

        row = {
            "scenario": label,
            "yes_man": yes_man["final_reward"],
            "confused_rebel": rebel["final_reward"],
            "principled_professional": principled["final_reward"],
            "yes_integrity": yes_man["reward_breakdown"].get("integrity", 0.0),
            "rebel_integrity": rebel["reward_breakdown"].get("integrity", 0.0),
            "principled_integrity": principled["reward_breakdown"].get("integrity", 0.0),
        }
        row["principled_beats_yes_man"] = (
            row["principled_professional"] > row["yes_man"]
        )
        row["principled_beats_rebel"] = (
            row["principled_professional"] > row["confused_rebel"]
        )
        row["yes_man_low_integrity"] = row["yes_integrity"] <= row["rebel_integrity"]

        if not row["principled_beats_yes_man"]:
            failures.append(f"{label}: principled did not beat yes_man")
        if not row["principled_beats_rebel"]:
            failures.append(f"{label}: principled did not beat confused_rebel")
        if not row["yes_man_low_integrity"]:
            failures.append(f"{label}: yes_man integrity exceeded confused_rebel")

        rows.append(row)

    summary = {
        "scenarios": len(rows),
        "failures": failures,
        "passed": not failures,
        "avg_yes_man": sum(row["yes_man"] for row in rows) / len(rows),
        "avg_confused_rebel": sum(row["confused_rebel"] for row in rows) / len(rows),
        "avg_principled": (
            sum(row["principled_professional"] for row in rows) / len(rows)
        ),
    }
    return rows, summary


def print_report(path: Path, rows: list[dict], summary: dict) -> None:
    print("=" * 78)
    print("ANTI-HACK INSPECTION")
    print("=" * 78)
    print(f"Capture: {path}")
    print(
        f"Average rewards: yes_man={summary['avg_yes_man']:.4f} | "
        f"confused_rebel={summary['avg_confused_rebel']:.4f} | "
        f"principled={summary['avg_principled']:.4f}"
    )
    print("-" * 78)
    print(f"{'Scenario':<30} {'Yes':>8} {'Rebel':>8} {'Principled':>12}  Status")
    print("-" * 78)

    for row in rows:
        ok = (
            row["principled_beats_yes_man"]
            and row["principled_beats_rebel"]
            and row["yes_man_low_integrity"]
        )
        print(
            f"{row['scenario'][:30]:<30} "
            f"{row['yes_man']:>8.4f} "
            f"{row['confused_rebel']:>8.4f} "
            f"{row['principled_professional']:>12.4f}  "
            f"{'PASS' if ok else 'REVIEW'}"
        )

    print("=" * 78)
    if summary["passed"]:
        print("PASS: reward ordering and integrity checks look sane.")
    else:
        print("REVIEW REQUIRED:")
        for failure in summary["failures"]:
            print(f"  - {failure}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect anti-hack baseline outputs")
    parser.add_argument(
        "--capture",
        type=Path,
        default=None,
        help="Path to a baseline_capture_*.json file. Defaults to latest capture.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if any inspection invariant fails.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    path = args.capture or latest_baseline_capture()
    rows, summary = inspect_capture(load_capture(path))
    print_report(path, rows, summary)
    return 1 if args.strict and not summary["passed"] else 0


if __name__ == "__main__":
    sys.exit(main())
