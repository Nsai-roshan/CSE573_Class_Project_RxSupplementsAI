"""Compare two evaluation reports using the PRD success-criterion formulas."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def score_change(baseline: float, candidate: float) -> float | None:
    """Return relative reduction, or None when the baseline is zero."""
    return None if baseline == 0 else (baseline - candidate) / baseline


def compare_reports(v1: dict[str, Any], v2: dict[str, Any]) -> dict[str, Any]:
    a, b = v1.get("summary", {}), v2.get("summary", {})
    h1, h2 = float(a.get("hallucination_rate", 0.0)), float(b.get("hallucination_rate", 0.0))
    c1, c2 = float(a.get("avg_cost_usd", 0.0)), float(b.get("avg_cost_usd", 0.0))
    sc1, sc2 = score_change(h1, h2), score_change(c1, c2)
    return {
        "SC-1": {"v1": h1, "v2": h2, "relative_reduction": sc1, "passed": sc1 is not None and sc1 >= 0.40},
        "SC-2": {"v1": c1, "v2": c2, "relative_reduction": sc2, "passed": sc2 is not None and sc2 >= 0.0},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("v1_report", type=Path)
    parser.add_argument("v2_report", type=Path)
    args = parser.parse_args()
    result = compare_reports(json.loads(args.v1_report.read_text()), json.loads(args.v2_report.read_text()))
    for name, data in result.items():
        print(f"{name}: v1={data['v1']:.6f} v2={data['v2']:.6f} relative_reduction={data['relative_reduction']} verdict={'PASS' if data['passed'] else 'FAIL'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
