#!/usr/bin/env python3
"""
Aggregate GPT-4.1 evaluation outputs.

This script scans JSON files produced by run_sysml_gpt41_eval.py, extracts the
precision/recall scores (looking for the `Score: X/Y` line), and reports
average precision, average recall, and average F1 across the selected models.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

SCORE_RE = re.compile(r"Score:\s*(\d+(?:\.\d+)?)/(\d+(?:\.\d+)?)", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize GPT-4.1 precision/recall scores.")
    default_root = detect_default_generated_root()
    parser.add_argument(
        "--root",
        type=Path,
        default=default_root,
        help="Base directory containing model subdirectories (default: %(default)s).",
    )
    parser.add_argument(
        "--start-id",
        type=int,
        default=1,
        help="First model id to include (default: %(default)s).",
    )
    parser.add_argument(
        "--end-id",
        type=int,
        default=151,
        help="Last model id to include (default: %(default)s).",
    )
    parser.add_argument(
        "--precision-suffix",
        default="_precision_gpt41.json",
        help="Filename suffix for precision results (default: %(default)s).",
    )
    parser.add_argument(
        "--recall-suffix",
        default="_recall_gpt41.json",
        help="Filename suffix for recall results (default: %(default)s).",
    )
    parser.add_argument(
        "--skip",
        type=int,
        nargs="*",
        default=[],
        help="Model ids to skip entirely.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-model logs; only print the final summary.",
    )
    return parser.parse_args()


def detect_default_generated_root() -> Path:
    candidates = [
        REPO_ROOT / "ai_agent" / "Generated_from_Prompts_AI_AGENT",
        REPO_ROOT / "api_loop" / "Generated_from_Prompts_API_LOOP",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def extract_score(payload: Dict) -> Optional[Tuple[float, float, float]]:
    """
    Return (score, numerator, denominator) if a Score line is present.
    """
    response = payload.get("response") or {}
    text = response.get("response_text") or ""
    match = SCORE_RE.search(text)
    if not match:
        return None
    num = float(match.group(1))
    denom = float(match.group(2))
    if denom == 0:
        return None
    return num / denom, num, denom


def load_json(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def mean_or_nan(values: List[float]) -> float:
    return float("nan") if not values else statistics.fmean(values)


def main() -> None:
    args = parse_args()
    precision_scores: List[float] = []
    recall_scores: List[float] = []
    f1_scores: List[float] = []

    skip = set(args.skip)

    for model_id in range(args.start_id, args.end_id + 1):
        if model_id in skip:
            if not args.quiet:
                print(f"[skip] {model_id}")
            continue

        model_dir = args.root / str(model_id)
        precision_path = model_dir / f"{model_id}{args.precision_suffix}"
        recall_path = model_dir / f"{model_id}{args.recall_suffix}"

        try:
            precision_data = extract_score(load_json(precision_path))
        except FileNotFoundError:
            if not args.quiet:
                print(f"[warn] Missing precision file: {precision_path}")
            continue
        except json.JSONDecodeError as exc:
            print(f"[warn] Invalid JSON in {precision_path}: {exc}")
            continue

        try:
            recall_data = extract_score(load_json(recall_path))
        except FileNotFoundError:
            if not args.quiet:
                print(f"[warn] Missing recall file: {recall_path}")
            continue
        except json.JSONDecodeError as exc:
            print(f"[warn] Invalid JSON in {recall_path}: {exc}")
            continue

        if not precision_data:
            if not args.quiet:
                print(f"[warn] Could not parse precision score in {precision_path}")
            continue
        if not recall_data:
            if not args.quiet:
                print(f"[warn] Could not parse recall score in {recall_path}")
            continue

        p_score, p_num, p_denom = precision_data
        r_score, r_num, r_denom = recall_data

        precision_scores.append(p_score)
        recall_scores.append(r_score)

        if p_score + r_score > 0:
            f1_scores.append(2 * p_score * r_score / (p_score + r_score))

        if not args.quiet:
            print(
                f"[ok] {model_id:>3}: precision={p_score:.3f} ({p_num}/{p_denom}), "
                f"recall={r_score:.3f} ({r_num}/{r_denom})"
            )

    avg_precision = mean_or_nan(precision_scores)
    avg_recall = mean_or_nan(recall_scores)
    avg_f1 = mean_or_nan(f1_scores)

    print("\n=== Summary ===")
    print(f"Models processed: {len(f1_scores)}")
    print(f"Average precision: {avg_precision:.4f}")
    print(f"Average recall:    {avg_recall:.4f}")
    print(f"Average F1:        {avg_f1:.4f}")


if __name__ == "__main__":
    main()
