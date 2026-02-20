#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

SCORE_RE = re.compile(
    r"Score:\s*[*_\s]*\s*(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)\s*[*_\s]*",
    re.IGNORECASE | re.DOTALL,
)


def detect_default_dataset_path() -> Path:
    candidates = [
        REPO_ROOT / "sysmbench_original_upstream" / "dataset" / "sysml" / "dataset.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def detect_default_scores_root() -> Path:
    candidates = [
        REPO_ROOT / "ai_agent" / "Generated_from_Prompts_AI_AGENT",
        REPO_ROOT / "api_loop" / "Generated_from_Prompts_API_LOOP",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def parse_args() -> argparse.Namespace:
    default_dataset = detect_default_dataset_path()
    default_scores = detect_default_scores_root()
    parser = argparse.ArgumentParser(description="Compute difficulty metrics from eval score JSONs.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=default_dataset,
        help="Path to SysMBench dataset.json (default: %(default)s).",
    )
    parser.add_argument(
        "--scores-root",
        type=Path,
        default=default_scores,
        help="Generated outputs root containing per-id score JSON files (default: %(default)s).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output JSON path. Default: <scores-root-parent>/difficult_result.json"
        ),
    )
    return parser.parse_args()


def extract_score(payload: Dict) -> Optional[Tuple[float, float, float]]:
    response = payload.get("response") or {}
    text = response.get("response_text") or ""
    text = text.replace("**\n", "** ").replace("**  ", "** ")
    match = SCORE_RE.search(text)
    if not match:
        return None
    num = float(match.group(1))
    denom = float(match.group(2))
    if denom == 0:
        return None
    return num / denom, num, denom


def load_scores_for_id(root: Path, model_id: int) -> Optional[Tuple[float, float]]:
    model_dir = root / str(model_id)
    p_path = model_dir / f"{model_id}_precision_gpt41.json"
    r_path = model_dir / f"{model_id}_recall_gpt41.json"
    if not (p_path.exists() and r_path.exists()):
        return None
    try:
        p_score = extract_score(json.loads(p_path.read_text(encoding="utf-8")))
        r_score = extract_score(json.loads(r_path.read_text(encoding="utf-8")))
    except json.JSONDecodeError:
        return None
    if not p_score or not r_score:
        return None
    return p_score[0], r_score[0]


def count_lines(code: str) -> int:
    if not code:
        return 0
    if "\n" not in code and "\\n" in code:
        code = code.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\r", "\n")
    code = code.replace("\r\n", "\n").replace("\r", "\n")
    return len(code.split("\n")) if code else 0


def difficult_id(dataset_path: Path) -> Dict[str, List[int]]:
    with dataset_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    result: Dict[str, List[int]] = {"1": [], "2": [], "3": [], "4": [], "5": []}
    for idx, sample in enumerate(data, start=1):
        sysm = sample["design"]
        line = count_lines(sysm)
        if line < 30:
            result["1"].append(idx)
        elif line < 60:
            result["2"].append(idx)
        elif line < 90:
            result["3"].append(idx)
        elif line < 120:
            result["4"].append(idx)
        else:
            result["5"].append(idx)
    return result


def get_distribution(result: Dict[str, List[int]]) -> None:
    for key, value in result.items():
        print(f"{key}:{len(value)}")


def get_metrics_various_difficulty(difficult2id: Dict[str, List[int]], scores_root: Path) -> Dict:
    diff_result = {}
    for bucket, sample_id_list in difficult2id.items():
        totals = {"precision": 0.0, "recall": 0.0}
        count = 0
        for sample_id in sample_id_list:
            pair = load_scores_for_id(scores_root, sample_id)
            if pair is None:
                continue
            prec, rec = pair
            totals["precision"] += prec
            totals["recall"] += rec
            count += 1
        if count == 0:
            continue
        diff_result[bucket] = {
            "precision": totals["precision"] / count,
            "recall": totals["recall"] / count,
            "count": count,
        }
    return diff_result


def main() -> None:
    args = parse_args()
    dataset_path = args.dataset.resolve()
    scores_root = args.scores_root.resolve()
    if args.output is None:
        result_path = scores_root.parent / "analysis" / "difficult_result.json"
    else:
        result_path = args.output.resolve()

    difficult2id = difficult_id(dataset_path)
    get_distribution(difficult2id)
    diff_metrics = get_metrics_various_difficulty(difficult2id, scores_root)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(diff_metrics, ensure_ascii=False, indent=4), encoding="utf-8")
    print(f"Difficulty metrics saved to {result_path} (from {scores_root})")


if __name__ == "__main__":
    main()
