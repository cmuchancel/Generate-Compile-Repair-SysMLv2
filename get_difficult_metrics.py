import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Difficulty buckets based on ground-truth design length; scores read from Generated_from_Prompts.


SCORE_RE = re.compile(
    r"Score:\s*[*_\s]*\s*(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)\s*[*_\s]*",
    re.IGNORECASE | re.DOTALL,
)


def extract_score(payload: Dict) -> Optional[Tuple[float, float, float]]:
    """Return (score, numerator, denominator) if a Score line is present."""
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
    """Load precision/recall scores for a model id if both files exist and parse."""
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
    """Count lines in a SysML string, handling escaped newlines."""
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
    for idx, sample in enumerate(data, start=1):  # 1-based ids
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


if __name__ == "__main__":
    dataset_path = Path("DesignBench/dataset/sysml/dataset.json")
    scores_root = Path("Generated_from_Prompts")
    difficult2id = difficult_id(dataset_path)
    get_distribution(difficult2id)
    diff_metrics = get_metrics_various_difficulty(difficult2id, scores_root)
    result_path = Path("difficult_result.json")
    result_path.write_text(json.dumps(diff_metrics, ensure_ascii=False, indent=4), encoding="utf-8")
    print(f"Difficulty metrics saved to {result_path} (from {scores_root})")
