import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# This version reads scores from the local Generated_from_Prompts structure
# (matching the run_sysml_gpt41_eval.py outputs) and aggregates by domain.


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


def get_domain_id(dataset_path: Path) -> Dict[str, List[int]]:
    with dataset_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    result: Dict[str, List[int]] = {}
    for idx, sample in enumerate(data, start=1):  # 1-based ids
        domain = sample["domain"]
        result.setdefault(domain, []).append(idx)
    return result


def get_metrics_various_domain(domain2sampleid: Dict[str, List[int]], scores_root: Path) -> Dict:
    domain_result = {}
    for domain, sample_id_list in domain2sampleid.items():
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
        domain_result[domain] = {
            "precision": totals["precision"] / count,
            "recall": totals["recall"] / count,
            "count": count,
        }
    return domain_result


if __name__ == "__main__":
    dataset_path = Path("DesignBench/dataset/sysml/dataset.json")
    scores_root = Path("Generated_from_Prompts")
    domain2sampleid = get_domain_id(dataset_path)
    result = get_metrics_various_domain(domain2sampleid, scores_root)
    result_path = Path("domain_result.json")
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=4), encoding="utf-8")
    print(f"Domain metrics saved to {result_path} (from {scores_root})")
