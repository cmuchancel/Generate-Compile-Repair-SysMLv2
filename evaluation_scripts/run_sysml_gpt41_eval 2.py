#!/usr/bin/env python3
"""
Batch evaluator that calls GPT-4.1 on generated SysML models.

For every model directory (default: Generated_from_Prompts/1..75) we:
  * Load the generated SysML file (e.g., 12/12.sysml)
  * Load the reference SysML file (e.g., 12/12_groundtruth.sysml or from samples)
  * Format the precision/recall prompts from sysm-eval-p.txt and sysm-eval-r.txt
  * Call GPT-4.1 with each prompt
  * Save the prompt + raw response JSON alongside the models

Usage example:
    OPENAI_API_KEY=... python run_sysml_gpt41_eval.py \\
        --generated-root ai_agent/Generated_from_Prompts_AI_AGENT \\
        --reference-root ai_agent/Generated_from_Prompts_AI_AGENT \\
        --precision-prompt evaluation_scripts/Evaluation_Prompts/sysm-eval-p.txt \\
        --recall-prompt evaluation_scripts/Evaluation_Prompts/sysm-eval-r.txt \\
        --start-id 1 --end-id 75 --skip 7 13
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

try:
    from openai import OpenAI
except ImportError as exc:  # pragma: no cover - make the error explicit for the user
    raise SystemExit(
        "The openai package is required. Install it via `pip install openai`."
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GPT-4.1 SysML precision/recall evals.")
    default_generated = detect_default_generated_root()
    parser.add_argument(
        "--generated-root",
        type=Path,
        default=default_generated,
        help="Directory containing generated model subdirectories (default: %(default)s).",
    )
    parser.add_argument(
        "--reference-root",
        type=Path,
        default=default_generated,
        help="Directory containing reference/ground-truth subdirectories (default: %(default)s).",
    )
    parser.add_argument(
        "--precision-prompt",
        type=Path,
        default=SCRIPT_DIR / "Evaluation_Prompts" / "sysm-eval-p.txt",
        help="Path to the precision prompt template.",
    )
    parser.add_argument(
        "--recall-prompt",
        type=Path,
        default=SCRIPT_DIR / "Evaluation_Prompts" / "sysm-eval-r.txt",
        help="Path to the recall prompt template.",
    )
    parser.add_argument(
        "--model",
        default="gpt-4.1",
        help="Which OpenAI model to call (default: %(default)s).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature passed to GPT-4.1 (default: %(default)s).",
    )
    parser.add_argument(
        "--start-id",
        type=int,
        default=1,
        help="Smallest model id to evaluate (default: %(default)s).",
    )
    parser.add_argument(
        "--end-id",
        type=int,
        default=75,
        help="Largest model id to evaluate (default: %(default)s).",
    )
    parser.add_argument(
        "--skip",
        type=int,
        nargs="*",
        default=[],
        help="List of model ids to skip (e.g., --skip 3 7 11).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="If set, do not call GPT-4.1. Prompts are still rendered and saved.",
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


def load_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return path.read_text(encoding="utf-8")


def find_generated_file(model_dir: Path, model_id: int) -> Path:
    preferred = model_dir / f"{model_id}.sysml"
    if preferred.exists():
        return preferred
    # Fallback: first .sysml that is not a ground-truth file.
    for candidate in sorted(model_dir.glob("*.sysml")):
        if "groundtruth" not in candidate.name.lower():
            return candidate
    raise FileNotFoundError(f"No generated .sysml found in {model_dir}")


def find_reference_file(reference_root: Path, model_id: int) -> Path:
    # Favor files colocated with the generated models.
    candidates = [
        reference_root / str(model_id) / f"{model_id}_groundtruth.sysml",
        reference_root / str(model_id) / "design.sysml",
        reference_root / f"{model_id:02d}" / f"{model_id}_groundtruth.sysml",
        reference_root / f"{model_id:02d}" / "design.sysml",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not locate a reference sysml for model {model_id}. "
        f"Tried: {', '.join(str(c) for c in candidates)}"
    )


def extract_text_from_response(response) -> str:
    text_chunks: List[str] = []
    # New Responses API exposes output_text for convenience.
    maybe_text = getattr(response, "output_text", None)
    if isinstance(maybe_text, str) and maybe_text.strip():
        return maybe_text
    for item in getattr(response, "output", []):
        for content in getattr(item, "content", []):
            text = getattr(content, "text", None)
            if text:
                text_chunks.append(text)
    return "\n".join(text_chunks).strip()


def call_model(
    client: Optional[OpenAI],
    prompt: str,
    model: str,
    temperature: float,
    dry_run: bool,
) -> Dict[str, object]:
    if dry_run:
        return {
            "dry_run": True,
            "response_text": "",
        }
    response = client.responses.create(
        model=model,
        input=prompt,
        temperature=temperature,
    )
    response_dict = response.model_dump()
    response_dict["response_text"] = extract_text_from_response(response)
    return response_dict


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    precision_template = load_text(args.precision_prompt)
    recall_template = load_text(args.recall_prompt)

    client = None if args.dry_run else OpenAI()
    skip_set = set(args.skip)

    for model_id in range(args.start_id, args.end_id + 1):
        if model_id in skip_set:
            print(f"[skip] model {model_id}")
            continue

        generated_dir = args.generated_root / str(model_id)
        if not generated_dir.exists():
            print(f"[warn] Generated dir missing for model {model_id}: {generated_dir}")
            continue

        try:
            generated_path = find_generated_file(generated_dir, model_id)
            reference_path = find_reference_file(args.reference_root, model_id)
        except FileNotFoundError as exc:
            print(f"[warn] {exc}")
            continue

        generated_text = load_text(generated_path)
        reference_text = load_text(reference_path)

        jobs = [
            ("precision", precision_template, generated_dir / f"{model_id}_precision_gpt41.json"),
            ("recall", recall_template, generated_dir / f"{model_id}_recall_gpt41.json"),
        ]

        for label, template, output_path in jobs:
            ensure_dir(output_path.parent)
            prompt = template.format(reference_model=reference_text, generated_model=generated_text)
            print(f"[run] model {model_id} {label} -> {output_path}")
            response_payload = call_model(
                client=client,
                prompt=prompt,
                model=args.model,
                temperature=args.temperature,
                dry_run=args.dry_run,
            )
            record = {
                "model_id": model_id,
                "evaluation": label,
                "model": args.model,
                "temperature": args.temperature,
                "prompt_file": str(args.precision_prompt if label == "precision" else args.recall_prompt),
                "prompt": prompt,
                "reference_path": str(reference_path),
                "generated_path": str(generated_path),
                "response": response_payload,
            }
            write_json(output_path, record)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("Interrupted by user.")
