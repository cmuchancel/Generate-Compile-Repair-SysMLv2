#!/usr/bin/env python3
"""Plot final SysML line count versus iterations-to-convergence."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    api_loop_root = repo_root / "api_loop"
    default_csv = api_loop_root / "analysis" / "sysml_lines_vs_iterations_all_models.csv"
    default_png = api_loop_root / "analysis" / "sysml_lines_vs_iterations_all_models.png"

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--api-loop-root",
        type=Path,
        default=api_loop_root,
        help="Directory containing Generated_from_Prompts_API_LOOP_* folders.",
    )
    parser.add_argument(
        "--generated-roots",
        type=Path,
        nargs="*",
        default=[],
        help=(
            "Optional explicit generated roots. "
            "If omitted, all Generated_from_Prompts_API_LOOP_* roots under --api-loop-root are used."
        ),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=default_csv,
        help="Output CSV path for extracted points.",
    )
    parser.add_argument(
        "--output-plot",
        type=Path,
        default=default_png,
        help="Output PNG path for scatter plot.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def count_lines(path: Path) -> int:
    return len(path.read_text(encoding="utf-8").splitlines())


def find_generated_file(case_dir: Path, manifest: Dict) -> Path:
    generated_path = manifest.get("generated_path")
    if isinstance(generated_path, str) and generated_path.strip():
        candidate = Path(generated_path)
        if candidate.exists():
            return candidate
    model_id = manifest.get("model_id")
    if model_id is not None:
        candidate = case_dir / f"{model_id}.sysml"
        if candidate.exists():
            return candidate
    candidates = sorted(p for p in case_dir.glob("*.sysml") if "groundtruth" not in p.name.lower())
    if not candidates:
        raise FileNotFoundError(f"No final .sysml file found under {case_dir}")
    return candidates[0]


def collect_points(generated_root: Path) -> List[Dict]:
    model_name = generated_root.name.removeprefix("Generated_from_Prompts_API_LOOP_")
    points: List[Dict] = []
    for manifest_path in sorted(generated_root.glob("*/*_refine_manifest.json")):
        case_dir = manifest_path.parent
        manifest = load_json(manifest_path)
        if not manifest.get("final_iteration_success", False):
            continue

        iterations = manifest.get("iterations_completed")
        if not isinstance(iterations, int):
            continue

        model_id = manifest.get("model_id")
        if model_id is None:
            model_id = case_dir.name

        final_sysml = find_generated_file(case_dir, manifest)
        lines = count_lines(final_sysml)

        points.append(
            {
                "model_name": model_name,
                "model_root": str(generated_root),
                "model_id": model_id,
                "iterations_to_converge": iterations,
                "sysml_line_count": lines,
                "sysml_path": str(final_sysml),
            }
        )
    return points


def write_csv(rows: List[Dict], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "model_name",
                "model_root",
                "model_id",
                "iterations_to_converge",
                "sysml_line_count",
                "sysml_path",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_plot(rows: List[Dict], output_plot: Path) -> None:
    output_plot.parent.mkdir(parents=True, exist_ok=True)
    groups: Dict[str, List[Dict]] = {}
    for row in rows:
        groups.setdefault(str(row["model_name"]), []).append(row)

    plt.figure(figsize=(10, 6), dpi=180)
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]
    for idx, model_name in enumerate(sorted(groups)):
        subset = groups[model_name]
        x = [r["sysml_line_count"] for r in subset]
        y = [r["iterations_to_converge"] for r in subset]
        plt.scatter(
            x,
            y,
            alpha=0.75,
            s=28,
            label=f"{model_name} (n={len(subset)})",
            color=colors[idx % len(colors)],
        )

    plt.title("Final SysML Line Count vs Iterations to Convergence (All Models)")
    plt.xlabel("Final SysML Line Count")
    plt.ylabel("Iterations to Convergence")
    plt.grid(alpha=0.25)
    plt.legend(frameon=False, fontsize=9)
    plt.tight_layout()
    plt.savefig(output_plot)
    plt.close()


def discover_generated_roots(api_loop_root: Path) -> Sequence[Path]:
    return sorted(
        p
        for p in api_loop_root.iterdir()
        if p.is_dir() and p.name.startswith("Generated_from_Prompts_API_LOOP_")
    )


def main() -> None:
    args = parse_args()
    api_loop_root = args.api_loop_root.resolve()
    if args.generated_roots:
        generated_roots = [p.resolve() for p in args.generated_roots]
    else:
        if not api_loop_root.exists():
            raise SystemExit(f"API loop root not found: {api_loop_root}")
        generated_roots = list(discover_generated_roots(api_loop_root))

    if not generated_roots:
        raise SystemExit("No generated roots found.")

    for root in generated_roots:
        if not root.exists():
            raise SystemExit(f"Generated root not found: {root}")

    rows: List[Dict] = []
    for root in generated_roots:
        rows.extend(collect_points(root))
    if not rows:
        raise SystemExit("No converged cases found; nothing to plot.")

    write_csv(rows, args.output_csv.resolve())
    write_plot(rows, args.output_plot.resolve())

    print(f"[ok] model roots: {len(generated_roots)}")
    print(f"[ok] points plotted: {len(rows)}")
    print(f"[ok] csv: {args.output_csv.resolve()}")
    print(f"[ok] plot: {args.output_plot.resolve()}")


if __name__ == "__main__":
    main()
