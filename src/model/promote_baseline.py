import argparse
import json
import logging
import os
import shutil


logger = logging.getLogger("promote_baseline")
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)


def read_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def ensure_parent_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def promote_baseline(comparison_path: str, candidate_path: str, baseline_path: str, force: bool = False) -> dict:
    comparison = read_json(comparison_path)
    recommendation = comparison.get("recommendation")

    if recommendation != "promote" and not force:
        raise ValueError(
            f"Candidate report is not eligible for promotion. Recommendation was '{recommendation}'. Use --force to override."
        )

    ensure_parent_dir(baseline_path)
    shutil.copyfile(candidate_path, baseline_path)
    promoted_report = read_json(candidate_path)

    result = {
        "status": "promoted",
        "baseline_path": baseline_path,
        "candidate_path": candidate_path,
        "comparison_path": comparison_path,
        "forced": force,
        "summary_metrics": promoted_report.get("summary_metrics", {}),
    }
    logger.debug("Promoted %s to %s", candidate_path, baseline_path)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Promote a candidate evaluation report to the baseline report.")
    parser.add_argument(
        "--comparison",
        default=os.path.join("reports", "model", "model_comparison.json"),
        help="Path to the model comparison report.",
    )
    parser.add_argument(
        "--candidate",
        default=os.path.join("reports", "model", "evaluation_metrics.json"),
        help="Path to the candidate evaluation report.",
    )
    parser.add_argument(
        "--baseline",
        default=os.path.join("reports", "model", "baseline_metrics.json"),
        help="Path to the baseline report to overwrite.",
    )
    parser.add_argument("--force", action="store_true", help="Promote even if the comparison says to hold.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = promote_baseline(args.comparison, args.candidate, args.baseline, force=args.force)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
