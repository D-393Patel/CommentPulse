import argparse
import csv
import json
import logging
import math
import os
import pickle
from typing import Any

import pandas as pd


logger = logging.getLogger("dataset_curation")
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)


VALID_LABELS = {"positive", "neutral", "negative"}
NUMERIC_TO_TEXT_LABEL = {-1: "negative", 0: "neutral", 1: "positive", 2: "positive"}
LABEL_PRIORITY_WEIGHT = {"negative": 1.0, "neutral": 1.15, "positive": 1.0, "": 1.05}


def project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))


def ensure_parent_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def load_project_params() -> dict[str, Any]:
    params_path = os.path.join(project_root(), "params.yaml")
    if not os.path.exists(params_path):
        return {}
    import yaml

    with open(params_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def load_comments_from_json(input_path: str) -> pd.DataFrame:
    with open(input_path, "r", encoding="utf-8") as file:
        payload = json.load(file)

    if isinstance(payload, dict) and "comments" in payload:
        comments = payload["comments"]
    elif isinstance(payload, list):
        comments = payload
    else:
        raise ValueError("Unsupported JSON structure. Expected a list or an object with a 'comments' field.")

    if comments and isinstance(comments[0], dict):
        frame = pd.DataFrame(comments)
        text_column = next((column for column in ["comment_text", "text", "comment", "clean_comment"] if column in frame.columns), None)
        if text_column is None:
            raise ValueError("Could not detect a text column in the JSON comment objects.")
        keep_columns = [text_column]
        for optional_column in ["timestamp", "video_id", "author", "like_count"]:
            if optional_column in frame.columns:
                keep_columns.append(optional_column)
        return frame.rename(columns={text_column: "comment_text"})[keep_columns].rename(columns={text_column: "comment_text"})

    return pd.DataFrame({"comment_text": comments})


def load_comments_from_csv(input_path: str) -> pd.DataFrame:
    frame = pd.read_csv(input_path)
    text_column = next((column for column in ["comment_text", "text", "comment", "clean_comment"] if column in frame.columns), None)
    if text_column is None:
        raise ValueError("Could not detect a text column in the CSV file.")
    keep_columns = [text_column]
    for optional_column in ["timestamp", "video_id", "author", "like_count"]:
        if optional_column in frame.columns:
            keep_columns.append(optional_column)
    return frame[keep_columns].rename(columns={text_column: "comment_text"})


def load_raw_comments(input_path: str) -> pd.DataFrame:
    if input_path.lower().endswith(".json"):
        return load_comments_from_json(input_path)
    if input_path.lower().endswith(".csv"):
        return load_comments_from_csv(input_path)
    raise ValueError("Supported input formats are .json and .csv")


def normalize_comment_series(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def normalize_comments(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    normalized["comment_text"] = normalize_comment_series(normalized["comment_text"])
    normalized = normalized[normalized["comment_text"] != ""].copy()
    normalized["normalized_comment"] = normalized["comment_text"].str.lower()
    return normalized


def load_existing_labeled_comments(existing_labeled_path: str | None) -> set[str]:
    if not existing_labeled_path or not os.path.exists(existing_labeled_path):
        return set()

    frame = pd.read_csv(existing_labeled_path)
    text_column = next((column for column in ["comment_text", "text", "comment", "clean_comment"] if column in frame.columns), None)
    if text_column is None:
        return set()
    return set(normalize_comment_series(frame[text_column]).str.lower().tolist())


def load_model_helpers() -> tuple[Any, Any] | tuple[None, None]:
    root = project_root()
    model_path = os.path.join(root, "lgbm_model.pkl")
    vectorizer_path = os.path.join(root, "tfidf_vectorizer.pkl")
    if not (os.path.exists(model_path) and os.path.exists(vectorizer_path)):
        return None, None

    with open(model_path, "rb") as model_file:
        model = pickle.load(model_file)
    with open(vectorizer_path, "rb") as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    return model, vectorizer


def map_numeric_prediction(prediction: Any) -> str:
    try:
        return NUMERIC_TO_TEXT_LABEL.get(int(prediction), str(prediction))
    except (TypeError, ValueError):
        return str(prediction)


def score_model_confidence(model: Any, matrix) -> list[float | None]:
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(matrix)
        return [round(float(row.max()), 6) for row in probabilities]
    if hasattr(model, "decision_function"):
        import numpy as np

        scores = model.decision_function(matrix)
        if len(getattr(scores, "shape", ())) == 1:
            scores = np.vstack([-scores, scores]).T
        margins = np.max(scores, axis=1) - np.partition(scores, -2, axis=1)[:, -2]
        return [round(float(margin), 6) for margin in margins]
    return [None] * matrix.shape[0]


def attach_model_assistance(frame: pd.DataFrame) -> pd.DataFrame:
    model, vectorizer = load_model_helpers()
    assisted = frame.copy()
    assisted["suggested_label"] = ""
    assisted["suggested_confidence"] = ""

    if model is None or vectorizer is None or assisted.empty:
        return assisted

    matrix = vectorizer.transform(assisted["comment_text"].tolist())
    try:
        feature_frame = pd.DataFrame.sparse.from_spmatrix(
            matrix,
            columns=vectorizer.get_feature_names_out(),
        )
        predictions = model.predict(feature_frame)
        confidences = score_model_confidence(model, feature_frame)
    except Exception:
        predictions = model.predict(matrix)
        confidences = score_model_confidence(model, matrix)

    assisted["suggested_label"] = [map_numeric_prediction(prediction) for prediction in predictions]
    assisted["suggested_confidence"] = [
        "" if confidence is None else confidence
        for confidence in confidences
    ]
    return assisted


def add_review_priority(frame: pd.DataFrame) -> pd.DataFrame:
    """Rank queue rows by likely human-review value."""
    prioritized = frame.copy()

    def _confidence_to_float(value: Any) -> float | None:
        try:
            if value == "":
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

    confidences = [_confidence_to_float(value) for value in prioritized.get("suggested_confidence", pd.Series([""] * len(prioritized)))]
    prioritized["_confidence"] = confidences
    prioritized["_uncertainty"] = [1.0 - confidence if confidence is not None else 0.75 for confidence in confidences]
    prioritized["_label_weight"] = [
        LABEL_PRIORITY_WEIGHT.get(str(label).strip().lower(), 1.0)
        for label in prioritized.get("suggested_label", pd.Series([""] * len(prioritized)))
    ]
    prioritized["review_priority_score"] = (
        prioritized["_uncertainty"] * 100 * prioritized["_label_weight"]
    ).round(4)
    prioritized["review_priority_reason"] = [
        "model_uncertain" if confidence is not None and confidence < 0.8 else
        "no_model_signal" if confidence is None else
        "high_confidence_spot_check"
        for confidence in confidences
    ]

    prioritized = prioritized.sort_values(
        by=["review_priority_score", "example_id"],
        ascending=[False, True],
    ).reset_index(drop=True)
    prioritized["review_priority_rank"] = prioritized.index + 1
    prioritized = prioritized.drop(columns=["_confidence", "_uncertainty", "_label_weight"])
    return prioritized


def prepare_labeling_queue(
    input_path: str,
    output_path: str,
    dataset_name: str,
    existing_labeled_path: str | None = None,
    batch_size: int = 0,
    add_model_suggestions: bool = False,
) -> dict[str, Any]:
    frame = load_raw_comments(input_path)
    frame = normalize_comments(frame)

    before_dedupe = len(frame)
    frame = frame.drop_duplicates(subset=["normalized_comment"]).copy()
    duplicates_removed = int(before_dedupe - len(frame))

    existing_comments = load_existing_labeled_comments(existing_labeled_path)
    if existing_comments:
        frame = frame[~frame["normalized_comment"].isin(existing_comments)].copy()

    frame = frame.reset_index(drop=True)
    frame["sentiment_label"] = ""
    frame["source_dataset"] = dataset_name
    frame["needs_review"] = True
    frame["labeler_notes"] = ""
    frame["example_id"] = [f"{dataset_name}-{index + 1:05d}" for index in range(len(frame))]
    frame["batch_id"] = ""

    if add_model_suggestions:
        frame = attach_model_assistance(frame)
    else:
        frame["suggested_label"] = ""
        frame["suggested_confidence"] = ""

    if batch_size and batch_size > 0 and not frame.empty:
        total_batches = math.ceil(len(frame) / batch_size)
        batch_ids = []
        for index in range(len(frame)):
            batch_number = (index // batch_size) + 1
            batch_ids.append(f"{dataset_name}-batch-{batch_number:02d}-of-{total_batches:02d}")
        frame["batch_id"] = batch_ids

    frame = add_review_priority(frame)

    frame = frame.drop(columns=["normalized_comment"])

    ensure_parent_dir(output_path)
    frame.to_csv(output_path, index=False, quoting=csv.QUOTE_MINIMAL)

    summary = {
        "action": "prepare",
        "input_path": input_path,
        "output_path": output_path,
        "dataset_name": dataset_name,
        "prepared_examples": int(len(frame)),
        "duplicates_removed": duplicates_removed,
        "already_labeled_skipped": int(before_dedupe - duplicates_removed - len(frame)),
        "existing_labeled_path": existing_labeled_path,
        "batch_size": int(batch_size),
        "batched": bool(batch_size and batch_size > 0),
        "model_suggestions_enabled": bool(add_model_suggestions),
    }
    logger.debug("Prepared %s comments for labeling at %s", len(frame), output_path)
    return summary


def normalize_labeled_frame(frame: pd.DataFrame) -> pd.DataFrame:
    text_column = next((column for column in ["comment_text", "text", "comment", "clean_comment"] if column in frame.columns), None)
    label_column = next((column for column in ["sentiment_label", "label", "sentiment", "category"] if column in frame.columns), None)
    if text_column is None or label_column is None:
        raise ValueError("The labeled file must contain a text column and a label column.")

    normalized = frame.rename(columns={text_column: "comment_text", label_column: "sentiment_label"}).copy()
    normalized["comment_text"] = normalize_comment_series(normalized["comment_text"])
    normalized["sentiment_label"] = normalized["sentiment_label"].fillna("").astype(str).str.strip().str.lower()
    return normalized


def merge_labeled_data(base_path: str, new_path: str, output_path: str) -> dict[str, Any]:
    base_frame = pd.read_csv(base_path) if os.path.exists(base_path) else pd.DataFrame(columns=["comment_text", "sentiment_label"])
    new_frame = pd.read_csv(new_path)

    normalized_new = normalize_labeled_frame(new_frame)
    normalized_new = normalized_new[
        (normalized_new["comment_text"] != "") & (normalized_new["sentiment_label"].isin(VALID_LABELS))
    ][["comment_text", "sentiment_label"]].copy()

    normalized_base = normalize_labeled_frame(base_frame)[["comment_text", "sentiment_label"]].copy() if not base_frame.empty else base_frame

    merged = pd.concat([normalized_base, normalized_new], ignore_index=True)
    merged["normalized_comment"] = merged["comment_text"].str.lower()
    merged = merged.drop_duplicates(subset=["normalized_comment"], keep="last").drop(columns=["normalized_comment"])
    merged = merged.sort_values(by=["sentiment_label", "comment_text"]).reset_index(drop=True)

    ensure_parent_dir(output_path)
    merged.to_csv(output_path, index=False, quoting=csv.QUOTE_MINIMAL)

    summary = {
        "action": "merge",
        "base_path": base_path,
        "new_path": new_path,
        "output_path": output_path,
        "base_examples": int(len(normalized_base)),
        "new_valid_examples": int(len(normalized_new)),
        "merged_examples": int(len(merged)),
        "class_distribution": {
            str(label): int(count)
            for label, count in merged["sentiment_label"].value_counts().sort_index().items()
        },
    }
    logger.debug("Merged labeled data into %s with %s total examples", output_path, len(merged))
    return summary


def bootstrap_pseudo_labels(
    input_path: str,
    output_path: str,
    min_confidence: float = 0.8,
) -> dict[str, Any]:
    frame = pd.read_csv(input_path)
    required_columns = {"comment_text", "suggested_label", "suggested_confidence"}
    missing_columns = required_columns - set(frame.columns)
    if missing_columns:
        raise ValueError(
            f"Pseudo-label bootstrap requires columns {sorted(required_columns)}. Missing: {sorted(missing_columns)}"
        )

    working = frame.copy()
    working["comment_text"] = normalize_comment_series(working["comment_text"])
    working["suggested_label"] = working["suggested_label"].fillna("").astype(str).str.strip().str.lower()
    working["suggested_confidence"] = pd.to_numeric(working["suggested_confidence"], errors="coerce")

    filtered = working[
        (working["comment_text"] != "")
        & (working["suggested_label"].isin(VALID_LABELS))
        & (working["suggested_confidence"] >= min_confidence)
    ].copy()

    filtered["sentiment_label"] = filtered["suggested_label"]
    filtered["label_origin"] = "pseudo_label"
    filtered["needs_review"] = True
    filtered["normalized_comment"] = filtered["comment_text"].str.lower()
    filtered = filtered.drop_duplicates(subset=["normalized_comment"]).drop(columns=["normalized_comment"])

    output_columns = [
        "comment_text",
        "sentiment_label",
        "label_origin",
        "suggested_confidence",
        "source_dataset",
        "batch_id",
        "example_id",
        "review_priority_rank",
        "review_priority_score",
        "review_priority_reason",
        "needs_review",
        "labeler_notes",
    ]
    available_columns = [column for column in output_columns if column in filtered.columns]

    ensure_parent_dir(output_path)
    filtered[available_columns].to_csv(output_path, index=False, quoting=csv.QUOTE_MINIMAL)

    summary = {
        "action": "bootstrap",
        "input_path": input_path,
        "output_path": output_path,
        "min_confidence": min_confidence,
        "candidate_examples": int(len(filtered)),
        "class_distribution": {
            str(label): int(count)
            for label, count in filtered["sentiment_label"].value_counts().sort_index().items()
        },
    }
    logger.debug("Bootstrapped %s pseudo-labeled examples into %s", len(filtered), output_path)
    return summary


def prepare_review_file(input_path: str, output_path: str) -> dict[str, Any]:
    frame = pd.read_csv(input_path)
    required_columns = {"comment_text", "sentiment_label"}
    missing_columns = required_columns - set(frame.columns)
    if missing_columns:
        raise ValueError(
            f"Review preparation requires columns {sorted(required_columns)}. Missing: {sorted(missing_columns)}"
        )

    review_frame = frame.copy()
    if "label_origin" not in review_frame.columns:
        review_frame["label_origin"] = "candidate_label"
    if "needs_review" not in review_frame.columns:
        review_frame["needs_review"] = True
    if "labeler_notes" not in review_frame.columns:
        review_frame["labeler_notes"] = ""
    if "review_priority_score" not in review_frame.columns:
        review_frame["review_priority_score"] = ""
    if "review_priority_rank" not in review_frame.columns:
        review_frame["review_priority_rank"] = ""
    if "review_priority_reason" not in review_frame.columns:
        review_frame["review_priority_reason"] = ""

    review_frame["review_status"] = "pending"
    review_frame["reviewed_label"] = ""
    review_frame["reviewed_by"] = ""
    review_frame["reviewed_at"] = ""

    ordered_columns = [
        "comment_text",
        "sentiment_label",
        "reviewed_label",
        "review_status",
        "label_origin",
        "suggested_confidence",
        "source_dataset",
        "batch_id",
        "example_id",
        "review_priority_rank",
        "review_priority_score",
        "review_priority_reason",
        "needs_review",
        "reviewed_by",
        "reviewed_at",
        "labeler_notes",
    ]
    available_columns = [column for column in ordered_columns if column in review_frame.columns]

    ensure_parent_dir(output_path)
    review_frame[available_columns].to_csv(output_path, index=False, quoting=csv.QUOTE_MINIMAL)

    summary = {
        "action": "review_prepare",
        "input_path": input_path,
        "output_path": output_path,
        "review_examples": int(len(review_frame)),
    }
    logger.debug("Prepared %s review rows at %s", len(review_frame), output_path)
    return summary


def audit_labeled_dataset(input_path: str) -> dict[str, Any]:
    frame = pd.read_csv(input_path)
    normalized = normalize_labeled_frame(frame)
    normalized["normalized_comment"] = normalized["comment_text"].str.lower()

    invalid_label_rows = normalized[~normalized["sentiment_label"].isin(VALID_LABELS)]
    duplicate_rows = normalized[normalized.duplicated(subset=["normalized_comment"], keep=False)]
    empty_comment_rows = normalized[normalized["comment_text"] == ""]

    class_distribution = {
        str(label): int(count)
        for label, count in normalized[normalized["sentiment_label"].isin(VALID_LABELS)]["sentiment_label"].value_counts().sort_index().items()
    }
    valid_count = sum(class_distribution.values())
    max_class = max(class_distribution.values()) if class_distribution else 0
    min_class = min(class_distribution.values()) if class_distribution else 0
    imbalance_ratio = round(max_class / min_class, 4) if min_class else None
    quality_params = load_project_params().get("data_quality", {})
    min_total_labeled_rows = int(quality_params.get("min_total_labeled_rows", 1000))
    min_examples_per_class = int(quality_params.get("min_examples_per_class", 200))
    meets_total_target = valid_count >= min_total_labeled_rows
    meets_class_target = all(count >= min_examples_per_class for count in class_distribution.values()) if class_distribution else False

    return {
        "action": "audit",
        "input_path": input_path,
        "total_rows": int(len(normalized)),
        "valid_labeled_rows": int(valid_count),
        "invalid_label_rows": int(len(invalid_label_rows)),
        "duplicate_comment_rows": int(len(duplicate_rows)),
        "empty_comment_rows": int(len(empty_comment_rows)),
        "class_distribution": class_distribution,
        "class_imbalance_ratio": imbalance_ratio,
        "dataset_readiness": {
            "min_total_labeled_rows": min_total_labeled_rows,
            "min_examples_per_class": min_examples_per_class,
            "meets_total_target": meets_total_target,
            "meets_class_target": meets_class_target,
        },
        "quality_checks": {
            "has_invalid_labels": bool(len(invalid_label_rows)),
            "has_duplicate_comments": bool(len(duplicate_rows)),
            "has_empty_comments": bool(len(empty_comment_rows)),
        },
    }


def write_curation_report(summary: dict[str, Any]) -> None:
    action = summary.get("action", "curation")
    if action == "audit":
        report_name = "dataset_audit_report.json"
    elif action == "merge":
        report_name = "dataset_merge_report.json"
    else:
        report_name = "dataset_curation_report.json"
    report_path = os.path.join(project_root(), "reports", "data_quality", report_name)
    ensure_parent_dir(report_path)
    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    logger.debug("Dataset curation report saved to %s", report_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare, audit, and merge locally curated YouTube sentiment datasets.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare", help="Prepare raw comments for manual labeling.")
    prepare_parser.add_argument("--input", required=True, help="Path to a JSON or CSV file containing raw comments.")
    prepare_parser.add_argument("--output", required=True, help="Path to the output CSV labeling queue.")
    prepare_parser.add_argument("--dataset-name", default="youtube_capture", help="Logical dataset name for tracking.")
    prepare_parser.add_argument(
        "--existing-labeled",
        default=os.path.join("data", "external", "youtube_comments_labeled.csv"),
        help="Existing labeled dataset used for dedupe.",
    )
    prepare_parser.add_argument("--batch-size", type=int, default=0, help="Optional batch size for labeling queue chunks.")
    prepare_parser.add_argument(
        "--add-model-suggestions",
        action="store_true",
        help="Populate suggested labels from the current trained model when artifacts are available.",
    )

    merge_parser = subparsers.add_parser("merge", help="Merge a newly labeled CSV into the training dataset.")
    merge_parser.add_argument("--base", required=True, help="Existing labeled dataset CSV.")
    merge_parser.add_argument("--new", required=True, help="Newly labeled CSV to merge.")
    merge_parser.add_argument("--output", required=True, help="Output path for the merged dataset CSV.")

    bootstrap_parser = subparsers.add_parser("bootstrap", help="Create a weakly labeled candidate dataset from model suggestions.")
    bootstrap_parser.add_argument("--input", required=True, help="Prepared labeling queue with suggested labels.")
    bootstrap_parser.add_argument("--output", required=True, help="Output path for the pseudo-labeled CSV.")
    bootstrap_parser.add_argument("--min-confidence", type=float, default=0.8, help="Minimum confidence required to keep a suggestion.")

    review_parser = subparsers.add_parser("review-prepare", help="Create a human-review file from candidate labels.")
    review_parser.add_argument("--input", required=True, help="Pseudo-labeled or candidate-labeled CSV.")
    review_parser.add_argument("--output", required=True, help="Output path for the review CSV.")

    audit_parser = subparsers.add_parser("audit", help="Audit a labeled dataset for QA issues.")
    audit_parser.add_argument("--input", required=True, help="Path to the labeled dataset CSV.")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "prepare":
        summary = prepare_labeling_queue(
            args.input,
            args.output,
            args.dataset_name,
            existing_labeled_path=args.existing_labeled,
            batch_size=args.batch_size,
            add_model_suggestions=args.add_model_suggestions,
        )
    elif args.command == "merge":
        summary = merge_labeled_data(args.base, args.new, args.output)
    elif args.command == "bootstrap":
        summary = bootstrap_pseudo_labels(args.input, args.output, args.min_confidence)
    elif args.command == "review-prepare":
        summary = prepare_review_file(args.input, args.output)
    else:
        summary = audit_labeled_dataset(args.input)
    write_curation_report(summary)


if __name__ == "__main__":
    main()
