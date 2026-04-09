import json
import logging
import os
import pickle
import tempfile
from typing import Any

import warnings
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))
warnings.filterwarnings(
    "ignore",
    message="Could not find the number of physical cores",
    category=UserWarning,
)

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
import seaborn as sns
import yaml
from mlflow.models import infer_signature
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


logger = logging.getLogger("model_evaluation")
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler("model_evaluation_errors.log")
    file_handler.setLevel(logging.ERROR)

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)


def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        df.fillna("", inplace=True)
        logger.debug("Data loaded and NaNs filled from %s", file_path)
        return df
    except Exception as error:
        logger.error("Error loading data from %s: %s", file_path, error)
        raise


def load_model(model_path: str):
    """Load the trained model."""
    try:
        with open(model_path, "rb") as file:
            model = pickle.load(file)
        logger.debug("Model loaded from %s", model_path)
        return model
    except Exception as error:
        logger.error("Error loading model from %s: %s", model_path, error)
        raise


def load_vectorizer(vectorizer_path: str) -> TfidfVectorizer:
    """Load the saved TF-IDF vectorizer."""
    try:
        with open(vectorizer_path, "rb") as file:
            vectorizer = pickle.load(file)
        logger.debug("TF-IDF vectorizer loaded from %s", vectorizer_path)
        return vectorizer
    except Exception as error:
        logger.error("Error loading vectorizer from %s: %s", vectorizer_path, error)
        raise


def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, "r", encoding="utf-8") as file:
            params = yaml.safe_load(file)
        logger.debug("Parameters loaded from %s", params_path)
        return params
    except Exception as error:
        logger.error("Error loading parameters from %s: %s", params_path, error)
        raise


def build_model_metrics(y_true, y_pred) -> dict[str, Any]:
    """Build a richer evaluation report for offline model comparison."""
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    metrics = {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
        "macro_precision": round(float(macro_precision), 6),
        "macro_recall": round(float(macro_recall), 6),
        "macro_f1": round(float(macro_f1), 6),
        "weighted_precision": round(float(weighted_precision), 6),
        "weighted_recall": round(float(weighted_recall), 6),
        "weighted_f1": round(float(weighted_f1), 6),
        "support": int(len(y_true)),
    }

    per_class_metrics = {}
    for label, label_metrics in report.items():
        if isinstance(label_metrics, dict):
            per_class_metrics[str(label)] = {
                "precision": round(float(label_metrics.get("precision", 0.0)), 6),
                "recall": round(float(label_metrics.get("recall", 0.0)), 6),
                "f1_score": round(float(label_metrics.get("f1-score", 0.0)), 6),
                "support": int(label_metrics.get("support", 0)),
            }

    return {
        "summary_metrics": metrics,
        "per_class_metrics": per_class_metrics,
        "confusion_matrix": cm.tolist(),
        "label_distribution": {
            "actual": {str(label): int(count) for label, count in pd.Series(y_true).value_counts().sort_index().items()},
            "predicted": {str(label): int(count) for label, count in pd.Series(y_pred).value_counts().sort_index().items()},
        },
    }


def save_json_report(report: dict[str, Any], report_path: str) -> None:
    """Persist a JSON report to disk."""
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
    logger.debug("Saved JSON report to %s", report_path)


def build_comparison_report(
    candidate_report: dict[str, Any],
    baseline_report_path: str,
    promotion_thresholds: dict[str, Any],
) -> dict[str, Any]:
    """Compare candidate metrics against a stored baseline report."""
    if not os.path.exists(baseline_report_path):
        return {
            "status": "baseline_missing",
            "baseline_report_path": baseline_report_path,
            "candidate_summary_metrics": candidate_report["summary_metrics"],
            "recommendation": "Set a baseline report before using automated promotion guidance.",
        }

    with open(baseline_report_path, "r", encoding="utf-8") as file:
        baseline_report = json.load(file)

    baseline_summary = baseline_report.get("summary_metrics", {})
    candidate_summary = candidate_report["summary_metrics"]
    deltas = {}
    for metric_name in [
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_precision",
        "weighted_recall",
        "weighted_f1",
    ]:
        baseline_value = float(baseline_summary.get(metric_name, 0.0))
        candidate_value = float(candidate_summary.get(metric_name, 0.0))
        deltas[metric_name] = round(candidate_value - baseline_value, 6)

    max_recall_regression = float(promotion_thresholds.get("max_recall_regression", 0.01))
    required_macro_f1_gain = float(promotion_thresholds.get("required_macro_f1_gain", 0.0))
    recall_regressions = {}

    baseline_classes = baseline_report.get("per_class_metrics", {})
    candidate_classes = candidate_report.get("per_class_metrics", {})
    for label, baseline_metrics in baseline_classes.items():
        if label in {"accuracy", "macro avg", "weighted avg"}:
            continue
        candidate_metrics = candidate_classes.get(label, {})
        recall_delta = round(
            float(candidate_metrics.get("recall", 0.0)) - float(baseline_metrics.get("recall", 0.0)),
            6,
        )
        recall_regressions[label] = recall_delta

    passes_macro_f1 = deltas["macro_f1"] >= required_macro_f1_gain
    passes_recall_guardrail = all(delta >= -max_recall_regression for delta in recall_regressions.values())
    promote = passes_macro_f1 and passes_recall_guardrail

    return {
        "status": "compared",
        "baseline_report_path": baseline_report_path,
        "candidate_summary_metrics": candidate_summary,
        "baseline_summary_metrics": baseline_summary,
        "delta_summary_metrics": deltas,
        "per_class_recall_deltas": recall_regressions,
        "promotion_policy": {
            "required_macro_f1_gain": required_macro_f1_gain,
            "max_recall_regression": max_recall_regression,
        },
        "recommendation": "promote" if promote else "hold",
        "promotion_checks": {
            "passes_macro_f1": passes_macro_f1,
            "passes_recall_guardrail": passes_recall_guardrail,
        },
    }


def log_confusion_matrix(cm, output_path: str, dataset_name: str) -> None:
    """Save and log confusion matrix as an artifact."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix for {dataset_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    mlflow.log_artifact(output_path)


def save_model_info(run_id: str, model_path: str, file_path: str) -> None:
    """Save the model run ID and path to a JSON file."""
    try:
        model_info = {
            "run_id": run_id,
            "model_path": model_path,
        }
        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(model_info, file, indent=4)
        logger.debug("Model info saved to %s", file_path)
    except Exception as error:
        logger.error("Error occurred while saving the model info: %s", error)
        raise


def configure_mlflow(project_root: str) -> None:
    """Configure MLflow to use an env override or a local SQLite store by default."""
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    else:
        mlflow.set_tracking_uri(f"sqlite:///{os.path.join(project_root, 'mlflow.db').replace(os.sep, '/')}")

    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "dvc-pipeline-runs")
    mlflow.set_experiment(experiment_name)


def configure_runtime_environment(project_root: str) -> None:
    """Use workspace-owned temp paths and stable CPU-count defaults on Windows."""
    temp_dir = os.path.join(project_root, ".tmp")
    os.makedirs(temp_dir, exist_ok=True)
    os.environ["TMP"] = temp_dir
    os.environ["TEMP"] = temp_dir
    tempfile.tempdir = temp_dir
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))


def export_model_for_mlflow(model, export_dir: str, signature, input_example) -> None:
    """Export the model into a workspace directory before logging artifacts to MLflow."""
    if os.path.exists(export_dir):
        for root, dirs, files in os.walk(export_dir, topdown=False):
            for file_name in files:
                os.remove(os.path.join(root, file_name))
            for dir_name in dirs:
                os.rmdir(os.path.join(root, dir_name))
    os.makedirs(export_dir, exist_ok=True)
    mlflow.sklearn.save_model(
        sk_model=model,
        path=export_dir,
        signature=signature,
        input_example=input_example,
        serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_PICKLE,
    )


def main():
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    configure_runtime_environment(root_dir)
    configure_mlflow(root_dir)

    with mlflow.start_run() as run:
        try:
            params = load_params(os.path.join(root_dir, "params.yaml"))
            evaluation_params = params.get("model_evaluation", {})

            for section_name, section_values in params.items():
                if isinstance(section_values, dict):
                    mlflow.log_params({f"{section_name}.{key}": value for key, value in section_values.items()})
                else:
                    mlflow.log_param(section_name, section_values)

            model = load_model(os.path.join(root_dir, "lgbm_model.pkl"))
            vectorizer = load_vectorizer(os.path.join(root_dir, "tfidf_vectorizer.pkl"))
            test_data = load_data(os.path.join(root_dir, "data", "interim", "test_processed.csv"))

            X_test_tfidf = vectorizer.transform(test_data["clean_comment"].values)
            X_test_frame = pd.DataFrame(
                X_test_tfidf.toarray(),
                columns=vectorizer.get_feature_names_out(),
            )
            y_test = test_data["category"].values
            y_pred = model.predict(X_test_frame)

            input_example = X_test_frame.head(5)
            signature = infer_signature(input_example, model.predict(input_example))

            model_path = "lgbm_model"
            save_model_info(run.info.run_id, model_path, os.path.join(root_dir, "experiment_info.json"))

            evaluation_report = build_model_metrics(y_test, y_pred)
            evaluation_report["dataset"] = {
                "name": "test_processed",
                "rows": int(len(test_data)),
                "source": "YouTube-style sentiment inference pipeline",
            }

            reports_dir = os.path.join(root_dir, "reports", "model")
            metrics_report_path = os.path.join(reports_dir, "evaluation_metrics.json")
            confusion_matrix_path = os.path.join(reports_dir, "confusion_matrix_test.png")
            comparison_report_path = os.path.join(reports_dir, "model_comparison.json")
            baseline_report_path = os.path.join(
                root_dir,
                evaluation_params.get("baseline_report_path", os.path.join("reports", "model", "baseline_metrics.json")),
            )

            save_json_report(evaluation_report, metrics_report_path)
            plt.figure(figsize=(8, 6))
            sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues")
            plt.title("Confusion Matrix for Test Data")
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.tight_layout()
            plt.savefig(confusion_matrix_path)
            plt.close()

            exported_model_dir = os.path.join(root_dir, ".tmp", "mlflow_model_export")
            try:
                export_model_for_mlflow(model, exported_model_dir, signature, input_example)
                mlflow.log_artifacts(exported_model_dir, artifact_path="lgbm_model")
            except Exception as error:
                logger.warning("Skipping MLflow model artifact logging because it failed locally: %s", error)

            for artifact_path in [
                os.path.join(root_dir, "tfidf_vectorizer.pkl"),
                metrics_report_path,
                confusion_matrix_path,
            ]:
                try:
                    mlflow.log_artifact(artifact_path)
                except Exception as error:
                    logger.warning("Skipping MLflow artifact logging for %s: %s", artifact_path, error)

            comparison_report = build_comparison_report(
                evaluation_report,
                baseline_report_path,
                evaluation_params.get("promotion_thresholds", {}),
            )
            save_json_report(comparison_report, comparison_report_path)
            try:
                mlflow.log_artifact(comparison_report_path)
            except Exception as error:
                logger.warning("Skipping MLflow artifact logging for %s: %s", comparison_report_path, error)

            for metric_name, metric_value in evaluation_report["summary_metrics"].items():
                if metric_name != "support":
                    mlflow.log_metric(f"test_{metric_name}", metric_value)

            for label, metrics in evaluation_report["per_class_metrics"].items():
                if label in {"accuracy", "macro avg", "weighted avg"}:
                    continue
                mlflow.log_metrics(
                    {
                        f"test_{label}_precision": metrics["precision"],
                        f"test_{label}_recall": metrics["recall"],
                        f"test_{label}_f1_score": metrics["f1_score"],
                    }
                )

            mlflow.set_tag("model_type", "LightGBM")
            mlflow.set_tag("task", "Sentiment Analysis")
            mlflow.set_tag("dataset_domain", "social_comment_text")
        except Exception as error:
            logger.error("Failed to complete model evaluation: %s", error)
            print(f"Error: {error}")


if __name__ == "__main__":
    main()
