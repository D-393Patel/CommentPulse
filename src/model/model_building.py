import json
import logging
import os
import pickle
from typing import Any

import warnings
import lightgbm as lgb
import pandas as pd
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC


os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
warnings.filterwarnings(
    "ignore",
    message="Could not find the number of physical cores",
    category=UserWarning,
)

logger = logging.getLogger("model_building")
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler("model_building_errors.log")
    file_handler.setLevel(logging.ERROR)

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)


def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, "r", encoding="utf-8") as file:
            params = yaml.safe_load(file)
        logger.debug("Parameters retrieved from %s", params_path)
        return params
    except FileNotFoundError:
        logger.error("File not found: %s", params_path)
        raise
    except yaml.YAMLError as error:
        logger.error("YAML error: %s", error)
        raise
    except Exception as error:
        logger.error("Unexpected error: %s", error)
        raise


def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        df.fillna("", inplace=True)
        logger.debug("Data loaded and NaNs filled from %s", file_path)
        return df
    except pd.errors.ParserError as error:
        logger.error("Failed to parse the CSV file: %s", error)
        raise
    except Exception as error:
        logger.error("Unexpected error occurred while loading the data: %s", error)
        raise


def apply_tfidf(train_data: pd.DataFrame, max_features: int, ngram_range: tuple) -> tuple:
    """Apply TF-IDF feature engineering on the training split."""
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
    features = vectorizer.fit_transform(train_data["clean_comment"].values)
    labels = train_data["category"].values

    logger.debug("TF-IDF transformation complete. Train shape: %s", features.shape)

    with open(os.path.join(get_root_directory(), "tfidf_vectorizer.pkl"), "wb") as file:
        pickle.dump(vectorizer, file)

    logger.debug("TF-IDF vectorizer saved to project root")
    return vectorizer, features, labels


def to_feature_frame(vectorizer: TfidfVectorizer, matrix) -> pd.DataFrame:
    """Build a sparse feature frame so estimators with feature-name tracking stay quiet."""
    return pd.DataFrame.sparse.from_spmatrix(
        matrix,
        columns=vectorizer.get_feature_names_out(),
    )


def build_lgbm_config(
    learning_rate: float,
    max_depth: int,
    n_estimators: int,
    training_rows: int,
) -> dict:
    """Create a LightGBM config that degrades gracefully on very small datasets."""
    config = {
        "objective": "multiclass",
        "num_class": 3,
        "metric": "multi_logloss",
        "is_unbalance": True,
        "class_weight": "balanced",
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "learning_rate": learning_rate,
        "max_depth": max_depth,
        "n_estimators": n_estimators,
        "verbosity": -1,
        "n_jobs": 1,
    }

    if training_rows < 100:
        config.update(
            {
                "min_data_in_leaf": 1,
                "min_data_in_bin": 1,
                "num_leaves": min(15, max(4, training_rows)),
            }
        )
        logger.warning(
            "Training on a very small dataset (%s rows). Using relaxed LightGBM leaf/bin constraints.",
            training_rows,
        )

    return config


def build_candidate_models(model_params: dict, training_rows: int) -> dict[str, Any]:
    """Create candidate text classifiers for offline model selection."""
    return {
        "logistic_regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=42,
        ),
        "linear_svc": LinearSVC(class_weight="balanced", random_state=42),
        "lightgbm": lgb.LGBMClassifier(
            **build_lgbm_config(
                learning_rate=model_params["learning_rate"],
                max_depth=model_params["max_depth"],
                n_estimators=model_params["n_estimators"],
                training_rows=training_rows,
            )
        ),
    }


def score_predictions(y_true, y_pred) -> dict[str, float]:
    """Return a compact set of model-selection metrics."""
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

    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
        "macro_precision": round(float(macro_precision), 6),
        "macro_recall": round(float(macro_recall), 6),
        "macro_f1": round(float(macro_f1), 6),
        "weighted_precision": round(float(weighted_precision), 6),
        "weighted_recall": round(float(weighted_recall), 6),
        "weighted_f1": round(float(weighted_f1), 6),
    }


def select_best_model(vectorizer: TfidfVectorizer, features, labels, model_params: dict) -> tuple[str, dict[str, Any]]:
    """Train candidate models on a validation split and return the best one."""
    validation_size = float(model_params.get("validation_size", 0.2))
    X_train, X_val, y_train, y_val = train_test_split(
        features,
        labels,
        test_size=validation_size,
        random_state=42,
        stratify=labels,
    )
    X_val_frame = to_feature_frame(vectorizer, X_val)

    candidate_names = model_params.get(
        "candidate_models",
        ["logistic_regression", "linear_svc", "lightgbm"],
    )
    candidates = build_candidate_models(model_params, len(y_train))
    results = {}

    for candidate_name in candidate_names:
        model = candidates[candidate_name]
        model.fit(X_train, y_train)
        predictions = model.predict(X_val_frame)
        metrics = score_predictions(y_val, predictions)
        results[candidate_name] = {
            "validation_metrics": metrics,
            "training_rows": int(len(y_train)),
            "validation_rows": int(len(y_val)),
        }
        logger.debug("Candidate %s validation metrics: %s", candidate_name, metrics)

    best_model_name = max(
        results,
        key=lambda name: (
            results[name]["validation_metrics"]["macro_f1"],
            results[name]["validation_metrics"]["weighted_f1"],
            results[name]["validation_metrics"]["accuracy"],
        ),
    )
    logger.info("Selected %s as the best validation model.", best_model_name)
    return best_model_name, results


def retrain_selected_model(model_name: str, model_params: dict, features, labels):
    """Fit the selected model on the full training split."""
    model = build_candidate_models(model_params, len(labels))[model_name]
    model.fit(features, labels)
    logger.debug("Retrained selected model %s on the full training data", model_name)
    return model


def save_model(model, file_path: str) -> None:
    """Save the trained model to a file."""
    with open(file_path, "wb") as file:
        pickle.dump(model, file)
    logger.debug("Model saved to %s", file_path)


def save_model_selection_report(report: dict[str, Any], report_path: str) -> None:
    """Persist model-selection diagnostics."""
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
    logger.debug("Model selection report saved to %s", report_path)


def get_root_directory() -> str:
    """Get the project root directory."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(current_dir, "../../"))


def main():
    try:
        root_dir = get_root_directory()
        params = load_params(os.path.join(root_dir, "params.yaml"))
        model_params = params["model_building"]

        train_data = load_data(os.path.join(root_dir, "data", "interim", "train_processed.csv"))
        vectorizer, features, labels = apply_tfidf(
            train_data,
            model_params["max_features"],
            tuple(model_params["ngram_range"]),
        )

        best_model_name, candidate_results = select_best_model(vectorizer, features, labels, model_params)
        selected_model = retrain_selected_model(best_model_name, model_params, features, labels)
        save_model(selected_model, os.path.join(root_dir, "lgbm_model.pkl"))

        model_selection_report = {
            "selected_model": best_model_name,
            "candidate_results": candidate_results,
            "training_rows": int(len(labels)),
            "feature_count": int(features.shape[1]),
        }
        save_model_selection_report(
            model_selection_report,
            os.path.join(root_dir, "reports", "model", "model_selection.json"),
        )
    except Exception as error:
        logger.error("Failed to complete the feature engineering and model building process: %s", error)
        print(f"Error: {error}")


if __name__ == "__main__":
    main()
