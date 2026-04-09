import json
import logging
import os
from typing import Any

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split


logger = logging.getLogger("data_ingestion")
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler("errors.log")
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


def load_data(data_url: str) -> pd.DataFrame:
    """Load data from a CSV file or URL."""
    try:
        df = pd.read_csv(data_url)
        logger.debug("Data loaded from %s", data_url)
        return df
    except pd.errors.ParserError as error:
        logger.error("Failed to parse the CSV file: %s", error)
        raise
    except Exception as error:
        logger.error("Unexpected error occurred while loading the data: %s", error)
        raise


def standardize_dataset_schema(df: pd.DataFrame, ingestion_params: dict) -> tuple[pd.DataFrame, dict[str, str]]:
    """Normalize source datasets to the clean_comment/category training schema."""
    text_candidates = [
        ingestion_params.get("text_column", "clean_comment"),
        "clean_comment",
        "comment_text",
        "comment",
        "text",
    ]
    label_candidates = [
        ingestion_params.get("label_column", "category"),
        "category",
        "sentiment_label",
        "label",
        "sentiment",
    ]
    label_mapping = ingestion_params.get("label_mapping") or {}

    text_column = next((column for column in text_candidates if column in df.columns), None)
    label_column = next((column for column in label_candidates if column in df.columns), None)

    if text_column is None:
        raise KeyError(ingestion_params.get("text_column", "clean_comment"))
    if label_column is None:
        raise KeyError(ingestion_params.get("label_column", "category"))

    standardized_df = df.copy()
    standardized_df = standardized_df.rename(
        columns={
            text_column: "clean_comment",
            label_column: "category",
        }
    )

    if label_mapping:
        normalized_mapping = {str(key).strip().lower(): value for key, value in label_mapping.items()}
        standardized_df["category"] = standardized_df["category"].apply(
            lambda value: normalized_mapping.get(str(value).strip().lower(), value)
        )

    standardized_df["clean_comment"] = standardized_df["clean_comment"].astype(str)
    try:
        standardized_df["category"] = pd.to_numeric(standardized_df["category"])
    except (TypeError, ValueError):
        logger.debug("Keeping non-numeric category labels after schema normalization.")
    return standardized_df, {
        "detected_text_column": text_column,
        "detected_label_column": label_column,
    }


def build_dataset_profile(df: pd.DataFrame, dataset_name: str) -> dict[str, Any]:
    """Create a compact data quality profile for a dataset."""
    row_count = int(len(df))
    empty_comment_rows = 0
    average_comment_length = 0.0
    median_comment_length = 0.0

    if "clean_comment" in df.columns:
        comments = df["clean_comment"].fillna("").astype(str)
        stripped_comments = comments.str.strip()
        comment_lengths = stripped_comments.str.len()
        empty_comment_rows = int((stripped_comments == "").sum())
        average_comment_length = round(float(comment_lengths.mean()), 2) if row_count else 0.0
        median_comment_length = round(float(comment_lengths.median()), 2) if row_count else 0.0

    class_distribution = {}
    if "category" in df.columns:
        class_distribution = {
            str(label): int(count)
            for label, count in df["category"].value_counts(dropna=False).sort_index().items()
        }

    missing_values = {
        str(column): int(value)
        for column, value in df.isna().sum().sort_index().items()
    }

    return {
        "dataset_name": dataset_name,
        "row_count": row_count,
        "column_count": int(len(df.columns)),
        "duplicate_rows": int(df.duplicated().sum()),
        "empty_comment_rows": empty_comment_rows,
        "missing_values_by_column": missing_values,
        "class_distribution": class_distribution,
        "comment_length": {
            "average": average_comment_length,
            "median": median_comment_length,
        },
    }


def preprocess_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Clean the raw dataset and summarize the effect of cleaning."""
    try:
        original_profile = build_dataset_profile(df, "raw_source")
        working_df = df.copy()

        missing_rows_removed = int(working_df.isna().any(axis=1).sum())
        working_df = working_df.dropna()

        duplicate_rows_removed = int(working_df.duplicated().sum())
        working_df = working_df.drop_duplicates()

        if "clean_comment" not in working_df.columns:
            raise KeyError("clean_comment")

        stripped_comments = working_df["clean_comment"].astype(str).str.strip()
        empty_comment_rows_removed = int((stripped_comments == "").sum())
        working_df = working_df[stripped_comments != ""].copy()

        final_profile = build_dataset_profile(working_df, "cleaned_source")
        cleaning_summary = {
            "rows_before_cleaning": original_profile["row_count"],
            "rows_after_cleaning": final_profile["row_count"],
            "rows_removed": int(original_profile["row_count"] - final_profile["row_count"]),
            "missing_rows_removed": missing_rows_removed,
            "duplicate_rows_removed": duplicate_rows_removed,
            "empty_comment_rows_removed": empty_comment_rows_removed,
        }

        logger.debug(
            "Data preprocessing completed: %s rows removed during cleaning.",
            cleaning_summary["rows_removed"],
        )
        return working_df, {
            "source_profile": original_profile,
            "cleaned_profile": final_profile,
            "cleaning_summary": cleaning_summary,
        }
    except KeyError as error:
        logger.error("Missing column in the dataframe: %s", error)
        raise
    except Exception as error:
        logger.error("Unexpected error during preprocessing: %s", error)
        raise


def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    """Save the train and test datasets, creating the raw folder if it does not exist."""
    try:
        raw_data_path = os.path.join(data_path, "raw")
        os.makedirs(raw_data_path, exist_ok=True)

        train_data.to_csv(os.path.join(raw_data_path, "train.csv"), index=False)
        test_data.to_csv(os.path.join(raw_data_path, "test.csv"), index=False)

        logger.debug("Train and test data saved to %s", raw_data_path)
    except Exception as error:
        logger.error("Unexpected error occurred while saving the data: %s", error)
        raise


def save_data_quality_report(report: dict[str, Any], project_root: str) -> None:
    """Persist ingestion quality reports under reports/data_quality."""
    reports_dir = os.path.join(project_root, "reports", "data_quality")
    os.makedirs(reports_dir, exist_ok=True)
    report_path = os.path.join(reports_dir, "ingestion_report.json")

    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    logger.debug("Data quality report saved to %s", report_path)


def main():
    try:
        project_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../"))
        params = load_params(params_path=os.path.join(project_root, "params.yaml"))
        ingestion_params = params["data_ingestion"]
        test_size = ingestion_params["test_size"]
        random_state = ingestion_params.get("random_state", 42)
        local_data_path = ingestion_params.get("local_data_path")
        data_url = ingestion_params["data_url"]
        dataset_name = ingestion_params.get("dataset_name", "social_comment_sentiment")

        if local_data_path:
            resolved_local_path = os.path.join(project_root, local_data_path)
        else:
            resolved_local_path = None

        if resolved_local_path and os.path.exists(resolved_local_path):
            source_location = resolved_local_path
            source_kind = "local_file"
        else:
            source_location = data_url
            source_kind = "remote_url"

        df = load_data(data_url=source_location)
        df, schema_details = standardize_dataset_schema(df, ingestion_params)
        final_df, quality_report = preprocess_data(df)

        train_data, test_data = train_test_split(
            final_df,
            test_size=test_size,
            random_state=random_state,
            stratify=final_df["category"] if "category" in final_df.columns else None,
        )

        quality_report["train_split_profile"] = build_dataset_profile(train_data, "train_split")
        quality_report["test_split_profile"] = build_dataset_profile(test_data, "test_split")
        quality_report["split_summary"] = {
            "test_size": test_size,
            "random_state": random_state,
            "train_rows": int(len(train_data)),
            "test_rows": int(len(test_data)),
        }
        quality_report["source"] = {
            "dataset_name": dataset_name,
            "source_kind": source_kind,
            "source_location": source_location,
            "configured_text_column": ingestion_params.get("text_column", "clean_comment"),
            "configured_label_column": ingestion_params.get("label_column", "category"),
            "detected_text_column": schema_details["detected_text_column"],
            "detected_label_column": schema_details["detected_label_column"],
            "local_source_available": bool(resolved_local_path and os.path.exists(resolved_local_path)),
        }

        save_data(train_data, test_data, data_path=os.path.join(project_root, "data"))
        save_data_quality_report(quality_report, project_root)
    except Exception as error:
        logger.error("Failed to complete the data ingestion process: %s", error)
        print(f"Error: {error}")


if __name__ == "__main__":
    main()
