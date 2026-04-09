import json
import logging
import os
import re
from typing import Any

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


logger = logging.getLogger("data_preprocessing")
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler("preprocessing_errors.log")
    file_handler.setLevel(logging.ERROR)

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)


LEMMATIZER = WordNetLemmatizer()
STOP_WORDS_TO_KEEP = {"not", "but", "however", "no", "yet"}


def ensure_nltk_resources() -> None:
    """Download required NLTK assets only when missing."""
    resources = {
        "corpora/wordnet": "wordnet",
        "corpora/stopwords": "stopwords",
    }
    for resource_path, download_name in resources.items():
        try:
            nltk.data.find(resource_path)
        except LookupError:
            nltk.download(download_name, quiet=True)


def build_text_profile(df: pd.DataFrame, dataset_name: str) -> dict[str, Any]:
    """Create text-focused quality metrics for a dataset."""
    comments = df["clean_comment"].fillna("").astype(str)
    token_counts = comments.str.split().str.len()
    char_counts = comments.str.len()

    return {
        "dataset_name": dataset_name,
        "row_count": int(len(df)),
        "empty_comment_rows": int((comments.str.strip() == "").sum()),
        "comment_length_chars": {
            "average": round(float(char_counts.mean()), 2) if len(df) else 0.0,
            "median": round(float(char_counts.median()), 2) if len(df) else 0.0,
        },
        "comment_length_tokens": {
            "average": round(float(token_counts.mean()), 2) if len(df) else 0.0,
            "median": round(float(token_counts.median()), 2) if len(df) else 0.0,
        },
    }


def preprocess_comment(comment: str) -> str:
    """Apply preprocessing transformations to a comment."""
    try:
        comment = str(comment).lower().strip()
        comment = re.sub(r"\n", " ", comment)
        comment = re.sub(r"[^A-Za-z0-9\s!?.,]", "", comment)

        stop_words = set(stopwords.words("english")) - STOP_WORDS_TO_KEEP
        comment = " ".join(word for word in comment.split() if word not in stop_words)
        comment = " ".join(LEMMATIZER.lemmatize(word) for word in comment.split())
        return comment
    except Exception as error:
        logger.error("Error in preprocessing comment: %s", error)
        return str(comment)


def normalize_text(df: pd.DataFrame) -> pd.DataFrame:
    """Apply preprocessing to the text data in the dataframe."""
    try:
        normalized_df = df.copy()
        normalized_df["clean_comment"] = normalized_df["clean_comment"].apply(preprocess_comment)
        logger.debug("Text normalization completed")
        return normalized_df
    except Exception as error:
        logger.error("Error during text normalization: %s", error)
        raise


def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    """Save the processed train and test datasets."""
    try:
        interim_data_path = os.path.join(data_path, "interim")
        os.makedirs(interim_data_path, exist_ok=True)

        train_data.to_csv(os.path.join(interim_data_path, "train_processed.csv"), index=False)
        test_data.to_csv(os.path.join(interim_data_path, "test_processed.csv"), index=False)

        logger.debug("Processed data saved to %s", interim_data_path)
    except Exception as error:
        logger.error("Error occurred while saving data: %s", error)
        raise


def save_preprocessing_report(report: dict[str, Any], project_root: str) -> None:
    """Persist preprocessing quality metrics under reports/data_quality."""
    reports_dir = os.path.join(project_root, "reports", "data_quality")
    os.makedirs(reports_dir, exist_ok=True)
    report_path = os.path.join(reports_dir, "preprocessing_report.json")

    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    logger.debug("Preprocessing report saved to %s", report_path)


def main():
    try:
        ensure_nltk_resources()
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

        logger.debug("Starting data preprocessing...")
        train_data = pd.read_csv(os.path.join(project_root, "data", "raw", "train.csv"))
        test_data = pd.read_csv(os.path.join(project_root, "data", "raw", "test.csv"))
        logger.debug("Data loaded successfully")

        train_before = build_text_profile(train_data, "train_raw")
        test_before = build_text_profile(test_data, "test_raw")

        train_processed_data = normalize_text(train_data)
        test_processed_data = normalize_text(test_data)

        train_after = build_text_profile(train_processed_data, "train_processed")
        test_after = build_text_profile(test_processed_data, "test_processed")

        save_data(train_processed_data, test_processed_data, data_path=os.path.join(project_root, "data"))
        save_preprocessing_report(
            {
                "train": {
                    "before": train_before,
                    "after": train_after,
                },
                "test": {
                    "before": test_before,
                    "after": test_after,
                },
            },
            project_root,
        )
    except Exception as error:
        logger.error("Failed to complete the data preprocessing process: %s", error)
        print(f"Error: {error}")


if __name__ == "__main__":
    main()
