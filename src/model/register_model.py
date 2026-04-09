import json
import logging
import os

import mlflow


logger = logging.getLogger("model_registration")
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler("model_registration_errors.log")
    file_handler.setLevel(logging.ERROR)

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)


def configure_mlflow(project_root: str) -> None:
    """Configure MLflow to use an env override or a local file store by default."""
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    else:
        mlflow.set_tracking_uri(f"file:///{os.path.join(project_root, 'mlruns').replace(os.sep, '/')}")


def load_model_info(file_path: str) -> dict:
    """Load the model info from a JSON file."""
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            model_info = json.load(file)
        logger.debug("Model info loaded from %s", file_path)
        return model_info
    except FileNotFoundError:
        logger.error("File not found: %s", file_path)
        raise
    except Exception as error:
        logger.error("Unexpected error occurred while loading the model info: %s", error)
        raise


def register_model(model_name: str, model_info: dict) -> None:
    """Register the model to the MLflow Model Registry."""
    try:
        model_uri = f"runs:/{model_info['run_id']}/{model_info['model_path']}"
        model_version = mlflow.register_model(model_uri, model_name)

        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage="Staging",
        )

        logger.debug(
            "Model %s version %s registered and transitioned to Staging.",
            model_name,
            model_version.version,
        )
    except Exception as error:
        logger.error("Error during model registration: %s", error)
        raise


def main():
    try:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
        configure_mlflow(project_root)

        model_info_path = os.path.join(project_root, "experiment_info.json")
        model_info = load_model_info(model_info_path)
        register_model("yt_chrome_plugin_model", model_info)
    except Exception as error:
        logger.error("Failed to complete the model registration process: %s", error)
        print(f"Error: {error}")


if __name__ == "__main__":
    main()
