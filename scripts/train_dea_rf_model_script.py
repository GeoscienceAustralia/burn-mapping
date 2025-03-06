# Import modules
import hashlib
import logging
import os
import sys

import click
import joblib
import pandas as pd
import requests
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, KFold
from tqdm import tqdm
from xgboost import XGBClassifier  # Import XGBoost

from dea_burn_cube import bc_io, helper

# Configure logging
logging.getLogger("botocore.credentials").setLevel(logging.WARNING)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def logging_setup():
    """Set up logging for all modules except sqlalchemy and boto."""
    loggers = [
        logging.getLogger(name)
        for name in logging.root.manager.loggerDict
        if not name.startswith("sqlalchemy") and not name.startswith("boto")
    ]

    stdout_hdlr = logging.StreamHandler(sys.stdout)
    for logger in loggers:
        logger.addHandler(stdout_hdlr)
        logger.propagate = False


class TQDMLogger:
    """Custom logger to integrate tqdm with the logging system."""

    def __init__(self, logger):
        self.logger = logger
        self.progress_bar = None

    def write(self, text):
        if "totalling" in text and "fits" in text:
            # Initialize tqdm based on the number of fits
            self.bar_size = int(text.split("totalling")[1].split("fits")[0][1:-1])
            self.progress_bar = tqdm(total=self.bar_size, desc="Grid Search Progress")
            return
        if "CV" in text and self.progress_bar:
            self.progress_bar.update(1)

    def flush(self):
        if self.progress_bar:
            self.progress_bar.close()


def fit_with_logging(model, X, y, logger):
    """Fit a model with progress bar redirected to logger."""
    default_stdout = sys.stdout
    sys.stdout = TQDMLogger(logger)
    try:
        model.verbose = 2
        model.fit(X, y)
    finally:
        sys.stdout = default_stdout


def download_file_from_s3_public(url, file_path):
    """Download a file from a public S3 URL."""
    response = requests.get(url)
    if response.status_code == 200:
        with open(file_path, "wb") as f:
            f.write(response.content)
        logger.info(f"File downloaded successfully from: {url}")
    else:
        logger.error(f"Failed to download file from: {url}")


@click.command(no_args_is_help=True)
@click.option(
    "--process-cfg-url",
    "-p",
    type=str,
    required=True,
    help="URL to Burn Cube process configuration file in YAML format.",
)
@click.option(
    "--overwrite/--no-overwrite",
    default=False,
    help="Rerun scenes that have already been processed.",
)
def dea_rf_training(process_cfg_url, overwrite):
    """
    Process FMC for a given configuration.

    Args:
        process_cfg_url (str): URL to the process configuration YAML file.
        overwrite (bool): Whether to overwrite existing results.
    """
    logging_setup()

    # Set AWS credentials for accessing public data
    os.environ["AWS_NO_SIGN_REQUEST"] = "Yes"

    # Load process configuration
    process_cfg = helper.load_yaml_remote(process_cfg_url)
    measurements_list = process_cfg["model_features"]
    training_model_url = process_cfg["model_path"]
    training_dataset_url = process_cfg["training_dataset_url"]
    param_grid = process_cfg["param_grid"]

    # Get the model type keyword from the configuration
    model_type = process_cfg.get("model_type", "RF").upper()

    # Convert dictionary to a sorted string representation to ensure consistent hash
    dict_string = str(sorted(process_cfg.items()))

    # Generate a hash key using SHA-256
    hash_key = hashlib.sha256(dict_string.encode()).hexdigest()

    # Keep only the last 4 digits of the hash
    last_4_digits = hash_key[-4:]

    training_dataset_file = (
        "RF_training_data_21_tiles_1000m_grid_3000m_to_7000m_buffer.csv"
    )
    download_file_from_s3_public(training_dataset_url, training_dataset_file)

    # Load the dataset
    data = pd.read_csv(training_dataset_file)

    # Separate features and target variable
    x = data.drop("class", axis=1)
    x = x[measurements_list]
    y = data["class"]

    # Set a random state for reproducibility
    random_state = 1234

    # Create an instance of KFold cross-validation
    kfold = KFold(n_splits=5, shuffle=True, random_state=random_state)

    # Initialize model and GridSearchCV based on the model type
    if model_type == "RF":
        base_model = RandomForestClassifier()
        logger.info("Using RandomForestClassifier.")
    elif model_type == "XGBOOST":
        base_model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
        logger.info("Using XGBoostClassifier.")
    else:
        logger.error(f"Unsupported model type: {model_type}")
        sys.exit(1)

    # Customize the GridSearchCV progress using a progress bar
    grid_search = GridSearchCV(
        estimator=base_model, param_grid=param_grid, cv=kfold, verbose=2
    )

    # Fit the model with progress displayed in the logger
    fit_with_logging(grid_search, x.values, y, logger)

    # Retrieve the best model
    best_model = grid_search.best_estimator_
    print(f"Best parameters: {grid_search.best_params_}")

    # Save the best model to a file
    model_filename = f"dea_ml_ba_{model_type.lower()}_model.joblib"
    joblib.dump(best_model, model_filename)
    print(f"Best model saved to {model_filename}")

    training_model_url = training_model_url.replace(
        ".joblib", f"-{last_4_digits}.joblib"
    )

    # Convert to S3 URI
    s3_uri = training_model_url.replace(
        "https://dea-public-data-dev.s3.ap-southeast-2.amazonaws.com",
        "s3://dea-public-data-dev",
    )

    helper.get_and_set_aws_credentials()
    bc_io.upload_object_to_s3(model_filename, s3_uri)
    logger.info(f"Uploaded result to: {training_model_url}")


if __name__ == "__main__":
    dea_rf_training()
