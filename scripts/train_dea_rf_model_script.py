# Import modules
import logging
import os
import sys

import click
import pandas as pd
import requests

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
    measurements_list = process_cfg["input_products"]["input_bands"]
    training_model_url = process_cfg["model_path"]
    training_dataset_url = process_cfg["training_dataset_url"]

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

    import joblib
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GridSearchCV, KFold
    from tqdm import tqdm

    # Set a random state for reproducibility
    random_state = 1234

    # Create an instance of KFold cross-validation
    kfold = KFold(n_splits=5, shuffle=True, random_state=random_state)

    # Define the parameter grid for RandomForestClassifier
    param_grid = {"n_estimators": [20, 30, 50, 100], "max_depth": [5, 15, 25]}

    # Create a RandomForestClassifier instance
    base_model = RandomForestClassifier()

    # Customize the GridSearchCV progress using a progress bar
    class TQDMGridSearchCV(GridSearchCV):
        def fit(self, x, y=None, **fit_params):
            total_fits = (
                len(self.cv.split(x, y))
                * len(self.param_grid["n_estimators"])
                * len(self.param_grid["max_depth"])
            )
            with tqdm(
                total=total_fits, desc="Grid Search Progress", unit="fit"
            ) as pbar:
                self._original_fit = super().fit

                def progress_callback(*args, **kwargs):
                    pbar.update()

                self.fit_callback = progress_callback
                result = self._original_fit(x, y, **fit_params)
                return result

    # Initialize the grid search with TQDMGridSearchCV
    grid_search = TQDMGridSearchCV(
        estimator=base_model, param_grid=param_grid, cv=kfold, verbose=0
    )

    # Fit the model and show progress
    print("Fitting the model...")
    grid_search.fit(x, y)

    # Retrieve the best model
    best_model = grid_search.best_estimator_
    print(f"Best parameters: {grid_search.best_params_}")

    # Save the best model to a file
    model_filename = "dea_ml_ba_rf_with_landcover_rf_model.joblib"
    joblib.dump(best_model, model_filename)
    print(f"Best model saved to {model_filename}")

    helper.get_and_set_aws_credentials()
    bc_io.upload_object_to_s3(model_filename, training_model_url)
    logger.info(f"Uploaded result to: {training_model_url}")


if __name__ == "__main__":
    dea_rf_training()
