
import mlflow

import glob
import os
import time
import pickle

from colorama import Fore, Style

from tensorflow import keras
from tensorflow.keras import Model


def save_model(model: Model = None,
               params: dict = None,
               metrics: dict = None) -> None:
    """
    persist trained model, params and metrics
    """

    if os.environ.get("MODEL_TARGET") == "mlflow":

        print(Fore.BLUE + "\nSave model to mlflow..." + Style.RESET_ALL)

        # retrieve mlflow env params
        mlflow_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
        mlflow_experiment = os.environ.get("MLFLOW_EXPERIMENT")
        mlflow_model_name = os.environ.get("MLFLOW_MODEL_NAME")

        mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment(experiment_name=mlflow_experiment)

        with mlflow.start_run():

            # push parameters to mlflow
            if params is not None:
                for param_name, param in params.items():
                    mlflow.log_param(param_name, param)

            # push metrics to mlflow
            if metrics is not None:
                for metric_name, metric in metrics.items():
                    mlflow.log_metric(metric_name, metric)

            # push model to mlflow
            if model is not None:

                mlflow.keras.log_model(keras_model=model,
                                       artifact_path="model",
                                       keras_module="tensorflow.keras",
                                       registered_model_name=mlflow_model_name)

        print("\n✅ data saved to mlflow")

        return

    print(Fore.BLUE + "\nSave model to local disk..." + Style.RESET_ALL)

    suffix = time.strftime("%Y%m%d-%H%M%S")

    # save params
    if params is not None:
        params_path = os.path.join(os.environ.get("LOCAL_REGISTRY_PATH"), "params", suffix + ".pickle")

        print(f"- params path: {params_path}")

        with open(params_path, "wb") as file:
            pickle.dump(params, file)

    # save metrics
    if metrics is not None:
        metrics_path = os.path.join(os.environ.get("LOCAL_REGISTRY_PATH"), "metrics", suffix + ".pickle")

        print(f"- metrics path: {metrics_path}")

        with open(metrics_path, "wb") as file:
            pickle.dump(metrics, file)

    # save model
    if model is not None:
        model_path = os.path.join(os.environ.get("LOCAL_REGISTRY_PATH"), "models", suffix + ".pickle")

        print(f"- model path: {model_path}")

        model.save(model_path)

    print("\n✅ data saved locally")


def load_model(
    stage="None"
) -> Model:
    """
    load the latest saved model
    """

    if os.environ.get("MODEL_TARGET") == "mlflow":

        print(Fore.BLUE + "\nLoad model from mlflow..." + Style.RESET_ALL)

        # load model from mlflow
        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI"))

        mlflow_model_name = os.environ.get("MLFLOW_MODEL_NAME")

        model_uri = f"models:/{mlflow_model_name}/{stage}"
        print(f"- uri: {model_uri}")

        model = mlflow.keras.load_model(model_uri=model_uri)
        print("\n✅ model loaded from mlflow")

        # raise exception if no model exists
        if model is None:
            raise NameError(f"No {mlflow_model_name} model in {stage} stage stored in mlflow")

        return model

    print(Fore.BLUE + "\nLoad model from local disk..." + Style.RESET_ALL)

    # get latest model version
    model_directory = os.path.join(os.environ.get("LOCAL_REGISTRY_PATH"), "models")

    model_path = sorted(glob.glob(f"{model_directory}/*"))[-1]
    print(f"- path: {model_path}")

    model = keras.models.load_model(model_path)
    print("\n✅ model loaded from disk")

    return model
