import json
import abc
from pathlib import Path
from typing import Callable, Dict, Union
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error as mae, mean_squared_error as mse
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline
from piven.metrics.numpy import coverage, pi_width, piven_loss as piven_loss_numpy
from piven.utils import save_piven_model, load_piven_model
from piven.scikit_learn.compose import PivenTransformedTargetRegressor
from piven.scikit_learn.wrappers import PivenKerasRegressor


# Helper
def _ifelse(condition, val_if_true, val_if_false):
    if condition:
        return val_if_true
    else:
        return val_if_false


class PivenBaseModel(metaclass=abc.ABCMeta):
    def __init__(self, **model_params):
        self.params = model_params
        self.model = None

    def __repr__(self):
        if self.model is None:
            return None
        else:
            return str(self.model)

    def fit(self, x: np.ndarray, y: np.ndarray, **kwargs):
        """
        Fit the piven model to data

        Parameters
        ----------
        x input data
        y targets
        kwargs any arguments passed to keras 'fit()' method

        Returns python class of model
        -------

        """
        self.model.fit(x, y, **kwargs)
        return self

    def predict(self, x: np.ndarray, return_prediction_intervals: bool = True):
        """
        Predict on (unseen) data

        Parameters
        ----------
        x input data
        return_prediction_intervals flag indicating whether to return lower & upper prediction intervals

        Returns either a numpy array with model predictions or a 3-dimensional array containing lower & upper
                 PI and point estimates.
        -------

        """
        if self.model is None:
            raise NotFittedError("Model has not yet been fitted.")
        return self.model.predict(x, return_prediction_intervals)

    def save(self, path: str):
        """
        Persist the model to disk

        Parameters
        ----------
        path directory in which the model should be saved.

        Returns class instance
        -------

        """
        if self.model is None:
            raise NotFittedError("Model has not yet been fitted.")
        with (Path(path) / "experiment_params.json").open("w") as outfile:
            json.dump(self.params, outfile)
        save_piven_model(self.model, path)
        return self

    def log(
        self, x: np.ndarray, y: np.ndarray, path: str, model=True, predictions=True
    ):
        """
        Log the results of the model and optionally the model and predictions to disk.

        Parameters
        ----------
        x input data
        y targets
        path directory in which to save the model, predictions and metrics.
        model flag indicating whether to persist the model on disk.
        predictions flag indicating whether to persist the predictions on disk.

        Returns class instance
        -------

        """
        ppath = Path(path)
        if not ppath.is_dir():
            raise NotADirectoryError(f"Path {path} is not a directory.")
        y_pred, y_pi_low, y_pi_high = self.predict(x, return_prediction_intervals=True)
        metrics = self.score(y, y_pred, y_pi_low, y_pi_high)
        with (ppath / "metrics.json").open("w") as outfile:
            json.dump(metrics, outfile)
        if model:
            mpath = ppath / "model"
            mpath.mkdir()
            self.save(str(mpath))
        if predictions:
            df = pd.DataFrame(
                data=np.column_stack([y, y_pred, y_pi_low, y_pi_high]),
                columns=["y_true", "y_pred", "y_pi_low", "y_pi_high"],
            )
            df.to_csv((ppath / "predictions.csv"))
        return self

    def score(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pi_low: np.ndarray,
        y_pi_high: np.ndarray,
    ) -> Dict:
        """
        Score the model on MAE, RMSE, % coverage and PI width.

        Parameters
        ----------
        y_true true target vaules
        y_pred predicted target values
        y_pi_low lower pi
        y_pi_high upper pi

        Returns dict containing metrics
        -------

        """
        return {
            "loss": piven_loss_numpy(
                y_true,
                y_pred,
                y_pi_low,
                y_pi_high,
                _ifelse(
                    self.params.get("lambda_") is None, 25.0, self.params.get("lambda_")
                ),
                _ifelse(
                    self.params.get("soften") is None, 160.0, self.params.get("soften")
                ),
                _ifelse(
                    self.params.get("alpha") is None, 0.05, self.params.get("alpha")
                ),
            ),
            "mae": mae(y_true, y_pred),
            "rmse": np.sqrt(mse(y_true, y_pred)),
            "coverage": coverage(y_true, y_pi_low, y_pi_high),
            "pi_width": pi_width(y_pi_low, y_pi_high),
        }

    @staticmethod
    def _load_model_config(path: str) -> Dict:
        if not (Path(path) / "experiment_params.json").is_file():
            raise FileNotFoundError(f"No experiment file found in {path}.")
        with (Path(path) / "experiment_params.json").open("r") as infile:
            params = json.load(infile)
        return params

    @staticmethod
    def _load_model_from_disk(
        build_fn: Callable, path: str
    ) -> Union[PivenTransformedTargetRegressor, PivenKerasRegressor, Pipeline]:
        return load_piven_model(build_fn, path)

    @classmethod
    def load(cls, path: str, build_fn: Callable):
        """
        Load a piven-based model from disk

        Parameters
        ----------
        path directory in which the model files were saved
        build_fn build function used to construct and compile the model

        Returns piven-based model
        -------

        """
        model_config = cls._load_model_config(path)
        model = cls._load_model_from_disk(build_fn, path)
        run = cls(**model_config)
        run.model = model
        return model

    @abc.abstractmethod
    def build(self, build_fn: Callable, **build_params):
        pass
