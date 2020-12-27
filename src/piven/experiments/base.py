import json
import abc
from pathlib import Path
from typing import Callable, Dict
import pandas as pd
import numpy as np
from sklearn.exceptions import NotFittedError
from piven.utils import save_piven_model, load_piven_model


class PivenExperiment(metaclass=abc.ABCMeta):
    def __init__(self, **model_params):
        self.params = model_params
        self.model = None

    def __repr__(self):
        if self.model is None:
            return None
        else:
            return str(self.model)

    def fit(self, x: np.ndarray, y: np.ndarray, **kwargs):
        self.model.fit(x, y, **kwargs)

    def predict(self, x: np.ndarray, return_prediction_intervals: bool = True):
        if self.model is None:
            raise NotFittedError("Model has not yet been fitted.")
        return self.model.predict(x, return_prediction_intervals)

    def save(self, path: str):
        if self.model is None:
            raise NotFittedError("Model has not yet been fitted.")
        ppath = Path(path)
        with (ppath / "experiment_params.json").open("w") as outfile:
            json.dump(self.params, outfile)
        save_piven_model(self.model, path)
        return self

    def log(
        self, x: np.ndarray, y: np.ndarray, path: str, model=True, predictions=True
    ):
        """
        Log the results of an experiment to disk
        :param x:
        :param y:
        :param path: directory in which to save results.
        :param model: save model to disk if True.
        :param predictions: save predictions to disk if True.
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
            pred_out = np.column_stack([y, y_pred, y_pi_low, y_pi_high])
            df = pd.DataFrame(
                data=pred_out, columns=["y_true", "y_pred", "y_pi_low", "y_pi_high"]
            )
            df.to_csv((ppath / "predictions.csv"))
        return self

    @staticmethod
    def load_experiment_config(path: str):
        ppath = Path(path)
        if not (ppath / "experiment_params.json").is_file():
            raise FileNotFoundError(f"No experiment file found in {path}.")
        with (ppath / "experiment_params.json").open("r") as infile:
            params = json.load(infile)
        return params

    @staticmethod
    def load_model_from_disk(build_fn: Callable, path: str):
        return load_piven_model(build_fn, path)

    @abc.abstractmethod
    def build_model(self, **build_params):
        pass

    @classmethod
    @abc.abstractmethod
    def load(cls, path: str):
        pass

    @abc.abstractmethod
    def score(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pi_low: np.ndarray,
        y_pi_high: np.ndarray,
    ) -> Dict:
        pass
