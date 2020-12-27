from pathlib import Path
from typing import Callable
import numpy as np
from sklearn.exceptions import NotFittedError
import json
import abc
from piven.utils import save_piven_model, load_piven_model


class PivenExperiment(metaclass=abc.ABCMeta):
    def __init__(self, **model_params):
        self.params = model_params
        self.model = None

    @abc.abstractmethod
    def build_model(self, **build_params):
        pass

    def fit(self, x: np.ndarray, y: np.ndarray, **kwargs):
        self.model.fit(x, y, **kwargs)

    def save(self, path: str):
        if self.model is None:
            raise NotFittedError("Model has not yet been fitted.")
        ppath = Path(path)
        with (ppath / "experiment_params.json").open("w") as outfile:
            json.dump(self.params, outfile)
        save_piven_model(self.model, path)

    @classmethod
    def load(cls, build_fn: Callable, path: str):
        ppath = Path(path)
        if not (ppath / "experiment_params.json").is_file():
            raise FileNotFoundError(f"No experiment file found in {path}.")
        with (ppath / "experiment_params.json").open("r") as infile:
            params = json.load(infile)
        model = load_piven_model(build_fn, path)
        run = cls(**params)
        run.model = model
        return run

    def predict(self, *args, **kwargs):
        if self.model is None:
            raise NotFittedError("Model has not yet been fitted.")
        return self.model.predict(*args, **kwargs)

    @abc.abstractmethod
    def score(self, *args, **kwargs):
        pass
