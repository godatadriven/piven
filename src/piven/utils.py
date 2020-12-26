import json
from pathlib import Path
import joblib
from typing import Union, Callable
import tensorflow as tf
from piven.loss import piven_loss
from piven.metrics.tensorflow import mpiw, picp
from piven.layers import Piven
from piven.scikit_learn.wrappers import PivenRegressor
from piven.scikit_learn.compose import PivenTransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone

# Dump custom metrics, loss and layers
# Need to do this when saving models.
tf.keras.utils.get_custom_objects().update(
    {"picp": picp, "mpiw": mpiw, "piven_loss": piven_loss, "Piven": Piven}
)


def _save_piven_model_wrapper(
    model: PivenRegressor, path: Path, file_name: str = None
) -> str:
    """Save a keras model that is wrapped in a PivenModelWrapper class"""
    if file_name is None:
        file_name = "piven_model.h5"
    tf.keras.models.save_model(model.model, path / file_name)
    with (path / "piven_model_config.json").open("w") as outfile:
        json.dump(model.sk_params, outfile)
    with (path / "piven_model_history.json").open("w") as outfile:
        json.dump(model.history, outfile)
    return str(path / "piven_model_config.json")


def _unset_sklearn_pipeline(obj: Pipeline, path: Path) -> Pipeline:
    """Unset a piven model from an sklearn pipeline and return the result"""
    for idx, step in enumerate(obj.steps):
        if isinstance(step[-1], PivenRegressor):
            _ = _save_piven_model_wrapper(step[-1], path=path)
            obj.steps[idx] = (step[0], None)
    return obj


def _save_model_pipeline(
    obj: Union[Pipeline, StandardScaler], path: Path, file_name: str
) -> str:
    """Save an sklearn pipeline"""
    # If piven model in pipeline
    if isinstance(obj, Pipeline):
        obj = _unset_sklearn_pipeline(obj, path=path)
    joblib.dump(obj, path / file_name)
    return str(path / file_name)


def _save_piven_transformed_target_regressor(
    regressor: PivenTransformedTargetRegressor, path: Path, file_name: str
) -> str:
    """Save a piven transformed target regressor"""
    regressor.regressor = None
    regressor.regressor_ = _unset_sklearn_pipeline(regressor.regressor_, path=path)
    joblib.dump(regressor, path / file_name)
    return str(path / file_name)


def save_piven_model(
    model: Union[PivenTransformedTargetRegressor, PivenRegressor, Pipeline], path: str
) -> str:
    """Save a piven model to a folder"""
    ppath = Path(path)
    if not ppath.is_dir():
        raise NotADirectoryError(f"Directory {path} does not exist.")
    if not isinstance(
        model, (PivenTransformedTargetRegressor, PivenRegressor, Pipeline)
    ):
        raise TypeError(
            "Model must be of type 'Pipeline', 'PivenModelWrapper'"
            + " or 'PivenTransformedTargetRegressor'"
        )
    # Model config
    config = {"type": str(type(model))}
    # Save config
    with (ppath / "config.json").open("w") as outfile:
        json.dump(config, outfile)
    # Dump model
    if isinstance(model, PivenTransformedTargetRegressor):
        _save_piven_transformed_target_regressor(model, ppath, "piven_ttr.joblib")
    elif isinstance(model, Pipeline):
        _save_model_pipeline(model, ppath, "piven_pipeline.joblib")
    elif isinstance(model, PivenRegressor):
        _save_piven_model_wrapper(model, ppath, "piven_model.h5")
    else:
        raise ValueError(
            "Model must be of type 'Pipeline', 'PivenModelWrapper'"
            + " or 'PivenTransformedTargetRegressor'"
        )
    # Return path
    return path


def _load_model_config(path: Path) -> dict:
    """Load PivenModelWrapper model config"""
    with (path / "piven_model_config.json").open("r") as infile:
        return json.load(infile)


def _load_piven_model_wrapper(
    path: Path, build_fn: Callable, model_config: dict
) -> PivenRegressor:
    """Load a keras model from disk and return a Piven wrapper"""
    model = tf.keras.models.load_model(path / "piven_model.h5")
    with (path / "piven_model_history.json").open("r") as infile:
        history = json.load(infile)
    pmw = PivenRegressor(build_fn, **model_config)
    pmw.history = history
    pmw.model = model
    return pmw


def _load_sklearn_pipeline(path: Path, build_fn: Callable) -> Pipeline:
    """Load a sklearn pipeline from disk"""
    pipeline = joblib.load(path / "piven_pipeline.joblib")
    model = _load_piven_model_wrapper(
        path, build_fn=build_fn, model_config=_load_model_config(path)
    )
    for idx, step in enumerate(pipeline.steps):
        if step[-1] is None:
            pipeline.steps[idx] = (step[0], model)
    return pipeline


def _load_piven_transformed_target_regressor(
    path: Path, build_fn: Callable
) -> PivenTransformedTargetRegressor:
    """Load a transformed target regressor from disk"""
    ttr = joblib.load(path / "piven_ttr.joblib")
    ttr.regressor = clone(ttr.regressor_)
    model_config = _load_model_config(path)
    pmw = _load_piven_model_wrapper(path, build_fn=build_fn, model_config=model_config)
    pmw_clone = PivenRegressor(build_fn, **model_config)
    for idx, step in enumerate(ttr.regressor_.steps):
        if step[-1] is None:
            ttr.regressor_.steps[idx] = (step[0], pmw)
            ttr.regressor.steps[idx] = (step[0], pmw_clone)
    return ttr


def load_piven_model(
    build_fn: Callable, path: str
) -> Union[PivenTransformedTargetRegressor, PivenRegressor, Pipeline]:
    """Load a piven model from disk"""
    ppath = Path(path)
    if not ppath.is_dir():
        raise NotADirectoryError(f"Directory {path} does not exist.")
    with (ppath / "config.json").open("r") as infile:
        config = json.load(infile)
        mtype = config.get("type").strip("'>")
    if mtype.endswith("PivenTransformedTargetRegressor"):
        m = _load_piven_transformed_target_regressor(ppath, build_fn=build_fn)
    elif mtype.endswith("Pipeline"):
        m = _load_sklearn_pipeline(ppath, build_fn=build_fn)
    elif mtype.endswith("PivenRegressor"):
        m = _load_piven_model_wrapper(
            ppath, build_fn=build_fn, model_config=_load_model_config(ppath)
        )
    else:
        raise TypeError(f"Don't know how to load {mtype} from disk")
    return m
