import json
from pathlib import Path
import joblib
from typing import Union, Callable, Tuple
import tensorflow as tf
from piven.loss import piven_loss
from piven.metrics.tensorflow import mpiw, picp
from piven.layers import Piven
from piven.scikit_learn.wrappers import PivenKerasRegressor
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
    model: PivenKerasRegressor, path: Path, file_name: str = None
) -> str:
    """Save a keras model that is wrapped in a PivenKerasRegressor class"""
    if file_name is None:
        file_name = "piven_model.h5"
    tf.keras.models.save_model(model.model, path / file_name)
    with (path / "piven_model_config.json").open("w") as outfile:
        json.dump(model.sk_params, outfile)
    with (path / "piven_model_history.json").open("w") as outfile:
        # From np array --> list
        history = {k: list(v) for k, v in model.history.items()}
        # I'm making this explicit because I've not seen this in other cases, but:
        #  when using ReduceLROnPlateau, the learning rate is added to the history
        #  dictionary. The values are np.float32 and these are not serializable.
        #  I also don't want to cast everything to floats, so I'm doing this explicitly
        #  this will likely fail when using other callbacks that do something similar.
        if history.get("lr") is not None:
            history["lr"] = [float(v) for v in history["lr"]]
        json.dump(history, outfile)
    return str(path / "piven_model_config.json")


def _unset_sklearn_pipeline(obj: Pipeline, path: Path) -> Pipeline:
    """Unset a piven model from an sklearn pipeline and return the result"""
    for idx, step in enumerate(obj.steps):
        if isinstance(step[-1], PivenKerasRegressor):
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


def _get_piven_model(
    regressor: Union[Pipeline, PivenTransformedTargetRegressor]
) -> PivenKerasRegressor:
    """Extract piven regressor from pipeline or PivenTransformedTargetRegressor"""
    model_copy = None
    if isinstance(regressor, Pipeline):
        for step in regressor.steps:
            if isinstance(step[-1], PivenKerasRegressor):
                model_copy = step
    if isinstance(regressor, PivenTransformedTargetRegressor):
        for step in regressor.regressor_.steps:
            if isinstance(step[-1], PivenKerasRegressor):
                model_copy = step
    if model_copy is None:
        raise ValueError(
            f"Could not retrieve piven model from model with type {type(regressor)}."
        )
    return model_copy


def _set_piven_model(
    regressor: Union[Pipeline, PivenTransformedTargetRegressor],
    piven_model: Tuple[str, PivenKerasRegressor],
) -> Union[Pipeline, PivenTransformedTargetRegressor]:
    """Set piven regressor from pipeline or PivenTransformedTargetRegressor"""
    if isinstance(regressor, Pipeline):
        for idx, step in enumerate(regressor.steps):
            if step[-1] is None:
                regressor.steps[idx] = piven_model
    if isinstance(regressor, PivenTransformedTargetRegressor):
        for idx, step in enumerate(regressor.regressor_.steps):
            if step[-1] is None:
                regressor.regressor_.steps[idx] = piven_model
    return regressor


def save_piven_model(
    model: Union[PivenTransformedTargetRegressor, PivenKerasRegressor, Pipeline],
    path: str,
) -> Union[PivenTransformedTargetRegressor, PivenKerasRegressor, Pipeline]:
    """Save a piven model to a folder"""
    ppath = Path(path)
    if not ppath.is_dir():
        raise NotADirectoryError(f"Directory {path} does not exist.")
    if not isinstance(
        model, (PivenTransformedTargetRegressor, PivenKerasRegressor, Pipeline)
    ):
        raise TypeError(
            "Model must be of type 'Pipeline', 'PivenKerasRegressor'"
            + " or 'PivenTransformedTargetRegressor'"
        )
    # Model config
    config = {"type": str(type(model))}
    # Save config
    with (ppath / "config.json").open("w") as outfile:
        json.dump(config, outfile)
    # Dump model
    if isinstance(model, PivenTransformedTargetRegressor):
        # Get piven model
        model_copy = _get_piven_model(model)
        _save_piven_transformed_target_regressor(model, ppath, "piven_ttr.joblib")
        # Insert model back into pipeline
        _set_piven_model(model, model_copy)
        # Clone regressor (just for printing purposes)
        model.regressor = clone(model.regressor_)
    elif isinstance(model, Pipeline):
        # Get piven model
        model_copy = _get_piven_model(model)
        _save_model_pipeline(model, ppath, "piven_pipeline.joblib")
        # Insert model
        _set_piven_model(model, model_copy)
    elif isinstance(model, PivenKerasRegressor):
        _save_piven_model_wrapper(model, ppath, "piven_model.h5")
    else:
        raise ValueError(
            "Model must be of type 'Pipeline', 'PivenKerasRegressor'"
            + " or 'PivenTransformedTargetRegressor'"
        )
    return model


def _load_model_config(path: Path) -> dict:
    """Load PivenKerasRegressor model config"""
    with (path / "piven_model_config.json").open("r") as infile:
        return json.load(infile)


def _load_piven_model_wrapper(
    path: Path, build_fn: Callable, model_config: dict
) -> PivenKerasRegressor:
    """Load a keras model from disk and return a Piven wrapper"""
    model = tf.keras.models.load_model(path / "piven_model.h5")
    with (path / "piven_model_history.json").open("r") as infile:
        history = json.load(infile)
    pmw = PivenKerasRegressor(build_fn, **model_config)
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
    pmw_clone = PivenKerasRegressor(build_fn, **model_config)
    for idx, step in enumerate(ttr.regressor_.steps):
        if step[-1] is None:
            ttr.regressor_.steps[idx] = (step[0], pmw)
            ttr.regressor.steps[idx] = (step[0], pmw_clone)
    return ttr


def load_piven_model(
    build_fn: Callable, path: str
) -> Union[PivenTransformedTargetRegressor, PivenKerasRegressor, Pipeline]:
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
    elif mtype.endswith("PivenKerasRegressor"):
        m = _load_piven_model_wrapper(
            ppath, build_fn=build_fn, model_config=_load_model_config(ppath)
        )
    else:
        raise TypeError(f"Don't know how to load {mtype} from disk")
    return m
