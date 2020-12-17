# from pathlib import Path
# import joblib
from typing import Union
import tensorflow as tf
from piven.loss import piven_loss
from piven.metrics import picp, mpiw
from piven.layers import Piven
from piven.wrappers import PivenModelWrapper
from piven.transformers import PivenTransformedTargetRegressor
from tensorflow.python.keras.engine.functional import Functional

# from sklearn.pipeline import Pipeline

# Dump custom metrics, loss and layers
# Need to do this when saving models.
tf.keras.utils.get_custom_objects().update(
    {"picp": picp, "mpiw": mpiw, "piven_loss": piven_loss, "Piven": Piven}
)


def save_piven_model(
    model: Union[PivenTransformedTargetRegressor, PivenModelWrapper, Functional] = None,
    path: str = None,
):
    # Record type and store in config
    # Check path is valid
    # Store:
    #  1. Keras model
    #  2. PivenModelWrapper --> params
    #  3. TransformedTargetRegressor --> Save Scaler
    #  4. Pipeline --> save preprocessing steps
    pass


def load_piven_model(build_fn=None, path: str = None):
    pass
