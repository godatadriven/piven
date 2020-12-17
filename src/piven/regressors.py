from typing import Tuple
from piven.layers import Piven
import tensorflow as tf


def build_keras_piven(
    input_dim: int,
    dense_units: Tuple[int, ...] = (64,),
    dropout_rate: Tuple[float, ...] = (0.1,),
    activation: str = "relu",
):  # -> tf.keras.engine.functional.Functional:
    """Create a PIVEN model"""
    input_variable = tf.keras.layers.Input(shape=(input_dim,))
    x = tf.keras.layers.Dense(
        dense_units[0],
        activation=activation,
        kernel_initializer=tf.keras.initializers.RandomNormal(0, 0.2),
        name="dense1",
    )(input_variable)
    if dropout_rate[0] > 0:
        x = tf.keras.layers.Dropout(dropout_rate[0])(x)
    for layer_idx, values in enumerate(zip(dense_units[1:], dropout_rate[1:])):
        units, dropout_rate_cur = values
        x = tf.keras.layers.Dense(
            units,
            activation=activation,
            kernel_initializer=tf.keras.initializers.RandomNormal(0, 0.2),
            name=f"dense{layer_idx+2}",
        )(x)
        if dropout_rate_cur > 0:
            x = tf.keras.layers.Dropout(dropout_rate_cur)(x)
    output = Piven()(x)
    model = tf.keras.models.Model(inputs=input_variable, outputs=[output], name="PIVEN")
    return model
