import tensorflow as tf
from piven.layers import Piven
from typing import Tuple


def piven_loss(eli, lambda_in=15.0, soften=160.0, alpha=0.05):
    # define loss fn
    def piven_loss(y_true, y_pred):
        y_u = y_pred[:, 0]
        y_l = y_pred[:, 1]
        y_t = y_true[:, 0]
        if eli:
            y_v = y_pred[:, 2]
        n_ = tf.cast(tf.size(y_t), tf.float32)  # batch size
        alpha_ = tf.constant(alpha)
        lambda_ = tf.constant(lambda_in)
        # soft uses sigmoid
        k_soft = tf.multiply(
            tf.sigmoid((y_u - y_t) * soften), tf.sigmoid((y_t - y_l) * soften)
        )
        # hard uses sign step function
        k_hard = tf.multiply(
            tf.maximum(0.0, tf.sign(y_u - y_t)), tf.maximum(0.0, tf.sign(y_t - y_l))
        )
        mpiw_capt = tf.divide(
            tf.reduce_sum(tf.abs(y_u - y_l) * k_hard), tf.reduce_sum(k_hard) + 0.001
        )
        picp_soft = tf.reduce_mean(k_soft)
        qd_rhs_soft = (
            lambda_
            * tf.sqrt(n_)
            * tf.square(tf.maximum(0.0, (1.0 - alpha_) - picp_soft))
        )
        piven_loss_ = mpiw_capt + qd_rhs_soft  # final qd loss form
        if eli:
            y_eli = y_v * y_u + (1 - y_v) * y_l
            y_eli = tf.reshape(y_eli, (-1, 1))
            piven_loss_ += tf.losses.mean_squared_error(y_true, y_eli)
        return piven_loss_

    return piven_loss


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
