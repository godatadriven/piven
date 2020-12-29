import tensorflow as tf
from tensorflow import keras


class Piven(keras.layers.Layer):
    def __init__(self, init_pi_lower=-3.0, init_pi_upper=3.0):
        """
        Piven keras layer

        Adapted from:

            Simhayev, Eli, Gilad Katz, and Lior Rokach. "PIVEN: A Deep
            Neural Network for Prediction Intervals with Specific Value
            Prediction." arXiv preprint arXiv:2006.05139 (2020).
        """
        if init_pi_lower >= init_pi_upper:
            raise ValueError(
                "Lower PI initial value cannot be larger"
                + "than upper PI initial value"
            )
        super(Piven, self).__init__()
        # Add PIVEN layers
        self.init_pi_upper = init_pi_upper
        self.init_pi_lower = init_pi_lower
        self.pi = tf.keras.layers.Dense(
            2,
            activation="linear",
            kernel_initializer=tf.keras.initializers.GlorotUniform(),
            bias_initializer=tf.keras.initializers.Constant(
                value=[init_pi_upper, init_pi_lower]
            ),
            name="pi",
        )
        self.v = tf.keras.layers.Dense(
            1,
            activation="sigmoid",
            name="v",
            kernel_initializer=tf.keras.initializers.GlorotNormal(),
        )
        self.output_layer = tf.keras.layers.Concatenate(name="output")

    def call(self, inputs) -> tf.keras.layers.Concatenate:
        pi = self.pi(inputs)
        v = self.v(inputs)
        return self.output_layer([pi, v])

    def get_config(self) -> dict:
        return {
            "init_pi_upper": self.init_pi_upper,
            "init_pi_lower": self.init_pi_lower,
        }
