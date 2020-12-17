from piven.wrappers import PivenModelWrapper
from piven.transformers import PivenTransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from piven.layers import Piven
from piven.loss import piven_loss
from piven.metrics import mpiw, picp


def test_piven_model_wrapper(mock_data):
    x_train, x_valid, y_train, y_valid = mock_data

    def keras_model_fn(input_shape, dropout_rate):
        i = tf.keras.layers.Input(shape=(input_shape,))
        x = tf.keras.layers.Dense(128)(i)
        x = tf.keras.layers.Dropout(dropout_rate)(x)
        o = Piven()(x)
        m = tf.keras.models.Model(inputs=i, outputs=[o])
        m.compile(
            optimizer="adam",
            loss=piven_loss(True, 15.0, 160.0, 0.05),
            metrics=[picp, mpiw],
        )
        return m

    # Wrap the model in a PivenModelWrapper
    pmw = PivenModelWrapper(
        build_fn=keras_model_fn, input_shape=x_train.shape[-1], dropout_rate=0.1
    )
    # Wrap the piven model in a pipeline
    model_pipeline = Pipeline([("scaler", StandardScaler()), ("model", pmw)])
    # Wrap the pipeline in a transformed target regressor
    regressor = PivenTransformedTargetRegressor(
        regressor=model_pipeline, transformer=StandardScaler()
    )
    # Fit regressor
    regressor.fit(x_train, y_train[:, 0], model__epochs=100, model__verbose=False)
    # Predict
    yhat = regressor.predict(x_valid, return_prediction_intervals=True)
    return yhat
